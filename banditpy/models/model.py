import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from banditpy.core import Bandit2Arm
from .policy.base import BasePolicy
from tqdm import tqdm
import os


def softmax_loglik(logits, choice, beta):
    z = beta * logits
    return z[choice] - logsumexp(z)


def softmax_sample(logits, beta, rng):
    z = beta * logits
    p = np.exp(z - logsumexp(z))
    return rng.choice(len(p), p=p)


def _get_slurm_cpus(default=1):
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return default


class DecisionModel:
    def __init__(self, task: Bandit2Arm, policy: BasePolicy, reset_mode="session"):
        self.task = task
        self.policy = policy

        self.reset_mode = reset_mode
        self.resets = self._compute_resets(task, reset_mode)

        self.choices = np.asarray(task.choices, int) - 1
        self.rewards = np.asarray(task.rewards, float)

        self.nll = None
        self.params = None
        self.fit_fvals = None
        self.fit_fval_mean = None
        self.fit_fval_std = None

    # -------------------- RESET MODE --------------------

    def _compute_resets(self, task, reset_mode):
        """
        reset_mode options:
            "session"  -> task.is_session_start
            "block"    -> task.is_block_start
            "window"   -> task.is_window_start
            ndarray    -> boolean/binary mask length == n_trials
        """

        # ---------- 1) ARRAY CASE FIRST ----------
        # (Supports bool, int {0,1}, numpy, list)
        if hasattr(reset_mode, "__array__") or isinstance(reset_mode, (list, tuple)):
            mask = np.asarray(reset_mode)

            if mask.shape[0] != task.n_trials:
                raise ValueError(
                    f"Custom reset mask must have length {task.n_trials}, "
                    f"got {mask.shape[0]}"
                )

            # allow bool or {0,1}
            if mask.dtype != bool:
                uniq = np.unique(mask)
                if not np.all(np.isin(uniq, (0, 1))):
                    raise ValueError(
                        "Custom reset mask must be boolean or contain only {0,1}"
                    )
                mask = mask.astype(bool)

            return mask

        # ---------- 2) SYMBOLIC RESET MODES ----------
        match reset_mode:
            case "session":
                return task.is_session_start
            case "block":
                return task.is_block_start
            case "window":
                return task.is_window_start
            case _:
                raise ValueError(
                    "reset_mode must be 'session', 'block', 'window', "
                    "or a boolean/0-1 mask array"
                )

    # -------------------- NLL --------------------

    def _nll(self, theta):
        names = self.policy.param_names()
        params = dict(zip(names, theta))

        self.policy.set_params(params)

        # reset after params are set so policies that read from self.params in reset don't KeyError
        self.policy.reset()

        beta = params["beta"]
        nll = 0.0

        for c, r, reset in zip(self.choices, self.rewards, self.resets):
            if reset:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            nll -= softmax_loglik(logits, c, beta)
            self.policy.update(c, r)

        return nll

    # -------------------- FIT --------------------

    def fit(
        self,
        n_starts=10,
        seed=None,
        progress=False,
        n_jobs=None,
        method="lbfgs",
        de_popsize=15,
        de_maxiter=200,
        de_tol=1e-6,
    ):
        from joblib import Parallel, delayed

        rng = np.random.default_rng(seed)

        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        n_jobs = max(1, min(n_jobs, n_starts))

        print(f"Using {n_jobs} workers")

        bounds_dict = self.policy.get_bounds()
        names = self.policy.param_names()
        bounds = [bounds_dict[n] for n in names]

        seeds = rng.integers(0, 2**32 - 1, size=n_starts)

        def _run_single(start_seed):
            local_rng = np.random.default_rng(start_seed)

            if method.lower() == "de":
                from scipy.optimize import differential_evolution

                res = differential_evolution(
                    self._nll,
                    bounds=bounds,
                    maxiter=de_maxiter,
                    popsize=de_popsize,
                    tol=de_tol,
                    seed=int(local_rng.integers(0, 2**32 - 1)),
                    polish=True,
                )
                return res.fun, res.x

            x0 = np.array([local_rng.uniform(*b) for b in bounds])
            res = minimize(
                self._nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            return res.fun, res.x

        iterator = seeds
        if progress:
            try:
                iterator = tqdm(iterator, desc="Fitting DecisionModel")
            except Exception:
                pass

        if n_jobs == 1:
            results = [_run_single(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_single)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)

        self.params = dict(zip(names, best_x))
        self.nll = best_fun
        self.fit_fvals = fvals
        self.fit_fval_mean = float(fvals.mean())
        self.fit_fval_std = float(fvals.std())

        self.policy.set_params(self.params)

    # -------------------- POSTERIOR PREDICTIVE --------------------

    def simulate_posterior_predictive(self, seed=None) -> Bandit2Arm:
        if self.params is None:
            raise RuntimeError("Model must be fit before simulation.")

        rng = np.random.default_rng(seed)

        self.policy.set_params(self.params)
        self.policy.reset()

        task = self.task
        n_trials = task.n_trials

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            if self.resets[t]:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            c = softmax_sample(logits, beta=self.params["beta"], rng=rng)

            p = task.probs[t, c]
            r = int(rng.random() < p)

            self.policy.update(c, r)

            choices[t] = c + 1
            rewards[t] = r

        return Bandit2Arm(
            probs=task.probs.copy(),
            choices=choices,
            rewards=rewards,
            session_ids=task.session_ids.copy(),
            block_ids=None if task.block_ids is None else task.block_ids.copy(),
            window_ids=None if task.window_ids is None else task.window_ids.copy(),
            starts=None if task.starts is None else task.starts.copy(),
            stops=None if task.stops is None else task.stops.copy(),
            datetime=None if task.datetime is None else task.datetime.copy(),
            metadata=task.metadata,
        )

    # -------------------- POLICY SIMULATION --------------------

    @classmethod
    def simulate_policy(
        cls,
        policy,
        reward_schedule,
        trials_per_block,
        params,
        seed=None,
        metadata=None,
    ):
        rng = np.random.default_rng(seed)

        policy.set_params(params)
        policy.reset()

        if isinstance(trials_per_block, int):
            trials_per_block = [trials_per_block] * len(reward_schedule)

        assert len(trials_per_block) == len(reward_schedule)

        probs_list, choices, rewards = [], [], []
        session_ids, block_ids = [], []

        session_counter = 1
        block_counter = 1

        for (p1, p2), n_trials in zip(reward_schedule, trials_per_block):
            block_probs = np.tile([p1, p2], (n_trials, 1))

            for t in range(n_trials):
                logits = policy.logits()
                c = softmax_sample(logits, beta=params["beta"], rng=rng)
                r = int(rng.random() < block_probs[t, c])

                policy.update(c, r)

                probs_list.append(block_probs[t])
                choices.append(c + 1)
                rewards.append(r)
                session_ids.append(session_counter)
                block_ids.append(block_counter)

            session_counter += 1
            block_counter += 1
            policy.reset()

        return Bandit2Arm(
            probs=np.asarray(probs_list),
            choices=np.asarray(choices),
            rewards=np.asarray(rewards),
            session_ids=np.asarray(session_ids),
            block_ids=np.asarray(block_ids),
            window_ids=None,
            starts=None,
            stops=None,
            datetime=None,
            metadata=metadata,
        )

    # -------------------- GREEDY SIM --------------------

    def simulate_greedy(self):
        self.policy.reset()
        choices = []

        for c, r, reset in zip(self.task.choices, self.task.rewards, self.resets):
            if reset:
                self.policy.reset()

            logits = self.policy.logits()
            choice = np.argmax(logits)
            choices.append(choice + 1)

            self.policy.update(choice, r)

        return np.array(choices)

    # -------------------- METRICS / OUTPUT --------------------

    def bic(self):
        if self.nll is None:
            raise RuntimeError("Model must be fit before computing BIC.")
        k = len(self.params)
        n = len(self.choices)
        return k * np.log(n) + 2.0 * self.nll

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return

        print("Fitted parameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v:.4f}")
        print(f"NLL: {self.nll:.2f}")
        print(f"BIC: {self.bic():.2f}")
        if self.fit_fval_mean is not None:
            print(
                f"Restart NLL mean±SD: "
                f"{self.fit_fval_mean:.3f} ± {self.fit_fval_std:.3f}"
            )

    def describe(self):
        if self.params is None:
            print("Model is not fit yet. Call fit() first.")
            return

        specs = self.policy.parameter_specs()
        names = self.policy.param_names()

        print("\nParameters")
        print("-" * 80)

        # Column layout
        # name (14) | value (10) | bounds (18) | description (rest)
        header = f"{'name':<14} {'fitted':>10}   {'bounds':<18} description"
        print(header)
        print("-" * 80)

        for name in names:
            v = self.params[name]
            b = specs[name].bounds
            desc = specs[name].description or ""

            bounds_str = f"({b[0]:.4g}, {b[1]:.4g})"

            print(f"{name:<14} " f"{v:>10.4f}   " f"{bounds_str:<18}" f"{desc}")

        print("-" * 80)
        print(f"NLL: {self.nll:.3f}")
        print(f"BIC: {self.bic():.3f}")

    def to_dict(self):
        if self.params is None:
            raise RuntimeError("Model must be fit before calling to_dict().")

        out = dict(self.params)
        out.update(
            dict(
                nll=float(self.nll),
                bic=float(self.bic()),
                n_trials=int(len(self.choices)),
                fit_fval_mean=(
                    None if self.fit_fval_mean is None else float(self.fit_fval_mean)
                ),
                fit_fval_std=(
                    None if self.fit_fval_std is None else float(self.fit_fval_std)
                ),
                policy_type=self.policy.__class__.__name__,
            )
        )
        return out

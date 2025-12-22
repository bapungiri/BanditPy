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
    """
    Determine number of CPUs available under Slurm.
    Falls back to default if not running under Slurm.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return default


class DecisionModel:
    def __init__(self, task: Bandit2Arm, policy: BasePolicy):
        self.task = task
        self.policy = policy
        self.nll = None
        self.params = None
        self.fit_fvals = None  # all restart objectives
        self.fit_fval_mean = None
        self.fit_fval_std = None

        self.choices = np.asarray(task.choices, int) - 1
        self.rewards = np.asarray(task.rewards, float)
        self.resets = task.is_session_start.astype(bool)

    def _nll(self, theta):
        names = self.policy.param_names()
        params = dict(zip(names, theta))
        self.policy.set_params(params)

        nll = 0.0
        for c, r, reset in zip(self.choices, self.rewards, self.resets):
            if reset:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            nll -= softmax_loglik(logits, c, params["beta"])
            self.policy.update(c, r)

        return nll

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
        """
        Slurm-friendly multi-start optimization.

        Parameters
        ----------
        n_starts : int
            Number of random restarts
        seed : int or None
            RNG seed
        progress : bool
            Show progress bar (recommended only for interactive use)
        n_jobs : int or None
            Number of parallel workers.
            If None, auto-detects from Slurm or defaults to 1.
        """

        from joblib import Parallel, delayed

        rng = np.random.default_rng(seed)

        # ---- Slurm-aware ----
        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)

        # Avoid pathological cases
        n_jobs = max(1, min(n_jobs, n_starts))

        bounds = self.policy.bounds()
        names = self.policy.param_names()

        # Independent RNG seeds per restart
        seeds = rng.integers(0, 2**32 - 1, size=n_starts)

        def _run_single(start_seed):
            local_rng = np.random.default_rng(start_seed)

            if method.lower() == "de":
                res = minimize  # placeholder to keep scope clear
                from scipy.optimize import differential_evolution

                res = differential_evolution(
                    self._nll,
                    bounds=bounds,
                    maxiter=de_maxiter,
                    popsize=de_popsize,
                    tol=de_tol,
                    seed=int(local_rng.integers(0, 2**32 - 1)),
                    polish=True,
                    # workers=5,
                )
                return res.fun, res.x
            else:
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
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Fitting DecisionModel")
            except ImportError:
                pass

        if n_jobs == 1:
            results = [_run_single(s) for s in iterator]
        else:
            # Process-based parallelism (safe under Slurm)
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
            )(delayed(_run_single)(s) for s in iterator)

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)

        self.params = dict(zip(names, best_x))
        self.nll = best_fun
        self.fit_fvals = fvals
        self.fit_fval_mean = float(fvals.mean())
        self.fit_fval_std = float(fvals.std())
        self.policy.set_params(self.params)

    def simulate_posterior_predictive(self, seed=None) -> Bandit2Arm:
        """
        Posterior predictive simulation using fitted parameters
        and the original Bandit2Arm task structure.
        """
        if self.params is None:
            raise RuntimeError(
                "Model must be fit before posterior predictive simulation."
            )

        rng = np.random.default_rng(seed)

        # Reset policy with fitted parameters
        self.policy.set_params(self.params)
        self.policy.reset()

        task = self.task
        n_trials = task.n_trials
        n_arms = task.n_ports

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)

        # Use task-defined resets (sessions by default)
        resets = task.is_session_start

        for t in range(n_trials):
            if resets[t]:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()

            # sample choice via softmax
            c = softmax_sample(
                logits,
                beta=self.params["beta"],
                rng=rng,
            )

            # trial-specific reward probability
            p = task.probs[t, c]
            r = int(rng.random() < p)

            self.policy.update(c, r)

            choices[t] = c + 1  # back to 1-based
            rewards[t] = r

        # Reconstruct a new Bandit2Arm with identical structure
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
        """
        Simulate a policy learning in a synthetic Bandit2Arm task.

        Parameters
        ----------
        policy : BasePolicy
            Policy instance (e.g., EmpiricalUCB, RLUCB, BayesianUCB).
        reward_schedule : list of tuple(float, float)
            List of (p_arm1, p_arm2) for each block.
        trials_per_block : int or list[int]
            Number of trials per block. If int, same for all blocks.
        params : dict
            Policy parameters.
        seed : int, optional
            RNG seed.
        metadata : dict, optional
            Metadata passed to Bandit2Arm.

        Returns
        -------
        Bandit2Arm
            Simulated task.
        """

        rng = np.random.default_rng(seed)

        policy.set_params(params)
        policy.reset()

        # Normalize trials_per_block
        if isinstance(trials_per_block, int):
            trials_per_block = [trials_per_block] * len(reward_schedule)

        assert len(trials_per_block) == len(
            reward_schedule
        ), "trials_per_block must match reward_schedule length"

        probs_list = []
        choices = []
        rewards = []
        session_ids = []
        block_ids = []

        session_counter = 1
        block_counter = 1

        for (p1, p2), n_trials in zip(reward_schedule, trials_per_block):
            # Build per-trial probability matrix for this block
            block_probs = np.tile([p1, p2], (n_trials, 1))

            for t in range(n_trials):
                logits = policy.logits()
                c = softmax_sample(
                    logits,
                    beta=params["beta"],
                    rng=rng,
                )

                r = int(rng.random() < block_probs[t, c])

                policy.update(c, r)

                probs_list.append(block_probs[t])
                choices.append(c + 1)  # 1-based
                rewards.append(r)
                session_ids.append(session_counter)
                block_ids.append(block_counter)

            # New session for each block (consistent with your semantics)
            session_counter += 1
            block_counter += 1
            policy.reset()

        probs = np.asarray(probs_list)
        choices = np.asarray(choices)
        rewards = np.asarray(rewards)
        session_ids = np.asarray(session_ids)
        block_ids = np.asarray(block_ids)

        return Bandit2Arm(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            block_ids=block_ids,
            window_ids=None,
            starts=None,
            stops=None,
            datetime=None,
            metadata=metadata,
        )

    def simulate_greedy(self):
        self.policy.reset()
        choices = []

        for c, r, reset in zip(
            self.task.choices, self.task.rewards, self.task.is_session_start
        ):
            if reset:
                self.policy.reset()

            logits = self.policy.logits()
            choice = np.argmax(logits)
            choices.append(choice + 1)

            self.policy.update(choice, r)

        return np.array(choices)

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
                f"Restart NLL mean±SD: {self.fit_fval_mean:.3f} ± {self.fit_fval_std:.3f}"
            )

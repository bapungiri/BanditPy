import numpy as np
import pandas as pd
from scipy.special import logsumexp
from banditpy.core import Bandit2Arm
from .policy.base import BasePolicy
from tqdm import tqdm
import os
from .optim import resolve_optimizer


def _softmax_probs(logits, beta, epsilon=0.0):
    z = beta * logits
    p = np.exp(z - logsumexp(z))
    if epsilon > 0:
        p = (1 - epsilon) * p + epsilon / len(p)
    return p


def softmax_loglik(logits, choice, beta, epsilon=0.0):
    p = _softmax_probs(logits, beta, epsilon)
    return np.log(p[choice] + 1e-12)


def softmax_sample(logits, beta, rng, epsilon=0.0):
    p = _softmax_probs(logits, beta, epsilon)
    return rng.choice(len(p), p=p)


def _get_slurm_cpus(default=1):
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return default


def _logit_dynamics_dataframe(
    probs,
    choices,
    rewards,
    reset_mask,
    n_bins=15,
    logit_range=(-5.0, 5.0),
    min_trials_per_bin=5,
):
    """Conditional "logit-change" dynamics, as in Li et al., *Discovering
    cognitive strategies with tiny recurrent neural networks*.

    Shared by any fitted model that can produce a per-trial ``(n_trials, 2)``
    choice-probability array: ``logit_t = log(p1_t / p2_t)`` (teacher-forced
    on the real observed history) is binned by the action actually taken and
    the reward actually received at trial ``t`` (4 conditions: A1/R0, A1/R1,
    A2/R0, A2/R1). Transitions that cross a reset boundary (``reset_mask``)
    are excluded since ``logit_{t+1}`` would not be a genuine continuation.

    Parameters
    ----------
    probs : np.ndarray, shape (n_trials, 2)
        Model choice probabilities, teacher-forced on the real history.
    choices : np.ndarray, shape (n_trials,)
        1-indexed observed choice per trial.
    rewards : np.ndarray, shape (n_trials,)
        Observed reward per trial.
    reset_mask : np.ndarray of bool, shape (n_trials,)
        True at trial indices where the model's internal state was reset
        (session/block/window start).
    n_bins : int
        Number of bins spanning ``logit_range``.
    logit_range : (float, float)
        Lower/upper bound of the logit axis.
    min_trials_per_bin : int
        Bins with fewer than this many trials are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_center``, ``action`` (1 or 2), ``reward`` (0 or 1),
        ``mean_change``, ``sem_change``, ``n``.
    """
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(probs[:, 0] / probs[:, 1])  # prefer A1 > 0, prefer A2 < 0

    valid = ~np.asarray(reset_mask, dtype=bool)[1:]

    logit_t = logit[:-1][valid]
    logit_t1 = logit[1:][valid]
    change = logit_t1 - logit_t
    action_t = np.asarray(choices)[:-1][valid]
    reward_t = np.asarray(rewards)[:-1][valid]

    edges = np.linspace(logit_range[0], logit_range[1], n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(logit_t, edges) - 1, 0, n_bins - 1)

    records = []
    for a in (1, 2):
        for r in (0, 1):
            cond = (action_t == a) & (reward_t == r)
            for b in range(n_bins):
                sel = cond & (bin_idx == b)
                n = int(sel.sum())
                if n < min_trials_per_bin:
                    continue
                records.append(
                    {
                        "bin_center": bin_centers[b],
                        "action": a,
                        "reward": r,
                        "mean_change": change[sel].mean(),
                        "sem_change": (
                            change[sel].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                        ),
                        "n": n,
                    }
                )
    return pd.DataFrame.from_records(records)


def _plot_logit_dynamics_ax(df, ax=None):
    """Plot the conditional logit-change dynamics (see
    ``_logit_dynamics_dataframe``).

    Reproduces the style of Li et al.'s dynamical-portrait figure: one line
    per (action, reward) condition, colored by action and shaded by reward
    outcome. Where ``df`` has a ``sem_change`` column, a shaded band of
    +/- 1 SEM is drawn around each line (bins with a NaN SEM, e.g. a single
    contributing trial/animal, are drawn with zero band width).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``_logit_dynamics_dataframe``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    colors = {1: "tab:blue", 2: "tab:red"}
    alphas = {0: 0.4, 1: 1.0}
    labels = {
        (1, 0): "A1, R=0",
        (1, 1): "A1, R=1",
        (2, 0): "A2, R=0",
        (2, 1): "A2, R=1",
    }
    has_sem = "sem_change" in df.columns

    for a in (1, 2):
        for r in (0, 1):
            sub = df[(df["action"] == a) & (df["reward"] == r)].sort_values(
                "bin_center"
            )
            if sub.empty:
                continue
            ax.plot(
                sub["bin_center"],
                sub["mean_change"],
                color=colors[a],
                alpha=alphas[r],
                marker="o",
                ms=3,
                label=labels[(a, r)],
            )
            if has_sem and sub["sem_change"].notna().any():
                sem = sub["sem_change"].fillna(0.0)
                ax.fill_between(
                    sub["bin_center"],
                    sub["mean_change"] - sem,
                    sub["mean_change"] + sem,
                    color=colors[a],
                    alpha=0.15,
                    linewidth=0,
                )

    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Logit  (Prefer A2 <- 0 -> Prefer A1)")
    ax.set_ylabel("Logit change")
    ax.legend(fontsize=8, frameon=False)
    return ax


class DecisionModel:
    def __init__(
        self,
        task: Bandit2Arm,
        policy: BasePolicy,
        reset_mode="session",
    ):
        # Allow passing either an instance or a policy class; normalize to an instance.
        if isinstance(policy, type) and issubclass(policy, BasePolicy):
            policy = policy()

        if not isinstance(policy, BasePolicy):
            raise TypeError("policy must be a BasePolicy instance or subclass")

        self.task = task
        self.policy = policy
        self.beta_schedule = policy.beta_schedule  # convenience alias

        self.reset_mode = reset_mode
        self.resets = self._compute_resets(task, reset_mode)

        self.choices = np.asarray(task.choices, int) - 1  # Choices 0/1
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

    def _nll(
        self,
        theta,
        best_nll=None,
        warmup_trials=80,
        check_every=25,
        slack=0.02,
    ):
        policy_names = self.policy.active_parameter_names()
        all_params = dict(zip(policy_names, theta))

        # Fill defaults for inactive parameters
        for name, spec in self.policy.parameter_specs().items():
            if name not in all_params:
                all_params[name] = spec.default if spec.default is not None else 0.0

        self.policy.set_params(all_params)

        # reset after params are set so policies that read from self.params in reset don't KeyError
        self.policy.reset()
        self.beta_schedule.reset()

        nll = 0.0

        do_early_stop = (
            best_nll is not None
            and np.isfinite(best_nll)
            and check_every is not None
            and check_every > 0
        )

        for t, (c, r, reset) in enumerate(
            zip(self.choices, self.rewards, self.resets), start=1
        ):
            if reset:
                self.policy.reset()
                self.beta_schedule.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            nll -= softmax_loglik(
                logits,
                c,
                self.beta_schedule.get_beta(),
                self.beta_schedule.get_epsilon(),
            )
            self.policy.update(c, r)
            self.beta_schedule.update()

            # Cumulative NLL is monotonic, so this is a safe pruning criterion.
            if do_early_stop and t >= warmup_trials and (t % check_every == 0):
                if nll > best_nll * (1.0 + slack):
                    return nll

        return nll

    def get_trial_nll(self):
        """Return per-trial negative log-likelihood."""
        self.policy.set_params(self.params)
        self.policy.reset()
        self.beta_schedule.reset()

        trial_nlls = np.zeros(len(self.choices))

        for t, (c, r, reset) in enumerate(zip(self.choices, self.rewards, self.resets)):
            if reset:
                self.policy.reset()
                self.beta_schedule.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            trial_nlls[t] = -softmax_loglik(
                logits,
                c,
                self.beta_schedule.get_beta(),
                self.beta_schedule.get_epsilon(),
            )
            self.policy.update(c, r)
            self.beta_schedule.update()

        return trial_nlls

    def predict_proba(self) -> np.ndarray:
        """Choice probabilities for every trial in the original trial order.

        Teacher-forced on the real observed choice/reward history, mirroring
        ``get_trial_nll()``'s trial loop.

        Returns
        -------
        np.ndarray, shape (n_trials, 2)
            Softmax choice probabilities.
        """
        self.policy.set_params(self.params)
        self.policy.reset()
        self.beta_schedule.reset()

        probs = np.zeros((len(self.choices), 2))

        for t, (c, r, reset) in enumerate(zip(self.choices, self.rewards, self.resets)):
            if reset:
                self.policy.reset()
                self.beta_schedule.reset()
            else:
                self.policy.forget()

            probs[t] = _softmax_probs(
                self.policy.logits(),
                self.beta_schedule.get_beta(),
                self.beta_schedule.get_epsilon(),
            )
            self.policy.update(c, r)
            self.beta_schedule.update()

        return probs

    # -------------------- FIT --------------------

    def fit(
        self,
        n_starts=10,
        seed=None,
        progress=False,
        n_jobs=None,
        optimizer=None,
        early_stop=False,
        es_warmup_trials=3000,  # roughly 10% of total trials
        es_check_every=250,  # check every 250 trials after warmup
        es_slack=0.01,  # Keep if within 1% of best NLL seen so far
    ):
        rng = np.random.default_rng(seed)

        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        n_jobs = max(1, min(n_jobs, n_starts))

        print(f"Using {n_jobs} workers")

        policy_names = self.policy.active_parameter_names()
        all_bounds_dict = self.policy.get_bounds()
        bounds = [(n, all_bounds_dict[n]) for n in policy_names]

        seeds = rng.integers(0, 2**32 - 1, size=n_starts)

        opt = resolve_optimizer(optimizer)

        if early_stop:
            best_seen = [np.inf]

            def objective(theta):
                val = self._nll(
                    theta,
                    best_nll=best_seen[0],
                    warmup_trials=es_warmup_trials,
                    check_every=es_check_every,
                    slack=es_slack,
                )
                if np.isfinite(val) and val < best_seen[0]:
                    best_seen[0] = val
                return val

        else:
            objective = self._nll

        best_fun, best_x, fvals = opt.fit(
            objective=objective,
            bounds=bounds,
            seeds=seeds,
            n_jobs=n_jobs,
            progress=progress,
        )

        self.params = dict(zip(policy_names, best_x))

        # Fill defaults for inactive parameters
        for name, spec in self.policy.parameter_specs().items():
            if name not in self.params:
                self.params[name] = spec.default if spec.default is not None else 0.0

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
        self.beta_schedule.reset()

        task = self.task
        n_trials = task.n_trials

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            if self.resets[t]:
                self.policy.reset()
                self.beta_schedule.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            c = softmax_sample(
                logits,
                beta=self.beta_schedule.get_beta(),
                rng=rng,
                epsilon=self.beta_schedule.get_epsilon(),
            )

            p = task.probs[t, c]
            r = int(rng.random() < p)

            self.policy.update(c, r)
            self.beta_schedule.update()

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
        min_trials_per_block,
        params=None,
        prob_switch=1.0,
        seed=None,
        metadata=None,
    ):
        """
        Simulate a policy on a multi-block 2-armed bandit with optional variable block lengths.

        Args:
            policy: A `BasePolicy` instance (mutated in-place during simulation).
                Its ``beta_schedule`` attribute controls the softmax temperature.
            reward_schedule: Sequence of `(p1, p2)` tuples; one per block specifying reward probs.
            min_trials_per_block: Int or sequence giving the minimum trials to run per block.
            params: Optional flat dict of parameters for both policy and its beta_schedule.
                If None, assumes both objects were already configured via ``set_params()``.
            prob_switch: Float or sequence in (0, 1]; probability of switching after min trials.
                Example: `min_trials_per_block=100`, `prob_switch=0.02` yields median ~150 trials.
            seed: RNG seed for reproducibility.
            metadata: Optional metadata stored on the returned `Bandit2Arm` task.

        Returns:
            Bandit2Arm: Simulated task with probs, choices, rewards, and block/session ids.
        """
        beta_schedule = policy.beta_schedule

        rng = np.random.default_rng(seed)

        if params is not None:
            policy.set_params(params)
        elif not policy.params:
            raise ValueError(
                "params is None and policy has no parameters set; "
                "call policy.set_params(...) or provide params"
            )

        policy.reset()
        beta_schedule.reset()

        if isinstance(min_trials_per_block, int):
            min_trials_per_block = [min_trials_per_block] * len(reward_schedule)

        if isinstance(prob_switch, (int, float)):
            prob_switch = [prob_switch] * len(reward_schedule)

        assert len(min_trials_per_block) == len(reward_schedule)
        assert len(prob_switch) == len(reward_schedule)

        if not all(0 < p <= 1 for p in prob_switch):
            raise ValueError("prob_switch must be in (0, 1]")

        probs_list, choices, rewards = [], [], []
        session_ids, block_ids = [], []

        session_counter = 1
        block_counter = 1

        for (p1, p2), n_trials, p_switch in zip(
            reward_schedule, min_trials_per_block, prob_switch
        ):
            trials_in_block = 0

            while True:
                logits = policy.logits()
                c = softmax_sample(
                    logits,
                    beta=beta_schedule.get_beta(),
                    rng=rng,
                    epsilon=beta_schedule.get_epsilon(),
                )
                r = int(rng.random() < [p1, p2][c])

                policy.update(c, r)
                beta_schedule.update()

                probs_list.append([p1, p2])
                choices.append(c + 1)
                rewards.append(r)
                session_ids.append(session_counter)
                block_ids.append(block_counter)

                trials_in_block += 1

                if trials_in_block >= n_trials and rng.random() < p_switch:
                    break

            session_counter += 1
            block_counter += 1
            policy.reset()
            beta_schedule.reset()

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

    # -------------------- LOGIT DYNAMICS --------------------

    def compute_logit_dynamics(
        self,
        n_bins: int = 15,
        logit_range=(-5.0, 5.0),
        min_trials_per_bin: int = 5,
    ) -> pd.DataFrame:
        """Conditional "logit-change" dynamics, as in Li et al., *Discovering
        cognitive strategies with tiny recurrent neural networks*.

        See ``VanillaRNNFit2Arm.compute_logit_dynamics`` for the full
        description; this is the same analysis applied to this policy's
        fitted choice probabilities (``predict_proba()``).
        """
        return _logit_dynamics_dataframe(
            self.predict_proba(),
            self.task.choices,
            self.task.rewards,
            self.resets,
            n_bins=n_bins,
            logit_range=logit_range,
            min_trials_per_bin=min_trials_per_bin,
        )

    def plot_logit_dynamics(self, ax=None, n_bins: int = 15, logit_range=(-5.0, 5.0)):
        """Plot the conditional logit-change dynamics (see
        ``compute_logit_dynamics``)."""
        df = self.compute_logit_dynamics(n_bins=n_bins, logit_range=logit_range)
        return _plot_logit_dynamics_ax(df, ax=ax)

    # -------------------- METRICS / OUTPUT --------------------

    def bic(self):
        if self.nll is None:
            raise RuntimeError("Model must be fit before computing BIC.")
        k = len(self.policy.active_parameter_names())
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

        header = f"{'name':<14} {'fitted':>10}   {'bounds':<18} description"
        sep = "-" * 80

        print("\nPolicy parameters")
        print(sep)
        print(header)
        print(sep)
        specs = self.policy.parameter_specs()
        for name in self.policy.param_names():
            spec = specs[name]
            if spec.active:
                v_str = f"{self.params.get(name, float('nan')):>10.4f}"
            else:
                v_str = f"{'inactive':>10}"
            b = spec.bounds
            desc = spec.description or ""
            bounds_str = f"({b[0]:.4g}, {b[1]:.4g})"
            print(f"{name:<14} {v_str}   {bounds_str:<18}{desc}")

        print("\nBeta schedule parameters")
        print(sep)
        beta_specs = self.beta_schedule.parameter_specs()
        if beta_specs:
            print(header)
            print(sep)
            for name, spec in beta_specs.items():
                if spec.active:
                    v_str = f"{self.params.get(name, float('nan')):>10.4f}"
                else:
                    v_str = f"{'inactive':>10}"
                b = spec.bounds
                desc = spec.description or ""
                bounds_str = f"({b[0]:.4g}, {b[1]:.4g})"
                print(f"{name:<14} {v_str}   {bounds_str:<18}{desc}")
        else:
            print(f"  (none \u2014 {self.beta_schedule.__class__.__name__})")

        print(sep)
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
                beta_schedule_type=self.beta_schedule.__class__.__name__,
            )
        )
        return out

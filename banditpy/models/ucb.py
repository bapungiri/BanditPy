import numpy as np
from scipy.optimize import minimize
from banditpy.core import Bandit2Arm


class UCB2Arm:
    """
    Two-arm Upper Confidence Bound model with optional Bayesian variant,
    per-session resets, and exponential forgetting.

    mode:
      - "classic"  : empirical-mean UCB with exploration weight.
      - "bayesian" : Beta-Bernoulli posterior; uses mean plus c x posterior std.

    Fitted parameters (both modes):
      1. explore_param : classic -> c_explore (>0); bayesian -> c_sigma (>0).
      2. tau           : forgetting factor (0-1) applied between trials.
      3. q_init        : initial mean for values (classic) or prior mean (bayesian).
      4. lr_chosen     : learning rate for the chosen arm.
      5. lr_unchosen   : counterfactual learning rate for the unchosen arm.
    """

    def __init__(
        self,
        task: Bandit2Arm,
        reset_bool: (
            np.ndarray | None
        ) = None,  # Flags for resetting, default session_start
        mode: str = "classic",
    ):
        self.choices = np.asarray(task.choices, dtype=np.int32) - 1  # 0-based
        self.rewards = np.asarray(task.rewards, dtype=float)
        assert task.n_ports == 2, "UCB2Arm only supports 2-armed tasks."

        if reset_bool is None:
            self.reset_bool = task.is_session_start.astype(bool)
            print("Using session start flags for resets")
        else:
            self.reset_bool = np.asarray(reset_bool, dtype=bool)
            print("Using custom reset flags for resets")

        self.mode = mode.lower()
        assert self.mode in {"classic", "bayesian"}, "mode must be classic or bayesian"
        self.n_trials = self.choices.size
        self.params: dict[str, float] | None = None
        self.nll: float | None = None
        self._prior_strength = 2.0  # Bayesian pseudo-counts

    # ---------- Negative log-likelihood ----------
    def _calculate_log_likelihood(self, theta: np.ndarray) -> float:
        explore_param, tau, q_init, lr_chosen, lr_unchosen = theta
        nll = 0.0

        if self.mode == "classic":
            q_values = np.full(2, q_init, dtype=float)
            pulls = np.zeros(2, dtype=float)
            total_pulls = 0.0

            for choice, reward, reset in zip(
                self.choices, self.rewards, self.reset_bool
            ):
                if reset:
                    q_values.fill(q_init)
                    pulls.fill(0.0)
                    total_pulls = 0.0
                else:
                    q_values = q_init + tau * (q_values - q_init)
                    pulls *= tau
                    total_pulls *= tau

                denom = np.maximum(pulls, 1e-12)
                bonus = explore_param * np.sqrt(np.log(total_pulls + 1.0) / denom)
                scores = q_values + bonus
                chosen_hat = int(np.argmax(scores))

                p_choice = 1.0 if choice == chosen_hat else 1e-12
                nll -= np.log(p_choice)

                pulls[choice] += 1.0
                total_pulls += 1.0
                other = 1 - choice

                q_values[choice] += lr_chosen * (reward - q_values[choice])
                q_values[other] += lr_unchosen * (1 - reward - q_values[other])

        else:  # bayesian
            prior_alpha = np.full(2, self._prior_strength * q_init, dtype=float)
            prior_beta = np.full(2, self._prior_strength * (1.0 - q_init), dtype=float)
            alpha = prior_alpha.copy()
            beta = prior_beta.copy()

            for choice, reward, reset in zip(
                self.choices, self.rewards, self.reset_bool
            ):
                if reset:
                    alpha = prior_alpha.copy()
                    beta = prior_beta.copy()
                else:
                    alpha = prior_alpha + tau * (alpha - prior_alpha)
                    beta = prior_beta + tau * (beta - prior_beta)

                mean = alpha / (alpha + beta)
                var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
                std = np.sqrt(np.maximum(var, 1e-12))
                upper = mean + explore_param * std
                chosen_hat = int(np.argmax(upper))

                p_choice = 1.0 if choice == chosen_hat else 1e-12
                nll -= np.log(p_choice)

                if reward == 1.0:
                    alpha[choice] += lr_chosen
                else:
                    beta[choice] += lr_chosen

                other = 1 - choice
                alpha[other] += lr_unchosen * q_init
                beta[other] += lr_unchosen * (1.0 - q_init)

        return nll

    # ---------- Fit ----------
    def fit(
        self,
        explore_bounds: tuple[float, float] | None = None,
        tau_bounds: tuple[float, float] = (0.5, 0.999),
        q_init_bounds: tuple[float, float] = (0.0, 1.0),
        lr_bounds: tuple[float, float] = (1e-3, 1.0),
        n_starts: int = 10,
        seed: int | None = None,
    ):
        if explore_bounds is None:
            explore_bounds = (1e-3, 10.0)

        bounds = [
            explore_bounds,
            tau_bounds,
            q_init_bounds,
            lr_bounds,
            lr_bounds,
        ]
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(seed)
        best_val = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = lo + (hi - lo) * rng.random(len(bounds))
            res = minimize(
                self._calculate_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=1000, ftol=1e-8),
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        if best_x is None:
            raise RuntimeError("UCB optimisation failed to converge.")

        key = "c_explore" if self.mode == "classic" else "c_sigma"
        self.params = {
            key: float(best_x[0]),
            "tau": float(best_x[1]),
            "q_init": float(best_x[2]),
            "lr_chosen": float(best_x[3]),
            "lr_unchosen": float(best_x[4]),
        }
        self.nll = float(best_val)

    # ---------- Diagnostics ----------
    def bic(self) -> float:
        if self.nll is None:
            return np.nan
        k = 5  # number of free parameters
        return k * np.log(self.n_trials) + 2 * self.nll

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return
        key = "c_explore" if self.mode == "classic" else "c_sigma"
        print(
            f"{key}={self.params[key]:.4f}, "
            f"tau={self.params['tau']:.4f}, "
            f"q_init={self.params['q_init']:.4f}, "
            f"lr_chosen={self.params['lr_chosen']:.4f}, "
            f"lr_unchosen={self.params['lr_unchosen']:.4f}, "
            f"NLL={self.nll:.2f}"
        )

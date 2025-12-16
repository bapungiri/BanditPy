import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from banditpy.core import Bandit2Arm


class UCB2Arm:
    """
    Upper Confidence Bound (UCB) models for two-armed bandit with probabilistic (softmax) choice rule.

    Modes:
      "classic"     : empirical-mean UCB
                        This method keeps a frequentist running average of each arm and adds a confidence bonus proportional to sqrt(log t / n); exploration weight is a free constant and no prior belief enters.

      "bayesian"    : pure Bayesian UCB (Beta-Bernoulli)
                        This method maintains a posterior distribution for each arm (e.g., Beta for Bernoulli rewards), uses the posterior mean plus a multiple of posterior uncertainty, and naturally incorporates priors and forgetting via the posterior update.

      "rl-bayesian" : Bayesian UCB with RL-style updates
                        Similar to the Bayesian UCB model, but updates only the chosen arm's posterior with a learning rate (like in RL), instead of full Bayesian updating. This allows for more flexible adaptation to non-stationary environments.

      "rl"          : RL value learning with UCB exploration
                        This method maintains Q-values for each arm updated via a Rescorla-Wagner rule with a learning rate, and adds a UCB-style exploration bonus. This combines the strengths of RL learning and UCB exploration.


    Fitted parameters (fixed-length for all modes):
        explore_param : c_explore (classic, rl) or c_sigma (bayesian variants)
        tau           : forgetting factor
        q_init        : initial value or prior mean
        lr_chosen     : learning rate for chosen arm
        lr_unchosen   : counterfactual learning rate
        beta_softmax  : inverse temperature

    """

    def __init__(
        self,
        task: Bandit2Arm,
        reset_bool: np.ndarray | None = None,
        mode: str = "classic",
    ):
        self.choices = np.asarray(task.choices, dtype=int) - 1
        self.rewards = np.asarray(task.rewards, dtype=float)

        if reset_bool is None:
            self.reset_bool = task.is_session_start.astype(bool)
        else:
            self.reset_bool = np.asarray(reset_bool, dtype=bool)

        self.mode = mode.lower()
        assert self.mode in {
            "classic",
            "rl",
            "rl-bayesian",
            "bayesian",
        }

        self.n_trials = len(self.choices)
        self.params = None
        self.nll = None
        self._prior_strength = 2.0

    # --------------------------------------------------
    def _softmax_ll(self, scores, choice, beta):
        probs = softmax(beta * scores)
        return -np.log(np.maximum(probs[choice], 1e-12))

    # --------------------------------------------------
    def _calculate_log_likelihood(self, theta: np.ndarray) -> float:
        (
            explore_param,
            tau,
            q_init,
            lr_chosen,
            lr_unchosen,
            beta_softmax,
        ) = theta

        nll = 0.0

        match self.mode:

            # ==============================================
            case "classic":
                q = np.full(2, q_init)
                n = np.zeros(2)
                N = 0.0

                for c, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        q[:] = q_init
                        n[:] = 0
                        N = 0
                    else:
                        q = q_init + tau * (q - q_init)
                        n *= tau
                        N *= tau

                    bonus = explore_param * np.sqrt(
                        np.log(N + 1.0) / np.maximum(n, 1e-12)
                    )
                    scores = q + bonus
                    nll += self._softmax_ll(scores, c, beta_softmax)

                    n[c] += 1
                    N += 1
                    q[c] += (r - q[c]) / n[c]

            # ==============================================
            case "rl":
                q = np.full(2, q_init)
                n = np.zeros(2)
                N = 0.0

                for c, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        q[:] = q_init
                        n[:] = 0
                        N = 0
                    else:
                        q *= tau
                        n *= tau
                        N *= tau

                    bonus = explore_param * np.sqrt(
                        np.log(N + 1.0) / np.maximum(n, 1e-12)
                    )
                    scores = q + bonus
                    nll += self._softmax_ll(scores, c, beta_softmax)

                    n[c] += 1
                    N += 1
                    other = 1 - c

                    q[c] += lr_chosen * (r - q[c])
                    q[other] += lr_unchosen * ((1 - r) - q[other])

            # ==============================================
            case "bayesian":
                a0 = self._prior_strength * q_init
                b0 = self._prior_strength * (1 - q_init)

                alpha = np.full(2, a0)
                beta = np.full(2, b0)

                for c, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        alpha[:] = a0
                        beta[:] = b0
                    else:
                        alpha = a0 + tau * (alpha - a0)
                        beta = b0 + tau * (beta - b0)

                    mean = alpha / (alpha + beta)
                    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
                    scores = mean + explore_param * np.sqrt(var)
                    nll += self._softmax_ll(scores, c, beta_softmax)

                    alpha[c] += r
                    beta[c] += 1 - r

            # ==============================================
            case "rl-bayesian":
                a0 = self._prior_strength * q_init
                b0 = self._prior_strength * (1 - q_init)

                alpha = np.full(2, a0)
                beta = np.full(2, b0)

                for c, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        alpha[:] = a0
                        beta[:] = b0
                    else:
                        alpha = a0 + tau * (alpha - a0)
                        beta = b0 + tau * (beta - b0)

                    mean = alpha / (alpha + beta)
                    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
                    scores = mean + explore_param * np.sqrt(var)
                    nll += self._softmax_ll(scores, c, beta_softmax)

                    other = 1 - c
                    alpha[c] += lr_chosen * r
                    beta[c] += lr_chosen * (1 - r)

                    alpha[other] += lr_unchosen * (1 - r)
                    beta[other] += lr_unchosen * r

        return nll

    def fit(
        self,
        explore_bounds=(1e-3, 10.0),
        tau_bounds=(0.5, 0.999),
        q_init_bounds=(0.0, 1.0),
        lr_chosen_bounds=(-1.0, 1.0),
        lr_unchosen_bounds=(-1.0, 1.0),
        beta_bounds=(0.1, 20.0),
        n_starts=10,
        seed=None,
    ):
        """
        Explicit, named bounds for all parameters.
        Unused parameters are retained but marked as np.nan after fitting.
        """

        # ----DO NOT CHANGE the order of names ----
        param_names = [
            "explore_param",
            "tau",
            "q_init",
            "lr_chosen",
            "lr_unchosen",
            "beta_softmax",
        ]

        bounds_dict = {
            "explore_param": explore_bounds,
            "tau": tau_bounds,
            "q_init": q_init_bounds,
            "lr_chosen": lr_chosen_bounds,
            "lr_unchosen": lr_unchosen_bounds,
            "beta_softmax": beta_bounds,
        }

        bounds = [bounds_dict[p] for p in param_names]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

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
            raise RuntimeError("Optimization failed.")

        # ---- Store parameters ----
        self.params = dict(zip(param_names, best_x))
        self.nll = float(best_val)

        # ---- Mark unused parameters as NA ----
        match self.mode:
            case "classic":
                self.params["lr_chosen"] = np.nan
                self.params["lr_unchosen"] = np.nan
            case "bayesian":
                self.params["lr_chosen"] = np.nan
                self.params["lr_unchosen"] = np.nan
            case _:
                pass

    # --------------------------------------------------
    def bic(self):
        if self.nll is None:
            return np.nan
        k = 6
        return k * np.log(self.n_trials) + 2 * self.nll

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return

        def fmt(x):
            return "NA" if x is None or np.isnan(x) else f"{x:.4f}"

        print(f"mode = {self.mode}")
        print(
            f"explore_param = {fmt(self.params.get('explore_param'))}, "
            f"tau = {fmt(self.params.get('tau'))}, "
            f"q_init = {fmt(self.params.get('q_init'))}, "
            f"lr_chosen = {fmt(self.params.get('lr_chosen'))}, "
            f"lr_unchosen = {fmt(self.params.get('lr_unchosen'))}, "
            f"beta_softmax = {fmt(self.params.get('beta_softmax'))}, "
            f"NLL = {self.nll:.2f}"
        )

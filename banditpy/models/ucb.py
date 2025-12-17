import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from banditpy.core import Bandit2Arm

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from banditpy.core import Bandit2Arm


# ---------- Bounds Dataclass ----------
@dataclass
class Bounds:
    explore_param: tuple = (1e-3, 10.0)
    tau: tuple = (0.5, 0.999)
    q_init: tuple = (0.0, 1.0)
    lr_chosen: tuple = (0.0, 1.0)
    lr_unchosen: tuple = (0.0, 1.0)
    beta_softmax: tuple = (0.1, 20.0)


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

    PARAMS_BY_MODE = {
        "classic": ["explore_param", "tau", "q_init", "beta_softmax"],
        "bayesian": ["explore_param", "tau", "q_init", "beta_softmax"],
        "rl": [
            "explore_param",
            "tau",
            "q_init",
            "lr_chosen",
            "lr_unchosen",
            "beta_softmax",
        ],
        "rl-bayesian": [
            "explore_param",
            "tau",
            "q_init",
            "lr_chosen",
            "lr_unchosen",
            "beta_softmax",
        ],
    }

    def __init__(
        self,
        task: Bandit2Arm,
        reset_bool: np.ndarray | None = None,
        mode: str = "classic",
    ):
        self.choices = np.asarray(task.choices, dtype=np.int32) - 1
        self.rewards = np.asarray(task.rewards, dtype=float)
        assert task.n_ports == 2

        if reset_bool is None:
            self.reset_bool = task.is_session_start.astype(bool)
        else:
            self.reset_bool = np.asarray(reset_bool, dtype=bool)

        self.mode = mode.lower()
        assert self.mode in self.PARAMS_BY_MODE

        self.n_trials = self.choices.size
        self.params = None
        self.nll = None

        self.set_bounds = Bounds()
        self._prior_strength = 2.0

    # ---------- Utilities ----------
    @staticmethod
    def _softmax(x, beta):
        z = beta * (x - np.max(x))
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    # ---------- Negative log-likelihood ----------
    def _calculate_log_likelihood(self, theta: np.ndarray) -> float:
        names = self.PARAMS_BY_MODE[self.mode]
        p = dict(zip(names, theta))

        nll = 0.0

        match self.mode:

            case "classic":
                q = np.full(2, p["q_init"])
                pulls = np.zeros(2)
                total = 0.0

                for a, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        q[:] = p["q_init"]
                        pulls[:] = 0.0
                        total = 0.0
                    else:
                        q = p["q_init"] + p["tau"] * (q - p["q_init"])
                        pulls *= p["tau"]
                        total *= p["tau"]

                    bonus = p["explore_param"] * np.sqrt(
                        np.log(total + 1.0) / np.maximum(pulls, 1e-12)
                    )
                    scores = q + bonus
                    probs = self._softmax(scores, p["beta_softmax"])

                    nll -= np.log(probs[a] + 1e-12)

                    pulls[a] += 1.0
                    total += 1.0
                    q[a] += (r - q[a]) / pulls[a]

            case "rl":
                q = np.full(2, p["q_init"])
                pulls = np.zeros(2)
                total = 0.0

                for a, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if reset:
                        q[:] = p["q_init"]
                        pulls[:] = 0.0
                        total = 0.0
                    else:
                        q *= p["tau"]
                        pulls *= p["tau"]
                        total *= p["tau"]

                    bonus = p["explore_param"] * np.sqrt(
                        np.log(total + 1.0) / np.maximum(pulls, 1e-12)
                    )
                    scores = q + bonus
                    probs = self._softmax(scores, p["beta_softmax"])

                    nll -= np.log(probs[a] + 1e-12)

                    pulls[a] += 1.0
                    total += 1.0
                    other = 1 - a

                    q[a] += p["lr_chosen"] * (r - q[a])
                    q[other] += p["lr_unchosen"] * ((1 - r) - q[other])

            case "bayesian":
                alpha = np.full(2, self._prior_strength * p["q_init"])
                beta = np.full(2, self._prior_strength * (1 - p["q_init"]))

                for a, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if not reset:
                        alpha = self._prior_strength * p["q_init"] + p["tau"] * (
                            alpha - self._prior_strength * p["q_init"]
                        )
                        beta = self._prior_strength * (1 - p["q_init"]) + p["tau"] * (
                            beta - self._prior_strength * (1 - p["q_init"])
                        )

                    mean = alpha / (alpha + beta)
                    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
                    scores = mean + p["explore_param"] * np.sqrt(var)

                    probs = self._softmax(scores, p["beta_softmax"])
                    nll -= np.log(probs[a] + 1e-12)

                    alpha[a] += r
                    beta[a] += 1 - r

            case "rl-bayesian":
                alpha = np.full(2, self._prior_strength * p["q_init"])
                beta = np.full(2, self._prior_strength * (1 - p["q_init"]))

                for a, r, reset in zip(self.choices, self.rewards, self.reset_bool):
                    if not reset:
                        alpha = self._prior_strength * p["q_init"] + p["tau"] * (
                            alpha - self._prior_strength * p["q_init"]
                        )
                        beta = self._prior_strength * (1 - p["q_init"]) + p["tau"] * (
                            beta - self._prior_strength * (1 - p["q_init"])
                        )

                    mean = alpha / (alpha + beta)
                    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
                    scores = mean + p["explore_param"] * np.sqrt(var)

                    probs = self._softmax(scores, p["beta_softmax"])
                    nll -= np.log(probs[a] + 1e-12)

                    alpha[a] += p["lr_chosen"] * r
                    beta[a] += p["lr_chosen"] * (1 - r)

                    other = 1 - a
                    alpha[other] += p["lr_unchosen"] * p["q_init"]
                    beta[other] += p["lr_unchosen"] * (1 - p["q_init"])

        return nll

    # ---------- Fit ----------
    def fit(self, n_starts: int = 10, seed: int | None = None):
        rng = np.random.default_rng(seed)

        names = self.PARAMS_BY_MODE[self.mode]
        bounds = [getattr(self.set_bounds, k) for k in names]

        best_val = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(
                self._calculate_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        self.params = dict(zip(names, best_x))
        self.nll = float(best_val)

    # ---------- Diagnostics ----------
    def bic(self) -> float:
        if self.nll is None:
            return np.nan
        k = len(self.params)
        return k * np.log(self.n_trials) + 2 * self.nll

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return

        def fmt(x):
            return "NA" if x is None else f"{x:.4f}"

        print(f"mode = {self.mode}")
        for k in [
            "explore_param",
            "tau",
            "q_init",
            "lr_chosen",
            "lr_unchosen",
            "beta_softmax",
        ]:
            print(f"{k}: {fmt(self.params.get(k))}")
        print(f"NLL: {self.nll:.2f}")

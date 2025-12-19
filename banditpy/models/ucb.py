import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple
from banditpy.core import Bandit2Arm


def softmax_loglik(logits: np.ndarray, choice: int, beta: float) -> float:
    z = beta * logits
    z -= np.max(z)
    logp = z - np.log(np.sum(np.exp(z)))
    return logp[choice]


@dataclass
class Bounds:
    explore: Optional[Tuple[float, float]] = None
    tau: Optional[Tuple[float, float]] = None
    q_init: Optional[Tuple[float, float]] = None
    beta: Optional[Tuple[float, float]] = None
    lr: Optional[Tuple[float, float]] = None

    def get(self, name: str, default: Tuple[float, float]):
        val = getattr(self, name)
        return val if val is not None else default

    def warn_unused(self, active, mode):
        unused = [k for k, v in vars(self).items() if v is not None and k not in active]
        if unused:
            print(
                f"[UCB2Arm warning] In mode '{mode}', bounds for {unused} are ignored."
            )


class BaseUCBPolicy:
    def reset(self): ...
    def forget(self, tau): ...
    def logits(self, explore): ...
    def update(self, choice, reward, lr_chosen=None, lr_unchosen=None): ...


class EmpiricalUCB(BaseUCBPolicy):
    def __init__(self, q_init):
        self.q_init = q_init
        self.reset()

    def reset(self):
        self.q = np.full(2, self.q_init)
        self.n = np.zeros(2)
        self.t = 0.0

    def forget(self, tau):
        self.q = self.q_init + tau * (self.q - self.q_init)
        self.n *= tau
        self.t *= tau

    def logits(self, explore):
        denom = np.maximum(self.n, 1e-12)
        bonus = explore * np.sqrt(np.log(self.t + 1.0) / denom)
        return self.q + bonus

    def update(self, choice, reward, lr_chosen=None, lr_unchosen=None):
        self.t += 1.0
        self.n[choice] += 1.0

        if lr_chosen is None:
            self.q[choice] += (reward - self.q[choice]) / self.n[choice]
        else:
            self.q[choice] += lr_chosen * (reward - self.q[choice])

        if lr_unchosen is not None:
            other = 1 - choice
            self.q[other] += lr_unchosen * ((1 - reward) - self.q[other])


class BayesianUCB(BaseUCBPolicy):
    def __init__(self, q_init, strength):
        self.q_init = q_init
        self.strength = strength
        self.reset()

    def reset(self):
        self.alpha = np.full(2, self.strength * self.q_init)
        self.beta = np.full(2, self.strength * (1 - self.q_init))

    def forget(self, tau):
        pa = self.strength * self.q_init
        pb = self.strength * (1 - self.q_init)
        self.alpha = pa + tau * (self.alpha - pa)
        self.beta = pb + tau * (self.beta - pb)

    def logits(self, explore):
        mean = self.alpha / (self.alpha + self.beta)
        var = (
            self.alpha
            * self.beta
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )
        return mean + explore * np.sqrt(np.maximum(var, 1e-12))

    def update(self, choice, reward, lr_chosen=None, lr_unchosen=None):
        if lr_chosen is None:
            self.alpha[choice] += reward
            self.beta[choice] += 1 - reward
        else:
            self.alpha[choice] += lr_chosen * reward
            self.beta[choice] += lr_chosen * (1 - reward)

        if lr_unchosen is not None:
            other = 1 - choice
            self.alpha[other] += lr_unchosen * self.q_init
            self.beta[other] += lr_unchosen * (1 - self.q_init)


class UCB2Arm:
    """
    Modes:
      classic, bayesian, rl, rl-bayesian
    """

    MODE_PARAMS = {
        "classic": {"explore", "tau", "q_init", "beta"},
        "rl": {"explore", "tau", "q_init", "beta", "lr"},
        "bayesian": {"explore", "tau", "q_init", "beta"},
        "rl-bayesian": {"explore", "tau", "q_init", "beta", "lr"},
    }

    PARAM_ORDER = {
        "classic": ["explore", "tau", "q_init", "beta"],
        "bayesian": ["explore", "tau", "q_init", "beta"],
        "rl": ["explore", "tau", "q_init", "beta", "lr_chosen", "lr_unchosen"],
        "rl-bayesian": ["explore", "tau", "q_init", "beta", "lr_chosen", "lr_unchosen"],
    }

    def __init__(self, task: Bandit2Arm, reset_bool=None, mode="classic"):
        self.choices = np.asarray(task.choices, int) - 1
        self.rewards = np.asarray(task.rewards, float)
        self.reset_bool = (
            task.is_session_start.astype(bool)
            if reset_bool is None
            else np.asarray(reset_bool, bool)
        )

        self.mode = mode
        self.n_trials = len(self.choices)
        self.bounds = Bounds()
        self.params = None
        self.nll = None
        self._prior_strength = 2.0

    def _calculate_log_likelihood(self, theta):
        explore, tau, q_init, beta = theta[:4]

        lr_chosen = lr_unchosen = None
        if "lr" in self.MODE_PARAMS[self.mode]:
            lr_chosen, lr_unchosen = theta[4:6]

        if "bayesian" in self.mode:
            policy = BayesianUCB(q_init, self._prior_strength)
        else:
            policy = EmpiricalUCB(q_init)

        nll = 0.0

        for c, r, reset in zip(self.choices, self.rewards, self.reset_bool):
            if reset:
                policy.reset()
            else:
                policy.forget(tau)

            logits = policy.logits(explore)
            nll -= softmax_loglik(logits, c, beta)
            policy.update(c, r, lr_chosen, lr_unchosen)

        return nll

    def fit(self, n_starts=10, seed=None):
        active = self.MODE_PARAMS[self.mode]
        self.bounds.warn_unused(active, self.mode)

        bounds = [
            self.bounds.get("explore", (1e-3, 10.0)),
            self.bounds.get("tau", (0.5, 0.999)),
            self.bounds.get("q_init", (0.0, 1.0)),
            self.bounds.get("beta", (0.1, 10.0)),
        ]

        if "lr" in active:
            bounds += [
                self.bounds.get("lr", (-1.0, 1.0)),
                self.bounds.get("lr", (-1.0, 1.0)),
            ]

        rng = np.random.default_rng(seed)
        best = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = np.array([rng.uniform(*b) for b in bounds])
            res = minimize(
                self._calculate_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best:
                best = res.fun
                best_x = res.x

        self.nll = best
        names = self.PARAM_ORDER[self.mode]
        self.params = dict(zip(names, best_x))

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return

        print(f"Mode: {self.mode}")
        print(f"NLL : {self.nll:.2f}")
        print(f"BIC : {self.bic():.2f}")
        for k, v in self.params.items():
            print(f"{k:>12s} : {v:.4f}")

    def bic(self):
        if self.nll is None or self.params is None:
            raise RuntimeError("Model must be fit before computing BIC.")
        k = len(self.params)
        return k * np.log(self.n_trials) + 2.0 * self.nll

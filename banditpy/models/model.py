import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from banditpy.core import Bandit2Arm
from .policy.base import BasePolicy


def softmax_loglik(logits, choice, beta):
    z = beta * logits
    return z[choice] - logsumexp(z)


def softmax_sample(logits, beta, rng):
    z = beta * logits
    p = np.exp(z - logsumexp(z))
    return rng.choice(len(p), p=p)


class DecisionModel:
    def __init__(self, task: Bandit2Arm, policy: BasePolicy):
        self.task = task
        self.policy = policy
        self.nll = None
        self.params = None

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

    def fit(self, n_starts=10, seed=None):
        rng = np.random.default_rng(seed)

        bounds = self.policy.bounds()
        names = self.policy.param_names()

        best = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = np.array([rng.uniform(*b) for b in bounds])
            res = minimize(
                self._nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best:
                best = res.fun
                best_x = res.x

        self.params = dict(zip(names, best_x))
        self.nll = best
        self.policy.set_params(self.params)

    def simulate_posterior_predictive(self, seed=None) -> Bandit2Arm:
        if self.params is None:
            raise RuntimeError(
                "Model must be fit before posterior predictive simulation."
            )

        rng = np.random.default_rng(seed)

        self.policy.set_params(self.params)
        self.policy.reset()

        n_trials = len(self.task.choices)

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=float)
        resets = self.task.is_session_start.astype(bool)

        for t in range(n_trials):
            if resets[t]:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            c = softmax_sample(logits, self.params["beta"], rng)
            r = rng.random() < self.task.probas[c]

            self.policy.update(c, r)

            choices[t] = c + 1  # back to 1-based
            rewards[t] = r

        return Bandit2Arm(
            choices=choices,
            rewards=rewards,
            probs=self.task.probs.copy(),
            is_session_start=resets.copy(),
        )

    @classmethod
    def simulate_policy(
        cls,
        policy: BasePolicy,
        reward_schedule: list[tuple[float, float]],
        trials_per_block: int,
        params: dict,
        seed: int | None = None,
    ) -> Bandit2Arm:
        """
        Simulate a policy learning across blocks with specified reward probabilities.

        reward_schedule: list of (p_left, p_right)
        trials_per_block: number of trials per block
        params: policy parameters
        """

        rng = np.random.default_rng(seed)

        policy.set_params(params)
        policy.reset()

        n_blocks = len(reward_schedule)
        n_trials = n_blocks * trials_per_block

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=float)
        resets = np.zeros(n_trials, dtype=bool)

        t = 0
        for block_idx, (p0, p1) in enumerate(reward_schedule):
            probas = np.array([p0, p1])
            resets[t] = True
            policy.reset()

            for _ in range(trials_per_block):
                logits = policy.logits()
                c = softmax_sample(logits, params["beta"], rng)
                r = rng.random() < probas[c]

                policy.update(c, r)

                choices[t] = c + 1
                rewards[t] = r
                t += 1

        return Bandit2Arm(
            choices=choices,
            rewards=rewards,
            probs=None,  # block-varying, not a single vector
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

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta


class Thompson2Arm:
    def __init__(self, n_arms=2, n_sim=500):
        self.n_arms = n_arms
        self.n_sim = n_sim

    def _initialize_agent(self, alpha0, beta0):
        # Initialize Beta priors for each arm
        self.alpha = np.full(self.n_arms, alpha0, dtype=float)
        self.beta = np.full(self.n_arms, beta0, dtype=float)

    def _get_choice_probs(self):
        # Estimate choice probability by sampling from Beta posteriors
        samples = np.random.beta(
            self.alpha[:, None], self.beta[:, None], size=(self.n_arms, self.n_sim)
        )
        chosen = np.argmax(samples, axis=0)
        return np.array([(chosen == i).mean() for i in range(self.n_arms)])

    def _simulate_neg_log_likelihood(self, params, choices, rewards):
        alpha0, beta0 = params
        if alpha0 <= 0 or beta0 <= 0:
            return np.inf

        self._initialize_agent(alpha0, beta0)
        neg_log_likelihood = 0

        for t in range(len(choices)):
            choice_probs = self._get_choice_probs()
            choice = choices[t]
            reward = rewards[t]

            prob = np.clip(choice_probs[choice], 1e-6, 1.0)
            neg_log_likelihood -= np.log(prob)

            if reward == 1:
                self.alpha[choice] += 1
            else:
                self.beta[choice] += 1

        return neg_log_likelihood

    def fit(self, choices, rewards, initial_guess=(1.0, 1.0)):
        result = minimize(
            fun=self._simulate_neg_log_likelihood,
            x0=initial_guess,
            args=(choices, rewards),
            bounds=[(0.01, 10.0), (0.01, 10.0)],
            method="L-BFGS-B",
        )
        self.fitted_alpha0, self.fitted_beta0 = result.x
        self.nll = result.fun
        return result

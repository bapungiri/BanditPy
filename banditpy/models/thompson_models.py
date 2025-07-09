import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import beta
from banditpy.core import Bandit2Arm


class Thompson2Arm:
    def __init__(self, task: Bandit2Arm, n_sim=500):
        self.choices = np.array(task.choices) - 1  # Convert to 0,1 format
        self.rewards = np.array(task.rewards)
        self.n_sim = n_sim
        self.n_arms = task.n_ports

    def _calculate_log_likelihood(self, alpha0, beta0):
        alpha = np.full(self.n_arms, alpha0, dtype=float)
        beta_ = np.full(self.n_arms, beta0, dtype=float)
        neg_log_likelihood = 0

        for choice, reward in zip(self.choices, self.rewards):
            # Sample n_sim values for each arm from Beta distributions
            samples = np.random.beta(
                alpha[:, None], beta_[:, None], size=(self.n_arms, self.n_sim)
            )
            selected = np.argmax(samples, axis=0)
            choice_prob = (selected == choice).mean()

            # Avoid log(0)
            choice_prob = np.clip(choice_prob, 1e-6, 1.0)
            neg_log_likelihood -= np.log(choice_prob)

            # Bayesian update
            if reward == 1:
                alpha[choice] += 1
            else:
                beta_[choice] += 1

        return neg_log_likelihood

    def _objective(self, params):
        alpha0, beta0 = params
        if alpha0 <= 0 or beta0 <= 0:
            return np.inf
        return self._calculate_log_likelihood(alpha0, beta0)

    def fit(self, bounds=np.array([(0.01, 10.0), (0.01, 10.0)]), n_optimize=5):

        x_vec = np.zeros((n_optimize, 2))
        nll_vec = np.zeros(n_optimize)

        for i in range(n_optimize):
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=None,
                disp=False,
                polish=True,
                init="latinhypercube",
                updating="deferred",
                workers=1,
            )
            x_vec[i] = result.x
            nll_vec[i] = result.fun

        idx_best = np.argmin(nll_vec)
        self.alpha0, self.beta0 = x_vec[idx_best]
        self.nll = nll_vec[idx_best]

    def bic(self):
        """Calculate the Bayesian Information Criterion."""
        if not hasattr(self, "nll") or np.isnan(self.nll):
            return np.nan
        k = 2  # Number of parameters (alpha0, beta0)
        return k * np.log(len(self.choices)) + 2 * self.nll

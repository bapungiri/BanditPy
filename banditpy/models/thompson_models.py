import numpy as np
from scipy.optimize import minimize, differential_evolution
from banditpy.core import Bandit2Arm


class Thompson2Arm:
    def __init__(self, task: Bandit2Arm):
        self.choices = np.array(task.choices) - 1  # Convert to 0,1 format
        self.rewards = np.array(task.rewards)
        self.is_session_start = task.is_session_start.astype(bool)
        self.n_arms = task.n_ports
        self.delta_s = None
        self.delta_f = None
        self.tau = None
        self._base_rng = np.random.default_rng()

    def print_params(self):
        print(f"Delta Success: {self.delta_s}")
        print(f"Delta Failure: {self.delta_f}")
        print(f"Tau: {self.tau}")

    def _calculate_log_likelihood(self, params):
        delta_s, delta_f, tau = params
        # Constraints
        if delta_s <= 0 or delta_f <= 0 or not (0 < tau <= 1):
            return np.inf

        # Initial values of alpha=1, beta=1
        alpha = np.ones(self.n_arms, dtype=float)
        beta = np.ones(self.n_arms, dtype=float)
        neg_log_likelihood = 0

        for choice, reward, start_bool in zip(
            self.choices, self.rewards, self.is_session_start
        ):
            # Sample n_sim values for each arm from Beta distributions
            if start_bool:
                alpha = np.ones(self.n_arms, dtype=float)
                beta = np.ones(self.n_arms, dtype=float)

            samples = np.random.beta(
                alpha[:, None], beta[:, None], size=(self.n_arms, 500)
            )
            selected = np.argmax(samples, axis=0)
            choice_prob = (selected == choice).mean()

            # Avoid log(0)
            choice_prob = np.clip(choice_prob, 1e-6, 1.0)
            neg_log_likelihood -= np.log(choice_prob)

            # forgetting factor for limited memory
            # using only multiplicative factor makes alpha/beta zero
            # alpha = tau * alpha
            # beta = tau * beta
            alpha = 1.0 + (alpha - 1.0) * tau
            beta = 1.0 + (beta - 1.0) * tau

            # Bayesian update
            if reward == 1:
                alpha[choice] += delta_s  # Success increment
            else:
                beta[choice] += delta_f  # Failure increment

        return neg_log_likelihood

    # def _objective(self, params):
    #     alpha0, beta0 = params
    #     if alpha0 <= 0 or beta0 <= 0:
    #         return np.inf
    #     return self._calculate_log_likelihood(alpha0, beta0)

    # def fit(
    #     self, bounds=np.array([(0.01, 10.0), (0.01, 10.0), (0.7, 1)]), n_optimize=5
    # ):

    #     x_vec = np.zeros((n_optimize, 3))
    #     nll_vec = np.zeros(n_optimize)

    #     for i in range(n_optimize):
    #         result = differential_evolution(
    #             self._calculate_log_likelihood,
    #             bounds=bounds,
    #             strategy="best1bin",
    #             maxiter=1000,
    #             popsize=15,
    #             tol=0.01,
    #             mutation=(0.5, 1),
    #             recombination=0.7,
    #             seed=None,
    #             disp=False,
    #             polish=True,
    #             init="latinhypercube",
    #             updating="deferred",
    #             workers=1,
    #         )
    #         x_vec[i] = result.x
    #         nll_vec[i] = result.fun

    #     idx_best = np.argmin(nll_vec)
    #     self.delta_s, self.delta_f, self.tau = x_vec[idx_best]
    #     self.nll = nll_vec[idx_best]

    def fit(self, bounds=((0.01, 10.0), (0.01, 10.0), (0.7, 0.999)), n_starts=8):
        """
        Multi-start local optimization using scipy.minimize (L-BFGS-B).
        bounds: ((delta_s_lo, delta_s_hi), (delta_f_lo, delta_f_hi), (tau_lo, tau_hi))
        """
        best_val = np.inf
        best_x = None
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        for _ in range(n_starts):
            print(_)
            x0 = lo + (hi - lo) * self._base_rng.random(3)
            res = minimize(
                self._calculate_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=400, ftol=1e-6),
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        self.delta_s, self.delta_f, self.tau = best_x
        self.nll = best_val

    def bic(self):
        """Calculate the Bayesian Information Criterion."""
        if not hasattr(self, "nll") or np.isnan(self.nll):
            return np.nan
        k = 3  # Number of parameters (delta_s, delta_f, tau)
        return k * np.log(len(self.choices)) + 2 * self.nll

    def simulate_posteriors(self):
        """
        Replay trials with fitted (delta_s, delta_f, tau) to obtain trajectories.
        Returns:
            alpha_traj: (T+1, n_arms)
            beta_traj : (T+1, n_arms)
            mean_traj : (T+1, n_arms) posterior mean per arm
        """
        if any(p is None for p in (self.delta_s, self.delta_f, self.tau)):
            raise RuntimeError("Call fit() first.")
        alpha = np.ones(self.n_arms, dtype=float)
        beta = np.ones(self.n_arms, dtype=float)
        alpha_traj = [alpha.copy()]
        beta_traj = [beta.copy()]
        for choice, reward in zip(self.choices, self.rewards):
            # forgetting
            alpha *= self.tau
            beta *= self.tau
            # update
            if reward == 1:
                alpha[choice] += self.delta_s
            else:
                beta[choice] += self.delta_f
            alpha_traj.append(alpha.copy())
            beta_traj.append(beta.copy())
        alpha_traj = np.stack(alpha_traj)
        beta_traj = np.stack(beta_traj)
        mean_traj = alpha_traj / (alpha_traj + beta_traj)
        return alpha_traj, beta_traj, mean_traj

import numpy as np
import pandas as pd
from .. import core
from scipy.optimize import differential_evolution

from functools import partial


class QlearningEstimator:
    """Estimate Q-learning parameters for a multi-armed bandit task

    Vanilla Q-learning model:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])

    Perseverance model:
    In addition to the vanilla Q-learning update, we also added a persistence term to choose the same action as the previous one:
    H = H + alpha_h * (choice - H)
    """

    def __init__(
        self, mab: core.MultiArmedBandit, bounds, model="vanilla", n_optimize=1, n_cpu=1
    ):
        """
        Initialize the Q-learning estimator.

        Parameters
        ----------
        mab : core.MultiArmedBandit
            mab object containing the task data
        model : str, ("vanilla", "persev")
            which model to fit, by default "vanilla"
        """
        assert mab.n_ports == 2, "This task has more than 2 ports"

        if model == "vanilla":
            self.n_params = 3
        elif model == "persev":
            self.n_params = 5
        else:
            raise ValueError(f"Unknown model: {self.model}")

        self.choices = mab.get_binarized_choices().astype(int)
        self.rewards = mab.rewards
        self.is_session_start = mab.is_session_start
        self.session_ids = mab.session_ids
        self.estimated_params = None
        self.model = model
        self.n_cpu = n_cpu
        self.n_optimize = n_optimize

        if bounds is not None:
            self.optimize_func = partial(
                differential_evolution,
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
                workers=n_cpu,
            )
        else:
            self.optimize_func = None

    def print_params(self):
        if self.model == "vanilla":
            a, b, c = self.estimated_params.round(4)
            print(f"alpha_c: {a}, alpha_u: {b}, beta: {c}")

        elif self.model == "persev":
            a, b, c, d, e = self.estimated_params.round(4)
            print(f"alpha_c: {a}, alpha_u: {b},alpha_h: {c}, scaler: {d}, beta: {e}")

    def compute_q_values(self, alpha_params):
        """
        Compute Q-values for each action based on the choices and rewards.

        Note: Having initial Q-values of 0.5 instead of 0 and limiting Q-values to (0,1) helped with convergence. Not entirely sure why.
        """
        Q = 0.5 * np.ones(2)  # Initialize Q-values for two actions
        q_values = []

        if self.model == "vanilla":
            alpha_c, alpha_u = alpha_params
        elif self.model == "persev":
            alpha_c, alpha_u, alpha_h = alpha_params
            H = 0  # Initialize history for two actions
            h_values = []
        else:
            raise ValueError(f"Unknown model: {self.model}")

        for choice, reward, is_start in zip(
            self.choices, self.rewards, self.is_session_start
        ):
            if is_start:
                Q[:] = 0.5  # Reset Q-values at session start

                if self.model == "persev":
                    H = 0.5

            unchosen = 1 - choice

            # ===== Q-learning update ============
            Q[choice] += alpha_c * (reward - Q[choice])
            Q[unchosen] += alpha_u * (reward - Q[choice])
            # Q[unchosen] += alpha_u * (reward - Q[unchosen])
            # Q[unchosen] += alpha_u * ((1 - reward) - Q[unchosen])
            # Q[unchosen] += alpha_u * (Q[choice] - reward)

            if self.model == "persev":
                H += alpha_h * (choice - H)
                h_values.append(H.copy())

            q_values.append(Q.copy())

        q_values = np.clip(np.array(q_values), 0, 1)
        h_values = np.array(h_values)

        if self.model == "vanilla":
            return q_values
        elif self.model == "persev":
            return q_values, h_values

    def compute_probabilites(self, params):
        # Compute softmax probabilities
        if self.model == "vanilla":
            Q_values = self.compute_q_values(params[:-1])
            beta = params[-1]
            betaQ = beta * Q_values
            exp_Q = np.exp(betaQ)

        elif self.model == "persev":
            Q_values, H_values = self.compute_q_values(params[:-2])
            scaler, beta = params[-2], params[-1]
            betaQscalerH = beta * Q_values + scaler * H_values.reshape(-1, 1)
            exp_Q = np.exp(betaQscalerH)

        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        return probs

    def log_likelihood(self, params):

        probs = self.compute_probabilites(params)

        # Get the probability of the chosen action
        chosen_probs = probs[np.arange(len(self.choices)), self.choices]

        # Numerical stability
        eps = 1e-9
        chosen_probs = np.clip(chosen_probs, eps, 1 - eps)

        # Log-likelihood
        ll = np.nansum(np.log(chosen_probs))
        return -ll  # For minimization

    def fit(self):
        # Optimize params using a bounded method
        x_vec = np.zeros((self.n_optimize, self.n_params))
        fval_vec = np.zeros(self.n_optimize)

        for opt_i in range(self.n_optimize):
            result = self.optimize_func(self.log_likelihood)
            x_vec[opt_i] = result.x
            fval_vec[opt_i] = result.fun

        idx_best = np.argmin(fval_vec)
        self.estimated_params = x_vec[idx_best]

    def predict_choices(self, params, deterministic=True):
        """
        Predict choices using the estimated parameters.

        Parameters
        ----------
        deterministic : bool, optional (default=True)
            If True, chooses the action with the highest probability at each trial (argmax).
            If False, samples actions probabilistically from the softmax distribution.

        Returns
        -------
        np.ndarray
            Array of predicted choices (0 or 1) for each trial.
        """

        probs = self.compute_probabilites(params)

        if deterministic:
            predicted_choices = np.argmax(probs, axis=1)
        else:
            # Sample from probabilities
            predicted_choices = np.array(
                [np.random.choice([0, 1], p=prob) for prob in probs]
            )

        return predicted_choices

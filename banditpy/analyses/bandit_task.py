import numpy as np
import pandas as pd
from .. import core
from scipy.optimize import minimize, differential_evolution
from joblib import Parallel, delayed

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from pathlib import Path


def get_performance_2ab(
    mab: core.MultiArmedBandit,
    min_trials_per_sess=None,
    roll_window=80,
    delta_prob=None,
    smooth=2,
):
    """Get performance on two armed bandit task

    Parameters
    ----------
    df : csv file containing all data
        _description_
    min_trials_per_sess : _type_, optional
        sessions with more than this number of trials will excluded.
    roll_window : int, optional
        no.of sessions over which performance is calculated, by default 80
    roll_step : int, optional
        _description_, by default 40
    delta_prob : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    assert mab.n_ports == 2, "This task has more than 2 ports"
    if delta_prob is not None:
        prob_diff = np.abs(df["rewprobfull1"] - df["rewprobfull2"])
        df = df[prob_diff >= delta_prob]

    session_id = df["session#"].to_numpy()
    unq_session_id, n_trials = np.unique(session_id, return_counts=True)
    is_choice_high = df["is_choice_high"].to_numpy()

    if min_trials_per_sess is not None:
        bad_sessions = unq_session_id[n_trials < min_trials_per_sess]
        n_trials = n_trials[n_trials >= min_trials_per_sess]
        bad_trials = np.isin(session_id, bad_sessions)

        is_choice_high = is_choice_high[~bad_trials]

    # converting into n_sessions x n_trials dataframe
    is_choice_high_per_session = pd.DataFrame(
        np.split(is_choice_high.astype(int), np.cumsum(n_trials)[:-1])
    )
    prob_correct_per_trial = is_choice_high_per_session.mean(axis=0)
    trial_x = np.arange(len(prob_correct_per_trial)) + 1

    perf = np.array([np.mean(arr) for arr in is_choice_high_per_session])

    sess_div_perf = is_choice_high_per_session.rolling(
        window=roll_window, closed="right", min_periods=2
    ).mean()[roll_window - 1 :: roll_window]

    sess_div_perf_arr = sess_div_perf.to_numpy()

    sess_div_perf_arr = gaussian_filter1d(sess_div_perf_arr, axis=1, sigma=smooth)

    return sess_div_perf_arr


class HistoryBasedLogisticModel:
    """Based on Miller et al. 2021, "From predictive models to cognitive models....." """

    def __init__(self, mab: core.MultiArmedBandit, n_past=5):
        assert mab.n_ports == 2, "Only 2-armed bandit task is supported"
        self.choices, self.rewards = self._reformat_choices_rewards(
            mab.choices, mab.rewards
        )
        self.rewards = mab.rewards
        self.n_past = n_past
        self.model = LogisticRegression(solver="lbfgs")
        self.feature_names = []

    @property
    def coef(self):
        return self.model.coef_.squeeze().reshape(3, -1)

    @property
    def coef_names(self):
        return ["reward seeking", "choice preservation", "effect of outcome"]

    def get_coef_df(self, form="wide"):
        """Get the coefficients and their names as a DataFrame."""
        df = pd.DataFrame(data=self.coef.T, columns=self.coef_names)
        df["past_id"] = np.arange(1, self.n_past + 1)

        if form == "long":
            df = df.melt(id_vars="past_id", var_name="coef_name", value_name="coef")
        return df

    def _reformat_choices_rewards(self, choices, rewards):
        """
        Convert choices to -1 for left, 1 for right.
        Convert rewards to -1 for no reward, 1 for reward.
        """
        choices[choices == 1] = -1
        choices[choices == 2] = 1

        rewards[rewards == 0] = -1
        rewards[rewards == 1] = 1
        return choices, rewards

    def _prepare_features(self, choices, rewards):
        """
        Prepare lagged features using sliding_window_view.
        """
        # Generate sliding windows and flipping arrays from left to right so that the first column is the most recent choice
        C_windows = sliding_window_view(choices, window_shape=self.n_past)[:-1][:, ::-1]
        R_windows = sliding_window_view(rewards, window_shape=self.n_past)[:-1][:, ::-1]

        actual_choices = choices[self.n_past :]

        # Interaction term
        CxR_windows = C_windows * R_windows

        # Stack features: shape (n_samples, n_lags * 3)
        X = np.hstack([CxR_windows, C_windows, R_windows])
        y = actual_choices.astype(int)  # -1 for left , 1 for right

        return X, y

    def fit(self):
        X, y = self._prepare_features(self.choices, self.rewards)
        self.model.fit(X, y)
        return self

    # def predict_proba(self, choices, rewards):
    #     X, _ = self._prepare_features(choices, rewards)
    #     return self.model.predict_proba(X)[:, 1]  # Prob of choosing right

    def get_coef_dict(self):
        return dict(zip(self.coef_names, self.coef))


class QlearningEstimator:
    """Estimate Q-learning parameters for a multi-armed bandit task

    Vanilla Q-learning model:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])

    Perseverance model:
    In addition to the vanilla Q-learning update, we also added a persistence term to choose the same action as the previous one:
    H = H + alpha_h * (choice - H)
    """

    def __init__(self, mab: core.MultiArmedBandit, model="vanilla"):
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
        self.choices = mab.get_binarized_choices().astype(int)
        self.rewards = mab.rewards
        self.is_session_start = mab.is_session_start
        self.session_ids = mab.session_ids
        self.estimated_params = None
        self.model = model

    def print_params(self):
        if self.model == "vanilla":
            a, b, c = self.estimated_params.round(4)
            print(f"alpha_c: {a}, alpha_u: {b}, beta: {c}")

        elif self.model == "persev":
            a, b, c, d, e = self.estimated_params.round(4)
            print(f"alpha_c: {a}, alpha_u: {b},alpha_h: {c}, scaler: {d}, beta: {e}")

    def compute_q_values(self, params):
        """
        Compute Q-values for each action based on the choices and rewards.

        Note: Having initial Q-values of 0.5 instead of 0 and limiting Q-values to (0,1) helped with convergence. Not entirely sure why.
        """
        Q = 0.5 * np.ones(2)  # Initialize Q-values for two actions
        q_values = []

        if self.model == "vanilla":
            alpha_c, alpha_u = params
        elif self.model == "persev":
            alpha_c, alpha_u, alpha_h = params
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

            # Q-learning update
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

    def log_likelihood(self, params):

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
        # Get the probability of the chosen action
        chosen_probs = probs[np.arange(len(self.choices)), self.choices]

        # Numerical stability
        eps = 1e-9
        chosen_probs = np.clip(chosen_probs, eps, 1 - eps)

        # Log-likelihood
        ll = np.nansum(np.log(chosen_probs))
        return -ll  # For minimization

    def fit(self, bounds, x0=None, method="diff_evolution", n_opts=1, n_cpu=1):
        # Optimize params using a bounded method
        if method == "bads":
            from pybads import BADS

            pass

        elif method == "diff_evolution":
            x_vec = np.zeros((n_opts, bounds.shape[0]))
            fval_vec = np.zeros(n_opts)

            for opt_count in range(n_opts):
                result = differential_evolution(
                    self.log_likelihood,
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
                x_vec[opt_count] = result.x
                fval_vec[opt_count] = result.fun

            idx_best = np.argmin(fval_vec)
            self.estimated_params = x_vec[idx_best]

        else:
            result = minimize(
                self.log_likelihood,
                x0=[0.5, -0.5, 1],
                bounds=[(0, 1), (0, 1), (0, 10)],
                method="L-BFGS-B",
            )

            self.alpha_c, self.alpha_u, self.beta = result.x

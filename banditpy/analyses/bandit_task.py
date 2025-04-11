import numpy as np
import pandas as pd
from .. import core
from scipy.optimize import minimize
from joblib import Parallel, delayed

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression


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


def get_port_bias_2ab(df, min_trials=250):

    ntrials_by_session = get_trial_metrics(df)[0]

    delta_prob = df["rewprobfull1"] - df["rewprobfull2"]
    port_choice = df["port"].to_numpy()
    port_choice[port_choice == 2] = -1
    is_choice_high = df["is_choice_high"]
    max_diff = 80
    nbins = int(2 * max_diff / 10) + 1
    bins = np.linspace(-max_diff, max_diff, nbins)

    mean_choice = stats.binned_statistic(
        delta_prob, port_choice, bins=bins, statistic="mean"
    )[0]
    bin_centers = bins[:-1] + 5
    # mean_choice[bin_centers < 0] = -1 * mean_choice[bin_centers < 0]

    return mean_choice, bin_centers


class HistoryBasedLogisticModel:
    """Based on Miller et al. 2021, "From predictive models to cognitive models....." """

    def __init__(self, mab: core.MultiArmedBandit, n_hist=5):
        assert mab.n_ports == 2, "Only 2-armed bandit task is supported"
        self.choices, self.rewards = self._reformat_choices_rewards(
            mab.choices, mab.rewards
        )
        self.rewards = mab.rewards
        self.n_hist = n_hist
        self.model = LogisticRegression(solver="lbfgs")
        self.feature_names = []

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
        # Generate sliding windows
        C_windows = sliding_window_view(choices, window_shape=self.n_hist)[:-1]
        R_windows = sliding_window_view(rewards, window_shape=self.n_hist)[:-1]
        targets = choices[self.n_hist :]

        # Interaction term
        CxR_windows = C_windows * R_windows

        # Stack features: shape (n_samples, n_lags * 3)
        X = np.hstack([C_windows, R_windows, CxR_windows])
        y = (targets == 1).astype(int)  # 1 for right, 0 for left

        if not self.feature_names:
            for k in range(1, self.n_hist + 1):
                self.feature_names.extend([f"C_{k}", f"O_{k}", f"CxO_{k}"])

        return X, y

    def fit(self, choices, outcomes):
        X, y = self._prepare_features(choices, outcomes)
        self.model.fit(X, y)
        return self

    def predict_proba(self, choices, outcomes):
        X, _ = self._prepare_features(choices, outcomes)
        return self.model.predict_proba(X)[:, 1]  # Prob of choosing right

    def get_coefficients(self):
        return dict(zip(self.feature_names, self.model.coef_[0]))


class QlearningEstimator:
    """Estimate Q-learning parameters for a multi-armed bandit task"""

    def __init__(self, mab: core.MultiArmedBandit):
        assert mab.n_ports == 2, "This task has more than 2 ports"
        self.choices = mab.get_binarized_choices().astype(int)
        self.rewards = mab.rewards
        self.is_session_start = mab.is_session_start
        self.session_ids = mab.session_ids
        self.alpha_c = None  # Learning rate for chosen action
        self.alpha_u = None  # Learning rate for unchosen action
        self.beta = None  # Inverse temperature parameter

    def print_params(self):
        print(f"alpha_c: {self.alpha_c}, alpha_u: {self.alpha_u}, beta: {self.beta}")

    def compute_q_values(self, alpha_c, alpha_u):
        Q = np.zeros(2)  # Initialize Q-values for two actions
        q_values = []

        for choice, reward, is_start in zip(
            self.choices, self.rewards, self.is_session_start
        ):
            if is_start:
                Q[:] = 0.0  # Reset Q-values at session start

            unchosen = 1 - choice

            # Q-learning update
            Q[choice] += alpha_c * (reward - Q[choice])
            Q[unchosen] += alpha_u * (reward - Q[choice])
            # Q[unchosen] += alpha_u * ((1 - reward) - Q[unchosen])
            # Q[unchosen] += alpha_u * (Q[choice] - reward)

            q_values.append(Q.copy())

        return np.array(q_values)

    def log_likelihood(self, params):
        alpha_c, alpha_u, beta = params
        Q_values = self.compute_q_values(alpha_c, alpha_u)
        # Compute softmax probabilities
        betaQ = beta * Q_values
        betaQ = np.clip(betaQ, -500, 500)  # Prevent overflow
        exp_Q = np.exp(betaQ)
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        # Get the probability of the chosen action
        chosen_probs = probs[np.arange(len(self.choices)), self.choices]

        # Numerical stability
        eps = 1e-9
        chosen_probs = np.clip(chosen_probs, eps, 1 - eps)

        # Log-likelihood
        ll = np.sum(np.log(chosen_probs))
        return -ll  # For minimization

    def fit(self, lb, ub, x0=None, plb=None, pub=None, method="bads", num_opts=10):
        # Optimize params using a bounded method
        if method == "bads":
            from pybads import BADS

            optimize_results = []
            x_vec = np.zeros((num_opts, lb.shape[0]))
            fval_vec = np.zeros(num_opts)
            options = {"display": "off", "uncertainty_handling": True}

            # bads_list = [
            #     BADS(
            #         fun=self.log_likelihood,
            #         x0=None,
            #         lower_bounds=lb,
            #         upper_bounds=ub,
            #         plausible_lower_bounds=plb,
            #         plausible_upper_bounds=pub,
            #         options={
            #             "display": "off",
            #             "uncertainty_handling": True,
            #             "random_seed": opt_count,
            #         },
            #     )
            #     for opt_count in range(num_opts)
            # ]

            # with Pool(4) as p:  # 4 is the number of parallel processes
            #     results = p.map(lambda obj: obj.optimize(), bads_list)

            # def optimize_bads(bads_obj):
            #     return bads_obj.optimize()

            # with multiprocessing.Pool(4) as pool:
            #     results = pool.map(optimize_bads, bads_list)

            # print(results[0])
            # results = Parallel(n_jobs=4)(delayed(optimize_bads)(bd) for bd in bads_list)

            for opt_count in range(num_opts):
                print("Running optimization " + str(opt_count) + "...")
                options["random_seed"] = opt_count
                bads = BADS(
                    self.log_likelihood,
                    x0=None,
                    lower_bounds=lb,
                    upper_bounds=ub,
                    plausible_lower_bounds=plb,
                    plausible_upper_bounds=pub,
                    options=options,
                )
                optimize_results.append(bads.optimize())
                x_vec[opt_count] = optimize_results[opt_count].x
                fval_vec[opt_count] = optimize_results[opt_count].fval

            idx_best = np.argmin(fval_vec)
            result_best = optimize_results[idx_best]
            x_min = result_best["x"]
            self.alpha_c, self.alpha_u, self.beta = x_min

        else:
            result = minimize(
                self.log_likelihood,
                x0=[0.5, -0.5, 1],
                bounds=[(0, 1), (0, 1), (0, 10)],
                method="L-BFGS-B",
            )

            self.alpha_c, self.alpha_u, self.beta = result.x

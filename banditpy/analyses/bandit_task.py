import numpy as np
import pandas as pd
from .. import core
from scipy.optimize import minimize


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


class QlearningEstimator:
    """Estimate Q-learning parameters for a multi-armed bandit task"""

    def __init__(self, mab: core.MultiArmedBandit):
        self.mab = mab

    def compute_q_values(self, alpha_c, alpha_u):
        Q = np.zeros(2)  # Q-values: Q[0] for Left, Q[1] for Right
        q_values = []

        for choice, reward in zip(self.mab.choices, self.mab.rewards):
            # If Left (0) is chosen, Right (1) is unchosen, and vice versa
            unchosen = 1 - choice

            # Update Q-values for chosen and unchosen arms
            Q[choice] += alpha_c * (reward - Q[choice])  # Chosen action update
            Q[unchosen] += alpha_u * (0 - Q[unchosen])  # Unchosen action decay

            q_values.append(Q.copy())

        return np.array(q_values)

    def log_likelihood(self, params):
        # Log-likelihood function to optimize alpha_L and alpha_R
        alpha_L, alpha_R, beta = params
        Q_values = self.compute_q_values(alpha_L, alpha_R)

        # Predictor for logistic regression: difference in Q-values (Q_right - Q_left)
        # X = (Q_values[:, 1] - Q_values[:, 0]).reshape(-1, 1)
        # model = LogisticRegression()
        # model.fit(X, choices)  # Fit logistic regression on choice data
        # probs = model.predict_proba(X)[:, 1]  # Probability of choosing right (1)

        Q_diff = Q_values[:, 1] - Q_values[:, 0]
        probs = 1 / (1 + np.exp(-beta * Q_diff))  # Softmax choice probability

        # Compute log-likelihood
        ll = np.sum(choices * np.log(probs) + (1 - choices) * np.log(1 - probs))
        return -ll  # Negative for minimization

    def fit(self):
        # Optimize alpha_L and alpha_R using a bounded method
        result = minimize(
            self.log_likelihood,
            x0=[0.5, -0.5, 1],
            bounds=[(0, 1), (0, 1), (0, 10)],
            method="L-BFGS-B",
        )

        estimated_params.append(result.x)
        alpha_L_est, alpha_R_est, beta = result.x
        print(
            f"Estimated alpha_L: {alpha_L_est:.4f}, Estimated alpha_R: {alpha_R_est:.4f}, Estimated: beta: {beta}"
        )

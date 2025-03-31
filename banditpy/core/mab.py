import numpy as np
from .datawriter import DataWriter
from scipy import stats


class MultiArmedBandit(DataWriter):
    """
    A class to hold a multi-armed bandit task data.

    """

    def __init__(
        self,
        probs,
        port_choice,
        prob_choice,
        is_reward,
        trial_session_id,
        starts,
        stops,
        datetime,
        metadata=None,
    ):

        super().__init__(metadata=metadata)

        self.probs = probs
        self.port_choice = port_choice
        self.prob_choice = prob_choice
        self.is_reward = is_reward
        self.starts = starts
        self.stops = stops
        self.trial_session_id = trial_session_id
        self.trial_session_id = trial_session_id
        self.datetime = datetime

        self.session_ids, self.ntrials_session = np.unique(
            self.trial_session_id, return_counts=True
        )

    @property
    def n_ports(self):
        return self.probs.shape[0]

    @property
    def is_choice_high(self):
        return np.max(self.probs, axis=0) == self.prob_choice

    @property
    def mean_ntrials(self):
        return np.mean(self.ntrials_session)

    @property
    def min_ntrials(self):
        return np.min(self.ntrials_session)

    @property
    def max_ntrials(self):
        return np.max(self.ntrials_session)

    def from_csv(self):
        return None

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

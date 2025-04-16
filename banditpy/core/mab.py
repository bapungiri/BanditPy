import numpy as np
from neuropy.core import DataWriter
from scipy import stats
import pandas as pd


class MultiArmedBandit(DataWriter):
    """
    A class to hold a multi-armed bandit task data.

    """

    def __init__(
        self,
        probs,  # array size n_trials x n_ports
        choices,  # array size n_trials
        rewards,  # array size n_trials
        session_ids,
        starts,
        stops,
        datetime,
        metadata=None,
    ):
        super().__init__(metadata=metadata)
        assert probs.ndim == 2, "probs must be 2D array"
        assert (
            probs.shape[1] == 2
        ), "probs must be 2D array with 2 columns, This is implemented for 2AB task only"
        assert probs.shape[0] == len(choices), "probs and choices must have same length"
        assert len(choices) == len(rewards), "choices and rewards must have same length"
        assert len(rewards) == len(
            session_ids
        ), "choices and session_ids must have same length"

        self.probs = probs
        self.choices = choices.astype(int)
        self.rewards = rewards.astype(int)
        self.starts = starts
        self.stops = stops
        self.session_ids = session_ids
        self.datetime = datetime

        self.sessions, self.ntrials_session = np.unique(
            self.session_ids, return_counts=True
        )

    @property
    def n_ports(self):
        return self.probs.shape[1]

    @property
    def is_choice_high(self):
        return np.max(self.probs, axis=1) == self.choices

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

    @property
    def probs_corr(self):
        return stats.pearsonr(self.probs[:, 0], self.probs[:, 1])[0]

    @property
    def is_structured(self):
        return np.abs(self.probs_corr) > 0.9

    @property
    def is_session_start(self):
        session_starts = np.diff(self.session_ids, prepend=self.session_ids[0])
        session_starts = np.clip(session_starts, 0, 1)
        session_starts[0] = 1  # First trial is always a start
        return session_starts

    def get_binarized_choices(self):
        """Get binarized choices for the task

        Returns
        -------
        _type_
            _description_
        """
        return np.where(self.choices == 1, 1, 0)

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

    @staticmethod
    def from_csv(fp):
        """This function primarily written to handle data from anirudh's bandit task/processed data

        Parameters
        ----------
        fp : .csv file name
            File path to the csv file that contains the data

        Returns
        -------
        _type_
            _description_
        """
        df = pd.read_csv(fp)
        return MultiArmedBandit(
            probs=df.loc[:, ["rewprobfull1", "rewprobfull2"]].to_numpy(),
            choices=df["port"].to_numpy(),
            rewards=df["reward"].to_numpy(),
            session_ids=df["session#"].to_numpy(),
            starts=df["trialstart"].to_numpy(),
            stops=df["trialend"].to_numpy(),
            datetime=df["datetime"].to_numpy(),
            metadata=None,
        )

    def keep_by_trials(self, min_trials=100, clip_max=None):
        """Keep only sessions with more than min_trials and optionally clip to max trials.

        Parameters
        ----------
        min_trials : int, optional
            Minimum number of trials in a session, by default 100
        clip_max : int, optional
            Maximum number of trials to keep per session, by default 100

        Returns
        -------
        MultiArmedBandit
            A new MultiArmedBandit instance with filtered sessions and trials.
        """
        ntrials_by_session = self.ntrials_session
        valid_sessions = self.sessions[ntrials_by_session >= min_trials]
        mask = np.isin(self.session_ids, valid_sessions)

        # Filter data by valid sessions
        filtered_probs = self.probs[mask]
        filtered_choices = self.choices[mask]
        filtered_rewards = self.rewards[mask]
        filtered_session_ids = self.session_ids[mask]
        filtered_starts = self.starts[mask]
        filtered_stops = self.stops[mask]
        filtered_datetime = self.datetime[mask]

        # Clip to max trials per session if clip_max is specified
        if clip_max is not None:
            clipped_mask = []
            for session in valid_sessions:
                session_mask = filtered_session_ids == session
                session_indices = np.where(session_mask)[0][:clip_max]
                clipped_mask.extend(session_indices)

            clipped_mask = np.array(clipped_mask)
            filtered_probs = filtered_probs[clipped_mask]
            filtered_choices = filtered_choices[clipped_mask]
            filtered_rewards = filtered_rewards[clipped_mask]
            filtered_session_ids = filtered_session_ids[clipped_mask]
            filtered_starts = filtered_starts[clipped_mask]
            filtered_stops = filtered_stops[clipped_mask]
            filtered_datetime = filtered_datetime[clipped_mask]

        return MultiArmedBandit(
            probs=filtered_probs,
            choices=filtered_choices,
            rewards=filtered_rewards,
            session_ids=filtered_session_ids,
            starts=filtered_starts,
            stops=filtered_stops,
            datetime=filtered_datetime,
            metadata=None,
        )

    def trim_sessions(self, trial_start, trial_stop):
        pass

    def keep_by_deltaprob(probs, session_ids, min_diff=20):
        """
        Select sessions where the probability difference between two ports exceeds a threshold.

        Parameters:
            probs (array-like): Array of shape (n_trials, 2) containing probabilities for two ports.
            session_ids (array-like): Array of session IDs corresponding to each trial.
            min_diff (float): Minimum probability difference threshold. Default is 0.2.

        Returns:
            array-like: Array of session IDs that meet the criteria.
        """
        # Calculate the absolute difference between probabilities of the two ports
        prob_diff = np.abs(probs[:, 0] - probs[:, 1])

        # Identify sessions where the probability difference exceeds the threshold
        valid_sessions = np.unique(session_ids[prob_diff > min_diff])

        return valid_sessions

import numpy as np
from neuropy.core import DataWriter
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class BanditTask(DataWriter):
    def __init__(self, metadata=None):
        super().__init__(metadata)


class TwoArmedBandit(DataWriter):
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

        self.probs = self._fix_probs(probs)
        self.choices = self._fix_choices(choices.astype(int))
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
        return (np.argmax(self.probs, axis=1) + 1) == self.choices

    @property
    def mean_ntrials(self):
        return np.mean(self.ntrials_session)

    @property
    def min_ntrials(self):
        return np.min(self.ntrials_session)

    @property
    def max_ntrials(self):
        return np.max(self.ntrials_session)

    @property
    def n_sessions(self):
        return len(self.sessions)

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
        return session_starts.astype(bool)

    @property
    def session_probs(self):
        """Get the probabilities for each session.

        Returns
        -------
        _type_
            _description_
        """
        return np.array([self.probs[self.session_ids == s] for s in self.sessions])

    def _fix_probs(self, probs):
        """Fix the probs to be between 0 and 1. Most of our sessions have probabilities in 0-100 range.

        Parameters
        ----------
        probs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if probs.max() > 1:
            probs = probs / 100

        # probs = np.clip(probs, 0, 1)
        return probs

    def _fix_choices(self, choices):
        """Fix choices to start from 1,2.....

        Parameters
        ----------
        probs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        choices = choices - choices.min() + 1

        return choices

    def get_binarized_choices(self):
        """get choices coded as 0 and 1

        Returns
        -------
        _type_
            _description_
        """
        assert self.n_ports == 2, "Only implemented for 2AB task"
        return np.where(self.choices == 2, 1, 0)  # Port 1: 0 and Port 2: 1

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

    def filter_by_trials(self, min_trials=100, clip_max=None):
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

    def filter_by_deltaprob(self, delta_min, delta_max=None):
        """Keep only sessions with delta probabilities between min and max.

        Note: Only implemented for 2AB task

        Parameters
        ----------
        min : float
            Minimum delta probability to keep
        max : float, optional
            Maximum delta probability to keep, by default None

        Returns
        -------
        Multi
        """
        assert self.n_ports == 2, "This method is only implemented for 2AB task"

        if delta_max is not None:
            assert delta_min < delta_max, "min should be less than max"
        else:
            delta_max = 1

        # Calculate the absolute difference between probabilities of the two ports
        prob_diff = np.abs(np.diff(self.probs, axis=1).flatten())

        # Identify sessions where the probability difference exceeds the threshold
        delta_bool = (prob_diff >= delta_min) & (prob_diff <= delta_max)
        valid_sessions = np.unique(self.session_ids[delta_bool])

        return self.filter_by_session_id(valid_sessions)

    def filter_by_session_id(self, ids):
        """Keep only sessions with the specified IDs.

        Parameters
        ----------
        ids : list of int
            list of session IDs to keep

        Returns
        -------
        MultiArmedBandit
            A new MultiArmedBandit instance with filtered sessions.
        """

        mask = np.isin(self.session_ids, ids)

        # Filter data by valid sessions
        filtered_probs = self.probs[mask]
        filtered_choices = self.choices[mask]
        filtered_rewards = self.rewards[mask]
        filtered_session_ids = self.session_ids[mask]
        filtered_starts = self.starts[mask]
        filtered_stops = self.stops[mask]
        filtered_datetime = self.datetime[mask]

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

    def get_switch_prob(self, session_id=None):
        """Get the probability of switching between two ports in a session.

        Parameters
        ----------
        session_id : int,array-like, optional
            The session IDs to analyze. If None, all sessions are analyzed.

        Returns
        -------
        float
            The probability of switching between two ports in the specified session.
        """

        if session_id is None:
            print("Calculating switch probability for all sessions")
            session_id = self.sessions
            # Ignore the first trial of each session
            valid_indices = ~self.is_session_start

            # Calculate switches (change in choices)
            switches = np.diff(self.choices, prepend=self.choices[0]) != 0

            # Calculate switching probability
            switch_probability = np.mean(switches[valid_indices])

        elif isinstance(session_id, (list, np.ndarray)):
            print(f"Calculating switch probability for sessions: {session_id}")
            mask = self.session_ids == session_id
            choices = self.choices[mask]

            # Calculate the number of switches and total trials in the session
            switches = np.sum(np.diff(choices) != 0)
            total_trials = len(choices) - 1

            # Calculate the switch probability
            switch_probability = switches / total_trials if total_trials > 0 else 0

        return switch_probability

    def get_switch_prob_by_trial(self):
        """Get the probability of switching between ports as a function of trials.

        Returns
        -------
        array-like
            The probability of switching between two ports in the specified session.
        """

        # Calculate switches (change in choices)
        switches = np.diff(self.choices, prepend=self.choices[0]) != 0

        # convert to 2D array of shape (n_sessions, n_trials)
        switches = pd.DataFrame(
            np.split(switches, np.cumsum(self.ntrials_session)[:-1])
        ).to_numpy()

        # Calculate switch probability across session and ignore the first trial
        switch_prob = np.nanmean(switches, axis=0)[1:]

        return switch_prob

    def get_switch_prob_by_history(self, n_past):
        """Get the probability of switching between ports as a function of history.

        References
        ----------
        Beron et al. 2022

        Parameters
        ----------
        n_past : int, optional
            History length

        Returns
        -------
        array
            The probability of switching on next action. If your n_past is 3, then probability of switching on next choice given unique sequence of 3 past actions/rewards.
        array_like
            Unique sequences.


        """

        assert self.n_ports == 2, "Only implemented for 2AB task"
        # Calculate switches (change in choices)
        choices = self.choices.copy()
        rewards = self.rewards.copy()
        rewards[rewards == 0] = -1

        def merge_symmetry(arr):
            if arr[0] == 2:
                new_arr = arr.copy()
                indx1 = np.where(arr == 1)[0]
                indx2 = np.where(arr == 2)[0]
                new_arr[indx1] = 2
                new_arr[indx2] = 1
                return new_arr
            else:
                return arr

        # Converting to history view slices
        choices_history = sliding_window_view(choices, n_past)[:-1]
        choices_history = np.array([merge_symmetry(_) for _ in choices_history])
        rewards_history = sliding_window_view(rewards, n_past)[:-1]

        # Encoding reward in choices
        choices_x_reward = choices_history * rewards_history

        switches = (np.diff(choices, prepend=choices[0]) != 0)[n_past : len(choices)]

        unq_history = np.unique(choices_x_reward, axis=0)
        unq_indx = [
            np.where(np.all(choices_x_reward == _, axis=1))[0] for _ in unq_history
        ]
        switch_prob = np.array([np.mean(switches[_]) for _ in unq_indx])

        return switch_prob, unq_history

    def get_performance(self, bin_size=None, as_df=False):
        """Get performance on two armed bandit task

        Parameters
        ----------
        bin_size : int, optional
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

        assert self.n_ports == 2, "This method is only implemented for 2AB task"

        # converting into n_sessions x n_trials dataframe and then into numpy array, converting to dataframe automatically handles the NaN values for sessions which has fewer trials
        is_choice_high_per_session = pd.DataFrame(
            np.split(
                self.is_choice_high.astype(int), np.cumsum(self.ntrials_session)[:-1]
            )
        ).to_numpy()

        assert (
            is_choice_high_per_session.shape[0] == self.n_sessions
        ), f"Number of sessions not matching"

        if bin_size is not None:

            sess_div_perf = np.array_split(
                is_choice_high_per_session, self.n_sessions // bin_size, axis=0
            )
            sess_div_perf = np.array([np.nanmean(_, axis=0) for _ in sess_div_perf])
        else:
            sess_div_perf = np.nanmean(is_choice_high_per_session, axis=0)

        return sess_div_perf

    def get_cummulative_reward(self):
        """Get the cumulative rewards for each session.

        Returns
        -------
        array-like
            The cumulative rewards for each session.
        """
        return np.array(
            [np.cumsum(session_rewards) for session_rewards in self.rewards]
        )

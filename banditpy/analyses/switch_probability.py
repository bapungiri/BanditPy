import numpy as np
import pandas as pd
from ..core import TwoArmedBandit
from numpy.lib.stride_tricks import sliding_window_view


class SwitchProbability2AB:
    def __init__(self, task: TwoArmedBandit):
        assert isinstance(task, TwoArmedBandit), "task must be a TwoArmedBandit object"
        self.task = task

    def by_session(self, session_id=None):
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
            session_id = self.task.sessions
            # Ignore the first trial of each session
            valid_indices = ~self.task.is_session_start

            # Calculate switches (change in choices)
            switches = np.diff(self.task.choices, prepend=self.task.choices[0]) != 0

            # Calculate switching probability
            switch_probability = np.mean(switches[valid_indices])

        elif isinstance(session_id, (list, np.ndarray)):
            print(f"Calculating switch probability for sessions: {session_id}")
            mask = self.task.session_ids == session_id
            choices = self.task.choices[mask]

            # Calculate the number of switches and total trials in the session
            switches = np.sum(np.diff(choices) != 0)
            total_trials = len(choices) - 1

            # Calculate the switch probability
            switch_probability = switches / total_trials if total_trials > 0 else 0

        return switch_probability

    def by_trial(self):
        """Get the probability of switching between ports as a function of trials.

        Returns
        -------
        array-like
            The probability of switching between two ports in the specified session.
        """

        # Calculate switches (change in choices)
        switches = np.diff(self.task.choices, prepend=self.task.choices[0]) != 0

        # convert to 2D array of shape (n_sessions, n_trials)
        switches = pd.DataFrame(
            np.split(switches, np.cumsum(self.task.ntrials_session)[:-1])
        ).to_numpy()

        # Calculate switch probability across session and ignore the first trial
        switch_prob = np.nanmean(switches, axis=0)[1:]

        return switch_prob

    def by_history(self, n_past, history_as_str=False):
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
            Unique sequences. Coded as 1,-1,2,-2. Where positive numbers are rewarded and negative numbers are unrewarded. 1 -> animal stayed with previous choice, 2 -> switched to the other port.


        """

        assert self.task.n_ports == 2, "Only implemented for 2AB task"
        # Calculate switches (change in choices)
        choices = self.task.choices.copy()
        rewards = self.task.rewards.copy()
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

        if history_as_str:
            seq = unq_history.astype(str)
            seq[seq == "-1"] = "a"
            seq[seq == "1"] = "A"
            seq[seq == "-2"] = "b"
            seq[seq == "2"] = "B"

            seq = np.array(["".join(map(str, _)) for _ in seq])

            return switch_prob, seq

        else:
            return switch_prob, unq_history

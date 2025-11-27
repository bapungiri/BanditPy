import numpy as np
from .data_manager import DataManager
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class BanditTask(DataManager):
    """
    Base class for multi-armed bandit task data.

    Structure:
    ----------
    - Each trial contains: reward probabilities (probs), choices, rewards, session_ids, and optionally block_ids, window_ids, start/stop indices, and datetimes.
    - Trials are grouped into sessions (session_ids), blocks (block_ids), and windows (window_ids) for flexible analysis.

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (n_trials, n_arms) with reward probabilities for each arm at each trial.
    choices : np.ndarray
        Array of shape (n_trials,) with chosen arm indices (1-based).
    rewards : np.ndarray
        Array of shape (n_trials,) with reward outcomes for each trial.
    session_ids : np.ndarray
        Array of shape (n_trials,) with session identifiers (monotonically increasing, unique for each reward probability combination).
    block_ids : np.ndarray, optional
        Array of shape (n_trials,) with block identifiers (reset to 1 at the start of each experiment window, increment for each uncued reward probability change).
    window_ids : np.ndarray, optional
        Array of shape (n_trials,) with window identifiers.
    starts : np.ndarray, optional
        Array of shape (n_trials,) indicating trial start indices.
    stops : np.ndarray, optional
        Array of shape (n_trials,) indicating trial stop indices.
    datetime : np.ndarray, optional
        Array of shape (n_trials,) with datetime information for each trial.
    metadata : dict, optional
        Dictionary of additional metadata.

    Key Properties
    --------------
    n_ports : int
        Number of arms in the bandit task.
    n_trials : int
        Number of trials in the dataset.
    mean_ntrials, min_ntrials, max_ntrials : float/int
        Statistics on number of trials per session.
    n_sessions : int
        Number of unique sessions.
    sessions : np.ndarray
        Unique session IDs.
    ntrials_session : np.ndarray
        Number of trials per session.
    is_session_start : np.ndarray
        Boolean array, True at the start of each session.
    is_window_start : np.ndarray
        Boolean array, True at the start of each window (requires window_ids).

    Methods
    -------
    filter_by_trials(min_trials=100, clip_max=None)
        Filter sessions by minimum number of trials and optionally clip to max trials per session.
    filter_by_session_id(ids)
        Filter trials by session IDs.
    to_df()
        Convert the bandit task data to a pandas DataFrame.
    auto_block_window_ids(time_window_min=40)
        Auto-generate block_ids and window_ids using datetime and reward probability changes.
    _filtered(mask)
        Return a new BanditTask instance with filtered data.

    Usage Example
    -------------
    >>> task = BanditTask(probs, choices, rewards, session_ids, datetime=datetimes)
    >>> block_ids, window_ids = task.auto_block_window_ids(time_window_min=40)
    >>> df = task.to_df()
    >>> filtered_task = task.filter_by_trials(min_trials=50)

    Notes
    -----
    - session_ids should be unique for each reward probability combination and increment globally.
    - block_ids reset at the start of each window and increment for each uncued reward probability change.
    - window_ids reset at the start of each experiment window (e.g., every 40 minutes).
    - Use is_session_start, is_window_start for custom resetting logic in models.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(metadata=metadata)
        assert probs.ndim == 2, "probs must be (n_trials, n_arms)"
        assert probs.shape[0] > probs.shape[1], "n_arms can't be greater than n_trials"
        assert (
            probs.shape[0] == len(choices) == len(rewards) == len(session_ids)
        ), "Mismatch in trials length"

        self.probs = self._fix_probs(probs)
        self.choices = self._fix_choices(choices.astype(int))
        self.rewards = rewards.astype(int)
        self.session_ids = session_ids

        if starts is not None:
            assert starts.shape[0] == probs.shape[0], "starts should be of same length"
            self.starts = starts
        else:
            self.starts = None

        if stops is not None:
            assert stops.shape[0] == probs.shape[0], "stops should be of same length"
            self.stops = stops
        else:
            self.stops = None

        self.window_ids = window_ids
        self.block_ids = block_ids
        self.datetime = datetime

        self.sessions, self.ntrials_session = np.unique(
            self.session_ids, return_counts=True
        )

    @staticmethod
    def _fix_probs(probs):
        return probs / 100 if probs.max() > 1 else probs

    @staticmethod
    def _fix_choices(choices):
        return choices - choices.min() + 1

    @property
    def n_ports(self):
        return self.probs.shape[1]

    @property
    def n_trials(self):
        return self.probs.shape[0]

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

    @property
    def is_session_start(self):
        session_starts = np.diff(self.session_ids, prepend=self.session_ids[0])
        session_starts = np.clip(session_starts, 0, 1)
        session_starts[0] = 1
        return session_starts.astype(bool)

    @property
    def is_window_start(self):
        assert self.window_ids is not None, "window_ids must be set"
        window_starts = np.diff(self.window_ids, prepend=self.window_ids[0])
        window_starts = np.clip(window_starts, 0, 1)
        window_starts[0] = 1
        return window_starts.astype(bool)

    def filter_by_trials(self, min_trials=100, clip_max=None):
        valid_sessions = self.sessions[self.ntrials_session >= min_trials]
        mask = np.isin(self.session_ids, valid_sessions)

        if clip_max is not None:
            clipped_mask = []
            for session in valid_sessions:
                session_mask = self.session_ids == session
                session_indices = np.where(session_mask)[0][:clip_max]
                clipped_mask.extend(session_indices)
            mask = np.zeros_like(self.session_ids, dtype=bool)
            mask[clipped_mask] = True

        return self._filtered(mask)

    def filter_by_session_id(self, ids):
        mask = np.isin(self.session_ids, ids)
        return self._filtered(mask)

    def _filtered(self, mask):
        """Return a new instance of the same class with filtered data."""
        return self.__class__(
            probs=self.probs[mask],
            choices=self.choices[mask],
            rewards=self.rewards[mask],
            session_ids=self.session_ids[mask],
            block_ids=None if self.block_ids is None else self.block_ids[mask],
            window_ids=None if self.window_ids is None else self.window_ids[mask],
            starts=None if self.starts is None else self.starts[mask],
            stops=None if self.stops is None else self.stops[mask],
            datetime=None if self.datetime is None else self.datetime[mask],
            metadata=self.metadata,
        )

    @classmethod
    def from_csv(cls):
        pass

    def to_df(self):
        """Convert the bandit task data to a pandas DataFrame."""
        data = pd.DataFrame(
            {
                "choices": self.choices,
                "rewards": self.rewards,
                "session_ids": self.session_ids,
                "block_ids": self.block_ids,
                "window_ids": self.window_ids,
                "starts": self.starts,
                "stops": self.stops,
                "datetime": self.datetime,
            }
        )

        for i in range(self.probs.shape[1]):
            data.insert(i + 2, f"probs_{i+1}", self.probs[:, i])

        return data

    def auto_block_window_ids(self, time_window_min=40):
        """
        Auto-generate window_ids and block_ids for each trial.

        - A block is defined as a contiguous set of trials with fixed reward probabilities.
        - block_ids reset to 1 at the start of each experiment window (e.g., 40-min period).
        - session_ids are global and unique for each reward probability combination (do not reset).

        Parameters
        ----------
        time_window_min : int
            Time window in minutes for splitting blocks by datetime.

        Returns
        -------
        block_ids : np.ndarray
            Array of block IDs for each trial (starts at 1 for each experiment window).
        window_ids : np.ndarray
            Array of window IDs for each trial (starts at 1 for first window).
        """
        n_trials = len(self.probs)
        session_ids = self.session_ids
        block_ids = np.zeros(n_trials, dtype=int)
        window_ids = np.zeros(n_trials, dtype=int)

        # If datetime is provided, split windows by time gaps > time_window_min
        assert (
            self.datetime is not None
        ), "Datetime must be provided for window splitting."
        dt = self.datetime.astype("datetime64[s]")
        gap = np.diff(dt, prepend=dt[0]).astype("timedelta64[s]").astype(int) / 60
        window_starts_bool = gap > time_window_min
        window_starts_bool[0] = 1  # Ensure first trial is a window start
        window_ids = np.cumsum(window_starts_bool)

        _, counts = np.unique(window_ids, return_counts=True)
        chunks = np.split(session_ids, np.cumsum(counts)[:-1])
        block_ids = np.concatenate([chunk - chunk[0] + 1 for chunk in chunks])

        self.block_ids = block_ids
        self.window_ids = window_ids


class Bandit2Arm(BanditTask):
    """
    A class to hold a two armed bandit task data.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            block_ids=block_ids,
            window_ids=window_ids,
            starts=starts,
            stops=stops,
            datetime=datetime,
            metadata=metadata,
        )
        assert self.n_ports == 2, "TwoArmedBandit requires exactly 2 arms"

    @property
    def is_block_start(self):
        """Boolean array, True at the start of each block. Requires block_ids."""
        assert self.block_ids is not None, "block_ids must be set"
        block_starts = np.diff(self.block_ids, prepend=self.block_ids[0])
        block_starts = np.clip(block_starts, 0, 1)
        block_starts[0] = 1
        return block_starts.astype(bool)

    @property
    def is_choice_high(self):
        # return (np.argmax(self.probs, axis=1) + 1) == self.choices
        return (
            np.max(self.probs, axis=1)
            == self.probs[np.arange(self.n_trials), self.choices - 1]
        )

    @property
    def probs_corr(self):
        return stats.pearsonr(self.probs[:, 0], self.probs[:, 1])[0]

    @property
    def is_structured(self):
        # return np.abs(self.probs_corr) > 0.9
        return self.probs_corr < -0.9

    @property
    def session_probs(self):
        """Get the probabilities for each session.

        Returns
        -------
        _type_
            _description_
        """
        return self.probs[self.is_session_start, :]

    def get_binarized_choices(self):
        """get choices coded as 0 and 1

        Returns
        -------
        _type_
            _description_
        """
        assert self.n_ports == 2, "Only implemented for 2AB task"
        return np.where(self.choices == 2, 1, 0)  # Port 1: 0 and Port 2: 1

    def get_port_bias(df, min_trials=250):

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
    def from_csv(
        fp,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
    ):
        """Build Bandit2Arm from a CSV file.

        probs: list[str] column names for per-arm probabilities.
        choices, rewards, session_ids, block_ids, window_ids, starts, stops, datetime: column names (str).
        """
        df = pd.read_csv(fp)
        return Bandit2Arm(
            probs=df.loc[:, probs].to_numpy(),
            choices=df[choices].to_numpy(),
            rewards=df[rewards].to_numpy(),
            session_ids=df[session_ids].to_numpy(),
            block_ids=None if block_ids is None else df[block_ids].to_numpy(),
            window_ids=None if window_ids is None else df[window_ids].to_numpy(),
            starts=None if starts is None else df[starts].to_numpy(),
            stops=None if stops is None else df[stops].to_numpy(),
            datetime=None if datetime is None else df[datetime].to_numpy(),
            metadata=None,
        )

    @staticmethod
    def from_df(
        df,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
    ):
        """Build Bandit2Arm from an existing DataFrame."""
        return Bandit2Arm(
            probs=df.loc[:, probs].to_numpy(),
            choices=df[choices].to_numpy(),
            rewards=df[rewards].to_numpy(),
            session_ids=df[session_ids].to_numpy(),
            block_ids=None if block_ids is None else df[block_ids].to_numpy(),
            window_ids=None if window_ids is None else df[window_ids].to_numpy(),
            starts=None if starts is None else df[starts].to_numpy(),
            stops=None if stops is None else df[stops].to_numpy(),
            datetime=None if datetime is None else df[datetime].to_numpy(),
            metadata=None,
        )

    def trim_sessions(self, trial_start, trial_stop):
        pass

    def filter_by_probs(self, probs):
        """Keep only sessions with probabilities that match the given probabilities.

        Parameters
        ----------
        probs : list of length or 2D numpy array, optional
            Probabilities to match. If a list, it should contain two elements representing the probabilities for each port. If a 2D numpy array, it should have shape (n, 2) where each row represents the probabilities for each port.

        Returns
        -------
        Bandit2Arm with filtered data
        """
        probs = np.array(probs).reshape(-1, 2)
        assert probs.shape[1] == 2, "This method is only implemented for 2AB task"
        mask = (self.probs[:, None, :] == probs[None, :, :]).all(axis=2).any(axis=1)
        return self._filtered(mask)

    def filter_by_deltaprob(self, delta_min, delta_max=None):
        """Keep only sessions with delta probabilities between min and max.

        Note: Only implemented for 2AB task

        Parameters
        ----------
        delta_min : float
            Minimum delta probability to keep
        delta_max : float, optional
            Maximum delta probability to keep, by default None

        Returns
        -------
        Multi
        """
        assert self.n_ports == 2, "This method is only implemented for 2AB task"

        if delta_max is not None:
            assert delta_min <= delta_max, "min should be less than max"
        else:
            delta_max = 1

        # Calculate the absolute difference between probabilities of the two ports
        prob_diff = np.abs(np.diff(self.probs, axis=1).flatten())

        # Identify sessions where the probability difference exceeds the threshold
        delta_bool = (prob_diff >= delta_min) & (prob_diff <= delta_max)
        valid_sessions = np.unique(self.session_ids[delta_bool])

        return self.filter_by_session_id(valid_sessions)

    def get_optimal_choice_probability(self, bin_size=None, as_df=False):
        """Get probability of choosing high arm on two armed bandit task

        Parameters
        ----------
        bin_size : int, optional
            no.of sessions over which performance is calculated, by default None
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

    def get_reward_probability(self):
        """Get the reward probabilities for each session.

        Returns
        -------
        array-like
            The reward probabilities for each session.
        """
        session_rewards = pd.DataFrame(
            np.hsplit(self.rewards, np.cumsum(self.ntrials_session)[:-1])
        ).to_numpy()

        return np.nanmean(session_rewards, axis=0)

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

    def get_choice_entropy(self):
        """Calculate the entropy of the probabilities for each trial.

        Returns
        -------
        array-like
            The entropy values for each trial.
        """
        choices = self.choices - 1
        session_choices = pd.DataFrame(
            np.hsplit(choices, np.cumsum(self.ntrials_session)[:-1])
        ).to_numpy()

        port2_prob = np.nanmean(session_choices, axis=0)
        port1_prob = 1 - port2_prob
        choices_prob = np.vstack((port1_prob, port2_prob))

        return stats.entropy(choices_prob, base=2, axis=0)

    def get_prob_hist_2d(self, stat="count"):
        """Get the probability matrix for each session. Calculates a 2D histogram of the reward probabilities.

        Returns
        -------
        2d array-like
            The probability matrix for each session.
        """
        probs_combinations = self.probs[self.is_session_start, :]
        p1_bins = np.linspace(0, 0.9, 10) + 0.05
        p2_bins = np.linspace(0, 0.9, 10) + 0.05
        H, xedges, yedges, _ = stats.binned_statistic_2d(
            probs_combinations[:, 0].astype(float),
            probs_combinations[:, 1].astype(float),
            values=probs_combinations[:, 0],
            statistic="count",
            bins=[p1_bins, p2_bins],
        )

        if stat == "prop":
            H = H / probs_combinations.shape[0]

        return H.T, xedges, yedges

    def get_trials_hist(self, bin_size=10):
        """Get histogram of number of trials per session."""

        ntrials_per_session = self.ntrials_session
        min_trials = ntrials_per_session.min()
        max_trials = ntrials_per_session.max()

        h, bins = np.histogram(
            ntrials_per_session,
            bins=np.arange(min_trials, max_trials + bin_size, bin_size),
        )

        return h, bins[:-1] + bin_size / 2

    def get_performance_prob_grid(self):
        """Get performance grid based on reward probabilities.

        Parameters
        ----------
        bin_size : float, optional
            Size of the bins for reward probabilities, by default 0.1

        Returns
        -------
        performance_grid : 2D array
            Grid of performance values.
        xedges : 1D array
            Edges of the bins along the x-axis.
        yedges : 1D array
            Edges of the bins along the y-axis.
        """
        probs_combinations = self.probs[self.is_session_start, :]
        unique_combinations = np.unique(probs_combinations, axis=0)
        bin_size = np.min(np.diff(probs_combinations[:, 0]))
        performance = self.get_optimal_choice_probability()

        p_bins = np.arange(0, 1 + bin_size, bin_size)
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(
            probs_combinations[:, 0].astype(float),
            probs_combinations[:, 1].astype(float),
            values=performance,
            statistic="mean",
            bins=[p_bins, p_bins],
        )

        return H.T, xedges, yedges


class Bandit4Arm(BanditTask):
    """
    A class to hold a multi-armed bandit task data.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            starts=starts,
            stops=stops,
            datetime=datetime,
            metadata=metadata,
        )
        assert self.n_ports == 4, "Bandit4Arm requires exactly 4 arms"

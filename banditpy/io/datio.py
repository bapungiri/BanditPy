import numpy as np
import pandas as pd

from ._common import TRIAL_COLUMNS, assemble, resolve_files

# Event codes in the Teensy log.
_SESSION_PROB = "83"  # arg 1/2 sets the reward probability of that port
_PORT_ENTRY = "81"  # arg 1/2 is the port the animal chose
_REWARD = {"51": 1, "-51": 0}  # -52 is a timeout, counted as neither
_TIMEOUT = "-52"


def dat_trials(src):
    """Read '.dat' logs into the canonical per-trial table.

    The files are parsed as one continuous event stream rather than one at a
    time, because the AutoTrainer rotates its log mid-session: a session's
    reward probabilities and start time are set in one file and its trials
    can land in the next. Parsing per file would discard every trial that
    precedes the first probability event of a rotated file.

    See _common.py for the column contract.

    Parameters
    ----------
    src : Path, str, or iterable of Path/str
        Folder containing .dat files, a single .dat file, or a list of them.

    Returns
    -------
    pd.DataFrame
    """
    files = resolve_files(src, "*.dat")

    frames = []
    for fp in files:
        frame = pd.read_csv(fp, sep=",", header=None, low_memory=False)
        frame["_stem"] = fp.stem
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    code = data[0].astype(str).to_numpy()
    arg = pd.to_numeric(data[1], errors="coerce").to_numpy()
    prob = pd.to_numeric(data[4], errors="coerce").to_numpy()
    dttime = data[5].to_numpy()
    stems = data["_stem"].to_numpy()

    trials = []

    session_id = -1
    p1 = np.nan
    p2 = np.nan
    current_port = np.nan
    last_port_idx = -1
    last_outcome_idx = -1
    current_start = np.nan
    session_start_ts = np.nan

    for idx, (c, a, prb, ts) in enumerate(zip(code, arg, prob, dttime)):
        ts_val = float(ts) if not pd.isna(ts) else np.nan

        if c == _SESSION_PROB and not np.isnan(a):
            if int(a) == 1:
                session_id += 1
                session_start_ts = ts_val
                p1 = prb
            elif int(a) == 2:
                p2 = prb
            continue

        if c == _PORT_ENTRY and np.isin(a, [1, 2]):
            current_port = int(a)
            last_port_idx = idx
            current_start = ts_val
            continue

        if c in _REWARD:
            stop_time = ts_val
            if (
                not np.isnan(current_port)
                and last_port_idx > last_outcome_idx
                and not np.isnan(p1)
                and not np.isnan(p2)
                and not np.isnan(current_start)
                and not np.isnan(session_start_ts)
                and not np.isnan(stop_time)
            ):
                trials.append(
                    (
                        int(current_port),
                        _REWARD[c],
                        float(p1),
                        float(p2),
                        f"dat:{session_id}",
                        stems[idx],
                        (current_start - session_start_ts) * 1000.0,
                        (stop_time - session_start_ts) * 1000.0,
                        float(stop_time),
                        np.nan,  # the .dat log carries no block structure
                    )
                )
            current_port = np.nan
            current_start = np.nan
            last_outcome_idx = idx

        elif c == _TIMEOUT:
            last_outcome_idx = idx
            current_port = np.nan
            current_start = np.nan

    return pd.DataFrame(trials, columns=TRIAL_COLUMNS)


def dat2ArmIO(src, metadata=None):
    """Build a 'Bandit2Arm' from AutoTrainer '.dat' logs.

    Trial 'starts' and 'stops' are milliseconds relative to the beginning of
    each session; 'datetime' is the absolute trial-end timestamp.

    Parameters
    ----------
    src : Path, str, or iterable of Path/str
        Folder containing .dat files, a single .dat file, or a list of them.
    metadata : dict, optional
        Passed through to 'Bandit2Arm'.

    Returns
    -------
    Bandit2Arm

    Notes
    -----
    The .dat log carries no block structure, so 'block_ids' is left unset.
    Derive it with 'Bandit2Arm.auto_block_window_ids' when it is needed.

    See Also
    --------
    raw2ArmIO : prefers the .csv of each session and falls back to this.
    """
    return assemble([dat_trials(src)], metadata=metadata)

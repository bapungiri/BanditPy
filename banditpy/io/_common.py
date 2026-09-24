"""Shared plumbing for the task-file readers in this package.

Both readers ('csv2ArmIO', 'dat2ArmIO') follow the same three steps: resolve
whatever the caller passed into a sorted list of files, turn each file into a
canonical per-trial table, then assemble those tables into one 'Bandit2Arm'.
Keeping those steps here is what lets 'raw2ArmIO' interleave the two readers
on a per-session basis -- see rawio.py.

The canonical per-trial table has one row per trial and these columns:

    port    chosen port, 1 or 2
    reward  outcome, 0 or 1
    p1, p2  reward probability of each port
    skey    opaque key identifying the reward-probability epoch this trial
            belongs to; unique per reader and per epoch, renumbered into
            global 'session_ids' by 'assemble'
    stem    source session, e.g. 'BGF0-2025-10-04-17-12-22', so a caller can
            select which reader supplies which session
    start   trial start, ms relative to the start of its recording session
    stop    trial end, same reference as 'start'
    dt      trial end as unix epoch seconds
    block   recorded block id, or NaN for sources that carry no block
            structure (the .dat log does not)

'dt' is the column that makes the two readers interchangeable: the .csv and
.dat written for the same session agree on it to the second.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Bandit2Arm

TRIAL_COLUMNS = [
    "port",
    "reward",
    "p1",
    "p2",
    "skey",
    "stem",
    "start",
    "stop",
    "dt",
    "block",
]


def resolve_files(src, pattern):
    """Return a sorted list of files from a folder, a single file, or an iterable.

    Parameters
    ----------
    src : Path, str, or iterable of Path/str
        A directory to glob with 'pattern', one file, or an explicit list.
    pattern : str
        Glob applied when 'src' is a directory, e.g. '*.csv'.

    Returns
    -------
    list of Path
    """
    if isinstance(src, (str, Path)):
        src = Path(src)
        files = sorted(src.glob(pattern)) if src.is_dir() else [src]
    else:
        files = sorted(Path(f) for f in src)

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {src}")
    return files


def assemble(frames, sort=False, metadata=None):
    """Concatenate canonical per-trial tables into one 'Bandit2Arm'.

    Sessions are renumbered globally by counting changes in 'skey', which
    each reader makes unique per epoch. 'skey' is prefixed per reader, so
    frames from different readers can be mixed without their keys colliding.

    'block_ids' is only set when every trial carries one. A mixed-source
    folder has blocks for its .csv trials and none for its .dat trials, and
    half a block numbering is worse than none -- the caller should derive a
    consistent one with 'Bandit2Arm.auto_block_window_ids'.

    Parameters
    ----------
    frames : list of pd.DataFrame
        Tables with the columns listed in 'TRIAL_COLUMNS'.
    sort : bool, optional
        Order trials by 'dt' before numbering sessions. Needed when frames
        come from more than one reader, since each reader contributes a
        separate stretch of the same timeline. Default False keeps the
        caller's order.
    metadata : dict, optional
        Passed through to 'Bandit2Arm'.

    Returns
    -------
    Bandit2Arm
    """
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError("No trials recovered from any input file")

    data = pd.concat(frames, ignore_index=True)
    if sort:
        data = data.sort_values("dt", kind="stable").reset_index(drop=True)

    skey = data["skey"].to_numpy()
    is_new = skey[1:] != skey[:-1]
    session_ids = np.concatenate([[0], np.cumsum(is_new)])

    block = data["block"]
    block_ids = block.to_numpy() if block.notna().all() else None

    return Bandit2Arm(
        probs=data[["p1", "p2"]].to_numpy(),
        choices=data["port"].to_numpy(),
        rewards=data["reward"].to_numpy(),
        session_ids=session_ids,
        block_ids=block_ids,
        starts=data["start"].to_numpy(),
        stops=data["stop"].to_numpy(),
        datetime=data["dt"].to_numpy(),
        metadata=metadata,
    )

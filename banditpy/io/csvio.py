import numpy as np
import pandas as pd

from ._common import TRIAL_COLUMNS, assemble, resolve_files

CSV_SUFFIX = ".trial.csv"


def csv_trials(fp):
    """Read one '.trial.csv' file into the canonical per-trial table.

    See _common.py for the column contract.

    Parameters
    ----------
    fp : Path
        A '.trial.csv' written by the AutoTrainer.

    Returns
    -------
    pd.DataFrame
    """
    data = pd.read_csv(fp, sep=",")
    data = data[
        (data["eventCode"].astype(str).str.contains("200"))
        & (data["chosenPort"].isin([1, 2]))
    ]

    # An aborted session leaves a .csv with a header and no scored trials.
    if data.empty:
        return pd.DataFrame(columns=TRIAL_COLUMNS)

    block_id = data["blockId"].to_numpy()
    session_start = data["sessionStartEpochMs"].to_numpy()
    trial_end = data["trialEndRelMs"].to_numpy()
    stem = fp.name[: -len(CSV_SUFFIX)] if fp.name.endswith(CSV_SUFFIX) else fp.stem
    # A new reward-probability epoch starts wherever 'blockId' changes.
    epoch = np.cumsum(np.diff(block_id, prepend=block_id[0] - 1) != 0) - 1

    out = pd.DataFrame(
        {
            "port": data["chosenPort"].to_numpy(),
            "reward": data["rewarded"].to_numpy(),
            "p1": data["port1Prob"].to_numpy(),
            "p2": data["port2Prob"].to_numpy(),
            "skey": [f"csv:{stem}:{e}" for e in epoch],
            "stem": stem,
            "start": data["trialStartRelMs"].to_numpy(),
            "stop": trial_end,
            # Per-trial wall-clock. 'sessionStartEpochMs' alone is constant for
            # a whole session, which leaves every trial in it sharing one
            # timestamp -- too coarse to split 40-min windows, and impossible
            # to interleave with .dat trials. Adding the trial offset fixes
            # both; the result matches the .dat clock to the second.
            "dt": (session_start + trial_end) / 1000,
            "block": block_id,
        }
    )
    return out[TRIAL_COLUMNS]


def csv2ArmIO(src, metadata=None):
    """Build a 'Bandit2Arm' from AutoTrainer '.trial.csv' files.

    Expects these columns:
    eventCode: 200 or 201,
    port1Prob: Probability of reward at port 1,
    port2Prob: Probability of reward at port 2,
    chosenPort: Port chosen (1 or 2),
    rewarded: Reward outcome (0 or 1),
    trialId: Trial identifier,
    blockId: Block identifier,
    unstructuredProb: Percentage of independent reward combinations,
    sessionStartEpochMs: Start of session in epoch ms,
    blockStartRelMs: Start of block in ms relative to session start,
    trialStartRelMs: Start of trial in ms relative to session start,
    trialEndRelMs: End of trial in ms relative to session start,

    Parameters
    ----------
    src : Path, str, or iterable of Path/str
        Folder containing .csv files, a single .csv file, or a list of them.
    metadata : dict, optional
        Passed through to 'Bandit2Arm'.

    Returns
    -------
    Bandit2Arm

    See Also
    --------
    raw2ArmIO : same, but falls back to the .dat of any session whose .csv
        is missing.
    """
    files = resolve_files(src, "*.csv")
    return assemble([csv_trials(fp) for fp in files], metadata=metadata)

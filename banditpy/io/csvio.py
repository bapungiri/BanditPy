import numpy as np
import pandas as pd
from pathlib import Path
from ..core import Bandit2Arm


def csv2ArmIO(folder: Path) -> Bandit2Arm:
    """Extracting trial info from csv files with following columns:
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
    folder : Path
        Folder containing .csv files

    Returns
    -------
    Bandit2Arm

    """
    # Concatenate all .csv files
    print(sorted(folder.glob("*.csv")))
    dfs = [pd.read_csv(fp, sep=",") for fp in sorted(folder.glob("*.csv"))]
    print(f"nfiles={len(dfs)}")
    data = pd.concat(dfs, ignore_index=True)
    data = data[
        (data["eventCode"].astype(str).str.contains("200"))
        & (data["chosenPort"].isin([1, 2]))
    ]
    trial_starts = data["trialStartRelMs"].to_numpy()
    trial_stops = data["trialEndRelMs"].to_numpy()
    dt_time = data["sessionStartEpochMs"].to_numpy() / 1000

    blockId = data["blockId"].to_numpy()
    block_starts = np.where(np.diff(blockId, prepend=-1) != 0, 1, 0)
    sessionId = np.cumsum(block_starts)

    print(data["chosenPort"].unique())
    print(data.size)

    return Bandit2Arm(
        probs=data[["port1Prob", "port2Prob"]].to_numpy(),
        choices=data["chosenPort"].to_numpy(),
        rewards=data["rewarded"].to_numpy(),
        block_ids=blockId,
        session_ids=sessionId,
        # window_ids=data["window_id"].to_numpy(),
        datetime=dt_time.astype(int),
        starts=trial_starts,
        stops=trial_stops,
    )

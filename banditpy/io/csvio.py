import numpy as np
import pandas as pd
from pathlib import Path
from ..core import Bandit2Arm


def csv2ArmIO(folder: Path, code=200) -> Bandit2Arm:
    # Concatenate all .csv files
    print(sorted(folder.glob("*.csv")))
    dfs = [pd.read_csv(fp, sep=",") for fp in sorted(folder.glob("*.csv"))]
    print(f"nfiles={len(dfs)}")
    data = pd.concat(dfs, ignore_index=True)
    data = data[
        (data["eventCode"].astype(str).str.contains("200"))
        & (data["chosenPort"].isin([1, 2]))
    ]

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
        # datetime=data["datetime"].to_numpy(),
    )

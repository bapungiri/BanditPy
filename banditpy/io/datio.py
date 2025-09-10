import numpy as np
import pandas as pd
from pathlib import Path
from ..core import Bandit2Arm


def dat2ArmIO(folder: Path) -> Bandit2Arm:
    # Concatenate all .dat files
    dfs = [pd.read_csv(fp, sep=",", header=None) for fp in sorted(folder.glob("*.dat"))]
    data = pd.concat(dfs, ignore_index=True)

    code = data[0].to_numpy()
    arg = data[1].to_numpy()
    prob = data[4].to_numpy()
    dttime = data[5].to_numpy()

    data["datetime"] = dttime
    # Probability updates
    data["p1_update"] = np.where((code == 83) & (arg == 1), prob, np.nan)
    data["p2_update"] = np.where((code == 83) & (arg == 2), prob, np.nan)
    data["p1"] = data["p1_update"].ffill()
    data["p2"] = data["p2_update"].ffill()

    # Session id increments when port1 prob line appears
    data["session_id"] = np.cumsum((code == 83) & (arg == 1))
    # Ensure first session starts at 1 (only if any updates exist)
    if data["session_id"].gt(0).any():
        data.loc[data["session_id"] > 0, "session_id"] += 0

    # Port poke updates
    data["port_update"] = np.where((code == 81) & np.isin(arg, [1, 2]), arg, np.nan)
    data["port"] = data["port_update"].ffill()

    # Reward outcome mapping
    reward_map = {52: 1, -51: 0, -52: -1}
    is_outcome = np.isin(code, list(reward_map.keys()))
    outcomes = data.loc[is_outcome, ["port", "p1", "p2", "session_id", "datetime", 0]]

    print(outcomes.columns)

    # Drop rows before first probabilities (port/prob may be NaN)
    outcomes = outcomes.dropna(subset=["port", "p1", "p2", "session_id"])

    outcomes["port"] = outcomes["port"].astype(int)
    outcomes["reward"] = outcomes[0].map(reward_map)
    behav = outcomes.loc[
        :, ["port", "reward", "p1", "p2", "session_id", "datetime"]
    ].reset_index(drop=True)
    behav[behav["reward"].isin([0, 1])]

    return Bandit2Arm(
        probs=behav[["p1", "p2"]].to_numpy(),
        choices=behav["port"].to_numpy(),
        rewards=behav["reward"].to_numpy(),
        session_ids=behav["session_id"].to_numpy(),
        datetime=behav["datetime"].to_numpy(),
    )

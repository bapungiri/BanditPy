import numpy as np
import pandas as pd
from pathlib import Path
from ..core import Bandit2Arm


def _load_dat_frames(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"No .dat files found in {folder}")
    dfs = [pd.read_csv(fp, sep=",", header=None) for fp in files]
    return pd.concat(dfs, ignore_index=True)


def dat2ArmIO(folder: Path) -> Bandit2Arm:
    """Two-armed bandit task stored as `.dat` logs.

    Parameters
    ----------
    folder : Path
        Directory containing one or more `.dat` files. Files are expected
        to encode event `code`, `arg`, `prob`, and `dttime` columns.

    Returns
    -------
    Bandit2Arm
        Two-armed bandit task with trial choices, rewards, and arm probabilities.
        Trial `starts` and `stops` are expressed in milliseconds relative to the
        beginning of each session, while `datetime` retains the absolute outcome
        timestamp (seconds since the recording clock origin).
    """
    data = _load_dat_frames(folder)

    code = data[0].astype(str).to_numpy()
    arg = pd.to_numeric(data[1], errors="coerce").to_numpy()
    prob = pd.to_numeric(data[4], errors="coerce").to_numpy()
    dttime = data[5].to_numpy()

    reward_map = {"51": 1, "-51": 0}  # ignore -52 (timeouts)
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

        if c == "83" and not np.isnan(a):
            if int(a) == 1:
                session_id += 1
                session_start_ts = ts_val
                p1 = prb
            elif int(a) == 2:
                p2 = prb
            continue

        if c == "81" and np.isin(a, [1, 2]):
            current_port = int(a)
            last_port_idx = idx
            current_start = ts_val
            continue

        if c in reward_map:
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
                start_ms = (current_start - session_start_ts) * 1000.0
                stop_ms = (stop_time - session_start_ts) * 1000.0
                trials.append(
                    (
                        int(current_port),
                        reward_map[c],
                        float(p1),
                        float(p2),
                        int(session_id),
                        float(start_ms),
                        float(stop_ms),
                        float(stop_time),
                    )
                )
            current_port = np.nan
            current_start = np.nan
            last_outcome_idx = idx

        elif c == "-52":
            last_outcome_idx = idx
            current_port = np.nan
            current_start = np.nan

    behav = pd.DataFrame(
        trials,
        columns=[
            "port",
            "reward",
            "p1",
            "p2",
            "session_id",
            "start",
            "stop",
            "stop_time",
        ],
    )

    return Bandit2Arm(
        probs=behav[["p1", "p2"]].to_numpy(),
        choices=behav["port"].to_numpy(),
        rewards=behav["reward"].to_numpy(),
        session_ids=behav["session_id"].to_numpy(),
        starts=behav["start"].to_numpy(),
        stops=behav["stop"].to_numpy(),
        datetime=behav["stop_time"].to_numpy(),
    )

"""Reader for a raw_data folder that holds a mix of .csv and .dat sessions.

The AutoTrainer writes both a '.trial.csv' and a '.dat' per session, but the
.csv only started being written partway through some animals' histories, so a
folder can hold sessions that exist as .dat only. Reading such a folder with
'csv2ArmIO' silently drops those sessions; reading it with 'dat2ArmIO' drops
any session whose .dat is missing instead. 'raw2ArmIO' reads each session
from whichever file it has, preferring the .csv.
"""

import warnings
from pathlib import Path

from ._common import assemble
from .csvio import CSV_SUFFIX, csv_trials
from .datio import dat_trials


def session_sources(folder):
    """Map each session in 'folder' to the file it should be read from.

    Parameters
    ----------
    folder : Path or str
        A raw_data folder holding '<stem>.trial.csv' and/or '<stem>.dat'.

    Returns
    -------
    dict
        Session stem -> Path, ordered by stem. Stems are of the form
        '<animal>-<YYYY>-<MM>-<DD>-<HH>-<MM>-<SS>', so ordering by stem is
        chronological.
    """
    folder = Path(folder)
    csvs = {p.name[: -len(CSV_SUFFIX)]: p for p in folder.glob(f"*{CSV_SUFFIX}")}
    dats = {p.stem: p for p in folder.glob("*.dat")}
    return {stem: csvs.get(stem, dats.get(stem)) for stem in sorted(csvs | dats)}


def raw2ArmIO(folder, auto_blocks=True, time_window_min=40, metadata=None):
    """Build a 'Bandit2Arm' from every session in a raw_data folder.

    Each session is read from its '.trial.csv' when present and from its
    '.dat' otherwise, so no session is dropped for having only one of the
    two. The two readers agree on the absolute clock, so trials from both
    interleave correctly once ordered by time.

    The .dat files are parsed together as one event stream (see 'dat_trials')
    and only then filtered down to the sessions that have no .csv, rather
    than being parsed individually. Parsing them individually would lose the
    trials of every log the AutoTrainer rotated mid-session.

    Parameters
    ----------
    folder : Path or str
        A raw_data folder.
    auto_blocks : bool, optional
        Derive 'block_ids' and 'window_ids' with
        'Bandit2Arm.auto_block_window_ids'. Default True. The .dat log carries
        no block structure of its own, so deriving them is the only way a
        mixed folder gets a consistent block numbering.
    time_window_min : int, optional
        Window length passed to 'auto_block_window_ids'. Default 40.
    metadata : dict, optional
        Passed through to 'Bandit2Arm'. A 'sources' entry recording which file
        each session was read from is added to it.

    Returns
    -------
    Bandit2Arm

    See Also
    --------
    csv2ArmIO, dat2ArmIO : read one kind of file only.
    """
    folder = Path(folder)
    sources = session_sources(folder)
    if not sources:
        raise FileNotFoundError(f"No .csv or .dat files in {folder}")

    csv_files = [fp for fp in sources.values() if fp.name.endswith(CSV_SUFFIX)]
    dat_files = [fp for fp in sources.values() if fp.suffix == ".dat"]

    frames, from_csv = [], set()
    for fp in csv_files:
        stem = fp.name[: -len(CSV_SUFFIX)]
        try:
            frame = csv_trials(fp)
        except Exception as e:
            # Older sessions were written in a .csv layout this reader does
            # not understand. Leaving the stem out of 'from_csv' is what
            # makes its .dat get used instead, so nothing is lost.
            warnings.warn(
                f"{fp.name}: {type(e).__name__}: {e} -- falling back to .dat",
                stacklevel=2,
            )
            continue
        if len(frame):
            frames.append(frame)
            from_csv.add(stem)

    used = {stem: fp.name for stem, fp in sources.items() if stem in from_csv}
    if dat_files or from_csv != set(sources):
        # Parse every .dat the folder has, so mid-session rotations stay
        # stitched, then keep only the sessions no usable .csv covers.
        every_dat = sorted(folder.glob("*.dat"))
        if every_dat:
            table = dat_trials(every_dat)
            table = table[~table["stem"].isin(from_csv)]
            frames.append(table)
            used.update({stem: f"{stem}.dat" for stem in table["stem"].unique()})

    task = assemble(frames, sort=True, metadata={**(metadata or {}), "sources": used})
    if auto_blocks:
        task.auto_block_window_ids(time_window_min=time_window_min)
    return task

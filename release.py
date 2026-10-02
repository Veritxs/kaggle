"""Align train titles to their WIDE release day (what test D1 represents).

58% of train titles first appear as a 1-cinema preview days before the wide
opening. Test D1 is the wide opening (median 70 cinemas, Wed/Thu). Define wide
D1 as the first date within 21 days of first appearance on which the title plays
at >= 30% of its peak cinema count, and drop the earlier preview rows.
"""
import pandas as pd


def wide_release_dates(raw: pd.DataFrame, frac: float = 0.3) -> pd.Series:
    raw = raw.copy()
    raw["movie_title"] = raw.movie_title.str.upper().str.strip()
    nc = raw.groupby(["movie_title", "date_show"]).cinema_ids.nunique().rename("n").reset_index()
    first = nc.groupby("movie_title").date_show.transform("min")
    nc = nc[nc.date_show <= first + pd.Timedelta(days=21)]
    peak = nc.groupby("movie_title").n.transform("max")
    return nc[nc.n >= frac * peak].groupby("movie_title").date_show.min()


def align_to_wide_release(raw: pd.DataFrame, frac: float = 0.3) -> pd.DataFrame:
    raw = raw.copy()
    raw["movie_title"] = raw.movie_title.str.upper().str.strip()
    first_seen = raw.groupby("movie_title").date_show.min()
    d1 = wide_release_dates(raw, frac)
    out = raw[raw.date_show >= raw.movie_title.map(d1)]
    # titles already running when data starts are left-censored: keep their
    # rows (useful as competition context) but they never form a cohort.
    censored = first_seen[first_seen == raw.date_show.min()].index
    return out, set(censored)

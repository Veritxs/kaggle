"""V7 feature additions on top of v6 (all available at test time, no future labels).

- capacity: each cinema's typical daily screen supply (from train.csv only) and the
  film's / new releases' share of it.
- dow_rel: prior weekday factor of the target day divided by the mean factor of the
  three history days (fixes weekend-heavy vs weekday-heavy opening windows).
- school holidays: Indonesian school break periods (calendar knowledge, not data).
- present-day base: mean of days the film was actually showing (late-joining pairs).
- movie squeeze: share of the film's cinemas that already had <=2 shows on D3.
"""
import numpy as np
import pandas as pd

import features_v6 as F

SCHOOL_HOLIDAYS = [
    ("2025-06-21", "2025-07-13"),   # end of 2024/25 school year
    ("2025-12-20", "2026-01-04"),   # semester break
    ("2026-03-14", "2026-03-29"),   # Idulfitri 1447 collective leave / school break
]
# Prior weekday factors (Mon..Sun) measured on train in the step-by-step session
DOW_PRIOR = np.array([0.700, 0.640, 0.795, 0.953, 0.986, 1.177, 1.098])


def school_flag(dates: pd.Series) -> np.ndarray:
    out = np.zeros(len(dates), dtype="int8")
    for a, b in SCHOOL_HOLIDAYS:
        out |= dates.between(pd.Timestamp(a), pd.Timestamp(b)).to_numpy().astype("int8")
    return out


def cinema_capacity(train_raw: pd.DataFrame) -> pd.DataFrame:
    daily = train_raw.groupby(["cinema_ids", "date_show"]).total_show.sum()
    cap = daily.groupby("cinema_ids").quantile(0.9).rename("cinema_capacity")
    titles = train_raw.groupby(["cinema_ids", "date_show"]).movie_title.nunique()
    nt = titles.groupby("cinema_ids").median().rename("cinema_typical_titles")
    return pd.concat([cap, nt], axis=1).reset_index()


def add_v7(df: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(cap, on="cinema_ids", how="left")
    df["cinema_capacity"] = df.cinema_capacity.fillna(df.cinema_capacity.median())
    df["cinema_typical_titles"] = df.cinema_typical_titles.fillna(df.cinema_typical_titles.median())
    c = df.cinema_capacity.clip(lower=1)
    df["own_d3_vs_cap"] = df.total_show_d3 / c
    df["new_shows_vs_cap"] = df.cum_new_shows / c
    df["comp_shows_vs_cap"] = df.comp_shows / c

    rd = df.release_dow.to_numpy()
    hist_f = (DOW_PRIOR[rd % 7] + DOW_PRIOR[(rd + 1) % 7] + DOW_PRIOR[(rd + 2) % 7]) / 3
    df["dow_rel"] = DOW_PRIOR[df.target_dow.to_numpy()] / hist_f

    df["school_target"] = school_flag(df.target_date)
    df["school_hist"] = school_flag(df.release_date + pd.Timedelta(days=1))

    t = df[["total_ticket_d1", "total_ticket_d2", "total_ticket_d3"]].to_numpy()
    df["n_present"] = (t > 0).sum(1).clip(min=1)
    df["base_present"] = np.maximum(t.sum(1) / df.n_present, 1)
    df["present_vs_scale"] = df.base_present / df.scale

    low = (df.total_show_d3 <= 2).astype(float)
    df["movie_low_show_share"] = low.groupby(df.movie).transform("mean")
    df["movie_occ_slope"] = df.groupby("movie").occ_slope.transform("mean")
    return df


V7_EXTRA = ["own_d3_vs_cap", "new_shows_vs_cap", "comp_shows_vs_cap", "cinema_capacity",
            "cinema_typical_titles", "dow_rel", "school_target", "school_hist", "n_present",
            "present_vs_scale", "movie_low_show_share", "movie_occ_slope"]
FEATURES = F.FEATURES + V7_EXTRA
CAT = F.CAT

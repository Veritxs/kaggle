"""V6 features: everything in honest_experiments + screen-competition and
movie-level signals derived ONLY from D1-D3 history windows of all titles.

At test time test_history.csv shows the D1-D3 rows of every title, including
titles that open DURING another title's D4-D10 window. That tells us when a
new film arrives at a cinema and how many shows it takes. Train features are
built identically from the D1-D3 windows of train titles, so there is no
train/test asymmetry and no future label leakage.
"""
import numpy as np
import pandas as pd

import honest_experiments as hx


def history_windows(raw: pd.DataFrame, min_d1=None) -> pd.DataFrame:
    """Rows within each title's own calendar D1-D3 (what test_history exposes)."""
    raw = raw.copy()
    raw["movie_title"] = raw.movie_title.str.upper().str.strip()
    d1 = raw.groupby("movie_title").date_show.min()
    raw["d1"] = raw.movie_title.map(d1)
    raw["day"] = (raw.date_show - raw.d1).dt.days + 1
    h = raw[raw.day.between(1, 3)]
    if min_d1 is not None:
        h = h[h.d1 >= min_d1]
    return h


def add_competition(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Add competition features keyed on (cinema, target_date) and nationally."""
    df = df.copy()
    h = hist.rename(columns={"date_show": "target_date"})

    # --- per cinema-date: other titles that are in their D1-D3 window ---
    cell = (h.groupby(["cinema_ids", "target_date", "movie_title"])
              .agg(shows=("total_show", "sum"), tix=("total_ticket", "sum"),
                   opening=("day", lambda s: int((s == 1).any())))
              .reset_index())
    key = df[["cinema_ids", "target_date", "movie"]].drop_duplicates()
    j = key.merge(cell, on=["cinema_ids", "target_date"], how="left")
    j = j[j.movie_title.notna() & (j.movie_title != j.movie)]
    agg = (j.groupby(["cinema_ids", "target_date", "movie"])
             .agg(comp_titles=("movie_title", "nunique"), comp_shows=("shows", "sum"),
                  comp_tix=("tix", "sum"), comp_openings=("opening", "sum"))
             .reset_index())
    df = df.merge(agg, on=["cinema_ids", "target_date", "movie"], how="left")

    # cumulative new openings at this cinema between D3 and target date
    opens = h[h.day == 1][["cinema_ids", "target_date", "movie_title", "total_show"]]
    opens = opens.rename(columns={"target_date": "open_date", "total_show": "open_shows"})
    k2 = df[["cinema_ids", "movie", "release_date", "target_date"]].drop_duplicates()
    k2 = k2.merge(opens, on="cinema_ids", how="left")
    k2 = k2[(k2.movie_title != k2.movie)
            & (k2.open_date > k2.release_date + pd.Timedelta(days=2))
            & (k2.open_date <= k2.target_date)]
    cum = (k2.groupby(["cinema_ids", "movie", "target_date"])
             .agg(cum_new_titles=("movie_title", "nunique"), cum_new_shows=("open_shows", "sum"))
             .reset_index())
    df = df.merge(cum, on=["cinema_ids", "movie", "target_date"], how="left")

    # national: titles opening between D3 and target date, and their D1 strength
    nat = (h[h.day == 1].groupby(["movie_title", "target_date"])
             .agg(n_cin=("cinema_ids", "nunique"), d1_tix=("total_ticket", "sum"))
             .reset_index().rename(columns={"target_date": "open_date"}))
    k3 = df[["movie", "release_date", "target_date"]].drop_duplicates()
    k3 = k3.assign(_k=1).merge(nat.assign(_k=1), on="_k").drop(columns="_k")
    k3 = k3[(k3.movie_title != k3.movie)
            & (k3.open_date > k3.release_date + pd.Timedelta(days=2))
            & (k3.open_date <= k3.target_date)]
    nagg = (k3.groupby(["movie", "target_date"])
              .agg(nat_new_titles=("movie_title", "nunique"),
                   nat_new_cinemas=("n_cin", "sum"), nat_new_d1tix=("d1_tix", "sum"))
              .reset_index())
    df = df.merge(nagg, on=["movie", "target_date"], how="left")

    # own share of visible screen supply at this cinema on D3
    d3cell = (h.groupby(["cinema_ids", "target_date"]).total_show.sum()
                .rename("cinema_visible_shows_d3").reset_index()
                .rename(columns={"target_date": "d3_date"}))
    df["d3_date"] = df.release_date + pd.Timedelta(days=2)
    df = df.merge(d3cell, on=["cinema_ids", "d3_date"], how="left")
    df["own_show_share_d3"] = df.total_show_d3 / df.cinema_visible_shows_d3.clip(lower=1)

    fill0 = ["comp_titles", "comp_shows", "comp_tix", "comp_openings",
             "cum_new_titles", "cum_new_shows", "nat_new_titles", "nat_new_cinemas", "nat_new_d1tix"]
    df[fill0] = df[fill0].fillna(0)
    df["comp_shows_vs_own"] = df.comp_shows / df.total_show_d3.clip(lower=1)
    df["cum_new_shows_vs_own"] = df.cum_new_shows / df.total_show_d3.clip(lower=1)
    df["nat_new_d1tix_vs_movie"] = df.nat_new_d1tix / df.movie_total_d1.clip(lower=1)

    # movie-level breadth and per-cinema strength
    df["movie_opening_cinemas"] = df.movie_pair_count
    df["movie_mean_scale"] = df.groupby("movie").scale.transform("mean")
    df["movie_mean_occ_d3"] = df.groupby("movie").occupation_rate_d3.transform("mean")
    df["movie_d3_d1"] = df.movie_total_d3 / df.movie_total_d1.clip(lower=1)
    df["days_since_release"] = df.horizon + 2
    df["is_weekend"] = df.target_dow.ge(5).astype(int)
    df["week2"] = (df.target_date >= df.release_date + pd.Timedelta(days=7)).astype(int)
    return df


COMP = ["comp_titles", "comp_shows", "comp_tix", "comp_openings", "cum_new_titles",
        "cum_new_shows", "nat_new_titles", "nat_new_cinemas", "nat_new_d1tix",
        "own_show_share_d3", "cinema_visible_shows_d3", "comp_shows_vs_own",
        "cum_new_shows_vs_own", "nat_new_d1tix_vs_movie", "movie_mean_scale",
        "movie_mean_occ_d3", "movie_d3_d1", "is_weekend", "week2"]
FEATURES = hx.NUM_FEATURES + COMP + hx.CAT_FEATURES
CAT = hx.CAT_FEATURES


def build_train(raw):
    d = hx.add_aux(hx.build_dense_cohorts(raw))
    return add_competition(d, history_windows(raw, min_d1=raw.date_show.min() + pd.Timedelta(days=1)))


def build_test(history, test):
    d = hx.add_aux(hx.build_test_features(history, test))
    return add_competition(d, history_windows(history))

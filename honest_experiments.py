"""Rolling out-of-time experiments using exact calendar-aligned dense pseudo-tests.

This file deliberately does NOT use random folds. Each validation fold contains
whole movie release cohorts from a future calendar month, and training cohorts
must have completed D10 before validation starts.
"""
from __future__ import annotations

import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 2026
np.random.seed(SEED)
DATA = Path(__file__).resolve().parent


def norm_title(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper().str.strip()
            .str.replace(r"\s*\((?:3D|IMAX[^)]*)\)\s*$", "", regex=True)
            .str.strip())


def load_aux():
    hol = pd.read_csv(DATA / "holidays.csv", parse_dates=["date"])
    hol = hol.rename(columns={"date": "target_date"})
    hol["is_holiday"] = hol["holiday_tipe"].eq("holiday").astype("int8")
    hol = hol[["target_date", "is_holiday", "day_tipe"]]

    movies = pd.read_csv(DATA / "movies.csv")
    movies["base_title"] = norm_title(movies["original_title"])
    movies["genre_primary"] = movies["genre"].fillna("UNK").str.split(",").str[0].str.strip()
    movies = movies[["base_title", "age_rating", "genre_primary"]].drop_duplicates("base_title")

    prices = pd.read_csv(DATA / "ticket_prices.csv")
    prices["price_kind"] = prices["price_day"].str.lower()
    prices = prices.pivot_table(index="city_name", columns="price_kind", values="ceil", aggfunc="median")
    prices.columns = [f"price_{x}" for x in prices.columns]
    return hol, movies, prices.reset_index()


def build_dense_cohorts(raw: pd.DataFrame, require_complete_targets: bool = True) -> pd.DataFrame:
    """Build exact D1-D3 -> dense D4-D10 experiments for each movie cohort.

    A target pair is included iff it has an observed row on movie-calendar D3,
    matching the actual test construction. Missing calendar rows become zero.
    """
    raw = raw.copy()
    raw["movie_title"] = raw["movie_title"].str.upper().str.strip()
    raw["base_title"] = norm_title(raw["movie_title"])
    data_min, data_max = raw.date_show.min(), raw.date_show.max()
    out = []

    for movie, gm in raw.groupby("movie_title", sort=False):
        d1 = gm.date_show.min()
        d3, d10 = d1 + pd.Timedelta(days=2), d1 + pd.Timedelta(days=9)
        # Exclude left-censored cohorts and cohorts without all target dates.
        if d1 == data_min or (require_complete_targets and d10 > data_max):
            continue
        d3_rows = gm[gm.date_show.eq(d3)]
        if d3_rows.empty:
            continue
        pairs = d3_rows[["cinema_ids", "city_name"]].drop_duplicates("cinema_ids")
        pair_ids = pairs.cinema_ids.tolist()
        dates = pd.date_range(d1, d10)
        idx = pd.MultiIndex.from_product([pair_ids, dates], names=["cinema_ids", "date_show"])
        cols = ["total_ticket", "occupation_rate", "total_show"]
        panel = (gm.drop_duplicates(["cinema_ids", "date_show"], keep="last")
                 .set_index(["cinema_ids", "date_show"])[cols].reindex(idx).fillna(0.0))

        # D1-D3 pair history.
        hist = panel.loc[(slice(None), pd.date_range(d1, d3)), :].reset_index()
        hist["day"] = (hist.date_show - d1).dt.days + 1
        wide = hist.pivot(index="cinema_ids", columns="day", values=cols)
        wide.columns = [f"{name}_d{day}" for name, day in wide.columns]
        wide = wide.reset_index().merge(pairs, on="cinema_ids", how="left")
        for day in (1, 2, 3):
            for col in cols:
                key = f"{col}_d{day}"
                if key not in wide:
                    wide[key] = 0.0
        wide["scale"] = wide[[f"total_ticket_d{x}" for x in (1,2,3)]].sum(axis=1).div(3).clip(lower=1)
        for day in (1,2,3):
            wide[f"r_d{day}"] = wide[f"total_ticket_d{day}"] / wide.scale
            wide[f"present_d{day}"] = wide[f"total_ticket_d{day}"].gt(0).astype("int8")
        wide["r_slope"] = (wide.r_d3 - wide.r_d1) / 2
        wide["show_slope"] = (wide.total_show_d3 - wide.total_show_d1) / 2
        wide["occ_slope"] = (wide.occupation_rate_d3 - wide.occupation_rate_d1) / 2
        wide["tickets_per_show_d3"] = wide.total_ticket_d3 / wide.total_show_d3.clip(lower=1)
        wide["movie_pair_count"] = len(wide)
        wide["movie_total_d1"] = wide.total_ticket_d1.sum()
        wide["movie_total_d2"] = wide.total_ticket_d2.sum()
        wide["movie_total_d3"] = wide.total_ticket_d3.sum()
        wide["movie_active_d1"] = int(wide.present_d1.sum())
        wide["movie_active_d2"] = int(wide.present_d2.sum())
        wide["pair_scale_pct"] = wide.scale.rank(pct=True)
        wide["pair_scale_share"] = wide.scale / max(float(wide.scale.sum()), 1.0)
        wide["movie_r_slope"] = (wide.movie_total_d3 - wide.movie_total_d1) / max(wide.movie_total_d1.iloc[0], 1)
        wide["movie"] = movie
        wide["base_title"] = norm_title(pd.Series([movie])).iloc[0]
        wide["release_date"] = d1
        wide["release_dow"] = d1.dayofweek

        future = panel.loc[(slice(None), pd.date_range(d3 + pd.Timedelta(days=1), d10)), "total_ticket"].reset_index()
        future["horizon"] = (future.date_show - d3).dt.days
        future = future.rename(columns={"date_show": "target_date", "total_ticket": "y"})
        cohort = future.merge(wide, on="cinema_ids", how="left")
        cohort["target_dow"] = cohort.target_date.dt.dayofweek
        cohort["ratio"] = cohort.y / cohort.scale
        out.append(cohort)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_test_features(history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Create the same calendar-aligned features for the real test grid."""
    history = history.copy()
    test = test.copy()
    for frame in (history, test):
        frame["movie_title"] = frame.movie_title.str.upper().str.strip()
    out = []
    hist_cols = ["total_ticket", "occupation_rate", "total_show"]
    for movie, gt in test.groupby("movie_title", sort=False):
        gm = history[history.movie_title.eq(movie)]
        target_start = gt.date_show.min()
        d1, d3 = target_start - pd.Timedelta(days=3), target_start - pd.Timedelta(days=1)
        pairs = gt[["cinema_ids", "city_name"]].drop_duplicates("cinema_ids")
        pair_ids = pairs.cinema_ids.tolist()
        idx = pd.MultiIndex.from_product(
            [pair_ids, pd.date_range(d1, d3)], names=["cinema_ids", "date_show"]
        )
        panel = (gm.drop_duplicates(["cinema_ids", "date_show"], keep="last")
                 .set_index(["cinema_ids", "date_show"])[hist_cols]
                 .reindex(idx).fillna(0.0))
        hist = panel.reset_index()
        hist["day"] = (hist.date_show - d1).dt.days + 1
        wide = hist.pivot(index="cinema_ids", columns="day", values=hist_cols)
        wide.columns = [f"{name}_d{day}" for name, day in wide.columns]
        wide = wide.reset_index().merge(pairs, on="cinema_ids", how="left")
        for day in (1, 2, 3):
            for col in hist_cols:
                key = f"{col}_d{day}"
                if key not in wide:
                    wide[key] = 0.0
        wide["scale"] = wide[[f"total_ticket_d{x}" for x in (1, 2, 3)]].sum(axis=1).div(3).clip(lower=1)
        for day in (1, 2, 3):
            wide[f"r_d{day}"] = wide[f"total_ticket_d{day}"] / wide.scale
            wide[f"present_d{day}"] = wide[f"total_ticket_d{day}"].gt(0).astype("int8")
        wide["r_slope"] = (wide.r_d3 - wide.r_d1) / 2
        wide["show_slope"] = (wide.total_show_d3 - wide.total_show_d1) / 2
        wide["occ_slope"] = (wide.occupation_rate_d3 - wide.occupation_rate_d1) / 2
        wide["tickets_per_show_d3"] = wide.total_ticket_d3 / wide.total_show_d3.clip(lower=1)
        wide["movie_pair_count"] = len(wide)
        wide["movie_total_d1"] = wide.total_ticket_d1.sum()
        wide["movie_total_d2"] = wide.total_ticket_d2.sum()
        wide["movie_total_d3"] = wide.total_ticket_d3.sum()
        wide["movie_active_d1"] = int(wide.present_d1.sum())
        wide["movie_active_d2"] = int(wide.present_d2.sum())
        wide["pair_scale_pct"] = wide.scale.rank(pct=True)
        wide["pair_scale_share"] = wide.scale / max(float(wide.scale.sum()), 1.0)
        wide["movie_r_slope"] = (wide.movie_total_d3 - wide.movie_total_d1) / max(float(wide.movie_total_d1.iloc[0]), 1.0)
        wide["movie"] = movie
        wide["base_title"] = norm_title(pd.Series([movie])).iloc[0]
        wide["release_date"] = d1
        wide["release_dow"] = d1.dayofweek

        cohort = gt.rename(columns={"date_show": "target_date"}).merge(wide, on=["cinema_ids", "city_name"], how="left")
        cohort["horizon"] = (cohort.target_date - d3).dt.days
        cohort["target_dow"] = cohort.target_date.dt.dayofweek
        out.append(cohort)
    return pd.concat(out, ignore_index=True)


def add_aux(df: pd.DataFrame) -> pd.DataFrame:
    hol, movies, prices = load_aux()
    x = df.merge(hol, on="target_date", how="left")
    x = x.merge(movies, on="base_title", how="left")
    x = x.merge(prices, on="city_name", how="left")
    x["is_holiday"] = x.is_holiday.fillna(0).astype("int8")
    x["day_tipe"] = x.day_tipe.fillna("unknown")
    x["age_rating"] = x.age_rating.fillna("UNK")
    x["genre_primary"] = x.genre_primary.fillna("UNK")
    kind = np.where(x.target_dow.eq(4), "friday", np.where(x.target_dow.ge(5), "weekend", "weekday"))
    x["price_target"] = [x.iloc[i].get(f"price_{k}", np.nan) for i, k in enumerate(kind)]
    return x


NUM_FEATURES = [
    "scale", "r_d1", "r_d2", "r_d3", "r_slope",
    "present_d1", "present_d2", "present_d3",
    "occupation_rate_d1", "occupation_rate_d2", "occupation_rate_d3", "occ_slope",
    "total_show_d1", "total_show_d2", "total_show_d3", "show_slope", "tickets_per_show_d3",
    "movie_pair_count", "movie_total_d1", "movie_total_d2", "movie_total_d3",
    "movie_active_d1", "movie_active_d2", "pair_scale_pct", "pair_scale_share", "movie_r_slope",
    "horizon", "target_dow", "release_dow", "is_holiday", "price_target",
]
CAT_FEATURES = ["cinema_ids", "city_name", "genre_primary", "age_rating", "day_tipe"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def categorical_frames(train: pd.DataFrame, val: pd.DataFrame):
    a, b = train[FEATURES].copy(), val[FEATURES].copy()
    for c in CAT_FEATURES:
        cats = pd.Index(pd.concat([a[c], b[c]]).astype(str).unique())
        a[c] = pd.Categorical(a[c].astype(str), categories=cats)
        b[c] = pd.Categorical(b[c].astype(str), categories=cats)
    return a, b


def hierarchical_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Low-variance conditional medians with count shrinkage."""
    global_med = train.ratio.median()
    h = train.groupby("horizon").ratio.agg(["median", "count"])
    # Fine cell: release weekday + horizon + D3 presence pattern.
    keys = ["release_dow", "horizon", "present_d1", "present_d2"]
    fine = train.groupby(keys).ratio.agg(["median", "count"]).reset_index()
    z = val[keys].merge(fine, on=keys, how="left")
    hmed = val.horizon.map(h["median"]).fillna(global_med).to_numpy()
    count = z["count"].fillna(0).to_numpy()
    fine_med = z["median"].fillna(pd.Series(hmed)).to_numpy()
    weight = count / (count + 80.0)
    return weight * fine_med + (1 - weight) * hmed


def lgb_predict(train: pd.DataFrame, val: pd.DataFrame, params: dict) -> np.ndarray:
    Xtr, Xva = categorical_frames(train, val)
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, train.ratio, categorical_feature=CAT_FEATURES)
    return np.clip(model.predict(Xva), 0, None)


def folds(df: pd.DataFrame):
    for month in ["2025-06", "2025-07", "2025-08", "2025-09"]:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        tr = df[df.release_date + pd.Timedelta(days=9) < start]
        va = df[(df.release_date >= start) & (df.release_date < end)]
        if len(tr) and len(va):
            yield month, tr, va


def score(y, p):
    return float(np.mean(np.abs(y - p)))  # y and p are already scaled ratios


def main():
    raw = pd.read_csv(DATA / "train.csv", parse_dates=["date_show"])
    df = add_aux(build_dense_cohorts(raw))
    print(f"Dense pseudo-test: {len(df):,} rows, {df.movie.nunique()} movies, zero rate={df.y.eq(0).mean():.3f}")

    configs = {
        "lgb_tiny": dict(objective="regression_l1", n_estimators=250, learning_rate=.025,
                         num_leaves=7, max_depth=3, min_child_samples=120,
                         reg_lambda=20, reg_alpha=2, colsample_bytree=.75,
                         verbosity=-1, random_state=SEED, n_jobs=-1),
        "lgb_small": dict(objective="regression_l1", n_estimators=350, learning_rate=.025,
                          num_leaves=15, max_depth=4, min_child_samples=100,
                          reg_lambda=20, reg_alpha=2, colsample_bytree=.8,
                          verbosity=-1, random_state=SEED, n_jobs=-1),
    }
    all_results = []
    blend_grid = [0, .25, .5, .75, 1.0]
    for month, tr, va in folds(df):
        base = hierarchical_predict(tr, va)
        preds = {"hier": base}
        for name, params in configs.items():
            preds[name] = lgb_predict(tr, va, params)
        print(f"\nFold {month}: train movies={tr.movie.nunique()}, val movies={va.movie.nunique()}, rows={len(va):,}, zeros={va.y.eq(0).mean():.3f}")
        for name, p in preds.items():
            s = score(va.ratio.to_numpy(), p)
            all_results.append((month, name, s, len(va)))
            print(f"  {name:12s} {s:.5f}")
        for model_name in configs:
            for w in blend_grid:
                p = (1-w)*base + w*preds[model_name]
                name = f"blend_{model_name}_{w:.2f}"
                all_results.append((month, name, score(va.ratio.to_numpy(), p), len(va)))

    res = pd.DataFrame(all_results, columns=["fold", "model", "score", "rows"])
    summary = res.groupby("model").agg(mean=("score","mean"), median=("score","median"), worst=("score","max"), std=("score","std"))
    print("\n=== Equal-fold rolling summary (primary selection) ===")
    print(summary.sort_values(["median", "worst"]).head(15).to_string())
    res.to_csv(DATA / "honest_cv_results.csv", index=False)


if __name__ == "__main__":
    main()

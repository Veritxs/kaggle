"""V6 cinema ticket forecast.

Key changes vs v5 (each validated on rolling out-of-time folds):
1. Wide-release alignment. 58% of train titles first appear as a 1-cinema
   preview days before the real opening. Test D1 is the wide opening
   (median 70 cinemas, Wed/Thu). Train cohorts are re-anchored to the first
   day the title reaches >=30% of its peak cinema count. Cohorts: 110 -> 171
   movies, 3.1k -> 7.4k pairs, and release weekdays now match test.
2. Screen-competition features from D1-D3 history windows of OTHER titles:
   new openings at the same cinema between D3 and the target day, their show
   counts, national new-release strength, the film's D3 share of visible shows.
   test_history.csv exposes exactly the same information, so train and test
   features are built identically, with no label leakage.
3. Calendar: weekday, holiday flags from holidays.csv, horizon, release weekday,
   city price tier (ticket_prices.csv), genre/age rating (movies.csv).
4. Target = ticket / scale with L1 loss, so training MAE == competition MASE.
   Missing calendar rows (film pulled from the cinema) are zero.

Model: 0.6 LightGBM + 0.4 CatBoost, each averaged over 3 fixed seeds.
Reproducible: SEED = 2026.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import features_v6 as F
import honest_experiments as hx
import release as R

SEED = 2026
np.random.seed(SEED)
DATA = Path(__file__).resolve().parent
SEEDS = [SEED, SEED + 1, SEED + 2]
CAT_WEIGHT = 0.4
FOLDS = ["2025-06", "2025-07", "2025-08", "2025-09"]

LGB = dict(objective="regression_l1", n_estimators=300, learning_rate=0.03, num_leaves=15,
           min_child_samples=80, reg_lambda=20, colsample_bytree=0.7, subsample=0.8,
           subsample_freq=1, verbosity=-1, n_jobs=-1)
CAT = dict(loss_function="MAE", iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=10,
           verbose=False, thread_count=-1, allow_writing_files=False)


def build_train():
    raw = pd.read_csv(DATA / "train.csv", parse_dates=["date_show"])
    aligned, censored = R.align_to_wide_release(raw)
    d = hx.add_aux(hx.build_dense_cohorts(aligned))
    d = d[~d.movie.isin(censored)]
    return F.add_competition(d, F.history_windows(aligned))


def build_test():
    hist = pd.read_csv(DATA / "test_history.csv", parse_dates=["date_show"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date_show"])
    return F.build_test(hist, test)


def _lgb_frames(tr, va):
    a, b = tr[F.FEATURES].copy(), va[F.FEATURES].copy()
    for c in F.CAT:
        cats = pd.Index(pd.concat([a[c], b[c]]).astype(str).unique())
        a[c] = pd.Categorical(a[c].astype(str), categories=cats)
        b[c] = pd.Categorical(b[c].astype(str), categories=cats)
    return a, b


def _cat_frames(tr, va):
    a, b = tr[F.FEATURES].copy(), va[F.FEATURES].copy()
    for c in F.CAT:
        a[c] = a[c].fillna("UNK").astype(str)
        b[c] = b[c].fillna("UNK").astype(str)
    return a.fillna(-999), b.fillna(-999)


def fit_predict(tr, va):
    la, lb_ = _lgb_frames(tr, va)
    ca, cb = _cat_frames(tr, va)
    lp = np.zeros(len(va))
    cp = np.zeros(len(va))
    for s in SEEDS:
        lp += lgb.LGBMRegressor(**LGB, random_state=s).fit(
            la, tr.ratio, categorical_feature=F.CAT).predict(lb_)
        cp += CatBoostRegressor(**CAT, random_seed=s).fit(
            ca, tr.ratio, cat_features=F.CAT).predict(cb)
    lp, cp = np.clip(lp / len(SEEDS), 0, None), np.clip(cp / len(SEEDS), 0, None)
    return (1 - CAT_WEIGHT) * lp + CAT_WEIGHT * cp


def validate(d):
    rows = []
    for m in FOLDS:
        start = pd.Timestamp(m + "-01")
        end = start + pd.offsets.MonthBegin(1)
        tr = d[d.release_date + pd.Timedelta(days=9) < start]
        va = d[(d.release_date >= start) & (d.release_date < end)]
        p = fit_predict(tr, va)
        decay = va.horizon.map(tr.groupby("horizon").ratio.median()).to_numpy()
        rows.append(dict(fold=m, v6_mase=np.abs(va.ratio - p).mean(),
                         decay_baseline=np.abs(va.ratio - decay).mean(),
                         flat_baseline=np.abs(va.ratio - 1).mean(),
                         rows=len(va), movies=va.movie.nunique(), zero_rate=va.y.eq(0).mean()))
    rep = pd.DataFrame(rows)
    print(rep.round(4).to_string(index=False))
    print("mean:", rep[["v6_mase", "decay_baseline", "flat_baseline"]].mean().round(4).to_dict())
    return rep


def main():
    train = build_train()
    test = build_test()
    print(f"train cohorts: {len(train):,} rows, {train.movie.nunique()} movies | test: {len(test):,} rows")
    validate(train).to_csv(DATA / "rolling_cv_v6.csv", index=False)

    ratio = fit_predict(train, test)
    tickets = np.rint(ratio * test.scale.to_numpy()).clip(0).astype(int)
    pred = pd.DataFrame({"id": test.id.astype(int), "total_ticket": tickets})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if sub.total_ticket.isna().any():
        raise RuntimeError("missing predictions")
    sub.total_ticket = sub.total_ticket.astype(int)
    sub.to_csv(DATA / "submission_v6.csv", index=False)
    sub.to_csv(DATA / "submission.csv", index=False)
    print("saved submission_v6.csv / submission.csv", sub.shape)
    print(sub.total_ticket.describe().round(1).to_dict())


if __name__ == "__main__":
    main()

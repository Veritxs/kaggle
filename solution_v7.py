"""V7 cinema ticket forecast (builds on v6, LB 0.4777).

New vs v6 (rolling out-of-time backtest 0.410 -> 0.383, better in 3/4 months):
1. Offset-window augmentation: besides each film's true D1-D3 -> D4-D10 window,
   also train on windows starting 1-4 days later (history D1+k..D3+k), weight 0.5,
   with an `offset` feature. Validation/test always use offset 0.
2. Base = mean of history days the film actually showed (handles late-joining
   cinemas); sample weight base/scale keeps the loss exactly equal to MASE.
3. New features: cinema screen capacity and the film's/new releases' share of it,
   weekday factor relative to the opening window, school holidays, share of the
   film's cinemas already at <=2 shows on D3, movie occupancy trend.

Model: 0.7 LightGBM + 0.3 CatBoost, 3 fixed seeds each. SEED = 2026.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import features_v6 as F6
import features_v7 as F7
import honest_experiments as hx
import release as R

SEED = 2026
np.random.seed(SEED)
SEEDS = [SEED, SEED + 1, SEED + 2]
DATA = Path(__file__).resolve().parent
OFFSETS = [0, 1, 2, 3, 4]
AUG_WEIGHT = 0.5
CAT_WEIGHT = 0.3
BASE = "base_present"
FEATURES = F7.FEATURES + ["offset"]
CAT = F6.CAT
FOLDS = ["2025-06", "2025-07", "2025-08", "2025-09"]

LGB = dict(objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=15,
           min_child_samples=80, reg_lambda=20, colsample_bytree=0.7, subsample=0.8,
           subsample_freq=1, verbosity=-1, n_jobs=-1)
CATP = dict(loss_function="MAE", iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10,
            verbose=False, thread_count=-1, allow_writing_files=False)


def build_train():
    raw = pd.read_csv(DATA / "train.csv", parse_dates=["date_show"])
    aligned, censored = R.align_to_wide_release(raw)
    cap = F7.cinema_capacity(raw.assign(movie_title=raw.movie_title.str.upper().str.strip()))
    hist = F6.history_windows(aligned)
    d1 = aligned.groupby("movie_title").date_show.min()
    parts = []
    for k in OFFSETS:
        a = aligned[aligned.date_show >= aligned.movie_title.map(d1) + pd.Timedelta(days=k)]
        c = hx.add_aux(hx.build_dense_cohorts(a))
        c = c[~c.movie.isin(censored)]
        c = F7.add_v7(F6.add_competition(c, hist), cap)
        c["offset"] = k
        parts.append(c)
    return pd.concat(parts, ignore_index=True), cap


def build_test(cap):
    hist = pd.read_csv(DATA / "test_history.csv", parse_dates=["date_show"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date_show"])
    t = F7.add_v7(F6.build_test(hist, test), cap)
    t["offset"] = 0
    return t


def fit_predict(tr, va):
    w = (tr[BASE] / tr.scale) * np.where(tr.offset == 0, 1.0, AUG_WEIGHT)
    y = tr.y / tr[BASE]
    k = (va[BASE] / va.scale).to_numpy()

    a, b = tr[FEATURES].copy(), va[FEATURES].copy()
    for c in CAT:
        cc = pd.Index(pd.concat([a[c], b[c]]).astype(str).unique())
        a[c] = pd.Categorical(a[c].astype(str), cc)
        b[c] = pd.Categorical(b[c].astype(str), cc)
    a2, b2 = tr[FEATURES].copy(), va[FEATURES].copy()
    for c in CAT:
        a2[c] = a2[c].fillna("UNK").astype(str)
        b2[c] = b2[c].fillna("UNK").astype(str)
    a2, b2 = a2.fillna(-999), b2.fillna(-999)

    lp = cp = 0
    for s in SEEDS:
        lp = lp + lgb.LGBMRegressor(**LGB, random_state=s).fit(
            a, y, sample_weight=w, categorical_feature=CAT).predict(b)
        cp = cp + CatBoostRegressor(**CATP, random_seed=s).fit(
            a2, y, sample_weight=w, cat_features=CAT).predict(b2)
    lp = np.clip(lp / len(SEEDS), 0, None)
    cp = np.clip(cp / len(SEEDS), 0, None)
    return ((1 - CAT_WEIGHT) * lp + CAT_WEIGHT * cp) * k     # ratio of scale


def validate(d):
    rows = []
    d0 = d[d.offset == 0]
    for m in FOLDS:
        s = pd.Timestamp(m + "-01")
        e = s + pd.offsets.MonthBegin(1)
        va = d0[(d0.release_date >= s) & (d0.release_date < e)]
        tr = d[(d.release_date + pd.Timedelta(days=9) < s) & ~d.movie.isin(va.movie.unique())]
        p = fit_predict(tr, va)
        rows.append(dict(fold=m, v7_mase=np.abs(va.ratio - p).mean(), rows=len(va), movies=va.movie.nunique()))
    rep = pd.DataFrame(rows)
    print(rep.round(4).to_string(index=False))
    print("mean v7:", round(rep.v7_mase.mean(), 4))
    return rep


def main(run_validation=True):
    train, cap = build_train()
    test = build_test(cap)
    print(f"train rows {len(train):,} (offset0 {int((train.offset == 0).sum()):,}) | test rows {len(test):,}")
    if run_validation:
        validate(train).to_csv(DATA / "rolling_cv_v7.csv", index=False)

    ratio = fit_predict(train, test)
    tickets = np.rint(ratio * test.scale.to_numpy()).clip(0).astype(int)
    pred = pd.DataFrame({"id": test.id.astype(int), "total_ticket": tickets})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    assert sub.total_ticket.notna().all()
    sub.total_ticket = sub.total_ticket.astype(int)
    sub.to_csv(DATA / "submission_v7.csv", index=False)
    sub.to_csv(DATA / "submission.csv", index=False)
    pd.DataFrame({"id": test.id.astype(int), "ratio": ratio, "scale": test.scale.to_numpy()}).to_pickle(DATA / "v7_test_ratio.pkl")
    print("saved submission_v7.csv", sub.shape, sub.total_ticket.describe().round(1).to_dict())


if __name__ == "__main__":
    main()

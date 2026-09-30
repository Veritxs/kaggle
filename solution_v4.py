"""Robust dense-panel forecast selected by rolling out-of-time validation.

Why v4 exists:
- v2's random movie folds scored 0.33 offline but 0.60 on the leaderboard.
- That validation omitted missing/no-show D4-D10 rows and mixed future calendar
  cohorts into training.
- v4 reconstructs every movie's exact calendar D1-D10 panel. Pairs observed on
  D3 form a pseudo-test, missing D1/D2/D4-D10 rows are zero, and every D4-D10
  row is scored exactly once.
- Model and calibration are selected over four expanding future-month folds
  (June, July, August, September), considering median and worst-fold MASE.

The target is total_ticket / scale, so ordinary MAE is exactly competition MASE.
Final ensemble: 60% calibrated shallow LightGBM + 40% calibrated CatBoost.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import honest_experiments as hx

SEED = 2026
np.random.seed(SEED)
DATA = Path(__file__).resolve().parent

LGB_MULTIPLIER = 0.80
CAT_MULTIPLIER = 0.85
CAT_WEIGHT = 0.40

LGB_PARAMS = dict(
    objective="regression_l1",
    n_estimators=150,
    learning_rate=0.025,
    num_leaves=7,
    max_depth=3,
    min_child_samples=120,
    reg_lambda=20,
    reg_alpha=2,
    colsample_bytree=0.75,
    verbosity=-1,
    random_state=SEED,
    n_jobs=-1,
)

CAT_PARAMS = dict(
    loss_function="MAE",
    iterations=200,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=20,
    random_seed=SEED,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)


def cat_frames(train, val):
    a, b = train[hx.FEATURES].copy(), val[hx.FEATURES].copy()
    for col in hx.CAT_FEATURES:
        a[col] = a[col].fillna("UNK").astype(str)
        b[col] = b[col].fillna("UNK").astype(str)
    return a.fillna(-999), b.fillna(-999)


def fit_predict(train, val):
    lgb_train, lgb_val = hx.categorical_frames(train, val)
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(lgb_train, train.ratio, categorical_feature=hx.CAT_FEATURES)
    lgb_pred = np.clip(lgb_model.predict(lgb_val), 0, None)

    cat_train, cat_val = cat_frames(train, val)
    cat_model = CatBoostRegressor(**CAT_PARAMS)
    cat_model.fit(cat_train, train.ratio, cat_features=hx.CAT_FEATURES)
    cat_pred = np.clip(cat_model.predict(cat_val), 0, None)

    pred = ((1 - CAT_WEIGHT) * LGB_MULTIPLIER * lgb_pred
            + CAT_WEIGHT * CAT_MULTIPLIER * cat_pred)
    return np.clip(pred, 0, None), lgb_model, cat_model


def rolling_validation(train):
    rows = []
    for month, fold_train, fold_val in hx.folds(train):
        pred, _, _ = fit_predict(fold_train, fold_val)
        score = np.mean(np.abs(fold_val.ratio.to_numpy() - pred))

        # Re-score the prior v3 weekday/horizon rule on the SAME dense rows.
        dow = fold_train.groupby("target_dow").ratio.median()
        residual = fold_train.ratio / fold_train.target_dow.map(dow)
        horizon = 1 + 0.5 * (residual.groupby(fold_train.horizon).median() - 1)
        v3_pred = ((1 + 0.7 * (fold_val.target_dow.map(dow) - 1)).clip(lower=0.3)
                   * fold_val.horizon.map(horizon).fillna(1.0))
        v3_score = np.mean(np.abs(fold_val.ratio.to_numpy() - v3_pred.to_numpy()))
        unit_score = np.mean(np.abs(fold_val.ratio.to_numpy() - 1.0))
        rows.append((month, score, v3_score, unit_score, len(fold_val),
                     fold_val.movie.nunique(), fold_val.y.eq(0).mean()))
    report = pd.DataFrame(
        rows,
        columns=["fold", "v4_mase", "v3_dense_mase", "unit_baseline_mase",
                 "rows", "movies", "zero_rate"],
    )
    print("Rolling out-of-time validation (dense D4-D10 panels):")
    print(report.to_string(index=False, formatters={
        "v4_mase": "{:.5f}".format,
        "v3_dense_mase": "{:.5f}".format,
        "unit_baseline_mase": "{:.5f}".format,
        "zero_rate": "{:.3f}".format,
    }))
    print(f"V4 equal-fold mean={report.v4_mase.mean():.5f}, "
          f"median={report.v4_mase.median():.5f}, worst={report.v4_mase.max():.5f}")
    print(f"V3 dense equal-fold mean={report.v3_dense_mase.mean():.5f}")
    return report


def main():
    train_raw = pd.read_csv(DATA / "train.csv", parse_dates=["date_show"])
    history = pd.read_csv(DATA / "test_history.csv", parse_dates=["date_show"])
    test_raw = pd.read_csv(DATA / "test.csv", parse_dates=["date_show"])

    train = hx.add_aux(hx.build_dense_cohorts(train_raw))
    test = hx.add_aux(hx.build_test_features(history, test_raw))
    print(f"Training pseudo-test rows={len(train):,}, movies={train.movie.nunique()}, zero targets={train.y.eq(0).mean():.3f}")
    print(f"Real test rows={len(test):,}, movies={test.movie.nunique()}")

    report = rolling_validation(train)
    report.to_csv(DATA / "rolling_cv_v4.csv", index=False)

    pred_ratio, _, _ = fit_predict(train, test)
    pred_ticket = np.rint(pred_ratio * test.scale.to_numpy()).clip(0).astype(int)
    candidate = pd.DataFrame({"id": test.id.astype(int), "total_ticket": pred_ticket})

    sample = pd.read_csv(DATA / "sample_submission.csv")
    submission = sample[["id"]].merge(candidate, on="id", how="left", validate="one_to_one")
    if submission.total_ticket.isna().any():
        raise RuntimeError(f"Missing {submission.total_ticket.isna().sum()} test predictions")
    submission.total_ticket = submission.total_ticket.astype(int)

    # Preserve the previous candidate, then make v4 the current submission.
    old = DATA / "submission.csv"
    old_copy = DATA / "submission_v3.csv"
    if old.exists() and not old_copy.exists():
        old_copy.write_bytes(old.read_bytes())
    submission.to_csv(DATA / "submission_v4.csv", index=False)
    submission.to_csv(DATA / "submission.csv", index=False)
    print("Saved submission_v4.csv and submission.csv")
    print(submission.total_ticket.describe().to_string())


if __name__ == "__main__":
    main()

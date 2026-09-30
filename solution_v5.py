"""V5: horizon-aware forecast with an untouched final temporal backtest.

Model/horizon choices were selected using June-August rolling folds only.
September was held untouched until the policy was fixed. Compared with v4,
v5 improves the June-August mean and slightly improves September as well.

The dense pseudo-test and features come from honest_experiments.py. Missing
calendar rows are treated as zero/no-show days, and the direct ratio target
makes MAE exactly equal to competition MASE.
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

LGB_PARAMS = dict(
    objective="regression_l1", n_estimators=150, learning_rate=0.025,
    num_leaves=7, max_depth=3, min_child_samples=120,
    reg_lambda=20, reg_alpha=2, colsample_bytree=0.75,
    verbosity=-1, random_state=SEED, n_jobs=-1,
)
CAT_PARAMS = dict(
    loss_function="MAE", iterations=200, depth=6, learning_rate=0.03,
    l2_leaf_reg=20, random_seed=SEED, verbose=False, thread_count=-1,
    allow_writing_files=False,
)

# Fixed before evaluating September. H1 means D4, H7 means D10.
POLICY = {
    1: ("global_cat", 1.10),
    2: ("horizon_cat", 1.00),
    3: ("global_lgb", 1.10),
    4: ("global_lgb", 1.10),
    5: ("global_lgb", 0.70),
    6: ("global_cat", 0.50),
    7: ("global_lgb", 0.50),
}


def cat_frames(train, val):
    a, b = train[hx.FEATURES].copy(), val[hx.FEATURES].copy()
    for col in hx.CAT_FEATURES:
        a[col] = a[col].fillna("UNK").astype(str)
        b[col] = b[col].fillna("UNK").astype(str)
    return a.fillna(-999), b.fillna(-999)


def fit_predict(train, val):
    lgb_train, lgb_val = hx.categorical_frames(train, val)
    global_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    global_lgb.fit(lgb_train, train.ratio, categorical_feature=hx.CAT_FEATURES)
    lgb_pred = np.clip(global_lgb.predict(lgb_val), 0, None)

    cat_train, cat_val = cat_frames(train, val)
    global_cat = CatBoostRegressor(**CAT_PARAMS)
    global_cat.fit(cat_train, train.ratio, cat_features=hx.CAT_FEATURES)
    cat_pred = np.clip(global_cat.predict(cat_val), 0, None)

    # Only H2 earned a separate model in the pre-September selection.
    train_h2 = train[train.horizon.eq(2)]
    val_h2 = val[val.horizon.eq(2)]
    cat_h2_train, cat_h2_val = cat_frames(train_h2, val_h2)
    horizon_cat = CatBoostRegressor(**CAT_PARAMS)
    horizon_cat.fit(cat_h2_train, train_h2.ratio, cat_features=hx.CAT_FEATURES)
    h2_pred = np.clip(horizon_cat.predict(cat_h2_val), 0, None)

    pred = np.zeros(len(val))
    h2_positions = np.flatnonzero(val.horizon.to_numpy() == 2)
    sources = {"global_lgb": lgb_pred, "global_cat": cat_pred}
    for horizon, (source, multiplier) in POLICY.items():
        positions = np.flatnonzero(val.horizon.to_numpy() == horizon)
        if source == "horizon_cat":
            pred[positions] = multiplier * h2_pred
        else:
            pred[positions] = multiplier * sources[source][positions]
    return np.clip(pred, 0, None)


def v4_predict(train, val):
    """Reproduce v4 on the same rows for an apples-to-apples comparison."""
    lgb_train, lgb_val = hx.categorical_frames(train, val)
    lm = lgb.LGBMRegressor(**LGB_PARAMS)
    lm.fit(lgb_train, train.ratio, categorical_feature=hx.CAT_FEATURES)
    lp = np.clip(lm.predict(lgb_val), 0, None)
    cat_train, cat_val = cat_frames(train, val)
    cm = CatBoostRegressor(**CAT_PARAMS)
    cm.fit(cat_train, train.ratio, cat_features=hx.CAT_FEATURES)
    cp = np.clip(cm.predict(cat_val), 0, None)
    return 0.60 * 0.80 * lp + 0.40 * 0.85 * cp


def validate(train):
    rows = []
    for month, fold_train, fold_val in hx.folds(train):
        v5 = fit_predict(fold_train, fold_val)
        v4 = v4_predict(fold_train, fold_val)
        rows.append({
            "fold": month,
            "role": "untouched_final" if month == "2025-09" else "development",
            "v5_mase": np.mean(np.abs(fold_val.ratio.to_numpy() - v5)),
            "v4_mase": np.mean(np.abs(fold_val.ratio.to_numpy() - v4)),
            "rows": len(fold_val), "movies": fold_val.movie.nunique(),
            "zero_rate": fold_val.y.eq(0).mean(),
        })
    report = pd.DataFrame(rows)
    print(report.to_string(index=False, formatters={
        "v5_mase": "{:.5f}".format, "v4_mase": "{:.5f}".format,
        "zero_rate": "{:.3f}".format,
    }))
    development = report[report.role.eq("development")]
    final = report[report.role.eq("untouched_final")].iloc[0]
    print(f"Development mean: v5={development.v5_mase.mean():.5f}, v4={development.v4_mase.mean():.5f}")
    print(f"Untouched September: v5={final.v5_mase:.5f}, v4={final.v4_mase:.5f}")
    return report


def main():
    raw = pd.read_csv(DATA / "train.csv", parse_dates=["date_show"])
    history = pd.read_csv(DATA / "test_history.csv", parse_dates=["date_show"])
    test_raw = pd.read_csv(DATA / "test.csv", parse_dates=["date_show"])
    train = hx.add_aux(hx.build_dense_cohorts(raw))
    test = hx.add_aux(hx.build_test_features(history, test_raw))

    report = validate(train)
    report.to_csv(DATA / "rolling_cv_v5.csv", index=False)

    ratio = fit_predict(train, test)
    tickets = np.rint(ratio * test.scale.to_numpy()).clip(0).astype(int)
    pred = pd.DataFrame({"id": test.id.astype(int), "total_ticket": tickets})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    submission = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if submission.total_ticket.isna().any():
        raise RuntimeError("Some test IDs have no prediction")
    submission.total_ticket = submission.total_ticket.astype(int)
    submission.to_csv(DATA / "submission_v5.csv", index=False)
    submission.to_csv(DATA / "submission.csv", index=False)
    print("Saved submission_v5.csv and submission.csv")
    print(submission.total_ticket.describe().to_string())


if __name__ == "__main__":
    main()

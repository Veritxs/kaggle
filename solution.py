"""
Cinema ticket demand forecasting.

Task: for each (movie_title, cinema_ids) pair, given the first 3 days of sales
(D1-D3) predict daily total_ticket for the next 7 days (D4-D10).

Metric: MASE = MAE(y_true/scale, y_pred/scale), scale = sum(D1-D3)/3 per pair,
clipped to a minimum of 1.

We model the ratio r = total_ticket / scale with a LightGBM L1 (MAE) objective,
which aligns training with the competition metric and predicts the conditional
median. Final prediction = r_hat * scale, rounded and clipped to >= 1.

Reproducible: SEED = 2026 fixed on every stochastic component.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

SEED = 2026
np.random.seed(SEED)
DATA = "/projects/sandbox/Joints"
HORIZONS = [4, 5, 6, 7, 8, 9, 10]  # future day indices to predict (1-based within run)


# --------------------------------------------------------------------------- #
# Load auxiliary tables
# --------------------------------------------------------------------------- #
def load_aux():
    movies = pd.read_csv(f"{DATA}/movies.csv")
    movies = movies.rename(columns={"original_title": "movie_title"})
    movies["movie_title"] = movies["movie_title"].str.upper().str.strip()
    # keep compact movie features
    mv = movies[["movie_title", "age_rating", "genre"]].copy()
    # primary genre = first listed genre
    mv["genre_primary"] = mv["genre"].fillna("UNK").str.split(",").str[0].str.strip()
    mv["n_genre"] = mv["genre"].fillna("").apply(lambda s: len([x for x in str(s).split(",") if x.strip()]))
    mv = mv.drop(columns=["genre"]).drop_duplicates("movie_title")

    hol = pd.read_csv(f"{DATA}/holidays.csv", parse_dates=["date"])
    hol = hol.rename(columns={"date": "date_show"})
    hol["is_holiday"] = (hol["holiday_tipe"] == "holiday").astype(int)
    hol = hol[["date_show", "day_tipe", "is_holiday"]].drop_duplicates("date_show")

    prices = pd.read_csv(f"{DATA}/ticket_prices.csv")
    prices["day_kind"] = prices["price_day"].str.lower().map(
        {"weekday": "weekday", "friday": "friday", "weekend": "weekend"}
    )
    # pivot price per city per day kind
    pw = prices.pivot_table(index="city_name", columns="day_kind", values="ceil", aggfunc="mean")
    pw.columns = [f"price_{c}" for c in pw.columns]
    pw = pw.reset_index()
    return mv, hol, pw


MOVIES, HOLIDAYS, PRICES = load_aux()


def day_kind_from_dow(dow):
    # dow: Monday=0 ... Sunday=6
    if dow == 4:
        return "friday"
    if dow >= 5:
        return "weekend"
    return "weekday"


# --------------------------------------------------------------------------- #
# Feature engineering for one window (D1-D3 history) + one target day
# --------------------------------------------------------------------------- #
def build_features(history_df, target_rows):
    """
    history_df: rows for D1-D3 of a pair (>=1 row), columns include
                date_show,total_ticket,occupation_rate,total_show,city_name,movie_title
    target_rows: DataFrame with the future day(s) to predict; must have date_show,
                 and (for training) total_ticket.
    Returns a feature dict list, one per target row.
    """
    h = history_df.sort_values("date_show")
    tickets = h["total_ticket"].to_numpy(dtype=float)
    occ = h["occupation_rate"].to_numpy(dtype=float)
    shows = h["total_show"].to_numpy(dtype=float)

    n = len(tickets)
    scale = max(tickets.sum() / 3.0, 1.0)  # matches competition scale (sum/3, clip>=1)

    d_last = h["date_show"].max()
    t_first, t_last, t_mean = tickets[0], tickets[-1], tickets.mean()
    # trend / momentum
    slope = (t_last - t_first) / max(n - 1, 1)
    ratio_last_first = t_last / max(t_first, 1.0)
    ratio_last_mean = t_last / max(t_mean, 1.0)
    vol = tickets.std() if n > 1 else 0.0
    cv = vol / max(t_mean, 1.0)

    city = h["city_name"].iloc[0]
    movie = h["movie_title"].iloc[0]

    base = {
        "scale": scale,
        "h_sum": tickets.sum(),
        "h_mean": t_mean,
        "h_first": t_first,
        "h_last": t_last,
        "h_max": tickets.max(),
        "h_min": tickets.min(),
        "h_slope": slope,
        "h_ratio_lf": ratio_last_first,
        "h_ratio_lm": ratio_last_mean,
        "h_vol": vol,
        "h_cv": cv,
        "h_ndays": n,
        "occ_mean": occ.mean(),
        "occ_last": occ[-1],
        "occ_slope": (occ[-1] - occ[0]) / max(n - 1, 1),
        "shows_mean": shows.mean(),
        "shows_last": shows[-1],
        "ticket_per_show": tickets.sum() / max(shows.sum(), 1.0),
        "city_name": city,
        "movie_title": movie,
    }

    out = []
    for _, tr in target_rows.iterrows():
        d = tr["date_show"]
        horizon = (d - d_last).days  # days ahead of last history day
        dow = d.dayofweek
        row = dict(base)
        row["horizon"] = horizon
        row["dow"] = dow
        row["is_weekend"] = int(dow >= 5)
        row["is_friday"] = int(dow == 4)
        row["target_date"] = d
        if "total_ticket" in tr and pd.notna(tr["total_ticket"]):
            row["total_ticket"] = float(tr["total_ticket"])
        if "id" in tr and pd.notna(tr.get("id", np.nan)):
            row["id"] = tr["id"]
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Assemble training samples from train.csv (opening 3-day window per pair)
# --------------------------------------------------------------------------- #
def make_training_frame():
    tr = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date_show"])
    tr["movie_title"] = tr["movie_title"].str.upper().str.strip()
    rows = []
    for (movie, cinema), g in tr.groupby(["movie_title", "cinema_ids"]):
        g = g.sort_values("date_show").reset_index(drop=True)
        if len(g) < 4:
            continue  # need at least 3 history + 1 target
        # Slide the 3-day history window across the run to enrich training data.
        # start=0 replicates the test setup (opening window); later starts add
        # more horizon/day-of-week coverage. Cap number of windows per pair.
        n = len(g)
        max_start = min(n - 4, 6)  # limit windows so long runs don't dominate
        for start in range(0, max_start + 1):
            hist = g.iloc[start:start + 3]
            if len(hist) < 3:
                break
            d_last = hist["date_show"].max()
            fut = g.iloc[start + 3:].copy()
            fut["horizon"] = (fut["date_show"] - d_last).dt.days
            fut = fut[fut["horizon"].between(1, 7)]
            if fut.empty:
                continue
            rows.extend(build_features(hist, fut))
    df = pd.DataFrame(rows)
    return df


def make_test_frame():
    th = pd.read_csv(f"{DATA}/test_history.csv", parse_dates=["date_show"])
    te = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date_show"])
    th["movie_title"] = th["movie_title"].str.upper().str.strip()
    te["movie_title"] = te["movie_title"].str.upper().str.strip()
    rows = []
    te_g = dict(tuple(te.groupby(["movie_title", "cinema_ids"])))
    for (movie, cinema), g in th.groupby(["movie_title", "cinema_ids"]):
        g = g.sort_values("date_show").iloc[:3]  # first 3 history days
        key = (movie, cinema)
        if key not in te_g:
            continue
        rows.extend(build_features(g, te_g[key]))
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# Merge aux features + encode
# --------------------------------------------------------------------------- #
CAT_COLS = ["city_name", "genre_primary", "age_rating", "dow"]


def add_aux(df):
    df = df.merge(MOVIES[["movie_title", "age_rating", "genre_primary", "n_genre"]],
                  on="movie_title", how="left")
    df = df.merge(PRICES, on="city_name", how="left")
    # price for the target day kind
    def pick_price(r):
        k = day_kind_from_dow(r["dow"])
        return r.get(f"price_{k}", np.nan)
    df["price_target"] = df.apply(pick_price, axis=1)
    df["age_rating"] = df["age_rating"].fillna("UNK")
    df["genre_primary"] = df["genre_primary"].fillna("UNK")
    df["n_genre"] = df["n_genre"].fillna(0)
    return df


FEATURES = [
    "h_sum", "h_mean", "h_first", "h_last", "h_max", "h_min", "h_slope",
    "h_ratio_lf", "h_ratio_lm", "h_vol", "h_cv", "h_ndays",
    "occ_mean", "occ_last", "occ_slope", "shows_mean", "shows_last",
    "ticket_per_show", "horizon", "dow", "is_weekend", "is_friday",
    "n_genre", "price_target",
    "city_name", "genre_primary", "age_rating",
]


def encode(df, cat_maps=None, fit=False):
    df = df.copy()
    if fit:
        cat_maps = {}
    for c in ["city_name", "genre_primary", "age_rating"]:
        if fit:
            cats = df[c].astype("category")
            cat_maps[c] = {v: i for i, v in enumerate(cats.cat.categories)}
        df[c] = df[c].map(cat_maps[c]).fillna(-1).astype(int)
    return df, cat_maps


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def mase(y_true, y_pred, scale):
    return np.mean(np.abs(y_true / scale - y_pred / scale))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Building training frame...")
    train = make_training_frame()
    train = add_aux(train)
    print("train samples:", train.shape)

    print("Building test frame...")
    test = make_test_frame()
    test = add_aux(test)
    print("test samples:", test.shape)

    # target = ratio (aligns with MASE); train on log1p(ratio) for tail stability
    train["ratio"] = train["total_ticket"] / train["scale"]
    y = np.log1p(train["ratio"].to_numpy())

    train_enc, cat_maps = encode(train, fit=True)
    test_enc, _ = encode(test, cat_maps=cat_maps, fit=False)

    Xtr = train_enc[FEATURES]
    Xte = test_enc[FEATURES]
    groups = train_enc["movie_title"]

    params = dict(
        objective="regression_l1",   # MAE -> conditional median, matches metric
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    # CV grouped by movie to mimic unseen-movie generalization; report MASE
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(Xtr))
    for fold, (tri, vai) in enumerate(gkf.split(Xtr, y, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr.iloc[tri], y[tri],
              eval_set=[(Xtr.iloc[vai], y[vai])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = m.predict(Xtr.iloc[vai])
    oof = np.clip(np.expm1(oof), 0, None)  # back-transform log1p(ratio) -> ratio
    pred_tickets = oof * train_enc["scale"].to_numpy()
    cv_mase = mase(train_enc["total_ticket"].to_numpy(), pred_tickets, train_enc["scale"].to_numpy())
    # baseline: predict scale (i.e. mean of D1-D3) for every day
    base_mase = mase(train_enc["total_ticket"].to_numpy(),
                     train_enc["scale"].to_numpy(), train_enc["scale"].to_numpy())
    print(f"\nCV MASE (model):    {cv_mase:.4f}")
    print(f"CV MASE (baseline mean D1-D3): {base_mase:.4f}")

    # Fit on all data for final prediction; seed-average for robustness
    print("\nTraining final model on all data (seed-averaged)...")
    preds = np.zeros(len(Xte))
    seeds = [SEED, SEED + 1, SEED + 2]
    for s in seeds:
        p = dict(params, random_state=s)
        final = lgb.LGBMRegressor(**p)
        final.fit(Xtr, y)
        preds += np.clip(np.expm1(final.predict(Xte)), 0, None)
    ratio_pred = preds / len(seeds)
    tickets_pred = ratio_pred * test_enc["scale"].to_numpy()
    tickets_pred = np.clip(np.round(tickets_pred), 1, None).astype(int)

    sub = pd.DataFrame({"id": test_enc["id"].astype(int), "total_ticket": tickets_pred})
    # align to sample_submission ordering / completeness
    sample = pd.read_csv(f"{DATA}/sample_submission.csv")
    sub = sample[["id"]].merge(sub, on="id", how="left")
    missing = sub["total_ticket"].isna().sum()
    if missing:
        print(f"WARNING: {missing} ids missing predictions; filling with 1")
        sub["total_ticket"] = sub["total_ticket"].fillna(1)
    sub["total_ticket"] = sub["total_ticket"].astype(int)
    sub.to_csv(f"{DATA}/submission.csv", index=False)
    print("\nSaved submission.csv:", sub.shape)
    print(sub.head())

    # feature importance
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(15))


if __name__ == "__main__":
    main()

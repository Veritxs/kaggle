"""
Robust cinema ticket forecast (v3).

Lesson from v2: a heavy LightGBM overfit the training SEASON (CV 0.33 -> LB 0.60).
Under MASE, the dominant signal is the weekly rhythm (weekend vs weekday). A
simple, season-stable rule generalizes far better across the train->test gap.

Approach:
  prediction(day d) = scale * dow_factor[dow(d)] * horizon_decay[h]
  scale = sum(D1-D3)/3  (clip >=1), exactly the competition scale.
  dow_factor  : median of (ticket/scale) by day-of-week, learned on train.
  horizon_decay: gentle median (ticket/scale)/dow_factor by days-ahead, to
                 capture the mild fade after opening (shrunk toward 1 for safety).

Validated OUT-OF-TIME (latest 20% of train windows) so the offline number
tracks the leaderboard. Reproducible: SEED = 2026.
"""
import numpy as np, pandas as pd

SEED = 2026
np.random.seed(SEED)
DATA = "/projects/sandbox/Joints"


def build_windows(df, opening_only=True):
    rows = []
    for (m, c), g in df.groupby(["movie_title", "cinema_ids"]):
        g = g.sort_values("date_show").reset_index(drop=True)
        if len(g) < 4:
            continue
        hist = g.iloc[:3]
        dlast = hist["date_show"].max()
        scale = max(hist["total_ticket"].sum() / 3.0, 1.0)
        fut = g.iloc[3:].copy()
        fut["h"] = (fut["date_show"] - dlast).dt.days
        fut = fut[fut["h"].between(1, 7)]
        for _, r in fut.iterrows():
            rows.append(dict(movie=m, cinema=c, start=g["date_show"].iloc[0],
                             scale=scale, y=float(r["total_ticket"]),
                             dow=int(r["date_show"].dayofweek), h=int(r["h"])))
    return pd.DataFrame(rows)


def fit_factors(train_win):
    r = train_win.assign(ratio=train_win.y / train_win.scale)
    dow_factor = r.groupby("dow")["ratio"].median()
    # horizon effect after removing dow: gentle, shrunk toward 1
    r = r.assign(resid=r.ratio / r.dow.map(dow_factor))
    hz = r.groupby("h")["resid"].median()
    hz = 1.0 + 0.5 * (hz - 1.0)          # shrink 50% toward 1 for robustness
    hz = hz.clip(0.6, 1.6)
    return dow_factor, hz


def predict(win, dow_factor, hz):
    df = 1.0 + 0.7 * (win.dow.map(dow_factor) - 1.0)  # shrink dow effect 30%
    return win.scale * df.clip(lower=0.3) * win.h.map(hz).fillna(1.0)


def mase(y, p, s):
    return np.mean(np.abs(y / s - p / s))


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date_show"])
    tr["movie_title"] = tr["movie_title"].str.upper().str.strip()
    win = build_windows(tr)

    # --- out-of-time validation ---
    cut = win["start"].quantile(0.8)
    trn, val = win[win.start < cut], win[win.start >= cut]
    dow_f, hz = fit_factors(trn)
    val_pred = predict(val, dow_f, hz)
    print(f"OOT val MASE (v3 rule):   {mase(val.y, val_pred, val.scale):.4f}")
    print(f"OOT val MASE (mean base): {mase(val.y, val.scale, val.scale):.4f}")

    # --- refit on ALL train, predict test ---
    dow_f, hz = fit_factors(win)
    th = pd.read_csv(f"{DATA}/test_history.csv", parse_dates=["date_show"])
    te = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date_show"])
    th["movie_title"] = th["movie_title"].str.upper().str.strip()
    te["movie_title"] = te["movie_title"].str.upper().str.strip()

    scale = (th.sort_values("date_show").groupby(["movie_title", "cinema_ids"])
             .total_ticket.apply(lambda s: max(s.iloc[:3].sum() / 3.0, 1.0))
             .rename("scale").reset_index())
    tp = te.merge(scale, on=["movie_title", "cinema_ids"], how="left")
    tp["scale"] = tp["scale"].fillna(1.0)
    tp["dow"] = tp["date_show"].dt.dayofweek
    # horizon = days after last history day
    last_h = (th.groupby(["movie_title", "cinema_ids"]).date_show.max()
              .rename("last_hist").reset_index())
    tp = tp.merge(last_h, on=["movie_title", "cinema_ids"], how="left")
    tp["h"] = (tp["date_show"] - tp["last_hist"]).dt.days.clip(1, 7)

    tp["pred"] = predict(tp.rename(columns={}), dow_f, hz)
    tp["total_ticket"] = np.clip(np.round(tp["pred"]), 1, None).astype(int)

    samp = pd.read_csv(f"{DATA}/sample_submission.csv")
    sub = samp[["id"]].merge(tp[["id", "total_ticket"]], on="id", how="left")
    sub["total_ticket"] = sub["total_ticket"].fillna(1).astype(int)
    sub.to_csv(f"{DATA}/submission.csv", index=False)
    print("saved submission.csv", sub.shape)
    print(sub.total_ticket.describe())


if __name__ == "__main__":
    main()

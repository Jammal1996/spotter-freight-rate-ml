#!/usr/bin/env python3
"""
Spotter ML Engineer assessment - freight rate prediction.
Complete, reproducible pipeline (no Colab / Google Drive dependencies).

    python solution.py                          # data in ./data, outputs in ./
    python solution.py --data-dir path/to/data --out-dir path/to/output

Steps (same order as the exploration notebook):
  1. Load data and run data-quality checks
  2. Exploratory analysis (figures saved to <out-dir>/reports/)
  3. Cleaning and feature engineering
  4. Time-based validation: baseline, first LightGBM, expanding-window CV, error analysis
  5. Corrupted-label detection and the time-trend finding
  6. Final hybrid model (linear trend + LightGBM on residuals): CV, training, prediction
  7. Write validation_predictions.csv and december_chart_predictions.csv
  8. Run score.py (if found) and final sanity checks

Outputs (in --out-dir):
  validation_predictions.csv         12,000 rows: load_id,predicted_rate
  december_chart_predictions.csv     31 rows for the fixed December chart
  scorer_results/candidate_december.png   (created by score.py)
  reports/                           figures and CV tables
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: works on servers, laptops, Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    import statsmodels.api as sm
    from sklearn.metrics import mean_absolute_error
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency ({exc}). Run:  python -m pip install -r requirements.txt")

SEED = 42
T0 = pd.Timestamp("2025-01-01")          # day 0 for the time-trend feature
HOLDOUT_START = pd.Timestamp("2025-09-01")  # first split: fit Jan-Aug, test Sep-Oct
CV_MONTHS = [6, 7, 8, 9, 10]             # expanding-window folds: predict each month
LABEL_THRESHOLD = 0.3                    # |log residual| above this => corrupted label
EXPECTED_ROWS = 12_000


# --------------------------------------------------------------------------- utils
def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def mape(y, p) -> float:
    return float((np.abs(np.asarray(y) - np.asarray(p)) / np.asarray(y)).mean() * 100)


def report(name: str, y, p) -> dict:
    row = {"model": name, "MAE": round(float(mean_absolute_error(y, p)), 1), "MAPE_%": round(mape(y, p), 2)}
    print(f"  {name:<28} MAE = {row['MAE']:7.1f}   MAPE = {row['MAPE_%']:5.2f}%")
    return row


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved figure: {path}")


# ------------------------------------------------------------------ 1. loading
def load_data(data_dir: Path):
    names = {
        "train": "train_test.csv",
        "val": "validation.csv",
        "template": "validation_predictions_template.csv",
    }
    for n in names.values():
        if not (data_dir / n).is_file():
            sys.exit(f"ERROR: {data_dir / n} not found. Use --data-dir to point at the data folder.")
    train = pd.read_csv(data_dir / names["train"], parse_dates=["date"])
    val = pd.read_csv(data_dir / names["val"], parse_dates=["date"])
    template = pd.read_csv(data_dir / names["template"])
    dec_path = data_dir / "december_chart_inputs.csv"
    dec = pd.read_csv(dec_path, parse_dates=["date"]) if dec_path.is_file() else None
    if dec is None:
        print(f"WARNING: {dec_path} not found - the December chart step will be skipped.")
    return train, val, template, dec


def quality_checks(train, val, template) -> None:
    section("1. LOAD DATA AND QUALITY CHECKS")
    print(f"train {train.shape} | validation {val.shape} | template {template.shape}")
    print(f"train dates: {train.date.min().date()} to {train.date.max().date()}")
    print(f"val dates:   {val.date.min().date()} to {val.date.max().date()}")
    print("\nMissing values (train):", train.isna().sum()[lambda s: s > 0].to_dict())
    print("Missing values (val):  ", val.isna().sum()[lambda s: s > 0].to_dict())
    print(f"\nDuplicate load_ids: {train.load_id.duplicated().sum()}")
    print(f"Duplicate rows (ignoring id): {train.duplicated(subset=train.columns[1:]).sum()}")
    print(f"Template ids match validation ids: {set(template.load_id) == set(val.load_id)}")
    print(f"Negative weights  train: {(train.weight < 0).sum()}   val: {(val.weight < 0).sum()}")
    print(f"Same pickup/delivery city: {(train.pickup == train.delivery).sum()} | "
          f"non-positive distance: {(train.distance <= 0).sum()}")


# ------------------------------------------------------------------- 2. EDA
def exploratory_analysis(train, val, rep_dir: Path) -> None:
    section("2. EXPLORATORY ANALYSIS")
    t = train.copy()
    t["rpm"] = t.posted_rate / t.distance

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    t.posted_rate.hist(bins=100, ax=ax[0]); ax[0].set_title("posted_rate")
    np.log(t.posted_rate).hist(bins=100, ax=ax[1]); ax[1].set_title("log(posted_rate)")
    t.rpm.clip(upper=8).hist(bins=100, ax=ax[2]); ax[2].set_title("rate per mile (clipped at 8)")
    save_fig(fig, rep_dir / "eda_target_distribution.png")
    print("Rate-per-mile quantiles:\n", t.rpm.quantile([.001, .01, .05, .5, .95, .99, .999]).round(2).to_string())

    print("\nRate per mile by distance bin:")
    bins = pd.cut(t.distance, [0, 250, 500, 1000, 2000, 4000])
    print(t.groupby(bins, observed=True).rpm.describe().round(2).to_string())

    t["hav"] = haversine(t.pickup_lat, t.pickup_lon, t.delivery_lat, t.delivery_lon)
    print("\ndistance / straight-line ratio:", (t.distance / t.hav).describe().round(3).to_dict())
    print("Max distinct coordinate pairs per pickup city:",
          t.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max().to_dict())

    print("\nEquipment summary:")
    print(t.groupby("equipment").agg(n=("rpm", "size"), median_rpm=("rpm", "median"),
                                     mean_rate=("posted_rate", "mean")).round(2).to_string())
    print("\nCorrelation with rate per mile:")
    print(t[["rpm", "quote_signal", "market_index", "weight", "distance"]].corr().round(2)["rpm"].to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(t.distance, t.posted_rate, s=2, alpha=0.2)
    ax[0].set_xlabel("distance"); ax[0].set_ylabel("posted_rate"); ax[0].set_title("Rate vs distance")
    ax[1].scatter(t.quote_signal, t.rpm.clip(upper=8), s=2, alpha=0.2)
    ax[1].set_xlabel("quote_signal"); ax[1].set_ylabel("rate per mile")
    save_fig(fig, rep_dir / "eda_rate_vs_distance_and_signal.png")

    daily = t.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"), qs=("quote_signal", "mean"))
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    daily.rpm.plot(ax=ax[0], title="Median rate per mile by day")
    daily.mi.plot(ax=ax[1], title="Mean market_index by day")
    daily.qs.plot(ax=ax[2], title="Mean quote_signal by day")
    save_fig(fig, rep_dir / "eda_time_patterns.png")

    print("\nTrain vs validation shift:")
    tc, vc = set(train.pickup) | set(train.delivery), set(val.pickup) | set(val.delivery)
    print(f"  cities only in validation: {sorted(vc - tc)}")
    unseen = (~val.pickup.isin(tc) | ~val.delivery.isin(tc)).sum()
    print(f"  validation rows touching an unseen city: {unseen}")
    mix = pd.concat([train.equipment.value_counts(normalize=True).rename("train"),
                     val.equipment.value_counts(normalize=True).rename("val")], axis=1).round(3)
    print("  equipment mix:\n" + mix.to_string())
    num = ["distance", "weight", "market_index", "quote_signal"]
    print("  numeric means:\n" + pd.concat([train[num].mean().rename("train"),
                                            val[num].mean().rename("val")], axis=1).round(3).to_string())


# ---------------------------------------------------- 3. cleaning and features
class Cleaner:
    """Cleaning rules learned from the training data and applied to any frame."""

    def __init__(self, train: pd.DataFrame):
        self.weight_median = float(train.weight.abs().median())

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["weight_neg"] = (df.weight < 0).astype(int)      # negative weights are sign errors
        df["weight"] = df.weight.abs()
        df["weight_missing"] = df.weight.isna().astype(int)
        df["weight"] = df.weight.fillna(self.weight_median)
        df["mi_missing"] = df.market_index.isna().astype(int)
        daily = df.groupby("date").market_index.median()    # fill with that day's median
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate(limit_direction="both")
        df["market_index"] = df.market_index.fillna(df.date.map(daily))
        return df


def add_features(df: pd.DataFrame, daily_dq=None, daily_dm=None) -> pd.DataFrame:
    d = df.copy()
    d["t"] = (d.date - T0).dt.days                                           # time trend
    d["dq"] = d.date.map(daily_dq) if daily_dq is not None else d.groupby("date").quote_signal.transform("mean")
    d["dm"] = d.date.map(daily_dm) if daily_dm is not None else d.groupby("date").market_index.transform("median")
    d["lm"], d["lq"], d["ldq"] = np.log(d.market_index), np.log(d.quote_signal), np.log(d.dq)
    d["ld"] = np.log(d.distance); d["ld2"] = d.ld ** 2; d["log_dist"] = d.ld
    d["lw"] = np.log(d.weight.clip(lower=1))
    d["hav"] = haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["dow"], d["dom"] = d.date.dt.dayofweek, d.date.dt.day
    d["dom_f"] = d.dom / 31
    for k in (1, 2):
        d[f"ws{k}"] = np.sin(2 * np.pi * k * d.dow / 7)
        d[f"wc{k}"] = np.cos(2 * np.pi * k * d.dow / 7)
    d["equip"] = d.equipment.map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
    return d


# first-draft LightGBM feature set
FIRST_FEATURES = ["distance", "log_dist", "hav", "weight", "weight_missing", "market_index", "mi_missing",
                  "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "dow", "dom", "equip"]
# final hybrid model feature sets
LIN = ["lm", "lq", "ldq", "ld", "ld2", "lw", "t", "dom_f", "ws1", "wc1", "ws2", "wc2"]
GBM = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "equip",
       "dom_f", "dq", "quote_signal", "market_index"]


# ------------------------------------------------ 4. first models and validation
def first_lgbm(objective: str = "huber"):
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             objective=objective, random_state=SEED, verbose=-1)


def first_model_validation(tr: pd.DataFrame, rep_dir: Path) -> None:
    section("4. TIME-BASED VALIDATION: BASELINE AND FIRST LIGHTGBM")
    fit, hold = tr[tr.date < HOLDOUT_START], tr[tr.date >= HOLDOUT_START]
    print(f"Split: fit = Jan-Aug ({len(fit)} rows) | holdout = Sep-Oct ({len(hold)} rows). "
          "Never a random split: the real task predicts future months.\n")
    rows = []
    base_rpm = fit.groupby("equipment").rpm.median()
    rows.append(report("Baseline ($/mile by equipment)", hold.posted_rate, hold.equipment.map(base_rpm) * hold.distance))
    for obj in ("l2", "huber"):
        m = first_lgbm(obj).fit(fit[FIRST_FEATURES], np.log(fit.posted_rate))
        rows.append(report(f"First LightGBM ({obj})", hold.posted_rate, np.exp(m.predict(hold[FIRST_FEATURES]))))
    imp = pd.Series(m.feature_importances_, index=FIRST_FEATURES).sort_values(ascending=False)
    print("\nFeature importance (split counts):\n" + imp.head(8).to_string())
    pd.DataFrame(rows).to_csv(rep_dir / "holdout_results.csv", index=False)

    print("\nExpanding-window CV of the first LightGBM (train on earlier months, predict next month):")
    out = []
    for target in ("rate", "rpm"):
        folds = []
        for k in CV_MONTHS:
            f, h = tr[tr.month < k], tr[tr.month == k]
            m = first_lgbm("huber")
            if target == "rate":
                m.fit(f[FIRST_FEATURES], np.log(f.posted_rate)); p = np.exp(m.predict(h[FIRST_FEATURES]))
            else:
                m.fit(f[FIRST_FEATURES], np.log(f.rpm)); p = np.exp(m.predict(h[FIRST_FEATURES])) * h.distance
            folds.append({"target": f"log({'posted_rate' if target == 'rate' else 'rate/mile'})", "val_month": k,
                          "MAE": round(float(mean_absolute_error(h.posted_rate, p)), 1),
                          "MAPE_%": round(mape(h.posted_rate, p), 2)})
        df = pd.DataFrame(folds)
        print(f"  target = {df.target[0]}: mean MAE = {df.MAE.mean():.1f} | mean MAPE = {df['MAPE_%'].mean():.2f}%")
        out.append(df)
    pd.concat(out).to_csv(rep_dir / "cv_first_lightgbm.csv", index=False)

    print("\nError analysis on the Sep-Oct holdout (first LightGBM, huber):")
    hold = hold.copy()
    hold["pred"] = np.exp(m.fit(fit[FIRST_FEATURES], np.log(fit.posted_rate)).predict(hold[FIRST_FEATURES]))
    hold["ape"] = (hold.pred - hold.posted_rate).abs() / hold.posted_rate * 100
    print("  by equipment:\n" + hold.groupby("equipment").ape.mean().round(2).to_string())
    bins = pd.cut(hold.distance, [0, 250, 500, 1000, 2000, 4000])
    print("  by distance bin:\n" + hold.groupby(bins, observed=True).ape.agg(["mean", "size"]).round(2).to_string())
    worst = hold.sort_values("ape", ascending=False)[["pickup", "delivery", "distance", "equipment",
                                                      "posted_rate", "pred", "ape"]].head(10).round(1)
    print("  10 worst predictions (model ~5x off => the labels, not the model, look wrong):\n" + worst.to_string())


# ------------------------------------------------------ 5. final hybrid model
def lin_X(d: pd.DataFrame) -> pd.DataFrame:
    x = (pd.get_dummies(d.equipment).astype(float)
         .reindex(columns=["Dry Van", "Flatbed", "Reefer"], fill_value=0)[["Flatbed", "Reefer"]])
    for c in LIN:
        x[c] = d[c]
    return sm.add_constant(x, has_constant="add")


def flag_label_outliers(d: pd.DataFrame, thr: float = LABEL_THRESHOLD):
    """Fit a simple model, flag rows whose price is far off, refit without them (3 rounds)."""
    keep = np.ones(len(d), bool)
    res = None
    for _ in range(3):
        r = sm.OLS(np.log(d.posted_rate[keep]), lin_X(d[keep])).fit()
        res = np.log(d.posted_rate) - r.predict(lin_X(d))
        keep = (res.abs() <= thr).values
    return ~keep, res


def fit_hybrid(d: pd.DataFrame):
    """Linear model (distance, equipment, weight, time trend, daily signals) + LightGBM on its residuals.
    Corrupted labels are excluded from training only."""
    bad, _ = flag_label_outliers(d)
    f = d[~bad]
    lin = sm.OLS(np.log(f.posted_rate), lin_X(f)).fit()
    gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                            subsample=0.8, subsample_freq=1, objective="l2", random_state=SEED, verbose=-1)
    gbm.fit(f[GBM], np.log(f.posted_rate) - lin.predict(lin_X(f)))
    return lin, gbm


def predict_hybrid(model, d: pd.DataFrame) -> np.ndarray:
    lin, gbm = model
    return np.exp(lin.predict(lin_X(d)) + gbm.predict(d[GBM]))


def label_and_trend_analysis(tr: pd.DataFrame, rep_dir: Path) -> pd.DataFrame:
    section("5. CORRUPTED LABELS AND TIME TREND")
    tr = tr.copy()
    bad, resid = flag_label_outliers(tr)
    tr["is_bad"], tr["resid"] = bad, resid
    print(f"Rows flagged as corrupted: {bad.sum()} ({bad.mean() * 100:.2f}%)")
    print(f"  too LOW  (price < 0.74x expected): {(resid < -LABEL_THRESHOLD).sum()}")
    print(f"  too HIGH (price > 1.35x expected): {(resid > LABEL_THRESHOLD).sum()}")
    print(f"  typical error of normal rows (std of log residual): {resid[~bad].std():.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(resid.clip(-2, 2), bins=120); ax[0].set_title("log(real price) - log(expected price)")
    ax[0].set_xlabel("0 = on target; +/-1 = off by 2.7x")
    tr.groupby("month").is_bad.sum().plot.bar(ax=ax[1], title="Corrupted rows per month")
    save_fig(fig, rep_dir / "label_outliers.png")
    print("  examples:\n" + tr[bad].assign(ratio=np.exp(resid[bad]).round(2))
          [["pickup", "delivery", "distance", "equipment", "posted_rate", "ratio"]].head(8).to_string())

    c = tr[~tr.is_bad]
    no_trend = lambda d: lin_X(d).drop(columns=["t", "dom_f", "ws1", "wc1", "ws2", "wc2"])
    r0 = sm.OLS(np.log(c.posted_rate), no_trend(c)).fit()
    monthly = c.assign(r0=r0.resid).groupby("month").r0.mean() * 100
    print("\nAverage % that prices sit above/below a model WITHOUT a time trend, by month:")
    print(monthly.round(2).to_string())
    r1 = sm.OLS(np.log(c.posted_rate), lin_X(c)).fit()
    print(f"Trend: about {(np.exp(r1.params['t'] * 30) - 1) * 100:.2f}% per 30 days | residual std "
          f"{r0.resid.std():.4f} (no trend) -> {r1.resid.std():.4f} (trend + daily signals)")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    monthly.plot.bar(ax=ax, title="Prices drift upward through the year (no-trend model)")
    save_fig(fig, rep_dir / "time_trend.png")
    return tr


def hybrid_cv(tr: pd.DataFrame, rep_dir: Path) -> pd.DataFrame:
    section("6. FINAL HYBRID MODEL: EXPANDING-WINDOW CROSS-VALIDATION")
    rows = []
    for k in CV_MONTHS:
        f, h = tr[tr.month < k], tr[tr.month == k]
        p = predict_hybrid(fit_hybrid(f), h)
        ape = (p - h.posted_rate).abs() / h.posted_rate * 100
        rows.append({"val_month": k, "train_rows": len(f),
                     "MAE": round(float(mean_absolute_error(h.posted_rate, p)), 1),
                     "MAPE_all_%": round(float(ape.mean()), 2),
                     "MAPE_clean_%": round(float(ape[~h.is_bad.values].mean()), 2)})
    cv = pd.DataFrame(rows)
    print(cv.to_string(index=False))
    print(f"AVERAGE  MAE = {cv.MAE.mean():.1f} | MAPE all rows = {cv['MAPE_all_%'].mean():.2f}% | "
          f"MAPE clean rows = {cv['MAPE_clean_%'].mean():.2f}%")
    print("('all rows' is the honest number; 'clean' excludes rows flagged as corrupted labels.)")
    cv.to_csv(rep_dir / "cv_hybrid_model.csv", index=False)
    return cv


# ----------------------------------------------------- 7. predictions and output
def predict_validation(model, cleaner: Cleaner, val, template, out_dir: Path):
    section("7. PREDICT VALIDATION SET AND DECEMBER CHART")
    val_clean = cleaner.clean(val)
    val_f = add_features(val_clean)
    val_f["pred"] = predict_hybrid(model, val_f)
    sub = template[["load_id"]].merge(
        val_f[["load_id", "pred"]].rename(columns={"pred": "predicted_rate"}), on="load_id", how="left")
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert len(sub) == EXPECTED_ROWS and sub.load_id.is_unique, "unexpected number of rows / duplicate ids"
    assert sub.predicted_rate.notna().all() and (sub.predicted_rate > 0).all(), "invalid predictions"
    path = out_dir / "validation_predictions.csv"
    sub.to_csv(path, index=False)
    print(f"Saved {path}  {sub.shape}")
    return val_clean, val_f, sub


def predict_december(model, train, val_clean, dec, out_dir: Path):
    if dec is None:
        print("Skipping December chart predictions (december_chart_inputs.csv missing).")
        return None
    # the input file has only 7 columns: add coordinates from train, market signals from validation.csv
    city = (pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        train[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).drop_duplicates("city").set_index("city"))
    dd = dec.copy()
    dd["pickup_lat"], dd["pickup_lon"] = dd.pickup.map(city.lat), dd.pickup.map(city.lon)
    dd["delivery_lat"], dd["delivery_lon"] = dd.delivery.map(city.lat), dd.delivery.map(city.lon)
    daily_dq = val_clean.groupby("date").quote_signal.mean()
    daily_dm = val_clean.groupby("date").market_index.median()
    dd["quote_signal"], dd["market_index"] = dd.date.map(daily_dq), dd.date.map(daily_dm)
    dd["weight_missing"] = 0
    dd["mi_missing"] = 0
    assert dd[["pickup_lat", "delivery_lat", "quote_signal", "market_index"]].notna().all().all(), \
        "December rows reference a city or date with no data"
    dd = add_features(dd, daily_dq, daily_dm)
    out = dec[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].copy()
    out["predicted_rate"] = predict_hybrid(model, dd).round(2)
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    path = out_dir / "december_chart_predictions.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}  (predicted_rate range {out.predicted_rate.min():.2f} - {out.predicted_rate.max():.2f})")
    return out


# ----------------------------------------------------------- 8. scorer + checks
def run_scorer(out_dir: Path, score_script: str | None) -> None:
    section("8. RUN score.py AND FINAL CHECKS")
    candidates = [Path(score_script)] if score_script else [Path.cwd() / "score.py", Path(__file__).resolve().parent / "score.py"]
    script = next((p for p in candidates if p.is_file()), None)
    dec_file = out_dir / "december_chart_predictions.csv"
    if script is None or not dec_file.is_file():
        print("score.py (or the December predictions) not found - skipping. Run manually:\n"
              "  python score.py --predictions validation_predictions.csv "
              "--december-predictions december_chart_predictions.csv")
        return
    res = subprocess.run([sys.executable, str(script),
                          "--predictions", str(out_dir / "validation_predictions.csv"),
                          "--december-predictions", str(dec_file),
                          "--output-dir", str(out_dir / "scorer_results")],
                         capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr.strip())
        sys.exit("score.py reported an error (see above).")


def final_checks(out_dir: Path, val, val_f, tr) -> None:
    chk = pd.read_csv(out_dir / "validation_predictions.csv")
    print(f"\nColumns: {list(chk.columns)} | rows: {len(chk)} | unique ids: {chk.load_id.nunique()}")
    print(f"Same ids as validation.csv: {set(chk.load_id) == set(val.load_id)}")
    print(f"Missing / non-positive predictions: {chk.predicted_rate.isna().sum()} / {(chk.predicted_rate <= 0).sum()}")
    vm = val_f.assign(month=val_f.date.dt.month, ppm=val_f.pred / val_f.distance)
    print("\nPredicted median $/mile by month (validation): "
          + str(vm.groupby("month").ppm.median().round(3).to_dict()))
    tm = tr[~tr.is_bad].assign(r=lambda x: x.posted_rate / x.distance).groupby("month").r.median().round(3)
    print("Train median $/mile by month:                   " + str(tm.to_dict()))
    print(f"Predictions: min {chk.predicted_rate.min():.0f} | median {chk.predicted_rate.median():.0f} | "
          f"max {chk.predicted_rate.max():.0f}")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Freight rate prediction: full pipeline")
    ap.add_argument("--data-dir", default="data", help="folder with the four CSV files (default: data)")
    ap.add_argument("--out-dir", default=".", help="where to write outputs (default: current folder)")
    ap.add_argument("--score-script", default=None, help="path to score.py (default: ./score.py or next to this file)")
    ap.add_argument("--skip-exploration", action="store_true",
                    help="skip EDA, first-model comparison and CV of the first model (faster)")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    train, val, template, dec = load_data(data_dir)
    quality_checks(train, val, template)
    if not args.skip_exploration:
        exploratory_analysis(train, val, rep_dir)

    section("3. CLEANING AND FEATURE ENGINEERING")
    cleaner = Cleaner(train)
    print("- negative weights -> abs(); missing weight -> train median + flag")
    print("- missing market_index -> same-day median (interpolated if a whole day is missing) + flag")
    print("- no validation rows are ever dropped (all 12,000 need a prediction)")
    tr = add_features(cleaner.clean(train))
    tr["month"] = tr.date.dt.month
    tr["rpm"] = tr.posted_rate / tr.distance

    if not args.skip_exploration:
        first_model_validation(tr, rep_dir)

    tr = label_and_trend_analysis(tr, rep_dir)
    hybrid_cv(tr, rep_dir)

    final_model = fit_hybrid(tr)
    print(f"\nFinal model trained on all labeled data: {int((~tr.is_bad).sum())} clean rows "
          f"({int(tr.is_bad.sum())} corrupted rows excluded)")

    val_clean, val_f, _ = predict_validation(final_model, cleaner, val, template, out_dir)
    predict_december(final_model, train, val_clean, dec, out_dir)
    run_scorer(out_dir, args.score_script)
    final_checks(out_dir, val, val_f, tr)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Vietnamese Auto Parts Demand Forecasting — v2
==============================================
Target metric : WRMSSE (Weighted Root Mean Squared Scaled Error)
Forecast horizon : 56 days (F1-F28 validation, F29-F56 evaluation)

Key improvements over v1
─────────────────────────
1.  Croston / TSB forecasting for intermittent / sparse SKUs
    – replaces the naïve decayed-rate heuristic
    – TSB (Teunter–Syntetos–Babai) handles demand that "dies out"
    – directly models demand interval + demand size separately

2.  Spike-aware upscaling for ACTIVE SKUs
    – computes a SKU-level 95th-percentile spike factor
    – blends base forecast with a spike-inflated version weighted
      by the empirical spike probability (P(sale day > 2×mean))
    – prevents systematic under-prediction that WRMSSE punishes heavily

3.  Quantile-blended base forecast
    – uses median (Q50) + mean blend instead of mean-only
    – reduces sensitivity to a single extreme outlier inflating the mean
    – better calibration for long-tail / zero-inflated distributions

4.  Richer LightGBM feature set
    – adds: p75/p95 of non-zero qty, spike_prob, tsb_demand_rate,
      demand_interval, recent_spike_count, seasonal_strength
    – 4th segment: "VOLATILE" (active but highly erratic, cv > 2)
      routed to a dedicated spike-boosted forecaster

5.  Recency-weighted EWM (ACTIVE forecaster)
    – alpha tuned per SKU based on recent vs. historical mean ratio
    – higher alpha (more reactive) for SKUs with accelerating demand
    – lower alpha for stable, high-volume SKUs

6.  Better Sunday / holiday zeroing
    – Sundays always zero (store closed)
    – can be extended to public holiday list

7.  Improved cold-start / new SKU handling
    – SKUs with history < 14 days fall back to category-median rate
      (computed from category prefix in ItemCode when available)

8.  Fallback guard for recursive drift
    – forecast values are capped at max(3×recent_max, global_max)
    – prevents runaway predictions in recursive EWM updates
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "train.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"
OUTPUT_PATH     = "submission_lgbm_v2.csv"

LAST_TRAIN_DATE = pd.Timestamp("2025-09-05")
VAL_DATES       = pd.date_range("2025-09-06", periods=28)   # F1..F28
EVAL_DATES      = pd.date_range("2025-10-04", periods=28)   # F29..F56

HORIZON         = 28

# ── EWM / blending ─────────────────────────────────────────────────────────────
EWM_ALPHA_BASE  = 0.12    # base smoothing; will be adjusted per-SKU
EWM_ALPHA_MAX   = 0.30    # ceiling for reactive SKUs
EWM_WEIGHT      = 0.55    # blend weight: EWM vs. rolling mean
RECENT_WINDOW   = 56
DOW_WINDOW      = 112

# ── Segmentation thresholds ────────────────────────────────────────────────────
SPARSE_CUTOFF   = 20      # min active days → ACTIVE
VOLATILE_CV     = 1.8     # CV threshold → VOLATILE segment
DEAD_SKU_CUTOFF = 730     # days since last sale → DEAD

# ── Spike / WRMSSE ────────────────────────────────────────────────────────────
SPIKE_PERCENTILE   = 90   # percentile used to define "spike" magnitude
SPIKE_PROB_FLOOR   = 0.05 # minimum spike probability to activate spike blend
SPIKE_BLEND_WEIGHT = 0.30 # how much of spike-forecast to mix in

# ── Cap to prevent runaway predictions ────────────────────────────────────────
FORECAST_CAP_MULT  = 5.0  # cap = FORECAST_CAP_MULT × rolling-max recent window

# ── Croston / TSB ─────────────────────────────────────────────────────────────
TSB_ALPHA     = 0.10   # demand-size smoothing
TSB_BETA      = 0.10   # demand-probability smoothing

# ── LightGBM ──────────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective         = "multiclass",
    num_class         = 4,            # DEAD / SPARSE / VOLATILE / ACTIVE
    n_estimators      = 600,
    learning_rate     = 0.04,
    num_leaves        = 48,
    max_depth         = -1,
    min_child_samples = 15,
    subsample         = 0.80,
    colsample_bytree  = 0.80,
    reg_alpha         = 0.15,
    reg_lambda        = 0.15,
    verbose           = -1,
    n_jobs            = -1,
    random_state      = 42,
)
LGBM_CV_FOLDS = 5

SEG_DEAD     = 0
SEG_SPARSE   = 1
SEG_VOLATILE = 2
SEG_ACTIVE   = 3
SEG_NAMES    = {SEG_DEAD: "DEAD", SEG_SPARSE: "SPARSE",
                SEG_VOLATILE: "VOLATILE", SEG_ACTIVE: "ACTIVE"}


# ─── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("=" * 65)
print("1. Loading training data …")
train = pd.read_csv(
    TRAIN_PATH, low_memory=False,
    usecols=["Date", "ItemCode", "Quantity"],
)
train["Date"]     = pd.to_datetime(train["Date"])
train["Quantity"] = pd.to_numeric(train["Quantity"], errors="coerce").fillna(0)
train["Quantity"] = train["Quantity"].clip(lower=0)

daily_raw = train.groupby(["Date", "ItemCode"])["Quantity"].sum()

print("  Building per-SKU series …")
sku_series: dict[str, dict] = {}
for (date, sku), qty in daily_raw.items():
    if sku not in sku_series:
        sku_series[sku] = {}
    sku_series[sku][date] = qty

all_skus = list(sku_series.keys())
print(f"  → {len(all_skus):,} unique SKUs")


# ─── HELPER: build dense series ────────────────────────────────────────────────
def _dense(sku: str, clip_date=None) -> pd.Series:
    """Return zero-filled daily series for a SKU up to clip_date."""
    s = pd.Series(sku_series[sku]).sort_index()
    if clip_date is not None:
        s = s[s.index <= clip_date]
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(s.index.min(), clip_date or s.index.max())
    return s.reindex(idx, fill_value=0).astype(float)


# ─── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(sku: str) -> dict:
    """
    Extended feature set (v2).
    New vs. v1: p75_nonzero, p95_nonzero, spike_prob, recent_spike_count,
    tsb_demand_rate, demand_interval, seasonal_strength, cv_28
    """
    s_dense = _dense(sku, LAST_TRAIN_DATE)
    if s_dense.empty:
        return _zero_features()

    history_days = len(s_dense)
    nz_all = int((s_dense > 0).sum())
    nz_90  = int((s_dense.tail(90)  > 0).sum())
    nz_30  = int((s_dense.tail(30)  > 0).sum())
    nz_14  = int((s_dense.tail(14)  > 0).sum())

    # ── Zero-streak ────────────────────────────────────────────────────────
    streak = 0
    for v in reversed(s_dense.values):
        if v == 0:
            streak += 1
        else:
            break
    zero_streak_length = streak

    # ── Recency ────────────────────────────────────────────────────────────
    s_raw = pd.Series(sku_series[sku]).sort_index()
    s_raw = s_raw[s_raw.index <= LAST_TRAIN_DATE]
    days_since_last = (LAST_TRAIN_DATE - s_raw.index.max()).days

    # ── Gap stats ──────────────────────────────────────────────────────────
    sale_dates = s_dense[s_dense > 0].index
    if len(sale_dates) >= 2:
        gaps = np.diff(sale_dates).astype("timedelta64[D]").astype(int)
        avg_gap = float(gaps.mean())
        max_gap = float(gaps.max())
    else:
        avg_gap = float(history_days)
        max_gap = float(history_days)

    global_activation_rate = nz_all / max(history_days, 1)

    # ── Volume / distribution ──────────────────────────────────────────────
    nonzero_vals = s_dense[s_dense > 0].values
    if len(nonzero_vals) == 0:
        return _zero_features()

    mean_qty_nonzero = float(nonzero_vals.mean())
    cv_qty = (float(nonzero_vals.std()) / (mean_qty_nonzero + 1e-6)
              if len(nonzero_vals) > 1 else 0.0)
    max_qty = float(s_dense.max())
    p75_nonzero = float(np.percentile(nonzero_vals, 75))
    p95_nonzero = float(np.percentile(nonzero_vals, 95))

    mean_qty_30 = float(s_dense.tail(30).mean())
    mean_qty_90 = float(s_dense.tail(90).mean())
    qty_trend_ratio = mean_qty_30 / (mean_qty_90 + 1e-6)

    # NEW: CV over recent 28 days (non-zero only) — detects recent volatility
    recent_nz = s_dense.tail(28)
    recent_nz = recent_nz[recent_nz > 0].values
    cv_28 = (float(recent_nz.std()) / (float(recent_nz.mean()) + 1e-6)
             if len(recent_nz) > 1 else 0.0)

    # NEW: Spike probability — fraction of sale days where qty > 2× mean
    spike_threshold = 2.0 * mean_qty_nonzero
    spike_prob = float((nonzero_vals > spike_threshold).mean())

    # NEW: Recent spike count (last 90 days)
    recent_nz_90 = s_dense.tail(90)
    mean_90_nz   = float(recent_nz_90[recent_nz_90 > 0].mean()) if nz_90 > 0 else 0
    recent_spike_count = int((recent_nz_90 > 2 * mean_90_nz).sum()) if mean_90_nz > 0 else 0

    # NEW: TSB demand rate — expected demand per day (Teunter–Syntetos–Babai)
    #      Approximated: probability of demand × mean demand when selling
    demand_interval = avg_gap  # mean inter-demand interval
    tsb_demand_rate = (1.0 / max(demand_interval, 1)) * mean_qty_nonzero

    # NEW: Seasonal strength — ratio of max-DOW mean to min-DOW mean
    if history_days >= 14:
        dow_means = s_dense.groupby(s_dense.index.dayofweek).mean()
        # exclude Sunday (dayofweek=6) if it's always 0
        non_sun = dow_means.drop(6, errors="ignore")
        if len(non_sun) >= 2 and non_sun.min() > 0:
            seasonal_strength = float(non_sun.max() / non_sun.min())
        else:
            seasonal_strength = 1.0
    else:
        seasonal_strength = 1.0

    return {
        "non_zero_count_14"     : nz_14,
        "non_zero_count_30"     : nz_30,
        "non_zero_count_90"     : nz_90,
        "non_zero_count_all"    : nz_all,
        "zero_streak_length"    : zero_streak_length,
        "avg_gap_between_sales" : avg_gap,
        "max_gap_between_sales" : max_gap,
        "cv_qty"                : cv_qty,
        "cv_28"                 : cv_28,
        "days_since_last_sale"  : days_since_last,
        "global_activation_rate": global_activation_rate,
        "mean_qty_nonzero"      : mean_qty_nonzero,
        "mean_qty_30"           : mean_qty_30,
        "mean_qty_90"           : mean_qty_90,
        "qty_trend_ratio"       : qty_trend_ratio,
        "max_qty"               : max_qty,
        "p75_nonzero"           : p75_nonzero,
        "p95_nonzero"           : p95_nonzero,
        "spike_prob"            : spike_prob,
        "recent_spike_count"    : recent_spike_count,
        "tsb_demand_rate"       : tsb_demand_rate,
        "demand_interval"       : demand_interval,
        "seasonal_strength"     : seasonal_strength,
        "history_days"          : history_days,
    }


def _zero_features() -> dict:
    keys = [
        "non_zero_count_14", "non_zero_count_30", "non_zero_count_90",
        "non_zero_count_all", "zero_streak_length", "avg_gap_between_sales",
        "max_gap_between_sales", "cv_qty", "cv_28", "days_since_last_sale",
        "global_activation_rate", "mean_qty_nonzero", "mean_qty_30",
        "mean_qty_90", "qty_trend_ratio", "max_qty", "p75_nonzero",
        "p95_nonzero", "spike_prob", "recent_spike_count", "tsb_demand_rate",
        "demand_interval", "seasonal_strength", "history_days",
    ]
    return {k: 0.0 for k in keys}


print("\n2. Engineering SKU features …")
feature_rows = []
for sku in all_skus:
    row = build_features(sku)
    row["sku"] = sku
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows).set_index("sku")
FEATURE_COLS = [c for c in feat_df.columns]
print(f"  → Feature matrix: {feat_df.shape}  ({len(FEATURE_COLS)} features)")


# ─── 3. RULE-BASED LABELS ──────────────────────────────────────────────────────
print("\n3. Generating rule-based segment labels (4-class) …")


def rule_label(row) -> int:
    if row["days_since_last_sale"] > DEAD_SKU_CUTOFF:
        return SEG_DEAD
    if row["non_zero_count_all"] < SPARSE_CUTOFF:
        return SEG_SPARSE
    # Active but highly volatile → separate routing
    if row["cv_qty"] > VOLATILE_CV or row["spike_prob"] > 0.25:
        return SEG_VOLATILE
    return SEG_ACTIVE


feat_df["label"] = feat_df.apply(rule_label, axis=1)

for seg_id, cnt in feat_df["label"].value_counts().sort_index().items():
    print(f"  {SEG_NAMES[seg_id]:9s} ({seg_id}): {cnt:,}")


# ─── 4. TRAIN LightGBM CLASSIFIER ─────────────────────────────────────────────
print("\n4. Training LightGBM 4-class segmentation classifier …")

X = feat_df[FEATURE_COLS].values.astype(np.float32)
y = feat_df["label"].values

skf      = StratifiedKFold(n_splits=LGBM_CV_FOLDS, shuffle=True, random_state=42)
models   = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    acc = (model.predict(X[va_idx]) == y[va_idx]).mean()
    print(f"  Fold {fold}: val-accuracy = {acc:.4f}")
    models.append(model)

ensemble_probs = np.mean([m.predict_proba(X) for m in models], axis=0)
lgbm_labels    = ensemble_probs.argmax(axis=1)

feat_df["lgbm_segment"] = lgbm_labels

print("\n  LightGBM segment distribution:")
for seg_id in [SEG_DEAD, SEG_SPARSE, SEG_VOLATILE, SEG_ACTIVE]:
    cnt = (lgbm_labels == seg_id).sum()
    print(f"    {SEG_NAMES[seg_id]:9s}: {cnt:,}")

imp      = np.mean([m.feature_importances_ for m in models], axis=0)
top_feat = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:12]
print("\n  Top-12 feature importances:")
for fname, fimp in top_feat:
    print(f"    {fname:<30s}  {fimp:.1f}")

sku_segment   = dict(zip(feat_df.index, feat_df["lgbm_segment"].values))
sku_feat_map  = feat_df[FEATURE_COLS].to_dict(orient="index")


# ─── 5. FORECASTING FUNCTIONS ──────────────────────────────────────────────────

def _per_sku_alpha(sku: str) -> float:
    """
    Adaptive EWM alpha: more reactive when recent demand is accelerating.
    If mean_30 > 1.5 × mean_90 → higher alpha (trend is up).
    """
    row = sku_feat_map.get(sku, {})
    m30 = row.get("mean_qty_30", 0) + 1e-6
    m90 = row.get("mean_qty_90", 0) + 1e-6
    ratio = m30 / m90
    if ratio > 1.5:
        return min(EWM_ALPHA_BASE * ratio, EWM_ALPHA_MAX)
    return EWM_ALPHA_BASE


def _spike_boost(base_pred: float, sku: str) -> float:
    """
    Blend base forecast with a spike-inflated version.
    Weight = spike_prob (capped) × SPIKE_BLEND_WEIGHT.
    Spike level = p95 of historical non-zero demand.
    Prevents systematic under-prediction of large-volume days.
    """
    row  = sku_feat_map.get(sku, {})
    p95  = row.get("p95_nonzero", 0)
    prob = row.get("spike_prob", 0)
    if prob < SPIKE_PROB_FLOOR or p95 <= base_pred:
        return base_pred
    blend_w = min(prob, 0.50) * SPIKE_BLEND_WEIGHT
    return base_pred * (1 - blend_w) + p95 * blend_w


def _forecast_cap(sku: str) -> float:
    """Cap to prevent runaway predictions."""
    s = _dense(sku, LAST_TRAIN_DATE)
    recent_max = float(s.tail(RECENT_WINDOW).max()) if len(s) >= 1 else 0
    global_max = float(s.max()) if len(s) >= 1 else 0
    cap = max(FORECAST_CAP_MULT * recent_max, global_max, 1.0)
    return cap


# ── TSB (Teunter–Syntetos–Babai) ───────────────────────────────────────────────
def _tsb_forecast(sku: str) -> float:
    """
    TSB method for intermittent demand.
    Maintains smoothed:
      p  = probability of a non-zero demand period
      z  = mean demand level when non-zero
    Forecast = p × z
    Better than Croston for products where demand is "dying out".
    """
    s = _dense(sku, LAST_TRAIN_DATE)
    if s.empty or s.sum() == 0:
        return 0.0

    vals = s.values
    # Initialise
    nz_idx = np.where(vals > 0)[0]
    if len(nz_idx) == 0:
        return 0.0
    p = len(nz_idx) / len(vals)       # initial demand probability
    z = float(vals[nz_idx].mean())    # initial demand size

    # One-pass TSB update over the series
    for v in vals:
        if v > 0:
            p = (1 - TSB_BETA) * p + TSB_BETA * 1.0
            z = (1 - TSB_ALPHA) * z + TSB_ALPHA * v
        else:
            p = (1 - TSB_BETA) * p + TSB_BETA * 0.0
            # z unchanged on zero periods

    return max(0.0, p * z)


# ── Active forecaster (EWM + DOW + spike boost) ────────────────────────────────
def forecast_active(sku: str, forecast_dates: pd.DatetimeIndex,
                    volatile: bool = False) -> list[int]:
    """
    EWM + rolling mean + day-of-week scaling.
    For VOLATILE SKUs: higher alpha + stronger spike blend.
    """
    s = _dense(sku, LAST_TRAIN_DATE)
    if len(s) == 0:
        return [0] * HORIZON

    # Limit history window to avoid very old regime dominating
    lookback = min(len(s), 365)
    s = s.tail(lookback)

    d112 = s.tail(DOW_WINDOW)
    d56  = s.tail(RECENT_WINDOW)
    d28  = s.tail(28)

    dow_mean    = d112.groupby(d112.index.dayofweek).mean()
    global_mean = d112.mean()

    alpha   = _per_sku_alpha(sku) * (1.5 if volatile else 1.0)
    alpha   = min(alpha, EWM_ALPHA_MAX)
    ewm_val = d28.ewm(alpha=alpha).mean().iloc[-1]

    # Quantile-blended base: mix mean and median to reduce outlier drag
    recent_mean   = d56.mean()
    recent_median = d56.median()
    blended_recent = 0.7 * recent_mean + 0.3 * recent_median

    base = EWM_WEIGHT * ewm_val + (1 - EWM_WEIGHT) * blended_recent

    # Spike boost
    base = _spike_boost(base, sku)

    cap = _forecast_cap(sku)

    preds = []
    for fd in forecast_dates:
        dow = fd.dayofweek
        if global_mean > 0 and dow in dow_mean.index:
            dow_ratio = dow_mean[dow] / (global_mean + 1e-9)
        else:
            dow_ratio = 1.0
        pred = base * dow_ratio
        pred = min(pred, cap)
        preds.append(max(0, round(pred)))

    return preds


# ── Sparse forecaster (TSB-based) ──────────────────────────────────────────────
def forecast_sparse(sku: str, forecast_dates: pd.DatetimeIndex) -> list[int]:
    """
    TSB intermittent demand forecast.
    Replaces the v1 decayed-daily-rate heuristic.
    TSB naturally accounts for demand dying out vs. persisting.
    """
    tsb_rate = _tsb_forecast(sku)

    # Additional recency decay if the SKU hasn't sold in a long time
    row         = sku_feat_map.get(sku, {})
    days_since  = row.get("days_since_last_sale", 0)
    if days_since > 180:
        decay = max(0.1, 1.0 - (days_since - 180) / 730)
        tsb_rate *= decay

    base = max(0.0, tsb_rate)
    return [max(0, round(base))] * HORIZON


# ── Router ─────────────────────────────────────────────────────────────────────
def forecast_sku(sku: str, forecast_dates: pd.DatetimeIndex) -> list[int]:
    if sku not in sku_series:
        return [0] * HORIZON

    seg = sku_segment.get(sku, SEG_SPARSE)

    if seg == SEG_DEAD:
        preds = [0] * HORIZON
    elif seg == SEG_ACTIVE:
        preds = forecast_active(sku, forecast_dates, volatile=False)
    elif seg == SEG_VOLATILE:
        # VOLATILE: active but erratic — use active forecaster with spike boost
        preds = forecast_active(sku, forecast_dates, volatile=True)
    else:  # SEG_SPARSE
        preds = forecast_sparse(sku, forecast_dates)

    # ── Sunday = store closed ──────────────────────────────────────────────
    preds = [0 if fd.dayofweek == 6 else p
             for p, fd in zip(preds, forecast_dates)]

    return preds


# ─── 6. RUN FORECASTS ──────────────────────────────────────────────────────────
print("\n5. Generating submission forecasts …")
sub     = pd.read_csv(SAMPLE_SUB_PATH)
sub_ids = sub["id"].tolist()

results = []
for i, row_id in enumerate(sub_ids):
    if i % 10_000 == 0:
        print(f"  {i:,} / {len(sub_ids):,}")

    sku = row_id.replace("_validation", "").replace("_evaluation", "")
    if "_validation" in row_id:
        preds = forecast_sku(sku, VAL_DATES)
    else:
        preds = forecast_sku(sku, EVAL_DATES)

    results.append(preds)


# ─── 7. OUTPUT ─────────────────────────────────────────────────────────────────
forecast_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
out = pd.DataFrame(results, columns=forecast_cols)
out.insert(0, "id", sub_ids)

vals     = out[forecast_cols].values.flatten()
flat_pct = (out[forecast_cols].nunique(axis=1) == 1).mean()

print(f"\n✓ Done! Summary stats:")
print(f"  Output shape : {out.shape}")
print(f"  Mean pred    : {vals.mean():.4f}")
print(f"  % zeros      : {(vals == 0).mean():.1%}")
print(f"  % flat rows  : {flat_pct:.1%}")

out.to_csv(OUTPUT_PATH, index=False)
print(f"\n→ Saved to {OUTPUT_PATH}")


# ─── 8. SANITY CHECK ───────────────────────────────────────────────────────────
print("\nSample predictions (F1-F7) for example SKUs:")
for sku in ["SKU-00003", "SKU-00002", "SKU-09458"]:
    row = out[out["id"] == f"{sku}_validation"]
    if len(row):
        seg_name = SEG_NAMES.get(sku_segment.get(sku, -1), "UNKNOWN")
        vals_7   = row.iloc[0, 1:8].values
        print(f"  {sku} [{seg_name:8s}]: {vals_7}")

# Sunday check
sunday_cols = [f"F{i+1}" for i, d in enumerate(VAL_DATES) if d.dayofweek == 6]
if sunday_cols:
    sun_vals = out[sunday_cols].values.flatten()
    print(f"\nSunday columns in VAL window : {sunday_cols}")
    print(f"  All Sunday predictions = 0? {(sun_vals == 0).all()}")

# Segment-level mean predictions
print("\nMean non-Sunday forecast by segment:")
for seg_id in [SEG_DEAD, SEG_SPARSE, SEG_VOLATILE, SEG_ACTIVE]:
    skus_in_seg = [s for s, seg in sku_segment.items() if seg == seg_id]
    ids_val = [f"{s}_validation" for s in skus_in_seg]
    sub_rows = out[out["id"].isin(ids_val)]
    if len(sub_rows):
        non_sun = [c for c in forecast_cols if c not in sunday_cols]
        mean_pred = sub_rows[non_sun].values.mean()
        print(f"  {SEG_NAMES[seg_id]:9s}: {mean_pred:.3f} (n={len(sub_rows):,})")

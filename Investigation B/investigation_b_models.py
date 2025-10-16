#!/usr/bin/env python3
"""
Investigation B - Darwin Group 38
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

USING_STATSMODELS = True
try:
    import statsmodels.api as sm
    from patsy import dmatrices
except Exception:
    USING_STATSMODELS = False
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_args():
    ap = argparse.ArgumentParser(description="Investigation B portable analysis pipeline")
    # Defaults are RELATIVE to the script folder; we'll resolve them in main().
    ap.add_argument("--d1", default="dataset1_clean.csv", help="Path to dataset 1 (default: file next to this script)")
    ap.add_argument("--d2", default="dataset2_clean.csv", help="Path to dataset 2 (default: file next to this script)")
    ap.add_argument("--out", default="outputs", help="Output folder (default: ./outputs next to script)")
    ap.add_argument("--merge-mode", choices=["exact","window30"], default="exact", help="Merge mode")
    ap.add_argument("--iqr-threshold", type=float, default=1.5, help="IQR multiple for winsorization")
    return ap.parse_args()

def resolve_path(p_like: str) -> Path:
    """
    Resolve a user-provided path. If it's absolute, keep it.
    If it's relative, interpret it relative to the script's folder (BASE_DIR).
    """
    p = Path(p_like)
    if not p.is_absolute():
        p = BASE_DIR / p_like
    return p

# I/O helpers
def read_clean_d1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["start_time", "sunset_time", "rat_period_start", "rat_period_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["season", "month", "habit"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "start_time" in df.columns:
        df = df.dropna(subset=["start_time"])
    return df

def read_clean_d2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "month" in df.columns:
        df["month"] = df["month"].astype("category")
    if "bat_landing_number" in df.columns:
        df = df[df["bat_landing_number"].ge(0)]
    return df.dropna(subset=["time"]) if "time" in df.columns else df

# Merging and feature engineering
def merge_exact_keep_all(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    if "start_time" not in d1.columns or "time" not in d2.columns:
        d1c = d1.copy(); d1c["__row_id_d1"] = np.arange(len(d1c))
        d2c = d2.copy(); d2c["__row_id_d2"] = np.arange(len(d2c))
        merged = d1c.merge(d2c, left_on="__row_id_d1", right_on="__row_id_d2", how="outer", suffixes=("_d1","_d2"))
        merged.drop(columns=["__row_id_d1","__row_id_d2"], inplace=True, errors="ignore")
        merged["merge_time"] = pd.NaT
        return merged

    d1c = d1.copy()
    d2c = d2.copy()
    d1c["key_time"] = d1c["start_time"]
    d2c["key_time"] = d2c["time"]
    merged = pd.merge(d1c, d2c, on="key_time", how="outer", suffixes=("_d1", "_d2"), copy=True, sort=True)
    if "time" in merged.columns:
        merged["merge_time"] = merged["time"].combine_first(merged["start_time"] if "start_time" in merged.columns else pd.NaT)
    else:
        merged["merge_time"] = merged["start_time"] if "start_time" in merged.columns else merged["key_time"]
    return merged

def merge_window30(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    if "start_time" not in d1.columns or "time" not in d2.columns:
        return merge_exact_keep_all(d1, d2)
    d1c = d1.copy(); d2c = d2.copy()
    d1c["bin30"] = d1c["start_time"].dt.floor("30min")
    d2c["bin30"] = d2c["time"]
    merged = pd.merge(d1c, d2c, on="bin30", how="outer", suffixes=("_d1", "_d2"), copy=True, sort=True)
    merged["merge_time"] = merged["bin30"]
    merged.drop(columns=["bin30"], inplace=True, errors="ignore")
    return merged

def assign_season_from_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    for c in ["merge_time", "time", "start_time", "key_time"]:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        df["season"] = 1
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    winter_start = pd.Timestamp("2017-12-01")
    winter_end   = pd.Timestamp("2018-02-28")
    df["season"] = np.where((dt >= winter_start) & (dt <= winter_end), 0, 1).astype("int64")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def pick(col):
        if col in df.columns: return df[col]
        if f"{col}_d2" in df.columns: return df[f"{col}_d2"]
        if f"{col}_d1" in df.columns: return df[f"{col}_d1"]
        return pd.Series(index=df.index, dtype="float64")

    for col in ["bat_landing_number","rat_minutes","rat_arrival_number","food_availability","hours_after_sunset"]:
        if col not in df.columns:
            if f"{col}_d2" in df.columns:
                df[col] = df[f"{col}_d2"]
            elif f"{col}_d1" in df.columns:
                df[col] = df[f"{col}_d1"]

    if "season" in df.columns:
        df["season"] = df["season"].astype(int)

    df["rat_influence"] = pick("rat_minutes").fillna(0) * pick("rat_arrival_number").fillna(0)
    denom = pick("rat_minutes"); numer = pick("food_availability")
    df["risk_reward_ratio"] = numer / denom.replace(0, np.nan)

    if "season" in df.columns and "rat_arrival_number" in df.columns:
        for s in df["season"].dropna().astype(int).unique():
            df[f"season_{s}_x_rat_arrival"] = (df["season"].eq(s).astype(int) *
                                               df["rat_arrival_number"].fillna(0))

    for c in ["bat_landing_number","rat_minutes","food_availability","rat_arrival_number","rat_influence","risk_reward_ratio","risk"]:
        if c in df.columns:
            df.loc[np.isinf(df[c]), c] = np.nan

    return df

def cap_outliers_iqr(df: pd.DataFrame, cols, threshold: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        before_min, before_max = float(series.min()), float(series.max())
        df[col] = df[col].clip(lower, upper)
        after_min, after_max = float(df[col].min()), float(df[col].max())
        rows.append({
            "column": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_cap": lower, "upper_cap": upper,
            "min_before": before_min, "max_before": before_max,
            "min_after": after_min, "max_after": after_max
        })
    report = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column","q1","q3","iqr","lower_cap","upper_cap","min_before","max_before","min_after","max_after"])
    return df, report

# EDA Section
def run_eda(df: pd.DataFrame, outdir: Path):
    outdir = ensure_dir(outdir)

    numeric_cols = [
        c for c in [
            "bat_landing_number","rat_arrival_number","rat_minutes","food_availability",
            "hours_after_sunset","rat_influence","rat_influence_log",
            "risk_reward_ratio","risk_reward_ratio_log"
        ] if c in df.columns
    ]
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["skew"] = [skew(df[c].dropna()) if c in df.columns else np.nan for c in desc.index]
        desc["kurtosis"] = [kurtosis(df[c].dropna(), fisher=True) if c in df.columns else np.nan for c in desc.index]
        desc.to_csv(outdir / "eda_descriptive_stats.csv")

    corr_cols = [c for c in ["bat_landing_number","rat_arrival_number","food_availability","hours_after_sunset","rat_influence_log","risk_reward_ratio_log"] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(method="pearson")
        fig = plt.figure(figsize=(6,5))
        plt.imshow(corr.values, interpolation="nearest")
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.title("Correlation Heatmap (Pearson)")
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.values[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(outdir / "eda_corr_heatmap.png", dpi=150)
        plt.close(fig)

    if "season" in df.columns and "bat_landing_number" in df.columns and df["season"].notna().any():
        seasons_sorted = sorted(pd.Series(df["season"].dropna().unique()).astype(int).tolist())
        groups = [df.loc[df["season"] == s, "bat_landing_number"].dropna().values for s in seasons_sorted]
        if any(len(g) for g in groups):
            fig = plt.figure()
            plt.boxplot(groups, labels=[str(int(s)) for s in seasons_sorted], showfliers=True)
            plt.xlabel("season (0=winter, 1=spring)")
            plt.ylabel("bat_landing_number")
            plt.title("Bat Landings by Season")
            plt.tight_layout()
            plt.savefig(outdir / "eda_boxplot_bat_vs_season.png", dpi=150)
            plt.close(fig)

# Modelling and plots
def fit_models_and_plots(df: pd.DataFrame, outdir: Path):
    if "bat_landing_number" not in df.columns:
        raise ValueError("Missing bat_landing_number in merged data.")
    mdf = df.dropna(subset=["bat_landing_number"]).copy()

    has = mdf.columns.__contains__
    ycol = "bat_landing_number"

    m1_rhs = ["season"] if has("season") else []
    m2_rhs = (["season"] if has("season") else []) + [c for c in ["rat_minutes","rat_arrival_number","food_availability","hours_after_sunset"] if has(c)]
    base_m3 = [c for c in ["rat_arrival_number","food_availability","hours_after_sunset"] if has(c)]
    engineered_logs = [c for c in ["rat_influence_log","risk_reward_ratio_log"] if has(c)]
    interactions = [c for c in mdf.columns if c.startswith("season_") and c.endswith("_x_rat_arrival")]
    m3_rhs = (["season"] if has("season") else []) + base_m3 + engineered_logs + interactions

    comp_rows = []

    fig = plt.figure()
    mdf[ycol].plot(kind="hist", bins=20)
    plt.title("Histogram: bat_landing_number"); plt.tight_layout()
    plt.savefig(outdir / "hist_bat_landing_number.png", dpi=150); plt.close(fig)

    if has("season"):
        for s in sorted(mdf["season"].dropna().astype(int).unique()):
            sub = mdf.loc[mdf["season"] == s, ycol]
            if len(sub):
                fig = plt.figure()
                sub.plot(kind="hist", bins=20)
                plt.title(f"Histogram: bat_landing_number (season={s})"); plt.tight_layout()
                plt.savefig(outdir / f"hist_bat_landing_number_season_{s}.png", dpi=150); plt.close(fig)

    numeric_preds = ["rat_arrival_number","food_availability","hours_after_sunset"]
    for pred in [p for p in numeric_preds if has(p)] + [p for p in ["rat_influence_log","risk_reward_ratio_log"] if has(p)]:
        fig = plt.figure()
        plt.scatter(mdf[pred], mdf[ycol], alpha=0.6)
        plt.xlabel(pred); plt.ylabel(ycol); plt.title(f"{ycol} vs {pred}")
        try:
            mask = mdf[pred].notna() & mdf[ycol].notna()
            if mask.sum() >= 2:
                coef = np.polyfit(mdf.loc[mask, pred], mdf.loc[mask, ycol], 1)
                xline = np.linspace(mdf[pred].min(), mdf[pred].max(), 100)
                yline = coef[0]*xline + coef[1]
                plt.plot(xline, yline)
        except Exception:
            pass
        plt.tight_layout(); plt.savefig(outdir / f"scatter_{ycol}_vs_{pred}.png", dpi=150); plt.close(fig)

    if USING_STATSMODELS:
        def fit_ols(rhs, name):
            rhs_terms = " + ".join(rhs) if rhs else "1"
            formula = f"{ycol} ~ {rhs_terms}"
            y, X = dmatrices(formula, mdf, return_type="dataframe")
            model = sm.OLS(y, X, hasconst=True).fit()
            with open(outdir / f"{name}_summary.txt", "w") as f:
                f.write(model.summary().as_text())
            coefs = model.params.drop(labels=[i for i in model.params.index if "Intercept" in i], errors="ignore")
            if len(coefs):
                fig = plt.figure()
                coefs.abs().sort_values().plot(kind="barh")
                plt.title(f"{name.upper()} | |coefficients|")
                plt.tight_layout(); plt.savefig(outdir / f"{name}_coef_bars.png", dpi=150); plt.close(fig)
            return model

        m1 = fit_ols(m1_rhs, "model1")
        m2 = fit_ols(m2_rhs, "model2")
        m3 = fit_ols(m3_rhs, "model3")

        comp_rows.append({"model": "model1", "aic": m1.aic, "bic": m1.bic, "r2": m1.rsquared, "r2_adj": m1.rsquared_adj})
        comp_rows.append({"model": "model2", "aic": m2.aic, "bic": m2.bic, "r2": m2.rsquared, "r2_adj": m2.rsquared_adj})
        comp_rows.append({"model": "model3", "aic": m3.aic, "bic": m3.bic, "r2": m3.rsquared, "r2_adj": m3.rsquared_adj})
    else:
        def fit_lr(cols, name):
            X = mdf[cols].copy() if cols else pd.DataFrame(index=mdf.index)
            y = mdf[ycol].values
            if not cols:
                yhat = np.repeat(np.mean(y), len(y)); r2 = r2_score(y, yhat)
                return {"model": name, "r2": r2}
            lr = LinearRegression()
            lr.fit(X, y); r2 = lr.score(X, y)
            return {"model": name, "r2": r2}
        comp_rows.append(fit_lr(m1_rhs, "model1"))
        comp_rows.append(fit_lr(m2_rhs, "model2"))
        comp_rows.append(fit_lr(m3_rhs, "model3"))

    pd.DataFrame(comp_rows).to_csv(outdir / "model_comparison.csv", index=False)

def main():
    args = parse_args()
    # Resolve provided paths relative to the SCRIPT folder
    d1_path = resolve_path(args.d1)
    d2_path = resolve_path(args.d2)
    outdir = ensure_dir(resolve_path(args.out))

    # Load data
    d1 = read_clean_d1(str(d1_path))
    d2 = read_clean_d2(str(d2_path))

    # Merge
    if args.merge_mode == "exact":
        merged = merge_exact_keep_all(d1, d2)
    else:
        merged = merge_window30(d1, d2)

    (outdir / "merged_keep_as_is.csv").parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "merged_keep_as_is.csv", index=False)

    # Season tag & features
    merged = assign_season_from_date(merged)
    merged_fe = engineer_features(merged)

    # Outlier capping
    merged_fe, cap_report = cap_outliers_iqr(
        merged_fe,
        ["bat_landing_number", "rat_arrival_number", "rat_minutes", "risk_reward_ratio", "risk"],
        threshold=args.iqr_threshold
    )
    cap_report.to_csv(outdir / "outlier_capping_report.csv", index=False)

    # Log transforms
    if "rat_influence" in merged_fe.columns:
        merged_fe["rat_influence_log"] = np.log1p(merged_fe["rat_influence"].clip(lower=0))
    if "risk_reward_ratio" in merged_fe.columns:
        rr = merged_fe["risk_reward_ratio"].where(merged_fe["risk_reward_ratio"] >= 0, np.nan)
        merged_fe["risk_reward_ratio_log"] = np.log1p(rr)

    merged_fe.to_csv(outdir / "merged_with_features.csv", index=False)

    # EDA and Models
    run_eda(merged_fe, outdir)
    fit_models_and_plots(merged_fe, outdir)

    print(f"Done. Outputs written to: {outdir}")

if __name__ == "__main__":
    main()

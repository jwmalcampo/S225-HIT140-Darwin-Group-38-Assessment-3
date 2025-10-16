import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")
OUT_DIR = "A_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def _month_to_num(x):
    MAP = {'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
           'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
           'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12}
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    return MAP.get(s, pd.to_numeric(s, errors="coerce"))

def load_or_merge():
    if os.path.exists("merged_dataset.csv"):
        df = pd.read_csv("merged_dataset.csv")
        print("Loaded: merged_dataset.csv", df.shape)
        return df

    if not (os.path.exists("dataset1_clean.csv") and os.path.exists("dataset2_clean.csv")):
        raise FileNotFoundError("Missing merged_dataset.csv and/or (dataset1_clean.csv, dataset2_clean.csv).")
    d1 = pd.read_csv("dataset1_clean.csv")
    d2 = pd.read_csv("dataset2_clean.csv")

    for d in (d1, d2):
        if "month" in d.columns:
            d["month"] = d["month"].apply(_month_to_num)
        if "hours_after_sunset" in d.columns:
            d["hours_after_sunset"] = pd.to_numeric(d["hours_after_sunset"], errors="coerce")

    df = pd.merge(
        d1, d2,
        on=["month", "hours_after_sunset"],
        how="left",
        suffixes=("_bat","_obs")
    )
    print("Merged dataset1_clean + dataset2_clean:", df.shape)
    return df

df = load_or_merge()

keep_candidates = [
    "bat_landing_number", "bat_landing_to_food",
    "reward", "risk",
    "rat_arrival_number", "rat_minutes",
    "food_availability", "hours_after_sunset",
    "rat_present"
]

if "rat_present" not in df.columns and "rat_arrival_number" in df.columns:
    df["rat_present"] = (pd.to_numeric(df["rat_arrival_number"], errors="coerce").fillna(0) > 0).astype(int)

for c in keep_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

dfA = df.dropna(subset=["bat_landing_number", "rat_present"]).copy()
print("Rows available for Investigation A:", dfA.shape[0])

#EDA (descriptives + plots) - by rat presence

dfA["rat_present_label"] = dfA["rat_present"].map({0:"No Rat", 1:"Rat Present"})
ORDER = ["No Rat", "Rat Present"]
def iqr(s): return s.quantile(0.75) - s.quantile(0.25)

def summarize(series):
    if series not in dfA.columns: return None
    tbl = (dfA.groupby("rat_present_label")[series]
           .agg(count="count", mean="mean", median="median", std="std",
                q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), iqr=iqr)
           .reindex(ORDER).round(2))
    print(f"\n[EDA] {series} by rat presence:\n", tbl)
    return tbl

eda_tables = {}
for series in ["bat_landing_to_food", "bat_landing_number"]:
    t = summarize(series)
    if t is not None:
        eda_tables[f"{series}_by_rat"] = t

if "reward" in dfA.columns:
    eda_tables["reward_by_rat"] = (dfA.groupby("rat_present_label")["reward"]
                                   .agg(count="count", success_rate="mean")
                                   .reindex(ORDER).round(3))
    print("\n[EDA] Feeding success by rat presence:\n", eda_tables["reward_by_rat"])

if "risk" in dfA.columns:
    eda_tables["risk_by_rat"] = (dfA.groupby("rat_present_label")["risk"]
                                 .agg(count="count", risk_rate="mean")
                                 .reindex(ORDER).round(3))
    print("\n[EDA] Risk-taking by rat presence:\n", eda_tables["risk_by_rat"])

if "bat_landing_to_food" in dfA.columns:
    plt.figure()
    sns.boxplot(data=dfA, x="rat_present_label", y="bat_landing_to_food", order=ORDER, showfliers=True)
    sns.stripplot(data=dfA, x="rat_present_label", y="bat_landing_to_food",
                  order=ORDER, color="0.25", alpha=0.35, jitter=True)
    plt.title("Hesitation Time by Rat Presence")
    plt.xlabel("Rat Presence"); plt.ylabel("Seconds to Get Food");
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "A_hesitation_by_rat.png"), dpi=200); plt.close()

plt.figure()
sns.pointplot(data=dfA, x="rat_present_label", y="bat_landing_number", order=ORDER, errorbar=("ci",95))
plt.title("Mean Bat Landings by Rat Presence (95% CI)")
plt.xlabel("Rat Presence"); plt.ylabel("Bat Landings per Period");
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "A_landings_by_rat.png"), dpi=200); plt.close()

for col, title, fn in [("reward","Feeding Success Rate by Rat Presence","A_success_by_rat.png"),
                       ("risk","Risk-taking Rate by Rat Presence","A_risk_by_rat.png")]:
    if col in dfA.columns:
        plt.figure()
        sns.barplot(data=dfA, x="rat_present_label", y=col, order=ORDER, estimator="mean", errorbar=("ci",95))
        plt.title(title); plt.xlabel("Rat Presence"); plt.ylabel("Rate"); plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, fn), dpi=200); plt.close()

heat_cols = [c for c in [
    "bat_landing_number","bat_landing_to_food",
    "rat_arrival_number","rat_minutes",
    "food_availability","hours_after_sunset","reward","risk"
] if c in dfA.columns]
if len(heat_cols) >= 2:
    plt.figure(figsize=(8,6))
    sns.heatmap(dfA[heat_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix (Investigation A)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "A_corr_heatmap.png"), dpi=200); plt.close()

#Statistical Tests

print("\nSTATISTICAL TESTS (Investigation A)")

def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2)) if (nx+ny-2)>0 else np.nan
    return (x.mean() - y.mean()) / sp if sp>0 else np.nan

tests_out = {}

if "bat_landing_to_food" in dfA.columns:
    x0 = dfA.loc[dfA["rat_present"]==0, "bat_landing_to_food"].dropna().to_numpy()
    x1 = dfA.loc[dfA["rat_present"]==1, "bat_landing_to_food"].dropna().to_numpy()
    t_stat, p_val = stats.ttest_ind(x0, x1, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(x0, x1, alternative="two-sided")
    d = cohens_d(x0, x1)
    tests_out["hesitation_tests"] = pd.DataFrame({
        "test":["Welch t","Mann-Whitney U","Cohen d"],
        "stat":[t_stat, u_stat, d],
        "p":[p_val, p_u, np.nan]
    })
    print("\n[INFERENTIAL] Hesitation time:")
    print(f"  Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Mann–Whitney U: U={u_stat:.0f}, p={p_u:.4f}")
    print(f"  Cohen's d = {d:.3f}")

y0 = dfA.loc[dfA["rat_present"]==0, "bat_landing_number"].dropna().to_numpy()
y1 = dfA.loc[dfA["rat_present"]==1, "bat_landing_number"].dropna().to_numpy()
t2, p2 = stats.ttest_ind(y0, y1, equal_var=False)
d_land = cohens_d(y0, y1)
tests_out["landing_tests"] = pd.DataFrame({
    "test":["Welch t","Cohen d"],
    "stat":[t2, d_land],
    "p":[p2, np.nan]
})
print("\n[INFERENTIAL] Bat landings:")
print(f"  Welch t-test: t={t2:.3f}, p={p2:.4f}")
print(f"  Cohen's d = {d_land:.3f}")

if "reward" in dfA.columns:
    tab = pd.crosstab(dfA["rat_present"], dfA["reward"]).reindex(index=[0,1], columns=[0,1], fill_value=0)
    chi2, p_chi, dof, exp = stats.chi2_contingency(tab)
    n = tab.values.sum()
    phi2 = chi2/n
    r, c = tab.shape
    cv = np.sqrt(phi2 / (min(r-1, c-1)))
    p_no = tab.loc[0,1]/tab.loc[0].sum() if tab.loc[0].sum() else np.nan
    p_yes = tab.loc[1,1]/tab.loc[1].sum() if tab.loc[1].sum() else np.nan
    tests_out["reward_chi2"] = pd.DataFrame({
        "chi2":[chi2], "p":[p_chi], "cramers_v":[cv],
        "success_rate_no_rat":[p_no], "success_rate_rat_present":[p_yes]
    })
    print("\n[INFERENTIAL] Feeding success:")
    print(f"  Chi-square: chi2={chi2:.3f}, p={p_chi:.4f}, Cramér's V={cv:.3f}")

if "risk" in dfA.columns:
    tab = pd.crosstab(dfA["rat_present"], dfA["risk"]).reindex(index=[0,1], columns=[0,1], fill_value=0)
    chi2, p_chi, dof, exp = stats.chi2_contingency(tab)
    n = tab.values.sum()
    phi2 = chi2/n
    r, c = tab.shape
    cv = np.sqrt(phi2 / (min(r-1, c-1)))
    r_no = tab.loc[0,1]/tab.loc[0].sum() if tab.loc[0].sum() else np.nan
    r_yes = tab.loc[1,1]/tab.loc[1].sum() if tab.loc[1].sum() else np.nan
    tests_out["risk_chi2"] = pd.DataFrame({
        "chi2":[chi2], "p":[p_chi], "cramers_v":[cv],
        "risk_rate_no_rat":[r_no], "risk_rate_rat_present":[r_yes]
    })
    print("\n[INFERENTIAL] Risk-taking:")
    print(f"  Chi-square: chi2={chi2:.3f}, p={p_chi:.4f}, Cramér's V={cv:.3f}")

#Hierarchical Linear Regression

print("\nHIERARCHICAL LINEAR REGRESSION (Investigation A)")

def calc_metrics(y_true, y_pred, p):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_true.max() - y_true.min())
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - p - 1)
    return mae, mse, rmse, nrmse, r2, adj_r2

def show_metrics(name, y_true, y_pred, p):
    mae, mse, rmse, nrmse, r2, adj_r2 = calc_metrics(y_true, y_pred, p)
    print(f"\n{name}")
    print(f"  MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}  NRMSE={nrmse:.3f}  R²={r2:.3f}  AdjR²={adj_r2:.3f}")
    return {"MAE":mae,"MSE":mse,"RMSE":rmse,"NRMSE":nrmse,"R2":r2,"AdjR2":adj_r2}

y_full = dfA["bat_landing_number"].astype(float)

# Model 1
M1 = ["rat_present"]
X1 = dfA[M1].fillna(0.0)
X1_tr, X1_te, y_tr, y_te = train_test_split(X1, y_full, test_size=0.30, random_state=42)
m1 = LinearRegression().fit(X1_tr, y_tr)
y1_hat = m1.predict(X1_te)
m1_res = show_metrics("Model 1: Simple (rat_present)", y_te, y1_hat, p=X1_te.shape[1])
print("  Coefficients:", dict(zip(M1, m1.coef_)))

# Model 2
M2_cont = [c for c in ["rat_arrival_number","rat_minutes","food_availability","hours_after_sunset"] if c in dfA.columns]
M2 = ["rat_present"] + M2_cont
X2 = dfA[M2].fillna(0.0)
X2_scaled = X2.copy()
if M2_cont:
    sc2 = StandardScaler().fit(X2[M2_cont])
    X2_scaled[M2_cont] = sc2.transform(X2[M2_cont])
X2_tr, X2_te, y_tr, y_te = train_test_split(X2_scaled, y_full, test_size=0.30, random_state=42)
m2 = LinearRegression().fit(X2_tr, y_tr)
y2_hat = m2.predict(X2_te)
m2_res = show_metrics("Model 2: Multiple (rat + food + time)", y_te, y2_hat, p=X2_te.shape[1])
print("  Coefficients:", dict(zip(M2, m2.coef_)))

# Model 3 (feature engineered)
dfA["rat_influence"] = dfA.get("rat_arrival_number",0.0) * dfA.get("rat_minutes",0.0)
dfA["rats_x_food"]   = dfA.get("rat_arrival_number",0.0) * dfA.get("food_availability",0.0)
M3_cont = [c for c in ["rat_arrival_number","rat_minutes","food_availability","hours_after_sunset","rat_influence","rats_x_food"] if c in dfA.columns]
M3 = ["rat_present"] + M3_cont
X3 = dfA[M3].fillna(0.0)
X3_scaled = X3.copy()
if M3_cont:
    sc3 = StandardScaler().fit(X3[M3_cont])
    X3_scaled[M3_cont] = sc3.transform(X3[M3_cont])
X3_tr, X3_te, y_tr, y_te = train_test_split(X3_scaled, y_full, test_size=0.30, random_state=42)
m3 = LinearRegression().fit(X3_tr, y_tr)
y3_hat = m3.predict(X3_te)
m3_res = show_metrics("Model 3: Feature-engineered (interactions)", y_te, y3_hat, p=X3_te.shape[1])
print("  Coefficients:", dict(zip(M3, m3.coef_)))

comp = pd.DataFrame([
    ["Model 1: rat_present", *m1_res.values()],
    ["Model 2: +rat+food+time", *m2_res.values()],
    ["Model 3: +interactions", *m3_res.values()],
], columns=["Model","MAE","MSE","RMSE","NRMSE","R2","AdjR2"]).round(3)
print("\nModel comparison (A):\n", comp)
comp.to_csv(os.path.join(OUT_DIR, "A_model_comparison.csv"), index=False)

#Residual diagnostics (Model 3)

residuals = y_te - y3_hat
plt.figure()
plt.scatter(y3_hat, residuals, alpha=0.5)
plt.axhline(0, ls="--")
plt.xlabel("Fitted values (Model 3)"); plt.ylabel("Residuals")
plt.title("Residuals vs Fitted — Model 3 (Investigation A)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "A_residuals_model3.png"), dpi=200); plt.close()

from scipy.stats import probplot
plt.figure()
probplot(residuals, dist="norm", plot=plt)
plt.title("Q–Q Plot of Residuals — Model 3 (Investigation A)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "A_QQ_model3.png"), dpi=200); plt.close()

#Save all tables to a single Excel workbook

xlsx_path = os.path.join(OUT_DIR, "A_summaries.xlsx")
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
    for name, tbl in eda_tables.items():
        tbl.to_excel(xw, sheet_name=name[:31])
    comp.to_excel(xw, sheet_name="model_comparison", index=False)
    for name, tbl in tests_out.items():
        tbl.to_excel(xw, sheet_name=name[:31], index=False)
print(f"\nSaved Excel summaries to: {xlsx_path}")

#Final printed conclusion for Investigation A

print("\nINVESTIGATION A - FINAL CONCLUSION")
print("Descriptive and inferential analyses show no suppression of bat foraging when rats are present.")
print("Hierarchical regression (Models 1–3) shows small linear effects of rat presence/activity;")
print("findings are consistent with competition rather than predation.")
print("Therefore, Investigation A supports: bats treat rats primarily as COMPETITORS for food, not PREDATORS.")

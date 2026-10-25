import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE_DIR = r"c:\Users\swsz9\Downloads\WindowexeHideWindow_2026-03-21\Distributables\ZindiMiva2026"
train_raw = pd.read_csv(os.path.join(BASE_DIR, "Train.csv"))
test_raw = pd.read_csv(os.path.join(BASE_DIR, "Test.csv"))
clim_raw = pd.read_csv(os.path.join(BASE_DIR, "climate_features.csv")).drop(columns=["deathdate"])

train_df = train_raw.merge(clim_raw, on="ID", how="left")
test_df = test_raw.merge(clim_raw, on="ID", how="left")

# Combine for unsupervised temporal/cohort/location baseline statistics
full_df = pd.concat([train_df.assign(_is_train=1), test_df.assign(_is_train=0, is_climate_sensitive=np.nan)], ignore_index=True)
full_df["deathdate_dt"] = pd.to_datetime(full_df["deathdate"])
full_df["year"] = full_df["deathdate_dt"].dt.year
full_df["month"] = full_df["deathdate_dt"].dt.month
full_df["day_of_year"] = full_df["deathdate_dt"].dt.dayofyear
full_df["day_of_week"] = full_df["deathdate_dt"].dt.dayofweek
full_df["quarter"] = full_df["deathdate_dt"].dt.quarter

# Cohort-level unsupervised coding statistics by year and (year, quarter)
# Captures the 2018-2021 surveillance shift where missing ages were coded as age=0
year_stats = full_df.groupby("year").agg(
    year_total_deaths=("ID", "count"),
    year_age0_rate=("age", lambda s: (s == 0).mean()),
    year_age_mean=("age", "mean"),
    year_age_median=("age", lambda s: s[s > 0].median() if (s > 0).any() else 0.0),
    year_rural_rate=("zone", lambda s: (s == "Rural").mean()),
).reset_index()
full_df = full_df.merge(year_stats, on="year", how="left")

yq_stats = full_df.groupby(["year", "quarter"]).agg(
    yq_total_deaths=("ID", "count"),
    yq_age0_rate=("age", lambda s: (s == 0).mean()),
).reset_index()
full_df = full_df.merge(yq_stats, on=["year", "quarter"], how="left")

# Rolling temporal density of mortality records around each record's date
date_counts = full_df.groupby("deathdate_dt").size().rename("same_day_deaths")
full_df = full_df.merge(date_counts, on="deathdate_dt", how="left")

sorted_dates = np.sort(full_df["deathdate_dt"].values.astype("datetime64[D]").astype(np.int64))
cur_dates = full_df["deathdate_dt"].values.astype("datetime64[D]").astype(np.int64)
for win in [7, 14, 30, 60]:
    left_idx = np.searchsorted(sorted_dates, cur_dates - win, side="left")
    right_idx = np.searchsorted(sorted_dates, cur_dates + win, side="right")
    full_df[f"cohort_deaths_pm_{win}d"] = right_idx - left_idx

# Location-level unsupervised climate baselines (so models see local anomalies rather than static village IDs)
loc_clim_cols = ["tavg_30d", "tavg_90d", "rain_sum_30d", "rain_sum_90d", "ndvi_30d", "ndvi_90d", "temp_range_mean_30d"]
loc_stats = full_df.groupby("location")[loc_clim_cols].transform("mean")
for c in loc_clim_cols:
    full_df[f"{c}_loc_anom"] = full_df[c] - loc_stats[c]

# District extraction from location string
def extract_district(loc_str):
    s = str(loc_str).lower()
    if "iganga" in s:
        return 0
    elif "mayuge" in s:
        return 1
    elif "bugweri" in s:
        return 2
    else:
        return 3

full_df["district_code"] = full_df["location"].apply(extract_district)

def build_features(df):
    X = pd.DataFrame(index=df.index)
    
    # 1. Demographic & Surveillance Regime Features
    X["age"] = df["age"].astype(float)
    X["age_log1p"] = np.log1p(np.maximum(0.0, X["age"]))
    X["is_age_zero"] = (df["age"] == 0).astype(float)
    X["is_child_1_4"] = ((df["age"] >= 1) & (df["age"] <= 4)).astype(float)
    X["is_child_u5"] = (df["age"] < 5).astype(float)
    X["is_child_5_14"] = ((df["age"] >= 5) & (df["age"] <= 14)).astype(float)
    X["is_adult_15_49"] = ((df["age"] >= 15) & (df["age"] <= 49)).astype(float)
    X["is_elderly_50p"] = (df["age"] >= 50).astype(float)
    X["is_elderly_65p"] = (df["age"] >= 65).astype(float)
    
    # Critical regime interaction: pre-2017 age==0 (96.9% positive) vs 2018-2021 age==0 (53.1% positive) vs 2022 age==0 (76.0% positive)
    X["year"] = df["year"].astype(float)
    X["is_pre2017"] = (df["year"] <= 2016).astype(float)
    X["is_2017"] = (df["year"] == 2017).astype(float)
    X["is_2018_2021"] = ((df["year"] >= 2018) & (df["year"] <= 2021)).astype(float)
    X["is_2022"] = (df["year"] >= 2022).astype(float)
    
    X["age0_x_pre2017"] = X["is_age_zero"] * X["is_pre2017"]
    X["age0_x_2018_2021"] = X["is_age_zero"] * X["is_2018_2021"]
    X["age0_x_2022"] = X["is_age_zero"] * X["is_2022"]
    X["age0_x_year_age0_rate"] = X["is_age_zero"] * df["year_age0_rate"]
    X["age0_x_yq_age0_rate"] = X["is_age_zero"] * df["yq_age0_rate"]
    X["age_pos_x_year"] = (1.0 - X["is_age_zero"]) * (df["year"] - 2007.0)
    
    X["year_total_deaths"] = df["year_total_deaths"].astype(float)
    X["year_age0_rate"] = df["year_age0_rate"].astype(float)
    X["yq_total_deaths"] = df["yq_total_deaths"].astype(float)
    X["yq_age0_rate"] = df["yq_age0_rate"].astype(float)
    X["same_day_deaths"] = df["same_day_deaths"].astype(float)
    for win in [7, 14, 30, 60]:
        X[f"cohort_deaths_pm_{win}d"] = df[f"cohort_deaths_pm_{win}d"].astype(float)
    
    X["is_female"] = (df["gender"] == "Female").astype(float)
    X["is_rural"] = (df["zone"] == "Rural").astype(float)
    X["female_x_age0"] = X["is_female"] * X["is_age_zero"]
    X["rural_x_age0"] = X["is_rural"] * X["is_age_zero"]
    X["rural_x_u5"] = X["is_rural"] * X["is_child_u5"]
    
    # 2. Temporal & Seasonal Harmonics (East African bimodal rainy seasons: Mar-May & Sep-Nov)
    X["month"] = df["month"].astype(float)
    X["day_of_year"] = df["day_of_year"].astype(float)
    for k in [1, 2, 3]:
        X[f"doy_sin_{k}"] = np.sin(2.0 * np.pi * k * X["day_of_year"] / 365.25)
        X[f"doy_cos_{k}"] = np.cos(2.0 * np.pi * k * X["day_of_year"] / 365.25)
    
    # 3. Raw Weather & Downloaded ERA5/CHIRPS/MODIS Climate Features
    raw_clim = [
        "avg_temperature", "max_temperature", "min_temperature", "precipitation",
        "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d", "max_daily_rain_30d",
        "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d", "hot_days_30d",
        "temp_range_mean_30d", "ndvi_30d", "ndvi_90d"
    ]
    for c in raw_clim:
        X[c] = df[c].astype(float)
    
    # 4. Physical & Biometeorological Engineered Features
    # Diurnal temperature ranges & thermal shocks
    X["dtr_day"] = X["max_temperature"] - X["min_temperature"]
    X["dtr_30d_extreme"] = X["tmax_30d"] - X["tmin_30d"]
    X["dtr_shock_vs_30d"] = X["dtr_day"] - X["temp_range_mean_30d"]
    X["temp_shock_day_vs_7d"] = X["avg_temperature"] - X["tavg_7d"]
    X["temp_shock_day_vs_30d"] = X["avg_temperature"] - X["tavg_30d"]
    X["temp_shock_7d_vs_30d"] = X["tavg_7d"] - X["tavg_30d"]
    X["temp_shock_7d_vs_90d"] = X["tavg_7d"] - X["tavg_90d"]
    X["temp_shock_30d_vs_90d"] = X["tavg_30d"] - X["tavg_90d"]
    X["max_temp_anomaly"] = X["max_temperature"] - X["tmax_30d"]
    X["min_temp_anomaly"] = X["min_temperature"] - X["tmin_30d"]
    
    # Hydrological / Rainfall intensity, concentration & accumulation lags
    X["is_rainy_day"] = (X["precipitation"] > 0).astype(float)
    X["rain_7_to_30_ratio"] = X["rain_sum_7d"] / (X["rain_sum_30d"] + 1.0)
    X["rain_30_to_90_ratio"] = X["rain_sum_30d"] / (X["rain_sum_90d"] + 1.0)
    X["rain_7_to_90_ratio"] = X["rain_sum_7d"] / (X["rain_sum_90d"] + 1.0)
    X["rain_mean_per_rainy_day_30d"] = X["rain_sum_30d"] / (X["rain_days_30d"] + 0.5)
    X["rain_max_share_30d"] = X["max_daily_rain_30d"] / (X["rain_sum_30d"] + 1.0)
    X["rain_days_frac_30d"] = X["rain_days_30d"] / 30.0
    X["rain_prev_8_to_30d"] = X["rain_sum_30d"] - X["rain_sum_7d"]
    X["rain_prev_31_to_90d"] = X["rain_sum_90d"] - X["rain_sum_30d"]
    X["precip_log1p"] = np.log1p(np.maximum(0.0, X["precipitation"]))
    X["rain_sum_7d_log1p"] = np.log1p(np.maximum(0.0, X["rain_sum_7d"]))
    X["rain_sum_30d_log1p"] = np.log1p(np.maximum(0.0, X["rain_sum_30d"]))
    X["rain_sum_90d_log1p"] = np.log1p(np.maximum(0.0, X["rain_sum_90d"]))
    
    # Vegetation / Vector Ecology (NDVI greening response + hydrothermal malaria index)
    X["ndvi_diff_30_90"] = X["ndvi_30d"] - X["ndvi_90d"]
    X["ndvi_ratio_30_90"] = (X["ndvi_30d"] + 1.0) / (X["ndvi_90d"] + 1.0)
    X["malaria_hydrothermal_30d"] = X["tavg_30d"] * np.log1p(np.maximum(0.0, X["rain_sum_30d"])) * X["ndvi_30d"]
    X["malaria_hydrothermal_90d"] = X["tavg_90d"] * np.log1p(np.maximum(0.0, X["rain_sum_90d"])) * X["ndvi_90d"]
    X["heat_drought_stress_30d"] = X["tmax_30d"] / (np.log1p(np.maximum(0.0, X["rain_sum_30d"])) + 1.0)
    
    # Age x Climate vulnerability interactions
    X["u5_x_rain30"] = X["is_child_u5"] * X["rain_sum_30d_log1p"]
    X["u5_x_ndvi30"] = X["is_child_u5"] * X["ndvi_30d"]
    X["elderly_x_tmax30"] = X["is_elderly_50p"] * X["tmax_30d"]
    X["elderly_x_dtr"] = X["is_elderly_50p"] * X["temp_range_mean_30d"]
    
    # Location-centered climate anomalies
    for c in loc_clim_cols:
        X[f"{c}_loc_anom"] = df[f"{c}_loc_anom"].astype(float)
        
    return X

X_all_spatial_inv = build_features(full_df)

# Add spatial/topographic features for View B
X_all_full = X_all_spatial_inv.copy()
X_all_full["elevation"] = full_df["elevation"].astype(float)
X_all_full["slope"] = full_df["slope"].astype(float)
X_all_full["elev_x_slope"] = X_all_full["elevation"] * X_all_full["slope"]
X_all_full["district_code"] = full_df["district_code"].astype(float)

train_mask = (full_df["_is_train"] == 1).values
X_train_inv = X_all_spatial_inv.loc[train_mask].reset_index(drop=True)
X_test_inv = X_all_spatial_inv.loc[~train_mask].reset_index(drop=True)

X_train_full = X_all_full.loc[train_mask].reset_index(drop=True)
X_test_full = X_all_full.loc[~train_mask].reset_index(drop=True)

y_train = train_df["is_climate_sensitive"].astype(int).values
groups_loc = train_df["location"].values

print(f"Engineered features: Spatial-Invariant View={X_train_inv.shape[1]}, Full View={X_train_full.shape[1]}")

def eval_combo(y_true, prob, thresh=0.5):
    pred = (prob >= thresh).astype(int)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, prob)
    return 0.60 * f1 + 0.40 * auc, f1, auc

def find_best_threshold(y_true, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.20, 0.70, 501):
        f1 = f1_score(y_true, (prob >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# Out-of-fold evaluation across 10-Fold StratifiedKFold AND 10-Fold Spatial GroupKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, n_split_args=None) if False else StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=10)

models_oof = {}
models_test = {}

for cv_name, splitter, X_tr, X_te in [
    ("skf_inv", skf.split(X_train_inv, y_train), X_train_inv, X_test_inv),
    ("gkf_inv", gkf.split(X_train_inv, y_train, groups_loc), X_train_inv, X_test_inv),
    ("skf_full", skf.split(X_train_full, y_train), X_train_full, X_test_full),
    ("gkf_full", gkf.split(X_train_full, y_train, groups_loc), X_train_full, X_test_full),
]:
    splits = list(splitter)
    
    # 1. LightGBM DART / GBDT shallow regularized
    oof_lgb = np.zeros(len(y_train))
    pred_lgb = np.zeros(len(X_te))
    
    # 2. CatBoost
    oof_cat = np.zeros(len(y_train))
    pred_cat = np.zeros(len(X_te))
    
    # 3. XGBoost
    oof_xgb = np.zeros(len(y_train))
    pred_xgb = np.zeros(len(X_te))
    
    # 4. Regularized Logistic + MLP
    oof_lin = np.zeros(len(y_train))
    pred_lin = np.zeros(len(X_te))
    
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_train[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y_train[va_idx]
        
        # LightGBM (multi-seed average per fold)
        for seed in [42, 2026]:
            m_lgb = lgb.LGBMClassifier(
                n_estimators=1200,
                learning_rate=0.02,
                num_leaves=15,
                max_depth=4,
                min_child_samples=25,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=2.0,
                reg_lambda=5.0,
                random_state=seed + fold,
                verbose=-1,
            )
            m_lgb.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                callbacks=[lgb.early_stopping(80, verbose=False)]
            )
            oof_lgb[va_idx] += m_lgb.predict_proba(X_va_f)[:, 1] / 2.0
            pred_lgb += m_lgb.predict_proba(X_te)[:, 1] / (2.0 * len(splits))
            
        # CatBoost
        m_cat = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.025,
            depth=5,
            l2_leaf_reg=6.0,
            random_seed=42 + fold,
            eval_metric="AUC",
            verbose=False
        )
        m_cat.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), early_stopping_rounds=100, verbose=False)
        oof_cat[va_idx] = m_cat.predict_proba(X_va_f)[:, 1]
        pred_cat += m_cat.predict_proba(X_te)[:, 1] / len(splits)
        
        # XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=5.0,
            eval_metric="auc",
            early_stopping_rounds=80,
            random_state=42 + fold,
            n_jobs=4
        )
        m_xgb.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb[va_idx] = m_xgb.predict_proba(X_va_f)[:, 1]
        pred_xgb += m_xgb.predict_proba(X_te)[:, 1] / len(splits)
        
        # Linear + MLP
        pipe_lr = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=1000))
        ])
        pipe_mlp = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=300, early_stopping=True, random_state=42+fold))
        ])
        pipe_lr.fit(X_tr_f, y_tr_f)
        pipe_mlp.fit(X_tr_f, y_tr_f)
        oof_lin[va_idx] = 0.5 * pipe_lr.predict_proba(X_va_f)[:, 1] + 0.5 * pipe_mlp.predict_proba(X_va_f)[:, 1]
        pred_lin += (0.5 * pipe_lr.predict_proba(X_te)[:, 1] + 0.5 * pipe_mlp.predict_proba(X_te)[:, 1]) / len(splits)

    for mname, oof_arr, pred_arr in [
        (f"{cv_name}_lgb", oof_lgb, pred_lgb),
        (f"{cv_name}_cat", oof_cat, pred_cat),
        (f"{cv_name}_xgb", oof_xgb, pred_xgb),
        (f"{cv_name}_lin", oof_lin, pred_lin),
    ]:
        models_oof[mname] = oof_arr
        models_test[mname] = pred_arr
        sc_raw, f1_raw, auc_raw = eval_combo(y_train, oof_arr, 0.5)
        bt, bf1 = find_best_threshold(y_train, oof_arr)
        sc_opt = 0.60 * bf1 + 0.40 * auc_raw
        print(f"  {mname:15s} | Raw@0.5: Score={sc_raw:.6f} (F1={f1_raw:.5f}, AUC={auc_raw:.5f}) | Calibrated@{bt:.3f}: Score={sc_opt:.6f} (F1={bf1:.5f})")

# Optimize ensemble weights across all OOF models
from scipy.optimize import minimize

names = list(models_oof.keys())
OOF_MAT = np.column_stack([models_oof[k] for k in names])
TEST_MAT = np.column_stack([models_test[k] for k in names])

# Find optimal blend weights for AUC + F1
best_ens_score = -1.0
best_w = None
best_thresh = 0.5

rng = np.random.default_rng(42)
for trial in range(5000):
    w = rng.dirichlet(np.ones(len(names)))
    blend_oof = OOF_MAT @ w
    auc = roc_auc_score(y_train, blend_oof)
    # Fast vectorized threshold search
    for t in [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
        f1 = f1_score(y_train, (blend_oof >= t).astype(int))
        sc = 0.60 * f1 + 0.40 * auc
        if sc > best_ens_score:
            best_ens_score = sc
            best_w = w
            best_thresh = t

blend_oof = OOF_MAT @ best_w
best_thresh, best_f1 = find_best_threshold(y_train, blend_oof)
best_auc = roc_auc_score(y_train, blend_oof)
best_ens_score = 0.60 * best_f1 + 0.40 * best_auc

print("\n=== OPTIMAL MULTI-VIEW ENSEMBLE ===")
for k, weight in zip(names, best_w):
    if weight > 0.02:
        print(f"  Weight {k:15s}: {weight:.4f}")
print(f"OOF ROC-AUC: {best_auc:.6f}")
print(f"OOF Raw@0.50 F1: {f1_score(y_train, (blend_oof >= 0.5).astype(int)):.6f} -> Raw Score: {eval_combo(y_train, blend_oof, 0.5)[0]:.6f}")
print(f"OOF Optimal Threshold t* = {best_thresh:.4f} -> Optimal F1: {best_f1:.6f} -> Final Score: {best_ens_score:.6f}")

# Monotonic Log-Odds Shift Calibration so that p == t* maps EXACTLY to 0.500000!
# Preserves ROC-AUC 100.0000% identically while making (TargetRAUC >= 0.5) achieve the optimal F1!
def calibrate_to_05(prob, t_star):
    eps = 1e-6
    p = np.clip(prob, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p))
    logit_t = np.log(t_star / (1.0 - t_star))
    return 1.0 / (1.0 + np.exp(-(logit_p - logit_t)))

blend_test_raw = TEST_MAT @ best_w
blend_oof_cal = calibrate_to_05(blend_oof, best_thresh)
blend_test_cal = calibrate_to_05(blend_test_raw, best_thresh)

print("Verify Calibrated OOF @ 0.50:", eval_combo(y_train, blend_oof_cal, 0.5))

# Also create a pure Spatial-Invariant Ensemble (View A only) and a Pure Raw-Threshold 0.5 Ensemble
inv_indices = [i for i, k in enumerate(names) if "_inv_" in k]
w_inv = np.array([best_w[i] if i in inv_indices else 0.0 for i in range(len(names))])
w_inv = w_inv / w_inv.sum()
oof_inv = OOF_MAT @ w_inv
test_inv_raw = TEST_MAT @ w_inv
t_inv, f1_inv = find_best_threshold(y_train, oof_inv)
test_inv_cal = calibrate_to_05(test_inv_raw, t_inv)
print(f"Spatial-Invariant View OOF: Score={0.60*f1_inv + 0.40*roc_auc_score(y_train, oof_inv):.6f} (F1={f1_inv:.5f}, AUC={roc_auc_score(y_train, oof_inv):.5f} @ t*={t_inv:.4f})")

# Save 3 submission candidates:
# 1. submission.csv (Primary: Calibrated Multi-View GBDT+Neural Ensemble where TargetF1 == (TargetRAUC >= 0.5) exactly!)
sub1 = pd.DataFrame({
    "ID": test_raw["ID"],
    "TargetF1": (blend_test_cal >= 0.5).astype(int),
    "TargetRAUC": np.round(blend_test_cal, 6)
})
sub1.to_csv(os.path.join(BASE_DIR, "submission.csv"), index=False)

# 2. submission_spatial_invariant.csv (Spatial-Invariant View A Calibrated Ensemble)
sub2 = pd.DataFrame({
    "ID": test_raw["ID"],
    "TargetF1": (test_inv_cal >= 0.5).astype(int),
    "TargetRAUC": np.round(test_inv_cal, 6)
})
sub2.to_csv(os.path.join(BASE_DIR, "submission_spatial_invariant.csv"), index=False)

# 3. submission_unshifted_05.csv (Unshifted probabilities at raw 0.5 threshold)
sub3 = pd.DataFrame({
    "ID": test_raw["ID"],
    "TargetF1": (blend_test_raw >= 0.5).astype(int),
    "TargetRAUC": np.round(blend_test_raw, 6)
})
sub3.to_csv(os.path.join(BASE_DIR, "submission_unshifted_05.csv"), index=False)

print("\nSaved submissions:")
for fn, s in [("submission.csv", sub1), ("submission_spatial_invariant.csv", sub2), ("submission_unshifted_05.csv", sub3)]:
    p = os.path.join(BASE_DIR, fn)
    print(f"  {fn}: {len(s)} rows, {os.path.getsize(p)} bytes, pos_rate={s['TargetF1'].mean():.4f}, mean_prob={s['TargetRAUC'].mean():.4f}")

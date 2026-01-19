"""
==============================================================
Rank-Preserving RLHF-Lite Calibration for Q-2 (Damped, 5-Anchor Stratified)
==============================================================
Stage 1:  4 model skorundan ensemble ridge -> reward scalar
Stage 2:  Rank-preserving isotonic mapping -> human-like scaling
Stage 3:  Damped + Hard Zero heuristic correction rules
==============================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import pickle

def rank_preserving_rlhf_q2_damped():

    excel_path = "total_scores_q2.xlsx"
    sheet = "Q-2"

    df = pd.read_excel(excel_path, sheet_name=sheet)

    # Model & target columns
    model_cols = ["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]
    target = "Expert Score"

    # ===== Stage 1: Ensemble Score =====
    X = df[model_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    ensemble = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 21)))
    ])
    ensemble.fit(X, y)
    ensemble_score = ensemble.predict(X)

    df["_ensemble_score"] = ensemble_score

    # ===== Select 5 Stratified Anchors (Median-per-bin) =====
    bins = [(0,4),(5,8),(9,12),(13,16),(17,20)]
    anchor_rows = []

    for low, high in bins:
        subset = df[(df[target] >= low) & (df[target] <= high)]
        if len(subset) == 0:
            continue
        med = subset[target].median()
        chosen = subset.iloc[(subset[target] - med).abs().argsort().iloc[0]]
        anchor_rows.append(chosen)

    anchor_df = pd.DataFrame(anchor_rows).reset_index(drop=True)

    # ===== Stage 1 Reward Model (fit only on anchors) =====
    R = ensemble.predict(anchor_df[model_cols])
    H = anchor_df[target].to_numpy(dtype=float)

    reward_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 21)))
    ])
    reward_model.fit(R.reshape(-1,1), H)

    reward_scores = reward_model.predict(ensemble_score.reshape(-1,1))

    # ===== Stage 2: Rank-Preserving Isotonic =====
    iso = IsotonicRegression(out_of_bounds="clip")
    iso_fitted = iso.fit_transform(reward_scores, y)  # map to human-like scale

    # Stretch to [0,20]
    lo, hi = float(np.min(iso_fitted)), float(np.max(iso_fitted))
    stretched = (iso_fitted - lo) / (hi - lo) * 20
    df["calibrated_q2"] = stretched

    # ===== Stage 3: Damped Rule (Low Scores Gain ≤ +5) =====
    low_mask = df[target] <= 5
    df.loc[low_mask, "calibrated_q2"] = np.minimum(df.loc[low_mask, "calibrated_q2"], df.loc[low_mask, target] + 5)

    # Hard-zero: if Expert Score == 0 -> calibrated = 0
    df.loc[df[target] == 0, "calibrated_q2"] = 0

    # Clip final just in case
    df["calibrated_q2"] = df["calibrated_q2"].clip(0,20)

    # ===== Save Results =====
    out_name = "Q-2-calibrated-damped_V3.xlsx"
    with pd.ExcelWriter(out_name) as writer:
        df.to_excel(writer, sheet_name="Calibrated Results", index=False)
        anchor_df.to_excel(writer, sheet_name="Human Reference (5 Anchors)", index=False)

    pickle.dump(ensemble, open("ensemble_model_ridge_v3.pkl", "wb"))
    pickle.dump(reward_model, open("reward_model_ridge_Q2_damped_v3.pkl", "wb"))
    pickle.dump(iso, open("rank_mapping_Q2_damped_v3.pkl", "wb"))

    print("✅ Dampened RLHF pipeline tamamlandı →", out_name)
    print("🏁 5 Anchor Samples Kullanıldı:")
    print(anchor_df[[target, "Student ID" if "Student ID" in anchor_df.columns else model_cols[0]]])

if __name__ == "__main__":
    rank_preserving_rlhf_q2_damped()

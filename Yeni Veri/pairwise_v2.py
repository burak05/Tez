# ===========================================================
# pairwise_calibration_v2.py
# Author: Burak Keskin & ChatGPT
# Description: Pairwise Logistic Calibration (Bradley–Terry)
#              + Damped Heuristic Rules for Q-2 dataset
# ===========================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import random

# ===========================================================
# 1. Load data
# ===========================================================
file_path = "total_sonuc_keywords.xlsx"
df = pd.read_excel(file_path, sheet_name="Q-2-results")
print(f"Loaded Q-2 data with {len(df)} samples.")

# ===========================================================
# 2. Select columns
# ===========================================================
teacher_col = "Score"
model_cols = ["Bert Score", "Roberta Score", "DistilBert Score", "T5 Score"]

df = df.dropna(subset=[teacher_col])
X = df[model_cols].fillna(df[model_cols].mean())
y = df[teacher_col]

# ===========================================================
# 3. Reference sample selection (one per bin)
# ===========================================================
bins = [(0,5), (6,9), (10,13), (14,17), (18,20)]
ref_indices = []

for low, high in bins:
    subset = df[(df[teacher_col] >= low) & (df[teacher_col] <= high)]
    if not subset.empty:
        ref_indices.append(subset.sample(1, random_state=42).index[0])

print("Selected reference indices:", ref_indices)
ref_df = df.loc[ref_indices]

# ===========================================================
# 4. Create pairwise comparison dataset
# ===========================================================
pairs = []
for i in range(len(ref_df)):
    for j in range(len(ref_df)):
        if i == j:
            continue
        ref_i = ref_df.iloc[i]
        ref_j = ref_df.iloc[j]
        if ref_i[teacher_col] == ref_j[teacher_col]:
            continue

        x_diff = (ref_i[model_cols] - ref_j[model_cols]).values
        label = 1 if ref_i[teacher_col] > ref_j[teacher_col] else 0
        pairs.append({"x_diff": x_diff, "label": label})

X_train = np.vstack([p["x_diff"] for p in pairs])
y_train = np.array([p["label"] for p in pairs])

# ===========================================================
# 5. Train pairwise logistic regression (Bradley–Terry)
# ===========================================================
logreg = LogisticRegression(max_iter=1000, solver='lbfgs')
logreg.fit(X_train, y_train)

w = logreg.coef_.flatten()
print("Learned weights:", w)

# ===========================================================
# 6. Compute calibrated scores (0–20 scaling)
# ===========================================================
raw_scores = np.dot(X.values, w)
calibrated_scores = 20 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
df["Pairwise-Calibrated Score"] = calibrated_scores

# ===========================================================
# 7. Apply Damped Heuristic Rules
# ===========================================================
def apply_damped_rules(row):
    score = row["Pairwise-Calibrated Score"]
    teacher = row["Score"]

    # Rule 1: exact zeros stay zero
    if teacher == 0 or score < 1:
        return 0

    # Rule 2: low-score answers (0–5) may increase up to +5 max
    if teacher <= 5:
        return min(score, teacher + 5)

    # Rule 3: high-score answers (18–20) cannot drop below 15
    if teacher >= 18:
        return max(score, 15)

    # Default: unchanged
    return score

df["Pairwise-Calibrated (Damped)"] = df.apply(apply_damped_rules, axis=1)

# ===========================================================
# 8. Evaluation
# ===========================================================
def evaluate_model(pred_col):
    r, _ = pearsonr(df[teacher_col], df[pred_col])
    mae = mean_absolute_error(df[teacher_col], df[pred_col])
    mse = mean_squared_error(df[teacher_col], df[pred_col])
    return r, mae, mse

results = []
for col in ["Pairwise-Calibrated Score", "Pairwise-Calibrated (Damped)"]:
    r, mae, mse = evaluate_model(col)
    results.append((col, r, mae, mse))

print("\n=== Pairwise Logistic Calibration (Q-2) ===")
for name, r, mae, mse in results:
    print(f"{name:30s} | r={r:.3f} | MAE={mae:.3f} | MSE={mse:.3f}")

# ===========================================================
# 9. Save results
# ===========================================================
output_path = "pairwise_calibrated_q2_damped_v2.xlsx"
df.to_excel(output_path, index=False)
print(f"\nSaved calibrated results to: {output_path}")

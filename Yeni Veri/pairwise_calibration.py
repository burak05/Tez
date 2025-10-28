# pairwise_logistic_calibration_q2.py
# Author: Burak Keskin & ChatGPT
# Purpose: Pairwise logistic calibration for Q-2 using teacher ("Score") and baseline model scores.

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


# Data is already loaded from Q-2 sheet, no need to filter again
print(f"Loaded Q-2 data with {len(df)} samples.")


# ===========================================================
# 2. Select teacher and baseline model scores
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


# Convert to matrix form
X_train = np.vstack([p["x_diff"] for p in pairs])
y_train = np.array([p["label"] for p in pairs])

# ===========================================================
# 5. Train pairwise logistic regression (Bradley–Terry)
# ===========================================================
logreg = LogisticRegression(max_iter=1000, solver='lbfgs')
logreg.fit(X_train, y_train)

# Reward model weights
w = logreg.coef_.flatten()
print("Learned weights:", w)

# ===========================================================
# 6. Compute calibrated scores
# ===========================================================
raw_scores = np.dot(X.values, w)
# Normalize to 0–20
calibrated_scores = 20 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
df["Pairwise-Calibrated Score"] = calibrated_scores

# ===========================================================
# 7. Evaluation
# ===========================================================
r, _ = pearsonr(df[teacher_col], df["Pairwise-Calibrated Score"])
mae = mean_absolute_error(df[teacher_col], df["Pairwise-Calibrated Score"])
mse = mean_squared_error(df[teacher_col], df["Pairwise-Calibrated Score"])

print("\n=== Pairwise Logistic Calibration Results (Q-2) ===")
print(f"Pearson r: {r:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")

# ===========================================================
# 8. Save output
# ===========================================================
output_path = "pairwise_calibrated_q2.xlsx"
df.to_excel(output_path, index=False)
print(f"\nSaved calibrated results to: {output_path}")

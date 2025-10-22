"""
==============================================================
Rank-Preserving RLHF-Lite Calibration for Q-2 (Strict Zero)
==============================================================
Bu sürüm, sıralamayı korurken hocanın "0" verdiği cevapları
veya boş/kopya içeren cevapları doğrudan 0 puan olarak sabitler.
==============================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression


def rank_preserving_rlhf_q2_strict(excel_path="total_sonuc_keywords.xlsx"):
    q_sheet = "Q-2"
    r_sheet = "Q-2-results"
    print(f"\n🔹 Rank-Preserving RLHF-Lite Calibration for {q_sheet} (Strict Zero Mode)")

    # Sayfaları oku
    q_df = pd.read_excel(excel_path, sheet_name=q_sheet)
    r_df = pd.read_excel(excel_path, sheet_name=r_sheet)
    q_df = q_df.dropna(subset=["Answer"]).copy()

    # ----------------------------------------------------------
    # 🔸 İnsan örnekleri (5 adet)
    # ----------------------------------------------------------
    bins = [0, 5, 9, 13, 17, 21]
    labels = ["very_low", "low", "medium", "high", "very_high"]
    q_df["level"] = pd.cut(q_df["Score"], bins=bins, labels=labels, include_lowest=True)

    rep_list = []
    for label in labels:
        group = q_df[q_df["level"] == label]
        if not group.empty:
            rep_list.append(group.sample(1, random_state=42))
    rep = pd.concat(rep_list).reset_index(drop=True)

    print("🧠 Selected human reference samples:")
    print(rep[["StudentID", "Score", "Answer"]])

    # ----------------------------------------------------------
    # 🔸 Reward (Ridge) Modeli
    # ----------------------------------------------------------
    merged = r_df.merge(rep[["StudentID", "Score"]], on="StudentID", how="inner", suffixes=("", "_Human"))
    merged.rename(columns={"Score_Human": "Human Score"}, inplace=True)

    X = merged[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    y = merged["Score"].values

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ])
    ridge_pipe.fit(X, y)

    # ----------------------------------------------------------
    # 🔸 Ridge + Isotonic Regression (Rank-Preserving)
    # ----------------------------------------------------------
    X_all = r_df[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    ridge_pred = ridge_pipe.predict(X_all)

    human_sorted = np.sort(rep["Score"].values)
    ridge_sorted = np.sort(ridge_pred[: len(human_sorted)])

    iso = IsotonicRegression(y_min=0, y_max=20, increasing=True)
    iso.fit(ridge_sorted, human_sorted)

    calibrated = iso.predict(ridge_pred)
    r_df["RLHF-Calibrated Score"] = np.clip(calibrated, 0, 20)

    # ----------------------------------------------------------
    # 🔸 Strict-zero kuralı (boş, kopya, 0 alan)
    # ----------------------------------------------------------
    def is_copy_or_empty(text):
        if not isinstance(text, str):
            return True
        t = text.strip().lower()
        return t == "" or "copy" in t or "copied" in t

    zero_mask = (
        (r_df["Score"] == 0)
        | (r_df["Answer"].isna())
        | (r_df["Answer"].apply(is_copy_or_empty))
    )
    r_df.loc[zero_mask, "RLHF-Calibrated Score"] = 0

    # ----------------------------------------------------------
    # 🔸 Sonuçları kaydet
    # ----------------------------------------------------------
    human_doc = rep[["StudentID", "Score", "Answer"]].rename(columns={"Score": "Human Score"})

    with pd.ExcelWriter("Q-2-calibrated-strict.xlsx") as writer:
        r_df.to_excel(writer, sheet_name="Calibrated Results", index=False)
        human_doc.to_excel(writer, sheet_name="Human Reference (5 samples)", index=False)

    pickle.dump(ridge_pipe, open("reward_model_ridge_Q2_strict.pkl", "wb"))
    pickle.dump(iso, open("rank_mapping_Q2_strict.pkl", "wb"))

    print("✅ Q-2 strict rank-preserving calibration tamamlandı → 'Q-2-calibrated-strict.xlsx' oluşturuldu.")
    print("⚖️  Boş veya kopya cevaplar sabit 0 aldı; sıralama korundu.")


if __name__ == "__main__":
    rank_preserving_rlhf_q2_strict()

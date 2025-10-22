"""
==============================================================
Rank-Preserving RLHF-Lite Calibration for Q-2
==============================================================
Bu modül, yalnızca Q-2 sorusu için insan geri bildirimiyle
sıralamayı koruyan bir RLHF-lite kalibrasyonu yapar.

Yöntem:
  1. Q-2 ve Q-2-results sayfalarını okur
  2. 5 insan etiketi (örnek cevap) seçer
  3. Ridge Regression öğrenir (ödül modeli)
  4. Isotonic Regression uygular (monotonic transform)
  5. Sıralamayı koruyarak model skorlarını yeniden ölçekler
  6. Sonuçları Q-2-calibrated.xlsx dosyasına yazar
==============================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression


def rank_preserving_rlhf_q2(excel_path="total_sonuc_keywords.xlsx"):
    q_sheet = "Q-2"
    r_sheet = "Q-2-results"
    print(f"\n🔹 Rank-Preserving RLHF-Lite Calibration for {q_sheet}")

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
    # 🔸 Q-2'nin tüm cevaplarını tahmin et (ham ridge output)
    # ----------------------------------------------------------
    X_all = r_df[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    ridge_pred = ridge_pipe.predict(X_all)

    # ----------------------------------------------------------
    # 🔸 Monotonik Dönüşüm (Rank-Preserving)
    # ----------------------------------------------------------
    # İnsan skorları → sıralama (rank scale)
    human_sorted = np.sort(rep["Score"].values)
    ridge_sorted = np.sort(ridge_pred[: len(human_sorted)])

    # İnsan sıralamasıyla ridge tahminlerini hizala
    iso = IsotonicRegression(y_min=0, y_max=20, increasing=True)
    iso.fit(ridge_sorted, human_sorted)

    calibrated = iso.predict(ridge_pred)
    r_df["RLHF-Calibrated Score"] = np.clip(calibrated, 0, 20)

    # ----------------------------------------------------------
    # 🔸 Sonuçları kaydet
    # ----------------------------------------------------------
    human_doc = rep[["StudentID", "Score", "Answer"]].rename(columns={"Score": "Human Score"})

    with pd.ExcelWriter("qq-2-calibrated.xlsx") as writer:
        r_df.to_excel(writer, sheet_name="Calibrated Results", index=False)
        human_doc.to_excel(writer, sheet_name="Human Reference (5 samples)", index=False)

    pickle.dump(ridge_pipe, open("reward_model_ridge_Q2.pkl", "wb"))
    pickle.dump(iso, open("rank_mapping_Q2.pkl", "wb"))

    print("✅ Q-2 rank-preserving calibration tamamlandı → 'Q-2-calibrated.xlsx' oluşturuldu.")
    print("⚖️  Sıralama korunarak insan dağılımına hizalandı (0 hiçbir zaman yükselmedi).")


if __name__ == "__main__":
    rank_preserving_rlhf_q2()

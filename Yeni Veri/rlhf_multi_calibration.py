"""
==============================================================
RLHF-Lite Multi-Question Calibration Module
==============================================================
Bu modül, büyük dil modeli tabanlı değerlendirme çıktıları için
insan geri bildirimi (5 örnek cevap) ile hafif bir RLHF (fine-tuning)
benzetimi uygular.

Her Q-n sorusu için:
  1. "Q-n" ve "Q-n-results" sayfalarını Excel'den okur
  2. 5 insan etiketi seçer (her puan aralığından)
  3. Ridge Regression (reward model) ve Logistic Regression (preference model) eğitir
  4. Tüm cevapları kalibre eder (RLHF-Calibrated Score)
  5. Sonuçları "Q-n-calibrated.xlsx" dosyasına yazar:
     - "Calibrated Results" sayfası → tüm öğrenci cevapları + yeni puan
     - "Human Reference (5 samples)" sayfası → reinforce edilen 5 örnek
  6. Modelleri kaydeder: reward_model_ridge_Qn.pkl, reward_model_logit_Qn.pkl
==============================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.pipeline import Pipeline
from itertools import combinations


def build_rlhf_models(question_id: str, excel_path="total_sonuc_keywords.xlsx"):
    """
    Tek bir soru (örneğin Q-2) için RLHF-lite kalibrasyonu yapar.
    """
    q_sheet = f"Q-{question_id}"
    r_sheet = f"Q-{question_id}-results"
    print(f"\n🔹 Processing {q_sheet} / {r_sheet}")

    try:
        q_df = pd.read_excel(excel_path, sheet_name=q_sheet)
        r_df = pd.read_excel(excel_path, sheet_name=r_sheet)
    except Exception as e:
        print(f"⚠️ {q_sheet} veya {r_sheet} bulunamadı: {e}")
        return

    q_df = q_df.dropna(subset=["Answer"]).copy()

    # ---------------------------
    # 🔸 İnsan örneklerini seçme
    # ---------------------------
    bins = [0, 5, 9, 13, 17, 21]
    labels = ["very_low", "low", "medium", "high", "very_high"]
    q_df["level"] = pd.cut(q_df["Score"], bins=bins, labels=labels, include_lowest=True)

    rep_list = []
    for label in labels:
        group = q_df[q_df["level"] == label]
        if not group.empty:
            rep_list.append(group.sample(1, random_state=42))
    rep = pd.concat(rep_list).reset_index(drop=True)

    print("🧠 Selected human preference samples:")
    print(rep[["StudentID", "Score", "Answer"]])

    # ---------------------------
    # 🔸 Reward & Preference Modelleri
    # ---------------------------
    merged = r_df.merge(rep[["StudentID", "Score"]], on="StudentID", how="inner", suffixes=("", "_Human"))
    merged.rename(columns={"Score_Human": "Human Score"}, inplace=True)

    X = merged[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    y = merged["Score"].values

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ])
    ridge_pipe.fit(X, y)

    pairs = list(combinations(range(len(merged)), 2))
    X_pairs, y_pairs = [], []
    for i, j in pairs:
        diff = X[i] - X[j]
        X_pairs.append(diff)
        y_pairs.append(1 if y[i] > y[j] else 0)
    X_pairs, y_pairs = np.array(X_pairs), np.array(y_pairs)
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    logit.fit(X_pairs, y_pairs)

    # ---------------------------
    # 🔸 RLHF-lite Kalibrasyonu
    # ---------------------------
    all_X = r_df[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    ridge_pred = ridge_pipe.predict(all_X)
    ridge_scaled = (ridge_pred - ridge_pred.min()) / (ridge_pred.max() - ridge_pred.min())

    w = logit.named_steps["clf"].coef_.ravel()
    scale = logit.named_steps["scaler"].scale_
    w_orig = w / scale
    correction = np.dot(all_X, w_orig)
    correction_scaled = (correction - correction.min()) / (correction.max() - correction.min())

    calibrated = 0.7 * ridge_scaled + 0.3 * correction_scaled
    r_df["RLHF-Calibrated Score"] = calibrated * 20

    # ---------------------------
    # 🔸 Sonuçların kaydı
    # ---------------------------
    human_doc = rep[["StudentID", "Score", "Answer"]].rename(columns={"Score": "Human Score"})

    with pd.ExcelWriter(f"Q-{question_id}-calibrated.xlsx") as writer:
        r_df.to_excel(writer, sheet_name="Calibrated Results", index=False)
        human_doc.to_excel(writer, sheet_name="Human Reference (5 samples)", index=False)

    pickle.dump(ridge_pipe, open(f"reward_model_ridge_Q{question_id}.pkl", "wb"))
    pickle.dump(logit, open(f"reward_model_logit_Q{question_id}.pkl", "wb"))

    print(f"✅ Q-{question_id} tamamlandı → 'Q-{question_id}-calibrated.xlsx' oluşturuldu.")


def run_all_questions(question_list=None, excel_path="total_sonuc_keywords.xlsx"):
    """
    Birden fazla soruyu sırayla işler (varsayılan: Q-1 → Q-5).
    """
    if question_list is None:
        question_list = [1, 2, 3, 4, 5]

    for q in question_list:
        try:
            build_rlhf_models(q, excel_path)
        except Exception as e:
            print(f"⚠️ Skipped Q-{q} ({e})")

    print("\n🏁 Tüm RLHF-lite kalibrasyon işlemleri tamamlandı.")

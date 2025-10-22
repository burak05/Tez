"""
====================================================================
Rank-Preserving RLHF-Lite Calibration (All Questions + Exam Summary)
====================================================================

Özellikler:
 - Her soru için (Q-1 ... Q-5):
    • Sıralamayı korur (Isotonic Regression)
    • Boş, "copy" veya hocadan 0 alanlara 0 verir
    • ≤5 alanlara en fazla +5 artış uygular
 - Sonuçta her Q-n için `Q-n-calibrated-damped.xlsx` oluşturur
 - Tüm öğrencilerin toplam sınav puanlarını hesaplar:
    • Hoca toplamı vs Model toplamı
    • Ortalama, sapma ve korelasyon analizi
 - Final çıktısı: `Exam_Total_Comparison.xlsx`
====================================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression


def damp_low_scores(original, calibrated):
    """≤5 notlara en fazla +5 puan artış sınırı uygular."""
    if original <= 5:
        return min(calibrated, original + 5)
    return calibrated


def is_copy_or_empty(text):
    """Boş veya 'copy' içerikli cevapları tespit eder."""
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    return t == "" or "copy" in t or "copied" in t


def calibrate_question(q_num, excel_path="total_sonuc_keywords.xlsx"):
    """Tek bir soruyu işler ve kalibre edilmiş DataFrame döner."""
    q_sheet = f"Q-{q_num}"
    r_sheet = f"Q-{q_num}-results"

    try:
        q_df = pd.read_excel(excel_path, sheet_name=q_sheet)
        r_df = pd.read_excel(excel_path, sheet_name=r_sheet)
    except Exception as e:
        print(f"⚠️ {q_sheet} veya {r_sheet} bulunamadı: {e}")
        return None

    q_df = q_df.dropna(subset=["Answer"]).copy()

    # --- İnsan örnekleri (5 temsilî cevap) ---
    bins = [0, 5, 9, 13, 17, 21]
    labels = ["very_low", "low", "medium", "high", "very_high"]
    q_df["level"] = pd.cut(q_df["Score"], bins=bins, labels=labels, include_lowest=True)

    rep_list = []
    for label in labels:
        group = q_df[q_df["level"] == label]
        if not group.empty:
            rep_list.append(group.sample(1, random_state=42))
    rep = pd.concat(rep_list).reset_index(drop=True)

    # --- Ridge Model ---
    merged = r_df.merge(rep[["StudentID", "Score"]], on="StudentID", how="inner", suffixes=("", "_Human"))
    merged.rename(columns={"Score_Human": "Human Score"}, inplace=True)

    X = merged[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    y = merged["Score"].values

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ])
    ridge_pipe.fit(X, y)

    # --- Isotonic Regression (Rank-Preserving) ---
    X_all = r_df[["Roberta Score", "Bert Score", "DistilBert Score", "T5 Score"]].values
    ridge_pred = ridge_pipe.predict(X_all)

    human_sorted = np.sort(rep["Score"].values)
    ridge_sorted = np.sort(ridge_pred[: len(human_sorted)])

    iso = IsotonicRegression(y_min=0, y_max=20, increasing=True)
    iso.fit(ridge_sorted, human_sorted)

    calibrated = iso.predict(ridge_pred)
    r_df["RLHF-Calibrated Score"] = np.clip(calibrated, 0, 20)

    # --- Strict Zero Kuralı ---
    zero_mask = (
        (r_df["Score"] == 0)
        | (r_df["Answer"].isna())
        | (r_df["Answer"].apply(is_copy_or_empty))
    )
    r_df.loc[zero_mask, "RLHF-Calibrated Score"] = 0

    # --- Damping Katmanı ---
    r_df["RLHF-Calibrated Score"] = r_df.apply(
        lambda row: damp_low_scores(row["Score"], row["RLHF-Calibrated Score"]),
        axis=1
    )

    # --- Kaydet ---
    human_doc = rep[["StudentID", "Score", "Answer"]].rename(columns={"Score": "Human Score"})
    out_file = f"Q-{q_num}-calibrated-damped.xlsx"

    with pd.ExcelWriter(out_file) as writer:
        r_df.to_excel(writer, sheet_name="Calibrated Results", index=False)
        human_doc.to_excel(writer, sheet_name="Human Reference (5 samples)", index=False)

    pickle.dump(ridge_pipe, open(f"reward_model_ridge_Q{q_num}_damped.pkl", "wb"))
    pickle.dump(iso, open(f"rank_mapping_Q{q_num}_damped.pkl", "wb"))

    print(f"✅ Q-{q_num} tamamlandı → {out_file}")

    # Toplam puan analizine eklenmek üzere gerekli sütunları döndür
    return r_df[["StudentID", "Score", "RLHF-Calibrated Score"]].rename(
        columns={"Score": f"Q{q_num}_Human", "RLHF-Calibrated Score": f"Q{q_num}_Model"}
    )


def run_all_questions(excel_path="total_sonuc_keywords.xlsx", question_range=[1, 2, 3, 4, 5]):
    """Tüm soruları işler ve toplam sınav analizini yapar."""
    print("\n🔹 Rank-Preserving RLHF-Lite (All Questions) başlatılıyor...\n")
    all_results = []

    for q in question_range:
        df = calibrate_question(q, excel_path)
        if df is not None:
            all_results.append(df)

    if not all_results:
        print("❌ Hiçbir soru işlenemedi.")
        return

    # --- Tüm soruları birleştir ---
    merged_total = all_results[0]
    for df in all_results[1:]:
        merged_total = pd.merge(merged_total, df, on="StudentID", how="outer")

    # --- Toplam puanları hesapla ---
    merged_total["Human_Total"] = merged_total[[c for c in merged_total.columns if "_Human" in c]].sum(axis=1)
    merged_total["Model_Total"] = merged_total[[c for c in merged_total.columns if "_Model" in c]].sum(axis=1)

    # --- İstatistiksel karşılaştırma ---
    corr = merged_total["Human_Total"].corr(merged_total["Model_Total"])
    avg_diff = (merged_total["Model_Total"] - merged_total["Human_Total"]).mean()

    print("\n📊 Sınav Geneli Analiz:")
    print(f"• Korelasyon (Human vs Model): {corr:.3f}")
    print(f"• Ortalama fark (Model - Human): {avg_diff:.2f} puan")

    # --- Sonuçları kaydet ---
    with pd.ExcelWriter("Exam_Total_Comparison.xlsx") as writer:
        merged_total.to_excel(writer, sheet_name="Total Comparison", index=False)

    print("✅ 'Exam_Total_Comparison.xlsx' oluşturuldu (her öğrencinin toplamı).")


if __name__ == "__main__":
    run_all_questions()

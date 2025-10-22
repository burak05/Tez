from rlhf_calibration_block import rlhf_calibrated_score

df = pd.read_excel("total_sonuc_keywords.xlsx", sheet_name="Q-2-results")
calibrated_df = rlhf_calibrated_score(df)
calibrated_df.to_excel("Q-2-calibrated.xlsx", index=False)

import os
import pandas as pd
from glob import glob
import argparse

def extract_and_merge(folder, prefix):
    merged_df = None
    for filepath in glob(os.path.join(folder, '*.csv')):
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        df.columns = df.columns.str.strip()

        matching_cols = [col for col in df.columns if col.lower().startswith(prefix.lower())]
        if "participant_id" in df.columns and matching_cols:

            sub_df = df[["participant_id"] + matching_cols].copy()
            filename_tag = os.path.basename(filepath).replace(".csv", "")

            # Rename columns to include filename tag for uniqueness
            sub_df = sub_df.rename(columns={col: f"{col}_{filename_tag}" for col in matching_cols})

            if merged_df is None:
                merged_df = sub_df
            else:
                merged_df = pd.merge(merged_df, sub_df, on="participant_id", how="outer")
    return merged_df

def summarize_predictions(file_path, trait_name):
    df = pd.read_csv(file_path, dtype=str)
    df.columns = df.columns.str.strip()

    if "participant_id" not in df.columns:
        print(f"⚠️ No participant_id column in {file_path}, skipping.")
        return

    pred_cols = [col for col in df.columns if col != "participant_id"]
    # Count how many values are exactly "1" (string or int)
    count_ones = df[pred_cols].apply(lambda row: sum(str(val).strip() == "1" for val in row), axis=1)

    summary_df = pd.DataFrame({
        "participant_id": df["participant_id"],
        f"{trait_name}_count_ones": count_ones
    })

    output_path = f"{trait_name}_summarized.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")

def create_final_prediction(adhd_summary_path, sex_summary_path, n, m):
    adhd_df = pd.read_csv(adhd_summary_path)
    sex_df = pd.read_csv(sex_summary_path)

    # Merge on participant_id
    merged = pd.merge(adhd_df, sex_df, on="participant_id", how="inner")

    # Apply thresholds
    merged["ADHD_Outcome"] = (merged["adhd_count_ones"] > n).astype(int)
    merged["Sex_F"] = (merged["sex_count_ones"] > m).astype(int)

    # Select final columns
    final_df = merged[["participant_id", "ADHD_Outcome", "Sex_F"]]
    final_df.to_csv("final_prediction.csv", index=False)
    print(f"Saved final prediction to final_prediction.csv")

def main(ml_folder, gnn_folder):
    adhd_df_ml = extract_and_merge(ml_folder, "ADHD_Outcome")
    adhd_df_gnn = extract_and_merge(gnn_folder, "ADHD_Outcome")
    sex_df_ml = extract_and_merge(ml_folder, "Sex_F")
    sex_df_gnn = extract_and_merge(gnn_folder, "Sex_F")

    # Merge ML and GNN parts for each prediction type
    adhd_final = pd.merge(adhd_df_ml, adhd_df_gnn, on="participant_id", how="outer")
    sex_final = pd.merge(sex_df_ml, sex_df_gnn, on="participant_id", how="outer")

    # Sort and save
    adhd_final = adhd_final.sort_values("participant_id")
    sex_final = sex_final.sort_values("participant_id")

    adhd_path = "adhd_final_prediction.csv"
    sex_path = "sex_final_prediction.csv"
    adhd_final.to_csv(adhd_path, index=False)
    sex_final.to_csv(sex_path, index=False)

    print(f"Saved ADHD predictions: {adhd_final.shape}")
    print(f"Saved Sex predictions: {sex_final.shape}")

    # Summarize
    summarize_predictions(adhd_path, "adhd")
    summarize_predictions(sex_path, "sex")

    create_final_prediction("adhd_summarized.csv", "sex_summarized.csv", args.n, args.m)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_folder", required=True, help="Path to ML .csv files")
    parser.add_argument("--gnn_folder", required=True, help="Path to GNN .csv files")
    parser.add_argument("--m", type=int, required=True, help="Threshold for Sex prediction (e.g., 1)")
    parser.add_argument("--n", type=int, required=True, help="Threshold for ADHD prediction (e.g., 2)")
    args = parser.parse_args()
    main(args.ml_folder, args.gnn_folder)


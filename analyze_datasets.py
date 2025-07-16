import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIRS = [
    "datasets",
    "datasets/data7",
    "datasets/data8",
    "datasets/dataset/clinical_data"
]

SUMMARY = []

def classify_dataset(file_path, df):
    filename = os.path.basename(file_path).lower()

    if "cdc" in filename or "brfss" in filename:
        return "Survey"
    elif "diabetic_data" in filename or "hospital" in filename:
        return "Hospital"
    elif "diabetes-2" in filename or "lifestyle" in filename:
        return "Lifestyle"
    elif "diabetes-3" in filename or "pima" in filename:
        return "Pima EHR"
    elif "diabetes.csv" in filename:
        return "EHR"
    elif "clinical" in file_path or "person.csv" in filename or "measurement" in filename:
        return "Clinical"
    else:
        return "Unknown"

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        data_type = classify_dataset(file_path, df)

        info = {
            "File": os.path.basename(file_path),
            "Path": file_path,
            "Type": data_type,
            "Shape": f"{df.shape[0]} rows, {df.shape[1]} cols",
            "Missing %": round(df.isnull().mean().mean() * 100, 2),
            "Numeric cols": len(df.select_dtypes(include=np.number).columns),
            "Categorical cols": len(df.select_dtypes(include='object').columns)
        }

        SUMMARY.append(info)

        print(f"\n--- {os.path.basename(file_path)} ---")
        print(info)
        print(df.head(2))

        # Simple plot for numeric distributions
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 1:
            df[numeric_cols].hist(figsize=(10, 6), bins=30)
            plt.suptitle(os.path.basename(file_path))
            plt.tight_layout()
            plt.savefig(f"analysis_{os.path.basename(file_path)}.png")
            plt.close()

        # Correlation matrix if enough numerical columns
        if len(numeric_cols) >= 3:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm')
            plt.title(f"Correlation: {os.path.basename(file_path)}")
            plt.tight_layout()
            plt.savefig(f"corr_{os.path.basename(file_path)}.png")
            plt.close()

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

def run_analysis():
    print("=== Starting dataset analysis ===")
    for folder in DATASET_DIRS:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    analyze_file(os.path.join(folder, fname))
        else:
            print(f"[WARN] Folder does not exist: {folder}")

    # Export summary
    summary_df = pd.DataFrame(SUMMARY)
    summary_df.to_csv("dataset_summary.csv", index=False)
    print("\n Summary saved to 'dataset_summary.csv'")
    print(summary_df)

if __name__ == "__main__":
    run_analysis()

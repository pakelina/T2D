import pandas as pd
import numpy as np

def load_clinical_features_():
    # Load required CSV files
    person_df = pd.read_csv("datasets/dataset/clinical_data/person.csv")
    obs_df = pd.read_csv("datasets/dataset/clinical_data/observation.csv")
    meas_df = pd.read_csv("datasets/dataset/clinical_data/measurement.csv")
    cond_df = pd.read_csv("datasets/dataset/clinical_data/condition_occurrence.csv")

    print("person.csv:", person_df.shape)
    print("observation.csv:", obs_df.shape)
    print("measurement.csv:", meas_df.shape)
    print("condition_occurrence.csv:", cond_df.shape)

    # --- Demographics ---
    person_df = person_df[['person_id', 'gender_concept_id', 'year_of_birth']]
    person_df = person_df.rename(columns={
        'person_id': 'id',
        'gender_concept_id': 'gender',
        'year_of_birth': 'birth_year'
    })

    # --- Average observations ---
    obs_df = obs_df[pd.to_numeric(obs_df["value_as_number"], errors="coerce").notnull()]
    obs_df["value_as_number"] = pd.to_numeric(obs_df["value_as_number"])
    obs_agg = obs_df.groupby("person_id")["value_as_number"].mean().reset_index()
    obs_agg = obs_agg.rename(columns={"person_id": "id", "value_as_number": "avg_obs"})

    # --- Average measurements ---
    meas_df = meas_df[pd.to_numeric(meas_df["value_as_number"], errors="coerce").notnull()]
    meas_df["value_as_number"] = pd.to_numeric(meas_df["value_as_number"])
    meas_agg = meas_df.groupby("person_id")["value_as_number"].mean().reset_index()
    meas_agg = meas_agg.rename(columns={"person_id": "id", "value_as_number": "avg_meas"})

    # --- Count of conditions per person ---
    chronic_counts = cond_df.groupby("person_id").size().reset_index(name="condition_count")
    chronic_counts = chronic_counts.rename(columns={"person_id": "id"})

    print("Unique IDs:")
    print("  person_df:", person_df['id'].nunique())
    print("  obs_df:", obs_df['person_id'].nunique())
    print("  meas_df:", meas_df['person_id'].nunique())
    print("  cond_df:", cond_df['person_id'].nunique())

    # --- Merge all features ---
    df = person_df.merge(obs_agg, on="id", how="left")
    print("After merging obs_agg:", df.shape)

    df = df.merge(meas_agg, on="id", how="left")
    print("After merging meas_agg:", df.shape)

    df = df.merge(chronic_counts, on="id", how="left")
    print("After merging chronic_counts:", df.shape)

    print("Before mapping gender, unique values:", df["gender"].unique())
    print("Value counts:\n", df["gender"].value_counts())

    # --- Gender encoding (male=1, female=0) ---
    # Only map if gender values are concept IDs
    if df["gender"].isin([8507, 8532]).any():
        df["gender"] = df["gender"].map({8507: 1, 8532: 0})

    # Keep only rows with gender 0 or 1
    df = df[df["gender"].isin([0, 1])]

    print("After gender mapping and filtering:", df.shape)

    # --- Handle missing values ---
    df = df[df["gender"].isin([0, 1])]
    df = df[df["birth_year"] > 1900]  # Assuming birth year should be realistic
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print(f"Final clinical feature shape: {df.shape}")
    print("Final clinical feature DataFrame preview:")
    print(df.head())
    return df

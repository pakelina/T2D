import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from clinical_preprocessing import load_clinical_features_

def load_and_prepare_data():
    # Load and preprocess EHR dataset
    ehr_df = pd.read_csv('datasets/diabetes.csv').dropna().reset_index(drop=True)
    y = ehr_df['Outcome'].astype(int)
    ehr_features = ehr_df.drop(columns=['Outcome'])

    # Load and preprocess lifestyle dataset
    lifestyle_df = pd.read_csv('datasets/diabetes-2.csv').dropna().reset_index(drop=True)
    lifestyle_df.columns = lifestyle_df.columns.str.strip().str.lower()
    lifestyle_df = lifestyle_df[["age", "frame", "waist", "hip"]]
    lifestyle_df["frame"] = LabelEncoder().fit_transform(lifestyle_df["frame"])

    # Load and preprocess clinical dataset from clinical_preprocessing.py
    clinical_df = load_clinical_features_()
    if "id" in clinical_df.columns:
        clinical_df = clinical_df.drop(columns=["id"])

    # Load and preprocess PIMA dataset
    pima_df = pd.read_csv("datasets/diabetes-3.csv")
    pima_df = pima_df.dropna().reset_index(drop=True)
    pima_features = pima_df.drop(columns=['Outcome'])

    # Load and preprocess CDC dataset
    cdc_df = pd.read_csv("datasets/data8/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    cdc_df = cdc_df[[
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "PhysActivity", "Fruits", "Veggies", "GenHlth", "MentHlth"
    ]]
    cdc_df = cdc_df.dropna().reset_index(drop=True)

    # Load and preprocess Hospital dataset
    hosp_df = pd.read_csv("datasets/data7/diabetic_data.csv")
    hosp_df.replace("?", np.nan, inplace=True)
    hosp_df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], inplace=True, errors='ignore')
    hosp_df.dropna(inplace=True)
    for col in hosp_df.select_dtypes(include='object').columns:
        hosp_df[col] = LabelEncoder().fit_transform(hosp_df[col])
    hosp_df = hosp_df.drop(columns=["readmitted"], errors="ignore")
    hosp_df = hosp_df.select_dtypes(include=[np.number]).iloc[:, :10]

    # Align all dataset lengths
    min_len = min(len(ehr_features), len(lifestyle_df), len(clinical_df), len(pima_features), len(cdc_df), len(hosp_df))
    ehr_features = ehr_features.iloc[:min_len].reset_index(drop=True)
    lifestyle_df = lifestyle_df.iloc[:min_len].reset_index(drop=True)
    clinical_df = clinical_df.iloc[:min_len].reset_index(drop=True)
    pima_features = pima_features.iloc[:min_len].reset_index(drop=True)
    cdc_df = cdc_df.iloc[:min_len].reset_index(drop=True)
    hosp_df = hosp_df.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)

    # Fill missing values
    ehr_features = ehr_features.fillna(ehr_features.mean(numeric_only=True))
    lifestyle_df = lifestyle_df.fillna(lifestyle_df.mean(numeric_only=True))
    clinical_df = clinical_df.fillna(clinical_df.mean(numeric_only=True))
    pima_features = pima_features.fillna(pima_features.mean(numeric_only=True))
    cdc_df = cdc_df.fillna(cdc_df.mean(numeric_only=True))
    hosp_df = hosp_df.fillna(hosp_df.mean(numeric_only=True))

    # Standardize features
    ehr_scaled = StandardScaler().fit_transform(ehr_features)
    life_scaled = StandardScaler().fit_transform(lifestyle_df)
    clin_scaled = StandardScaler().fit_transform(clinical_df)
    pima_scaled = StandardScaler().fit_transform(pima_features)
    cdc_scaled = StandardScaler().fit_transform(cdc_df)
    hosp_scaled = StandardScaler().fit_transform(hosp_df)

    # Split data
    X_train = train_test_split(
        ehr_scaled, life_scaled, clin_scaled, pima_scaled, cdc_scaled, hosp_scaled, y.values,
        test_size=0.2, random_state=42
    )
    X_ehr_train, X_ehr_test, X_life_train, X_life_test, X_clin_train, X_clin_test, \
        X_pima_train, X_pima_test, X_cdc_train, X_cdc_test, X_hosp_train, X_hosp_test, \
        y_train, y_test = X_train

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    train_ds = TensorDataset(
        to_tensor(X_ehr_train), to_tensor(X_life_train), to_tensor(X_clin_train),
        to_tensor(X_pima_train), to_tensor(X_cdc_train), to_tensor(X_hosp_train),
        to_tensor(y_train).unsqueeze(1)
    )
    test_ds = TensorDataset(
        to_tensor(X_ehr_test), to_tensor(X_life_test), to_tensor(X_clin_test),
        to_tensor(X_pima_test), to_tensor(X_cdc_test), to_tensor(X_hosp_test),
        to_tensor(y_test).unsqueeze(1)
    )

    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(test_ds, batch_size=32)

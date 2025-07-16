# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_synthea_features():
    patients = pd.read_csv("datasets/csv/patients.csv")
    conditions = pd.read_csv("datasets/csv/conditions.csv")
    observations = pd.read_csv("datasets/csv/observations.csv")
    observations.columns = observations.columns.str.upper()

    observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')
    observations.dropna(subset=['VALUE'], inplace=True)

    important_obs = ["Body Weight", "Body Height", "BMI", "Hemoglobin A1c", "Systolic Blood Pressure", "Diastolic Blood Pressure"]
    obs_filtered = observations[observations['DESCRIPTION'].isin(important_obs)]
    obs_pivot = obs_filtered.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='mean').reset_index()
    chronic_conditions = conditions.groupby("PATIENT").size().reset_index(name="condition_count")
    patient_df = patients[['Id', 'BIRTHDATE', 'GENDER', 'RACE']]
    synthea_data = patient_df.merge(obs_pivot, left_on='Id', right_on='PATIENT', how='inner')
    synthea_data = synthea_data.merge(chronic_conditions, left_on='Id', right_on='PATIENT', how='left')
    synthea_data = synthea_data.drop(columns=[col for col in ['patient_x', 'patient_y'] if col in synthea_data.columns])

    synthea_data['GENDER'] = synthea_data['GENDER'].map({'M': 0, 'F': 1})
    synthea_data = pd.get_dummies(synthea_data, columns=['RACE'])

    synthea_data.fillna(synthea_data.mean(numeric_only=True), inplace=True)
    return synthea_data

def load_and_prepare_data():
    ehr_df = pd.read_csv('datasets/diabetes.csv').dropna().reset_index(drop=True)
    lifestyle_df = pd.read_csv('datasets/diabetes-2.csv').dropna().reset_index(drop=True)
    lifestyle_df = lifestyle_df.drop(columns=['id', 'location', 'gender'], errors='ignore')

    # Select only relevant 4 features for lifestyle
    lifestyle_df = pd.read_csv('datasets/diabetes-2.csv').dropna().reset_index(drop=True)
    lifestyle_df.columns = lifestyle_df.columns.str.strip().str.lower()
    lifestyle_df = lifestyle_df[["age", "frame", "waist", "hip"]]

    from sklearn.preprocessing import LabelEncoder
    lifestyle_df["frame"] = LabelEncoder().fit_transform(lifestyle_df["frame"])

    synthea_df = load_synthea_features()
    synthea_df = synthea_df.select_dtypes(include=[np.number])  # only numeric
    synthea_df = synthea_df.iloc[:, :5]  # pick first 5 numerical features

    pima_df = pd.read_csv("datasets/diabetes-3.csv")

    cdc_df = pd.read_csv("datasets/data8/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    cdc_df = cdc_df[[
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "PhysActivity", "Fruits", "Veggies", "GenHlth", "MentHlth"
    ]]

    hosp_df = pd.read_csv("datasets/data7/diabetic_data.csv")
    hosp_df.replace("?", np.nan, inplace=True)
    hosp_df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], inplace=True)
    hosp_df.dropna(inplace=True)
    for col in hosp_df.select_dtypes(include='object').columns:
        hosp_df[col] = LabelEncoder().fit_transform(hosp_df[col])
    hosp_df = hosp_df.drop(columns=["readmitted"], errors="ignore")
    hosp_df = hosp_df.select_dtypes(include=[np.number]).iloc[:, :10]  # select first 10 numerical features

    # Align length
    min_len = min(len(ehr_df), len(lifestyle_df), len(synthea_df), len(pima_df), len(cdc_df), len(hosp_df))
    ehr_df = ehr_df.iloc[:min_len].reset_index(drop=True)
    lifestyle_df = lifestyle_df.iloc[:min_len].reset_index(drop=True)
    synthea_df = synthea_df.iloc[:min_len].reset_index(drop=True)
    pima_df = pima_df.iloc[:min_len].reset_index(drop=True)
    cdc_df = cdc_df.iloc[:min_len].reset_index(drop=True)
    hosp_df = hosp_df.iloc[:min_len].reset_index(drop=True)

    ehr_target = ehr_df['Outcome'].astype(int)
    ehr_features = ehr_df.drop(columns=['Outcome'])

    # === Scale ===
    ehr_scaled = StandardScaler().fit_transform(ehr_features)
    life_scaled = StandardScaler().fit_transform(lifestyle_df)
    syn_scaled = StandardScaler().fit_transform(synthea_df)
    pima_scaled = StandardScaler().fit_transform(pima_df.drop(columns=['Outcome']))
    cdc_scaled = StandardScaler().fit_transform(cdc_df)
    hosp_scaled = StandardScaler().fit_transform(hosp_df)

    # === Train/test split ===
    X_ehr_train, X_ehr_test, X_life_train, X_life_test, X_syn_train, X_syn_test, X_pima_train, X_pima_test, X_cdc_train, X_cdc_test, X_hosp_train, X_hosp_test, y_train, y_test = train_test_split(
        ehr_scaled, life_scaled, syn_scaled, pima_scaled, cdc_scaled, hosp_scaled, ehr_target.values, test_size=0.2, random_state=42)

    # === Convert to PyTorch tensors ===
    def to_tensor(x): return torch.tensor(x, dtype=torch.float32)
    train_ds = TensorDataset(
        to_tensor(X_ehr_train), to_tensor(X_life_train), to_tensor(X_syn_train),
        to_tensor(X_pima_train), to_tensor(X_cdc_train), to_tensor(X_hosp_train),
        to_tensor(y_train).unsqueeze(1)
    )
    test_ds = TensorDataset(
        to_tensor(X_ehr_test), to_tensor(X_life_test), to_tensor(X_syn_test),
        to_tensor(X_pima_test), to_tensor(X_cdc_test), to_tensor(X_hosp_test),
        to_tensor(y_test).unsqueeze(1)
    )


    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(test_ds, batch_size=32)

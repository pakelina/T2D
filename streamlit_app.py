from model import MultimodalT2DPredictor
import streamlit as st
import numpy as np
import torch

st.set_page_config(page_title="T2D Risk Predictor", layout="centered")
st.title("🧬 Type 2 Diabetes Risk Predictor")

st.write("Please fill in your health data to estimate your risk of developing Type 2 Diabetes.")

# EHR Inputs
def input_ehr():
    return np.array([
        st.number_input("Glucose Level"),
        st.number_input("Systolic Blood Pressure"),
        st.number_input("Diastolic Blood Pressure"),
        st.number_input("Cholesterol Level"),
        st.number_input("BMI"),
        st.number_input("Height (cm)"),
        st.number_input("Weight (kg)"),
        st.number_input("Hemoglobin A1C")
    ], dtype=np.float32)

# Lifestyle Inputs
def input_lifestyle():
    age = st.number_input("Your Age")
    frame = st.selectbox("Body Frame Size", options=["small", "medium", "large"], index=1)
    waist = st.number_input("Waist Circumference (cm)")
    hip = st.number_input("Hip Circumference (cm)")

    # Encode string to integer
    frame_encoded = {"small": 0, "medium": 1, "large": 2}[frame]
    return np.array([age, frame_encoded, waist, hip], dtype=np.float32)

# Synthea Inputs
def input_synthea():
    return np.array([
        st.number_input("Chronic Conditions Count"),
        st.number_input("Medications Count"),
        st.number_input("Abnormal Lab Tests"),
        st.number_input("ER Visits Last Year"),
        st.number_input("Vaccination Count")
    ], dtype=np.float32)

# PIMA Inputs
def input_pima():
    return np.array([
        st.number_input("Pregnancies"),
        st.number_input("Plasma Glucose"),
        st.number_input("Blood Pressure"),
        st.number_input("Skin Thickness"),
        st.number_input("Insulin Level"),
        st.number_input("BMI (PIMA)"),
        st.number_input("Diabetes Pedigree Function"),
        st.number_input("Age")
    ], dtype=np.float32)

# CDC Inputs
def input_cdc():
    return np.array([
        st.number_input("High Blood Pressure (1=Yes, 0=No)"),
        st.number_input("High Cholesterol (1=Yes, 0=No)"),
        st.number_input("Cholesterol Check (1=Yes, 0=No)"),
        st.number_input("BMI (CDC)"),
        st.number_input("Smoker (1=Yes, 0=No)"),
        st.number_input("Physically Active (1=Yes, 0=No)"),
        st.number_input("Fruits Intake (1=Yes, 0=No)"),
        st.number_input("Vegetables Intake (1=Yes, 0=No)"),
        st.number_input("General Health (1=Excellent, 5=Poor)"),
        st.number_input("Days of Poor Mental Health (last 30)")
    ], dtype=np.float32)

# Hospital Inputs
def input_hospital():
    return np.array([
        st.number_input("Hospital Admissions (last year)"),
        st.number_input("ER Visits"),
        st.number_input("Avg Hospital Stay (days)"),
        st.number_input("Chronic Illnesses Treated"),
        st.number_input("Medications Prescribed"),
        st.number_input("Procedure Count"),
        st.number_input("Lab Tests Count"),
        st.number_input("Total Estimated Expense"),
        st.number_input("Follow-up Visits"),
        st.number_input("Had Surgery (1=Yes, 0=No)")
    ], dtype=np.float32)

# ---- FORM ----
with st.form("t2d_form"):
    st.subheader("📄 Electronic Health Record (EHR)")
    ehr = input_ehr()

    st.subheader("🏃 Lifestyle Information")
    lifestyle = input_lifestyle()

    st.subheader("🧬 Synthea Features")
    synthea = input_synthea()

    st.subheader("🧪 PIMA Dataset")
    pima = input_pima()

    st.subheader("📊 CDC Health Indicators")
    cdc = input_cdc()

    st.subheader("🏥 Hospital Utilization")
    hosp = input_hospital()

    submitted = st.form_submit_button("🔍 Predict Risk")

# ---- Inference ----
if submitted:
    # Convert all to tensors
    ehr_tensor = torch.tensor(ehr).unsqueeze(0)
    lifestyle_tensor = torch.tensor(lifestyle).unsqueeze(0)
    synthea_tensor = torch.tensor(synthea).unsqueeze(0)
    pima_tensor = torch.tensor(pima).unsqueeze(0)
    cdc_tensor = torch.tensor(cdc).unsqueeze(0)
    hosp_tensor = torch.tensor(hosp).unsqueeze(0)

    # Load model
    model = MultimodalT2DPredictor(
        ehr_dim=8,
        lifestyle_dim=4,
        synthea_dim=5,
        pima_dim=8,
        cdc_dim=10,
        hosp_dim=10
    )
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        output = model(ehr_tensor, lifestyle_tensor, synthea_tensor, pima_tensor, cdc_tensor, hosp_tensor)
        risk_score = torch.sigmoid(output).item()

    # Show result
    st.success(f"✅ Predicted Type 2 Diabetes Risk Score: **{risk_score:.4f}**")
    if risk_score < 0.46:
        st.info("🟢 Low Risk — Keep maintaining a healthy lifestyle.")
    elif 0.46 <= risk_score < 0.7:
        st.warning("⚠️ Moderate Risk — Consider talking to your doctor and monitoring health habits.")
    else:
        st.error("🔴 High Risk — Strongly recommended to consult a healthcare provider.")


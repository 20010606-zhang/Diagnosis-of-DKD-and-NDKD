# ============================================================
# DKD vs NDKD Prediction Web Application
# Final 7-feature Random Forest Model
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="DKD vs NDKD Prediction",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "random_forest_model.joblib"
    )

    imputer = joblib.load(
        "final_mean_imputer.joblib"
    )

    features = joblib.load(
        "final_feature_names.joblib"
    )

    return model, imputer, features



model, imputer, features = load_model()



# ============================================================
# Title
# ============================================================

st.title(
    "Non-invasive Prediction of DKD vs NDKD"
)


st.write(
    """
This tool predicts the probability of diabetic kidney disease (DKD)
among patients with type 2 diabetes using a seven-feature random forest model.

The prediction is based on routinely available clinical variables.
"""
)



# ============================================================
# Input variables
# ============================================================

st.sidebar.header(
    "Patient Clinical Information"
)



# Serum creatinine

scr = st.sidebar.number_input(
    "Serum creatinine (μmol/L)",
    min_value=0.0,
    value=100.0
)



# DR

dr = st.sidebar.selectbox(
    "Diabetic retinopathy status",
    options=[
        0,
        1
    ],
    format_func=lambda x:
        "Absent" if x == 0 else "Present"
)



# TC

tc = st.sidebar.number_input(
    "Total cholesterol (mmol/L)",
    min_value=0.0,
    value=4.5
)



# Duration DM

duration_dm = st.sidebar.number_input(
    "Duration of diabetes (years)",
    min_value=0.0,
    value=10.0
)



# FBG

fbg = st.sidebar.number_input(
    "Fasting blood glucose (mmol/L)",
    min_value=0.0,
    value=7.0
)



# urine protein

upe = st.sidebar.number_input(
    "24-hour urinary protein excretion (g/24h)",
    min_value=0.0,
    value=1.0
)



# LDL

ldl = st.sidebar.number_input(
    "LDL cholesterol (mmol/L)",
    min_value=0.0,
    value=2.5
)



# ============================================================
# Prediction
# ============================================================


if st.button(
    "Predict"
):


    input_data = pd.DataFrame(
        {
            "Serum creatinine":
                [scr],

            "DR":
                [dr],

            "TC":
                [tc],

            "Duration of DM":
                [duration_dm],

            "FBG":
                [fbg],

            "Urine protein excretion":
                [upe],

            "LDL":
                [ldl]
        }
    )


    # 保证变量顺序
    input_data = input_data[
        features
    ]


    # continuous variables
    continuous_features = [
        "Serum creatinine",
        "TC",
        "Duration of DM",
        "FBG",
        "Urine protein excretion",
        "LDL"
    ]


    # impute missing continuous values
    input_data[continuous_features] = (
        imputer.transform(
            input_data[continuous_features]
        )
    )



    # probability

    probability = model.predict_proba(
        input_data
    )[0,1]



    threshold = 0.45


    prediction = (
        "DKD"
        if probability >= threshold
        else
        "NDKD"
    )



    # ========================================================
    # Output
    # ========================================================


    st.subheader(
        "Prediction Result"
    )


    st.metric(
        "Predicted probability of DKD",
        f"{probability:.3f}"
    )



    if prediction == "DKD":

        st.error(
            f"Predicted class: {prediction}"
        )

    else:

        st.success(
            f"Predicted class: {prediction}"
        )



    st.write(
        f"""
Classification threshold:
**0.45**

A probability ≥0.45 indicates a higher predicted likelihood
of DKD within this model framework.
"""
    )



# ============================================================
# Model information
# ============================================================


st.divider()


st.caption(
    """
Model:
Seven-feature Random Forest

Predictors:
Serum creatinine, diabetic retinopathy,
total cholesterol, diabetes duration,
fasting blood glucose,
24-hour urinary protein excretion,
LDL cholesterol.

The model should support clinical assessment
and does not replace kidney biopsy.
"""
)
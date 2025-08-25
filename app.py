import streamlit as st
import matplotlib
import os
import requests
from io import BytesIO

matplotlib.use('Agg')  # Non-interactive backend to avoid plotting errors
import matplotlib.pyplot as plt
import shap
import numpy as np
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer  # Reuse local missing value handling logic
import warnings

warnings.filterwarnings("ignore")  # Ignore redundant warnings

# -------------------------- 1. Basic Configuration (Fonts, Paths, Feature Definitions) --------------------------
# Set matplotlib fonts and negative sign display (consistent with local training code)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Define features and target variable (must match local training code exactly!)
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
target_name = 'Pathology type'

# Streamlit app title
st.title("SHAP Model Visualization App (Based on Local Trained Model)")

# Define file paths (locally trained model and data files should be in the same directory as the Streamlit script)
local_model_path = "random_forest_model.joblib"  # Locally trained model
local_data_path = "your_data.csv"  # Preprocessed data saved locally (for loading SimpleImputer rules)
# If you don't have your_data.csv, you can use the original test1.xlsx, ensure the path is correct:
# local_data_path = "test1.xlsx"

# -------------------------- 2. Load Locally Trained Model and Preprocessing Components --------------------------
try:
    # 1. Load the locally trained random forest model
    if not os.path.exists(local_model_path):
        st.error(
            f"Local model file not found: {local_model_path}, please place the locally trained model in the same directory as the script!")
        st.stop()  # Stop running if model doesn't exist
    model = joblib.load(local_model_path)
    st.success("✅ Local random forest model loaded successfully!")

    # 2. Load local data and re-initialize SimpleImputer (consistent with local training logic)
    # (Purpose: Reuse mean imputation rules from training to avoid bias with user input or new data)
    if os.path.exists(local_data_path):
        if local_data_path.endswith(".csv"):
            data = pd.read_csv(local_data_path)
        else:  # For xlsx files
            data = pd.read_excel(local_data_path)
    else:
        st.error(f"Local data file not found: {local_data_path}, please check the path!")
        st.stop()

    # Initialize mean imputer (only for numerical features, exactly matching local code)
    mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
    mean_imputer = SimpleImputer(strategy='mean')
    mean_imputer.fit(data[mean_columns])  # Fit with means from local training data (ensure consistent rules)
    st.success("✅ Missing value handling component loaded successfully!")

    # 3. Generate SHAP explainer (based on loaded local model)
    explainer = shap.TreeExplainer(model)
    st.success("✅ SHAP explainer initialized successfully!")

except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# -------------------------- 3. Streamlit User Input Components --------------------------
st.subheader("Please Enter Patient Metrics")
# Generate user input fields (in feature order, supporting numerical input with step=0.01 for precision)
input_features = []
for feature in feature_names:
    # Set reasonable input ranges based on feature meaning (optional, improves user experience)
    if feature == "DR":  # Assuming DR is a categorical feature (e.g., 0/1)
        val = st.number_input(f"{feature} (0=Absent, 1=Present)", min_value=0, max_value=1, step=1, value=0)
    elif feature == "Duration of DM":  # Diabetes duration (years)
        val = st.number_input(f"{feature} (years)", min_value=0.0, max_value=50.0, step=0.1, value=10.0)
    elif feature in ["HbA1c", "FBG"]:  # Blood glucose related metrics
        val = st.number_input(f"{feature}", min_value=3.0, max_value=20.0, step=0.1, value=7.0)
    elif feature == "Serum creatinine":  # Serum creatinine (μmol/L)
        val = st.number_input(f"{feature} (μmol/L)", min_value=30.0, max_value=500.0, step=1.0, value=80.0)
    elif feature == "TC":  # Total cholesterol (mmol/L)
        val = st.number_input(f"{feature} (mmol/L)", min_value=2.0, max_value=10.0, step=0.1, value=5.0)
    elif feature == "Urine protein excretion":  # Urine protein excretion rate (g/24h)
        val = st.number_input(f"{feature} (g/24h)", min_value=0.0, max_value=10.0, step=0.01, value=0.5)
    elif feature == "BMI":  # Body mass index
        val = st.number_input(f"{feature}", min_value=15.0, max_value=40.0, step=0.1, value=25.0)
    input_features.append(val)

# -------------------------- 4. Model Prediction and SHAP Visualization --------------------------
if st.button("Generate Prediction Results and SHAP Analysis"):
    try:
        # 1. Process user input (convert to array, match model input format)
        input_arr = np.array(input_features).reshape(1, -1)
        input_df = pd.DataFrame(input_arr, columns=feature_names)

        # 2. Apply missing value imputation to user input (consistent with local training logic)
        input_df[mean_columns] = mean_imputer.transform(input_df[mean_columns])
        st.subheader("✅ Processed Input Features")
        st.dataframe(input_df.round(2))  # Display processed input for user confirmation

        # 3. Model prediction (output class and probability)
        y_pred = model.predict(input_df)[0]
        y_pred_proba = model.predict_proba(input_df)[0].max()  # Prediction probability (maximum probability)
        st.subheader("📊 Model Prediction Results")
        st.write(f"Predicted pathology type: **{y_pred}**")
        st.write(f"Prediction confidence: **{y_pred_proba:.2%}**")

        # 4. Calculate SHAP values (explain prediction results)
        shap_values = explainer.shap_values(input_df)
        # Handle SHAP value structure for multi-class/binary classification (compatible with different cases)
        if isinstance(shap_values, list):  # Multi-class: take SHAP values for positive class (or target class)
            if len(shap_values) > 1:
                sample_shap = shap_values[1].flatten()  # Assuming 1 is positive class, adjust as needed
            else:
                sample_shap = shap_values[0].flatten()
        else:  # Binary classification: directly take 2D array
            sample_shap = shap_values.flatten()

        # Check if SHAP values match number of features (avoid visualization errors)
        if len(sample_shap) != len(feature_names):
            st.error(
                f"SHAP value length ({len(sample_shap)}) does not match number of features ({len(feature_names)})!")
            st.stop()

        # 5. Plot and display SHAP waterfall plot (explain each feature's contribution)
        st.subheader("🔍 SHAP Waterfall Plot (Feature Contribution Analysis)")
        plt.figure(figsize=(10, 6))
        # Construct SHAP explanation object (base_values is model expected value)
        shap_exp = shap.Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                  list) else explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        )
        shap.plots.waterfall(shap_exp, show=False)  # Don't display automatically, controlled by Streamlit
        plt.tight_layout()  # Adjust layout to avoid label truncation
        waterfall_path = "shap_waterfall.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        st.image(waterfall_path,
                 caption="Contribution of each feature to prediction (red=positive contribution, blue=negative contribution)")

        # 6. Plot and display SHAP force plot (interactive HTML)
        st.subheader("🔍 SHAP Force Plot (Intuitive Prediction Logic)")
        # Calculate model expected value (base value)
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                               list) else explainer.expected_value
        # Generate force plot
        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=sample_shap,
            features=input_df.iloc[0],
            feature_names=feature_names,
            matplotlib=False  # Generate HTML format (interactive)
        )
        # Save as HTML and load in Streamlit
        force_html_path = "shap_force.html"
        shap.save_html(force_html_path, force_plot)
        # Display HTML with Streamlit component
        import streamlit.components.v1 as components

        with open(force_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=300)  # Adjust height as needed

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

# -------------------------- 5. Additional Tips (to help users) --------------------------
st.sidebar.title("Usage Tips")
st.sidebar.info(
    "1. Please ensure entered metrics are within clinically reasonable ranges;\n2. In SHAP plots, red features indicate positive contribution to prediction, blue indicates negative contribution;\n3. The model is based on local training data. To update the model, replace the random_forest_model.joblib file.")

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import numpy as np
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
import warnings
import os
from io import BytesIO
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SHAP Model Visualization Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS（保持美观）
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .title-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .title-container h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .title-container p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eaeef2;
    }
    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #eaeef2;
        padding-bottom: 0.5rem;
    }
    .stNumberInput label {
        font-weight: 500;
        color: #2c3e50;
    }
    .stButton button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(90deg, #163458 0%, #1f3f7a 100%);
    }
    .metric-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-label {
        font-size: 1rem;
        color: #64748b;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
        line-height: 1.2;
    }
    .metric-value small {
        font-size: 1rem;
        font-weight: 400;
        color: #64748b;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题区域
st.markdown("""
<div class="title-container">
    <h1>🔬 SHAP-based Differential Diagnosis: DKD vs NDKD</h1>
    <p>Explainable Machine Learning for Diabetic Kidney Disease Classification</p>
</div>
""", unsafe_allow_html=True)

# 全局设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
target_name = 'Pathology type'

local_model_path = "random_forest_model.joblib"
local_data_path = "your_data.csv"

@st.cache_resource
def load_model_and_explainer():
    if not os.path.exists(local_model_path):
        st.error(f"❌ Local model file not found: {local_model_path}. Please place it in the script directory.")
        st.stop()
    model = joblib.load(local_model_path)

    if os.path.exists(local_data_path):
        if local_data_path.endswith(".csv"):
            data = pd.read_csv(local_data_path)
        else:
            data = pd.read_excel(local_data_path)
    else:
        alt_path = "test1.xlsx"
        if os.path.exists(alt_path):
            data = pd.read_excel(alt_path)
        else:
            st.error("❌ Local data file not found. Please provide your_data.csv or test1.xlsx to obtain imputation rules.")
            st.stop()

    mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
    mean_imputer = SimpleImputer(strategy='mean')
    mean_imputer.fit(data[mean_columns])

    explainer = shap.TreeExplainer(model)
    return model, mean_imputer, explainer, mean_columns

with st.spinner("Loading model and explainer..."):
    model, mean_imputer, explainer, mean_columns = load_model_and_explainer()
st.sidebar.success("✅ Model and explainer loaded successfully")

# 侧边栏说明
with st.sidebar:
    st.markdown('<div class="sidebar-header">📋 Instructions</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Enter patient metrics in the main panel.
    2. Click the **Analyze** button.
    3. View prediction results and SHAP plots.
    4. Red features indicate positive contribution; blue features indicate negative contribution.

    **Feature Descriptions**:
    - **DR**: Diabetic Retinopathy (0=No, 1=Yes)
    - **Duration of DM**: Diabetes duration (years)
    - **HbA1c**: Glycated hemoglobin (%)
    - **Serum creatinine**: (μmol/L)
    - **TC**: Total cholesterol (mmol/L)
    - **Urine protein excretion**: 24h urine protein (g/24h)
    - **FBG**: Fasting blood glucose (mmol/L)
    - **BMI**: Body mass index
    """)
    st.divider()
    st.caption("© 2024 Powered by locally trained model")

# 输入区域
st.markdown('<div class="card"><div class="card-header">📝 Patient Metrics Input</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    dr = st.number_input("DR (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=0,
                         help="Diabetic Retinopathy: 0=absent, 1=present")
    duration = st.number_input("Duration of DM (years)", min_value=0.0, max_value=50.0, step=0.1, value=10.0,
                               help="Diabetes duration in years")
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, step=0.1, value=7.0,
                            help="Glycated hemoglobin")
    serum_creatinine = st.number_input("Serum creatinine (μmol/L)", min_value=30.0, max_value=500.0, step=1.0, value=80.0,
                                       help="Serum creatinine level")

with col2:
    tc = st.number_input("TC (mmol/L)", min_value=2.0, max_value=10.0, step=0.1, value=5.0,
                         help="Total cholesterol")
    urine_protein = st.number_input("Urine protein excretion (g/24h)", min_value=0.0, max_value=10.0, step=0.01, value=0.5,
                                    help="24-hour urine protein excretion")
    fbg = st.number_input("FBG (mmol/L)", min_value=3.0, max_value=20.0, step=0.1, value=7.0,
                          help="Fasting blood glucose")
    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, step=0.1, value=25.0,
                          help="Body mass index")

st.markdown('</div>', unsafe_allow_html=True)

input_features = [dr, duration, hba1c, serum_creatinine, tc, urine_protein, fbg, bmi]

analyze_clicked = st.button("🚀 Analyze", use_container_width=True)

if analyze_clicked:
    with st.spinner("Computing SHAP values..."):
        try:
            # 处理输入
            input_arr = np.array(input_features).reshape(1, -1)
            input_df = pd.DataFrame(input_arr, columns=feature_names)
            input_df[mean_columns] = mean_imputer.transform(input_df[mean_columns])

            # 预测
            y_pred = model.predict(input_df)[0]
            y_pred_proba = model.predict_proba(input_df)[0].max()

            # 显示处理后的特征
            st.markdown('<div class="card"><div class="card-header">📊 Processed Input Features</div>', unsafe_allow_html=True)
            st.dataframe(input_df.round(2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 预测结果卡片
            st.markdown('<div class="card"><div class="card-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Predicted Pathology Type</div>
                    <div class="metric-value">{y_pred}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{y_pred_proba:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_res3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value">Random Forest</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ================== 调试信息 ==================
            st.write("### 🔍 Debug Information")
            st.write(f"**Model expects** {model.n_features_in_} features.")
            shap_values = explainer.shap_values(input_df)
            st.write(f"**Type of shap_values:** {type(shap_values)}")
            if isinstance(shap_values, list):
                st.write(f"**Number of classes:** {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    st.write(f"**shap_values[{i}] shape:** {sv.shape}")
            elif isinstance(shap_values, np.ndarray):
                st.write(f"**shap_values shape:** {shap_values.shape}")

            # ================== 提取正确的SHAP值（长度为8）==================
            expected_feature_count = len(feature_names)  # 8
            sample_shap = None

            if isinstance(shap_values, list):
                # 多分类列表形式
                if len(shap_values) >= 2:
                    class_index = 1  # 默认取类别1（正类）
                else:
                    class_index = 0
                candidate = shap_values[class_index].flatten()
                if len(candidate) == expected_feature_count:
                    sample_shap = candidate
                    st.write(f"**Using SHAP values from class {class_index}.**")
                else:
                    st.error(f"SHAP value length from class {class_index} is {len(candidate)}, expected {expected_feature_count}.")
                    st.stop()
            elif isinstance(shap_values, np.ndarray):
                # 处理三维数组 (n_samples, n_features, n_classes)
                if shap_values.ndim == 3:
                    n_classes = shap_values.shape[2]
                    st.write(f"**SHAP array has {n_classes} classes.**")
                    # 默认使用类别1（索引1），如果类别数大于1；否则使用唯一类别
                    class_index = 1 if n_classes > 1 else 0
                    # 取出该类别的 SHAP 值（第一个样本）
                    class_shap = shap_values[0, :, class_index]  # 形状 (n_features,)
                    candidate = class_shap.flatten()
                    if len(candidate) == expected_feature_count:
                        sample_shap = candidate
                        st.write(f"**Using SHAP values from class {class_index}.**")
                    else:
                        st.error(f"SHAP value length from class {class_index} is {len(candidate)}, expected {expected_feature_count}.")
                        st.stop()
                else:
                    # 普通二维数组
                    candidate = shap_values.flatten()
                    if len(candidate) == expected_feature_count:
                        sample_shap = candidate
                    else:
                        st.error(f"SHAP value length is {len(candidate)}, expected {expected_feature_count}.")
                        st.stop()
            else:
                st.error("Unexpected type for shap_values.")
                st.stop()

            if sample_shap is None:
                st.error("Could not extract SHAP values matching feature count.")
                st.stop()

            # 获取base_value（与所选类别对应）
            if isinstance(explainer.expected_value, list):
                if len(explainer.expected_value) >= 2:
                    base_value = explainer.expected_value[1]  # 与 class_index 保持一致
                else:
                    base_value = explainer.expected_value[0]
            elif isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim == 1:
                # 如果 expected_value 是一维数组，长度等于类别数
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value

            # 创建SHAP Explanation对象
            shap_exp = shap.Explanation(
                values=sample_shap,
                base_values=base_value,
                data=input_df.iloc[0].values,
                feature_names=feature_names
            )

            # ================== SHAP 瀑布图 ==================
            st.markdown('<div class="card"><div class="card-header">🔍 SHAP Waterfall Plot (Feature Contribution)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_exp, show=False, max_display=len(feature_names))
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            st.image(buf, caption="Contribution of each feature to the prediction (red=positive, blue=negative)", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ================== SHAP 力图 ==================
            st.markdown('<div class="card"><div class="card-header">🔍 SHAP Force Plot (Prediction Explanation)</div>', unsafe_allow_html=True)
            force_plot = shap.force_plot(
                base_value=base_value,
                shap_values=sample_shap,
                features=input_df.iloc[0],
                feature_names=feature_names,
                matplotlib=False
            )
            html_string = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html_string, height=400, scrolling=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("Tip: To update the model, replace `random_forest_model.joblib` and restart the app.")

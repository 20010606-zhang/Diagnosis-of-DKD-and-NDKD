# ============================================================
# STAGE 15 + STAGE 16
#
# Individual SHAP explanations
#
# Stage 15:
# Internal-validation position 271
# Expected pathology: NDKD (label = 0)
#
# Stage 16:
# Internal-validation position 273
# Expected pathology: DKD (label = 1)
#
# Final 7-feature Random Forest
#
# Outputs:
# - SHAP force plot (HTML)
# - SHAP waterfall plot (PDF + PNG)
# - Patient-level SHAP source data (Excel)
#
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Global settings
# ============================================================

RANDOM_STATE = 45

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


# ============================================================
# 2. Matplotlib settings
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})


# ============================================================
# 3. File and variable settings
# ============================================================

DATA_FILE = "test1.xlsx"

TARGET = "Pathology type"


# Final 7 predictors
FINAL_FEATURES = [
    "Serum creatinine",
    "DR",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


# Continuous variables requiring mean imputation
CONTINUOUS_FEATURES = [
    "Serum creatinine",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


# ============================================================
# 4. Define Stage 15 and Stage 16 patients
#
# IMPORTANT:
# These are positions within the internal validation cohort,
# NOT original Excel row numbers.
# ============================================================

PATIENTS = [
    {
        "stage": "15",
        "name": "NDKD",
        "sample_index": 271,
        "expected_label": 0
    },
    {
        "stage": "16",
        "name": "DKD",
        "sample_index": 273,
        "expected_label": 1
    }
]


# ============================================================
# 5. Load data
# ============================================================

try:

    df = pd.read_excel(
        DATA_FILE
    )

except FileNotFoundError:

    print(
        "文件未找到，请检查文件路径。"
    )

    raise


print("\n============================================================")
print("STAGE 15 + STAGE 16")
print("FINAL 7-FEATURE RF INDIVIDUAL SHAP ANALYSIS")
print("============================================================")


print(
    f"\nData file: "
    f"{DATA_FILE}"
)

print(
    f"Total N = "
    f"{len(df)}"
)

print(
    f"Features ({len(FINAL_FEATURES)}):"
)


for feature in FINAL_FEATURES:

    print(
        f"  - {feature}"
    )


# ============================================================
# 6. Check required columns
# ============================================================

required_columns = (
    FINAL_FEATURES
    +
    [TARGET]
)


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if len(
    missing_columns
) > 0:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# 7. Check DR missingness
# ============================================================

n_missing_dr = int(
    df[
        "DR"
    ].isna().sum()
)


print(
    f"\nMissing DR values = "
    f"{n_missing_dr}"
)


if n_missing_dr > 0:

    raise ValueError(
        "DR contains missing values. "
        "The final primary pipeline does not impute DR."
    )


# ============================================================
# 8. Exact 80/20 stratified split
#
# IMPORTANT:
# Split BEFORE imputation.
# ============================================================

all_indices = np.arange(
    len(df)
)


y_all = df[
    TARGET
].astype(int).values


dev_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_all
)


df_dev = df.iloc[
    dev_indices
].copy()


df_val = df.iloc[
    val_indices
].copy()


y_dev = df_dev[
    TARGET
].astype(int).values


y_val = df_val[
    TARGET
].astype(int).values


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")


print(
    f"Development N = "
    f"{len(df_dev)}"
)

print(
    f"Internal validation N = "
    f"{len(df_val)}"
)

print(
    f"Development DKD = "
    f"{np.sum(y_dev == 1)}"
)

print(
    f"Development NDKD = "
    f"{np.sum(y_dev == 0)}"
)

print(
    f"Internal DKD = "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Internal NDKD = "
    f"{np.sum(y_val == 0)}"
)


# ============================================================
# 9. Leakage-free mean imputation
#
# Fit ONLY on development cohort.
# ============================================================

mean_imputer = SimpleImputer(
    strategy="mean"
)


X_dev_cont = pd.DataFrame(
    mean_imputer.fit_transform(
        df_dev[
            CONTINUOUS_FEATURES
        ]
    ),
    columns=CONTINUOUS_FEATURES,
    index=df_dev.index
)


X_val_cont = pd.DataFrame(
    mean_imputer.transform(
        df_val[
            CONTINUOUS_FEATURES
        ]
    ),
    columns=CONTINUOUS_FEATURES,
    index=df_val.index
)


# ============================================================
# 10. Add DR
# ============================================================

X_dev = X_dev_cont.copy()

X_val = X_val_cont.copy()


X_dev[
    "DR"
] = df_dev[
    "DR"
].values


X_val[
    "DR"
] = df_val[
    "DR"
].values


# Restore final feature order
X_dev = X_dev[
    FINAL_FEATURES
]


X_val = X_val[
    FINAL_FEATURES
]


# ============================================================
# 11. Train final RF
# ============================================================

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)


rf_classifier.fit(
    X_dev,
    y_dev
)


# ============================================================
# Save final 7-feature RF model for deployment
# ============================================================

import joblib


# save RF model
joblib.dump(
    rf_classifier,
    "random_forest_model.joblib"
)


# save imputer
joblib.dump(
    mean_imputer,
    "final_mean_imputer.joblib"
)


# save feature order
joblib.dump(
    FINAL_FEATURES,
    "final_feature_names.joblib"
)


print("\nDeployment files saved:")
print(" - random_forest_model.joblib")
print(" - final_mean_imputer.joblib")
print(" - final_feature_names.joblib")


# ============================================================
# 12. Reproduce internal validation performance
# ============================================================

y_prob = rf_classifier.predict_proba(
    X_val
)[:, 1]


auc = roc_auc_score(
    y_val,
    y_prob
)


print("\n============================================================")
print("SANITY CHECK")
print("============================================================")


print(
    "Expected final internal-validation AUC ≈ 0.8941"
)

print(
    f"Current internal-validation AUC = "
    f"{auc:.4f}"
)


if abs(
    auc - 0.8941
) <= 0.005:

    print(
        "Final RF reproduction check: PASS"
    )

else:

    print(
        "Final RF reproduction check: CHECK PIPELINE"
    )


# ============================================================
# 13. Calculate SHAP values
# ============================================================

explainer = shap.TreeExplainer(
    rf_classifier
)


raw_shap_values = explainer.shap_values(
    X_val
)


# ============================================================
# 14. Extract DKD class = 1 SHAP values
#
# Compatible with different SHAP versions
# ============================================================

if isinstance(
    raw_shap_values,
    list
):

    # Older SHAP versions
    shap_values = np.asarray(
        raw_shap_values[1]
    )


else:

    raw_shap_values = np.asarray(
        raw_shap_values
    )


    if raw_shap_values.ndim == 3:

        # New SHAP:
        # samples × features × classes
        shap_values = raw_shap_values[
            :,
            :,
            1
        ]


    elif raw_shap_values.ndim == 2:

        shap_values = raw_shap_values


    else:

        raise ValueError(
            "Unexpected SHAP value dimensions: "
            f"{raw_shap_values.shape}"
        )


# ============================================================
# 15. Extract expected value for DKD class
# ============================================================

expected_value = explainer.expected_value


if isinstance(
    expected_value,
    (list, tuple, np.ndarray)
):

    expected_value = np.asarray(
        expected_value
    ).reshape(-1)


    if len(
        expected_value
    ) >= 2:

        base_value = float(
            expected_value[1]
        )


    else:

        base_value = float(
            expected_value[0]
        )


else:

    base_value = float(
        expected_value
    )


print("\n============================================================")
print("SHAP INFORMATION")
print("============================================================")


print(
    f"SHAP matrix shape = "
    f"{shap_values.shape}"
)

print(
    f"Expected value for DKD class = "
    f"{base_value:.4f}"
)


# ============================================================
# 16. Waterfall plot saving function
# ============================================================

def save_shap_waterfall_plot(
    shap_values_sample,
    base_value,
    features_sample,
    feature_names,
    pdf_filename,
    png_filename
):

    print(
        f"\n正在保存SHAP瀑布图到 "
        f"{pdf_filename}..."
    )


    # Close previous figures
    plt.close(
        "all"
    )


    # SHAP Explanation
    explanation = shap.Explanation(
        values=np.asarray(
            shap_values_sample
        ),
        base_values=float(
            base_value
        ),
        data=np.asarray(
            features_sample
        ),
        feature_names=feature_names
    )


    # Let SHAP generate its own figure
    shap.plots.waterfall(
        explanation,
        max_display=len(
            feature_names
        ),
        show=False
    )


    fig = plt.gcf()


    fig.set_size_inches(
        8.5,
        6.5
    )


    # --------------------------------------------------------
    # Arial font
    # --------------------------------------------------------

    for text in fig.findobj(
        match=plt.Text
    ):

        try:

            text.set_fontfamily(
                "Arial"
            )

        except Exception:

            pass


    for ax in fig.axes:

        for tick in ax.get_xticklabels():

            tick.set_fontfamily(
                "Arial"
            )

            tick.set_fontsize(
                11
            )


        for tick in ax.get_yticklabels():

            tick.set_fontfamily(
                "Arial"
            )

            tick.set_fontsize(
                11
            )


    fig.tight_layout()

    fig.canvas.draw()


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    fig.savefig(
        pdf_filename,
        format="pdf",
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    fig.savefig(
        png_filename,
        format="png",
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"Saved PDF: "
        f"{pdf_filename}"
    )

    print(
        f"Saved PNG: "
        f"{png_filename}"
    )


# ============================================================
# 17. Process Stage 15 and Stage 16
# ============================================================

for patient in PATIENTS:

    stage = patient[
        "stage"
    ]

    pathology_name = patient[
        "name"
    ]

    sample_index = patient[
        "sample_index"
    ]

    expected_label = patient[
        "expected_label"
    ]


    print("\n")
    print("============================================================")
    print(
        f"STAGE {stage}: "
        f"{pathology_name} PATIENT"
    )
    print("============================================================")


    # --------------------------------------------------------
    # Check position
    # --------------------------------------------------------

    if sample_index >= len(
        X_val
    ):

        raise IndexError(
            f"sample_index={sample_index} exceeds "
            f"internal validation size "
            f"({len(X_val)})."
        )


    # --------------------------------------------------------
    # Extract patient
    # --------------------------------------------------------

    sample_features = X_val.iloc[
        sample_index
    ].copy()


    sample_shap_values = shap_values[
        sample_index
    ].copy()


    sample_true_label = int(
        y_val[
            sample_index
        ]
    )


    sample_predicted_probability = float(
        y_prob[
            sample_index
        ]
    )


    sample_original_index = df_val.index[
        sample_index
    ]


    # --------------------------------------------------------
    # Verify expected pathology
    # --------------------------------------------------------

    if sample_true_label != expected_label:

        raise ValueError(
            f"\nStage {stage} patient label mismatch!\n"
            f"Internal-validation position: "
            f"{sample_index}\n"
            f"Expected label: "
            f"{expected_label}\n"
            f"Actual label: "
            f"{sample_true_label}\n"
            f"Please check patient position."
        )


    # --------------------------------------------------------
    # Print patient information
    # --------------------------------------------------------

    print(
        f"Internal-validation position = "
        f"{sample_index}"
    )


    print(
        f"Original dataframe index = "
        f"{sample_original_index}"
    )


    print(
        f"True pathology label = "
        f"{sample_true_label}"
    )


    print(
        f"Pathology group = "
        f"{pathology_name}"
    )


    print(
        f"Predicted P(DKD) = "
        f"{sample_predicted_probability:.3f}"
    )


    print("\nFeature values:")


    for feature in FINAL_FEATURES:

        print(
            f"{feature}: "
            f"{sample_features[feature]}"
        )


    # ========================================================
    # 18. Save patient-level SHAP source data
    # ========================================================

    sample_output = pd.DataFrame(
        {
            "Feature":
                FINAL_FEATURES,

            "Feature_Value":
                sample_features.values,

            "SHAP_Value":
                sample_shap_values
        }
    )


    sample_output[
        "Stage"
    ] = stage


    sample_output[
        "Pathology_Group"
    ] = pathology_name


    sample_output[
        "True_Label"
    ] = sample_true_label


    sample_output[
        "Predicted_P_DKD"
    ] = sample_predicted_probability


    sample_output[
        "Internal_Validation_Position"
    ] = sample_index


    sample_output[
        "Original_Data_Index"
    ] = sample_original_index


    source_filename = (
        f"{stage}-{pathology_name}_"
        f"SHAP_Patient_{sample_index}_"
        f"Source_Data.xlsx"
    )


    sample_output.to_excel(
        source_filename,
        index=False
    )


    # ========================================================
    # 19. Save SHAP force plot as HTML
    # ========================================================

    force_plot = shap.force_plot(
        base_value,
        sample_shap_values,
        sample_features.values,
        feature_names=FINAL_FEATURES
    )


    force_filename = (
        f"{stage}-{pathology_name}_"
        f"SHAP_Force_Patient_"
        f"{sample_index}.html"
    )


    shap.save_html(
        force_filename,
        force_plot
    )


    print(
        f"Saved force plot: "
        f"{force_filename}"
    )


    # ========================================================
    # 20. Save waterfall plot
    # ========================================================

    waterfall_pdf = (
        f"{stage}-{pathology_name}_"
        f"SHAP_Waterfall_Patient_"
        f"{sample_index}.pdf"
    )


    waterfall_png = (
        f"{stage}-{pathology_name}_"
        f"SHAP_Waterfall_Patient_"
        f"{sample_index}.png"
    )


    save_shap_waterfall_plot(
        shap_values_sample=sample_shap_values,
        base_value=base_value,
        features_sample=sample_features.values,
        feature_names=FINAL_FEATURES,
        pdf_filename=waterfall_pdf,
        png_filename=waterfall_png
    )


# ============================================================
# 21. Final summary
# ============================================================

print("\n============================================================")
print("STAGE 15 + STAGE 16 COMPLETED")
print("============================================================")


print(
    f"\nFinal model: "
    f"7-feature Random Forest"
)


print(
    f"Internal-validation AUC = "
    f"{auc:.4f}"
)


print("\nStage 15:")

print(
    "Patient position 271 = NDKD"
)


print("\nStage 16:")

print(
    "Patient position 273 = DKD"
)


print("\nExpected output files:")


print(
    "15-NDKD_SHAP_Force_Patient_271.html"
)

print(
    "15-NDKD_SHAP_Waterfall_Patient_271.pdf"
)

print(
    "15-NDKD_SHAP_Waterfall_Patient_271.png"
)

print(
    "15-NDKD_SHAP_Patient_271_Source_Data.xlsx"
)


print(
    "\n16-DKD_SHAP_Force_Patient_273.html"
)

print(
    "16-DKD_SHAP_Waterfall_Patient_273.pdf"
)

print(
    "16-DKD_SHAP_Waterfall_Patient_273.png"
)

print(
    "16-DKD_SHAP_Patient_273_Source_Data.xlsx"
)


print(
    "\nFont: Arial"
)

print(
    "Waterfall PDF files are vector graphics."
)
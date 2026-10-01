# ============================================================
# STAGE 10
# Formal benchmark:
# Final 7-feature Random Forest vs Logistic Regression
#
# Purpose:
# Reviewer #3 requested a formal statistical comparison
# between RF and logistic regression.
#
# IMPORTANT:
# 1. Both models use EXACTLY the same 7 predictors.
# 2. Both models use EXACTLY the same development/internal
#    validation split.
# 3. Both models use the same preprocessing.
# 4. No feature selection is repeated here.
# 5. No threshold optimization is performed here.
# 6. Primary statistical comparison = paired DeLong test.
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

from scipy.stats import norm


# ============================================================
# 1. Basic settings
# ============================================================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 45

DATA_FILE = "test1.xlsx"

TARGET = "Pathology type"


# ============================================================
# 2. Final 7 predictors
#
# Must remain identical to the locked final model.
# ============================================================

FINAL_FEATURES = [
    "Serum creatinine",
    "DR",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


# Continuous predictors requiring mean imputation
CONTINUOUS_FEATURES = [
    "Serum creatinine",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


# DR is categorical/binary and had no missing values
CATEGORICAL_FEATURES = [
    "DR"
]


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_excel(DATA_FILE)

print("\n============================================================")
print("STAGE 10")
print("Final 7-feature RF vs Logistic Regression")
print("============================================================")

print(f"\nData file: {DATA_FILE}")
print(f"Total N = {len(df)}")


# ============================================================
# 4. Check required columns
# ============================================================

required_columns = FINAL_FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) > 0:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. Raw X and y
# ============================================================

X = df[FINAL_FEATURES].copy()

y = df[TARGET].astype(int).values


# ============================================================
# 6. EXACT same 80/20 stratified split
#
# random_state = 45
# stratify = y
# ============================================================

X_dev_raw, X_val_raw, y_dev, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")

print(f"Development N = {len(y_dev)}")
print(f"Internal validation N = {len(y_val)}")

print(
    f"Development DKD = "
    f"{np.sum(y_dev == 1)}"
)

print(
    f"Development NDKD = "
    f"{np.sum(y_dev == 0)}"
)

print(
    f"Internal validation DKD = "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Internal validation NDKD = "
    f"{np.sum(y_val == 0)}"
)


# ============================================================
# 7. Mean imputation
#
# Fit ONLY on development data.
# Apply same imputer to validation data.
# ============================================================

imputer = SimpleImputer(
    strategy="mean"
)

X_dev_cont = pd.DataFrame(
    imputer.fit_transform(
        X_dev_raw[CONTINUOUS_FEATURES]
    ),
    columns=CONTINUOUS_FEATURES,
    index=X_dev_raw.index
)

X_val_cont = pd.DataFrame(
    imputer.transform(
        X_val_raw[CONTINUOUS_FEATURES]
    ),
    columns=CONTINUOUS_FEATURES,
    index=X_val_raw.index
)


# ============================================================
# 8. Add DR back
# ============================================================

X_dev_processed = X_dev_cont.copy()
X_val_processed = X_val_cont.copy()

for col in CATEGORICAL_FEATURES:

    X_dev_processed[col] = (
        X_dev_raw[col].values
    )

    X_val_processed[col] = (
        X_val_raw[col].values
    )


# ============================================================
# 9. Restore exact final feature order
# ============================================================

X_dev_processed = X_dev_processed[
    FINAL_FEATURES
]

X_val_processed = X_val_processed[
    FINAL_FEATURES
]


# ============================================================
# 10. Standardization
#
# Same preprocessing framework as final pipeline.
#
# Fit scaler on development data only.
# ============================================================

scaler = StandardScaler()

X_dev_scaled = scaler.fit_transform(
    X_dev_processed
)

X_val_scaled = scaler.transform(
    X_val_processed
)


# ============================================================
# 11. Random Forest
#
# Same specification as final RF model.
# ============================================================

rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE
)

rf_model.fit(
    X_dev_scaled,
    y_dev
)

rf_prob = rf_model.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 12. Logistic Regression
#
# Same predictors, same patients, same preprocessing.
#
# No class weighting.
# ============================================================

lr_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

lr_model.fit(
    X_dev_scaled,
    y_dev
)

lr_prob = lr_model.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 13. Calculate AUCs
# ============================================================

rf_auc = roc_auc_score(
    y_val,
    rf_prob
)

lr_auc = roc_auc_score(
    y_val,
    lr_prob
)

delta_auc = rf_auc - lr_auc


# ============================================================
# 14. Bootstrap 95% CI
#
# Same bootstrap samples are used for RF and LR.
# This also allows paired bootstrap CI for ΔAUC.
# ============================================================

N_BOOTSTRAP = 2000

rng = np.random.RandomState(
    RANDOM_STATE
)

rf_boot_auc = []
lr_boot_auc = []
delta_boot_auc = []

n_val = len(y_val)


for i in range(N_BOOTSTRAP):

    indices = rng.choice(
        np.arange(n_val),
        size=n_val,
        replace=True
    )

    y_boot = y_val[indices]

    # Skip bootstrap samples containing only one class
    if len(np.unique(y_boot)) < 2:
        continue

    rf_boot = roc_auc_score(
        y_boot,
        rf_prob[indices]
    )

    lr_boot = roc_auc_score(
        y_boot,
        lr_prob[indices]
    )

    rf_boot_auc.append(
        rf_boot
    )

    lr_boot_auc.append(
        lr_boot
    )

    delta_boot_auc.append(
        rf_boot - lr_boot
    )


rf_boot_auc = np.array(
    rf_boot_auc
)

lr_boot_auc = np.array(
    lr_boot_auc
)

delta_boot_auc = np.array(
    delta_boot_auc
)


rf_ci_lower = np.percentile(
    rf_boot_auc,
    2.5
)

rf_ci_upper = np.percentile(
    rf_boot_auc,
    97.5
)


lr_ci_lower = np.percentile(
    lr_boot_auc,
    2.5
)

lr_ci_upper = np.percentile(
    lr_boot_auc,
    97.5
)


delta_ci_lower = np.percentile(
    delta_boot_auc,
    2.5
)

delta_ci_upper = np.percentile(
    delta_boot_auc,
    97.5
)


# ============================================================
# 15. DeLong functions
# ============================================================

def compute_midrank(x):

    J = np.argsort(x)

    Z = x[J]

    N = len(x)

    T = np.zeros(N, dtype=float)

    i = 0

    while i < N:

        j = i

        while (
            j < N
            and Z[j] == Z[i]
        ):
            j += 1

        T[i:j] = 0.5 * (
            i + j - 1
        )

        i = j

    T2 = np.empty(N, dtype=float)

    T2[J] = T + 1

    return T2


def fast_delong(
    predictions_sorted_transposed,
    label_1_count
):

    m = label_1_count

    n = (
        predictions_sorted_transposed.shape[1]
        - m
    )

    positive_examples = (
        predictions_sorted_transposed[
            :,
            :m
        ]
    )

    negative_examples = (
        predictions_sorted_transposed[
            :,
            m:
        ]
    )

    k = (
        predictions_sorted_transposed.shape[0]
    )

    tx = np.empty(
        [k, m]
    )

    ty = np.empty(
        [k, n]
    )

    tz = np.empty(
        [k, m + n]
    )

    for r in range(k):

        tx[r, :] = compute_midrank(
            positive_examples[r, :]
        )

        ty[r, :] = compute_midrank(
            negative_examples[r, :]
        )

        tz[r, :] = compute_midrank(
            predictions_sorted_transposed[
                r,
                :
            ]
        )

    aucs = (
        tz[:, :m].sum(axis=1)
        / m
        / n
        -
        (m + 1.0)
        / 2.0
        / n
    )

    v01 = (
        tz[:, :m]
        -
        tx
    ) / n

    v10 = (
        1.0
        -
        (
            tz[:, m:]
            -
            ty
        )
        / m
    )

    sx = np.cov(v01)
    sy = np.cov(v10)

    delong_cov = (
        sx / m
        +
        sy / n
    )

    return aucs, delong_cov


def paired_delong_test(
    y_true,
    pred_rf,
    pred_lr
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    pred_rf = np.asarray(
        pred_rf
    ).astype(float)

    pred_lr = np.asarray(
        pred_lr
    ).astype(float)

    # Positive cases first
    order = np.argsort(
        -y_true
    )

    label_1_count = int(
        np.sum(y_true == 1)
    )

    predictions_sorted = np.vstack(
        [
            pred_rf,
            pred_lr
        ]
    )[:, order]

    aucs, covariance = fast_delong(
        predictions_sorted,
        label_1_count
    )

    contrast = np.array(
        [1.0, -1.0]
    )

    variance = (
        contrast
        @ covariance
        @ contrast.T
    )

    variance = float(
        np.asarray(
            variance
        ).item()
    )

    if variance <= 0:

        z_value = np.nan
        p_value = np.nan

    else:

        z_value = (
            np.abs(
                aucs[0] - aucs[1]
            )
            /
            np.sqrt(
                variance
            )
        )

        p_value = (
            2
            *
            norm.sf(
                z_value
            )
        )

    return (
        aucs[0],
        aucs[1],
        aucs[0] - aucs[1],
        z_value,
        p_value
    )


# ============================================================
# 16. Paired DeLong test
# ============================================================

(
    delong_rf_auc,
    delong_lr_auc,
    delong_delta_auc,
    delong_z,
    delong_p
) = paired_delong_test(
    y_val,
    rf_prob,
    lr_prob
)


# ============================================================
# 17. Print results
# ============================================================

print("\n============================================================")
print("RF vs LOGISTIC REGRESSION")
print("============================================================")

print(
    f"Random Forest AUC = "
    f"{rf_auc:.2f} "
    f"(95% CI, "
    f"{rf_ci_lower:.2f}–"
    f"{rf_ci_upper:.2f})"
)

print(
    f"Logistic Regression AUC = "
    f"{lr_auc:.2f} "
    f"(95% CI, "
    f"{lr_ci_lower:.2f}–"
    f"{lr_ci_upper:.2f})"
)

print(
    f"ΔAUC (RF - LR) = "
    f"{delta_auc:.3f}"
)

print(
    f"Paired bootstrap 95% CI for ΔAUC = "
    f"{delta_ci_lower:.3f} to "
    f"{delta_ci_upper:.3f}"
)

if np.isnan(delong_p):

    print(
        "Paired DeLong P = NA"
    )

elif delong_p < 0.001:

    print(
        "Paired DeLong P < 0.001"
    )

else:

    print(
        f"Paired DeLong P = "
        f"{delong_p:.3f}"
    )


# ============================================================
# 18. Sanity check
#
# The RF AUC should reproduce the final internal validation
# AUC (~0.894).
# ============================================================

print("\n============================================================")
print("SANITY CHECK")
print("============================================================")

print(
    f"Expected final RF internal AUC "
    f"≈ 0.894"
)

print(
    f"Current RF internal AUC = "
    f"{rf_auc:.4f}"
)

if abs(
    rf_auc - 0.8941
) <= 0.005:

    print(
        "RF reproduction check: PASS"
    )

else:

    print(
        "RF reproduction check: CHECK PIPELINE"
    )


# ============================================================
# 19. ROC curves
# ============================================================

rf_fpr, rf_tpr, rf_thresholds = roc_curve(
    y_val,
    rf_prob
)

lr_fpr, lr_tpr, lr_thresholds = roc_curve(
    y_val,
    lr_prob
)


fig, ax = plt.subplots(
    figsize=(7.5, 7),
    dpi=300
)

ax.plot(
    rf_fpr,
    rf_tpr,
    linewidth=2.3,
    label=(
        "Random Forest "
        f"(AUC = {rf_auc:.2f})"
    )
)

ax.plot(
    lr_fpr,
    lr_tpr,
    linewidth=2.3,
    label=(
        "Logistic Regression "
        f"(AUC = {lr_auc:.2f})"
    )
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    label="Reference"
)

ax.set_xlabel(
    "1 - Specificity",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Sensitivity",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "Random Forest vs Logistic Regression",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.25
)

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# DeLong annotation
# ------------------------------------------------------------

if np.isnan(delong_p):

    p_text = "DeLong P = NA"

elif delong_p < 0.001:

    p_text = "DeLong P < 0.001"

else:

    p_text = (
        f"DeLong P = "
        f"{delong_p:.3f}"
    )


comparison_text = (
    f"ΔAUC (RF − LR) = "
    f"{delta_auc:.3f}\n"
    f"{p_text}"
)


ax.text(
    0.55,
    0.20,
    comparison_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)


ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "10-RF_vs_LR_Internal_Validation_ROC.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "10-RF_vs_LR_Internal_Validation_ROC.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. Save prediction-level data
# ============================================================

prediction_output_df = pd.DataFrame(
    {
        "True_Label":
            y_val,

        "RF_Predicted_Probability":
            rf_prob,

        "LR_Predicted_Probability":
            lr_prob
    },
    index=X_val_raw.index
)

prediction_output_df.to_excel(
    "10-RF_vs_LR_Internal_Validation_Predictions.xlsx",
    index=True
)


# ============================================================
# 21. Save raw performance table
# ============================================================

performance_df = pd.DataFrame(
    [
        {
            "Model":
                "Random Forest",

            "N":
                len(y_val),

            "AUC":
                rf_auc,

            "AUC_CI_Lower":
                rf_ci_lower,

            "AUC_CI_Upper":
                rf_ci_upper,

            "Delta_AUC_vs_RF":
                0.0,

            "DeLong_P_vs_RF":
                np.nan
        },

        {
            "Model":
                "Logistic Regression",

            "N":
                len(y_val),

            "AUC":
                lr_auc,

            "AUC_CI_Lower":
                lr_ci_lower,

            "AUC_CI_Upper":
                lr_ci_upper,

            "Delta_AUC_vs_RF":
                lr_auc - rf_auc,

            "DeLong_P_vs_RF":
                delong_p
        }
    ]
)

performance_df.to_excel(
    "10-RF_vs_LR_Internal_Validation_Performance.xlsx",
    index=False
)


# ============================================================
# 22. Save paired comparison statistics
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "Comparison":
                "Random Forest vs Logistic Regression",

            "RF_AUC":
                rf_auc,

            "LR_AUC":
                lr_auc,

            "Delta_AUC_RF_minus_LR":
                delta_auc,

            "Bootstrap_Delta_CI_Lower":
                delta_ci_lower,

            "Bootstrap_Delta_CI_Upper":
                delta_ci_upper,

            "DeLong_Z":
                delong_z,

            "DeLong_P":
                delong_p
        }
    ]
)

comparison_df.to_excel(
    "10-RF_vs_LR_Paired_Comparison.xlsx",
    index=False
)


# ============================================================
# 23. Save ROC source data
# ============================================================

rf_roc_df = pd.DataFrame(
    {
        "Model":
            "Random Forest",

        "FPR":
            rf_fpr,

        "TPR":
            rf_tpr,

        "Threshold":
            rf_thresholds
    }
)


lr_roc_df = pd.DataFrame(
    {
        "Model":
            "Logistic Regression",

        "FPR":
            lr_fpr,

        "TPR":
            lr_tpr,

        "Threshold":
            lr_thresholds
    }
)


roc_source_df = pd.concat(
    [
        rf_roc_df,
        lr_roc_df
    ],
    ignore_index=True
)


roc_source_df.to_excel(
    "10-RF_vs_LR_ROC_Source_Data.xlsx",
    index=False
)


# ============================================================
# 24. Manuscript-formatted table
# ============================================================

if np.isnan(delong_p):

    delong_p_formatted = "NA"

elif delong_p < 0.001:

    delong_p_formatted = "<0.001"

else:

    delong_p_formatted = (
        f"{delong_p:.3f}"
    )


manuscript_df = pd.DataFrame(
    [
        {
            "Model":
                "Random Forest",

            "Predictors":
                "Same 7 predictors",

            "AUC (95% CI)":
                (
                    f"{rf_auc:.2f} "
                    f"({rf_ci_lower:.2f}–"
                    f"{rf_ci_upper:.2f})"
                ),

            "ΔAUC vs RF":
                "Reference",

            "DeLong P":
                "—"
        },

        {
            "Model":
                "Logistic Regression",

            "Predictors":
                "Same 7 predictors",

            "AUC (95% CI)":
                (
                    f"{lr_auc:.2f} "
                    f"({lr_ci_lower:.2f}–"
                    f"{lr_ci_upper:.2f})"
                ),

            "ΔAUC vs RF":
                f"{lr_auc - rf_auc:.3f}",

            "DeLong P":
                delong_p_formatted
        }
    ]
)


manuscript_df.to_excel(
    "10-RF_vs_LR_Manuscript_Table.xlsx",
    index=False
)


# ============================================================
# 25. Final summary
# ============================================================

print("\n============================================================")
print("STAGE 10 COMPLETED")
print("============================================================")

print("\nSame 7 predictors:")
for feature in FINAL_FEATURES:
    print(f"  - {feature}")

print("\nInternal validation:")

print(
    f"RF AUC = "
    f"{rf_auc:.2f} "
    f"(95% CI, "
    f"{rf_ci_lower:.2f}–"
    f"{rf_ci_upper:.2f})"
)

print(
    f"LR AUC = "
    f"{lr_auc:.2f} "
    f"(95% CI, "
    f"{lr_ci_lower:.2f}–"
    f"{lr_ci_upper:.2f})"
)

print(
    f"ΔAUC (RF - LR) = "
    f"{delta_auc:.3f}"
)

print(
    f"Bootstrap 95% CI for ΔAUC = "
    f"{delta_ci_lower:.3f} to "
    f"{delta_ci_upper:.3f}"
)

if np.isnan(delong_p):

    print("Paired DeLong P = NA")

elif delong_p < 0.001:

    print("Paired DeLong P < 0.001")

else:

    print(
        f"Paired DeLong P = "
        f"{delong_p:.3f}"
    )

print("\nSaved files:")

print(
    "10-RF_vs_LR_Internal_Validation_ROC.pdf"
)

print(
    "10-RF_vs_LR_Internal_Validation_ROC.png"
)

print(
    "10-RF_vs_LR_Internal_Validation_Predictions.xlsx"
)

print(
    "10-RF_vs_LR_Internal_Validation_Performance.xlsx"
)

print(
    "10-RF_vs_LR_Paired_Comparison.xlsx"
)

print(
    "10-RF_vs_LR_ROC_Source_Data.xlsx"
)

print(
    "10-RF_vs_LR_Manuscript_Table.xlsx"
)
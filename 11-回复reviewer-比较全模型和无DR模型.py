# ============================================================
# STAGE 11
# Reviewer #3 - Ablation / Clinical Benchmark
#
# Comparisons:
# 1. Full 7-feature Random Forest
# 2. 6-feature Random Forest without DR
# 3. Simple clinical logistic benchmark:
#       DR + Duration of DM + ACR + eGFR
#
# Primary comparison:
# Paired DeLong test on the SAME internal validation cohort.
#
# IMPORTANT:
# - Same 80/20 split
# - random_state = 45
# - Training-only imputation/scaling
# - No feature selection is repeated
# - No threshold optimization in validation
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
# 1. Settings
# ============================================================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 45
N_BOOTSTRAP = 2000

DATA_FILE = "test1.xlsx"
TARGET = "Pathology type"


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_excel(DATA_FILE)

print("\n============================================================")
print("STAGE 11")
print("Full RF vs RF without DR vs Clinical Benchmark")
print("============================================================")

print(f"\nData file: {DATA_FILE}")
print(f"Total N = {len(df)}")


# ============================================================
# 3. Resolve ACR / eGFR column names
#
# Your previous dataset used ACR.
# eGFR was present before correlation filtering.
# ============================================================

def resolve_column(df, candidates, label):

    for col in candidates:

        if col in df.columns:
            return col

    raise ValueError(
        f"\nCould not find {label} column.\n"
        f"Tried: {candidates}\n\n"
        f"Available columns:\n"
        f"{df.columns.tolist()}"
    )


ACR_COL = resolve_column(
    df,
    [
        "ACR",
        "UACR",
        "Urine albumin creatinine ratio",
        "Urinary albumin creatinine ratio"
    ],
    "albuminuria/ACR"
)


EGFR_COL = resolve_column(
    df,
    [
        "eGFR",
        "EGFR",
        "Egfr",
        "Estimated GFR",
        "Estimated glomerular filtration rate"
    ],
    "eGFR"
)


print(f"\nACR column detected: {ACR_COL}")
print(f"eGFR column detected: {EGFR_COL}")


# ============================================================
# 4. Define model predictor sets
# ============================================================

FULL_RF_FEATURES = [
    "Serum creatinine",
    "DR",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


NO_DR_FEATURES = [
    "Serum creatinine",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


CLINICAL_FEATURES = [
    "DR",
    "Duration of DM",
    ACR_COL,
    EGFR_COL
]


print("\nFull RF predictors:")
for x in FULL_RF_FEATURES:
    print(f"  - {x}")

print("\nRF without DR predictors:")
for x in NO_DR_FEATURES:
    print(f"  - {x}")

print("\nSimple clinical benchmark predictors:")
for x in CLINICAL_FEATURES:
    print(f"  - {x}")


# ============================================================
# 5. Check required columns
# ============================================================

required_columns = list(
    dict.fromkeys(
        FULL_RF_FEATURES
        + CLINICAL_FEATURES
        + [TARGET]
    )
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) > 0:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 6. IMPORTANT:
# Create the split using a master X dataframe containing all
# variables required by the three models.
#
# This guarantees EXACTLY the same patients in all models.
# ============================================================

MASTER_FEATURES = list(
    dict.fromkeys(
        FULL_RF_FEATURES
        + CLINICAL_FEATURES
    )
)

X_master = df[
    MASTER_FEATURES
].copy()

y = df[
    TARGET
].astype(int).values


X_dev_raw, X_val_raw, y_dev, y_val = train_test_split(
    X_master,
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
    f"Internal DKD = "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Internal NDKD = "
    f"{np.sum(y_val == 0)}"
)


# ============================================================
# 7. Helper:
# preprocess one predictor set
#
# DR is binary and does not require scaling/imputation.
# All other variables are treated as continuous.
#
# Imputer/scaler are fit ONLY on development data.
# ============================================================

def preprocess_features(
    X_dev_raw,
    X_val_raw,
    feature_list
):

    categorical_features = [
        col for col in feature_list
        if col == "DR"
    ]

    continuous_features = [
        col for col in feature_list
        if col != "DR"
    ]


    # --------------------------------------------------------
    # Mean imputation for continuous variables
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="mean"
    )

    X_dev_cont = pd.DataFrame(
        imputer.fit_transform(
            X_dev_raw[
                continuous_features
            ]
        ),
        columns=continuous_features,
        index=X_dev_raw.index
    )

    X_val_cont = pd.DataFrame(
        imputer.transform(
            X_val_raw[
                continuous_features
            ]
        ),
        columns=continuous_features,
        index=X_val_raw.index
    )


    # --------------------------------------------------------
    # Add categorical variable DR
    # --------------------------------------------------------

    X_dev_processed = X_dev_cont.copy()
    X_val_processed = X_val_cont.copy()

    for col in categorical_features:

        X_dev_processed[col] = (
            X_dev_raw[col].values
        )

        X_val_processed[col] = (
            X_val_raw[col].values
        )


    # --------------------------------------------------------
    # Restore exact requested feature order
    # --------------------------------------------------------

    X_dev_processed = (
        X_dev_processed[
            feature_list
        ]
    )

    X_val_processed = (
        X_val_processed[
            feature_list
        ]
    )


    # --------------------------------------------------------
    # Scaling
    #
    # RF does not require scaling, but retaining the same
    # preprocessing framework reproduces the final pipeline.
    # It is required for LR.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev_processed
    )

    X_val_scaled = scaler.transform(
        X_val_processed
    )


    return (
        X_dev_scaled,
        X_val_scaled,
        imputer,
        scaler
    )


# ============================================================
# 8. FULL 7-FEATURE RF
# ============================================================

(
    X_dev_full,
    X_val_full,
    imputer_full,
    scaler_full
) = preprocess_features(
    X_dev_raw,
    X_val_raw,
    FULL_RF_FEATURES
)


full_rf = RandomForestClassifier(
    random_state=RANDOM_STATE
)

full_rf.fit(
    X_dev_full,
    y_dev
)

full_rf_prob = full_rf.predict_proba(
    X_val_full
)[:, 1]

full_rf_auc = roc_auc_score(
    y_val,
    full_rf_prob
)


# ============================================================
# 9. 6-FEATURE RF WITHOUT DR
# ============================================================

(
    X_dev_nodr,
    X_val_nodr,
    imputer_nodr,
    scaler_nodr
) = preprocess_features(
    X_dev_raw,
    X_val_raw,
    NO_DR_FEATURES
)


nodr_rf = RandomForestClassifier(
    random_state=RANDOM_STATE
)

nodr_rf.fit(
    X_dev_nodr,
    y_dev
)

nodr_rf_prob = nodr_rf.predict_proba(
    X_val_nodr
)[:, 1]

nodr_rf_auc = roc_auc_score(
    y_val,
    nodr_rf_prob
)


# ============================================================
# 10. SIMPLE CLINICAL LOGISTIC BENCHMARK
#
# DR + DM duration + ACR + eGFR
# ============================================================

(
    X_dev_clinical,
    X_val_clinical,
    imputer_clinical,
    scaler_clinical
) = preprocess_features(
    X_dev_raw,
    X_val_raw,
    CLINICAL_FEATURES
)


clinical_lr = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

clinical_lr.fit(
    X_dev_clinical,
    y_dev
)

clinical_prob = clinical_lr.predict_proba(
    X_val_clinical
)[:, 1]

clinical_auc = roc_auc_score(
    y_val,
    clinical_prob
)


# ============================================================
# 11. Bootstrap CI helper
# ============================================================

def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_bootstrap=2000,
    random_state=45
):

    rng = np.random.RandomState(
        random_state
    )

    boot_aucs = []

    n = len(y_true)

    for _ in range(n_bootstrap):

        indices = rng.choice(
            np.arange(n),
            size=n,
            replace=True
        )

        y_boot = y_true[
            indices
        ]

        if len(
            np.unique(y_boot)
        ) < 2:
            continue

        auc_boot = roc_auc_score(
            y_boot,
            y_prob[indices]
        )

        boot_aucs.append(
            auc_boot
        )


    boot_aucs = np.array(
        boot_aucs
    )

    lower = np.percentile(
        boot_aucs,
        2.5
    )

    upper = np.percentile(
        boot_aucs,
        97.5
    )

    return lower, upper


# ============================================================
# 12. Bootstrap CIs
# ============================================================

full_ci_lower, full_ci_upper = (
    bootstrap_auc_ci(
        y_val,
        full_rf_prob,
        N_BOOTSTRAP,
        RANDOM_STATE
    )
)

nodr_ci_lower, nodr_ci_upper = (
    bootstrap_auc_ci(
        y_val,
        nodr_rf_prob,
        N_BOOTSTRAP,
        RANDOM_STATE
    )
)

clinical_ci_lower, clinical_ci_upper = (
    bootstrap_auc_ci(
        y_val,
        clinical_prob,
        N_BOOTSTRAP,
        RANDOM_STATE
    )
)


# ============================================================
# 13. DeLong functions
# ============================================================

def compute_midrank(x):

    J = np.argsort(x)

    Z = x[J]

    N = len(x)

    T = np.zeros(
        N,
        dtype=float
    )

    i = 0

    while i < N:

        j = i

        while (
            j < N
            and Z[j] == Z[i]
        ):
            j += 1

        T[i:j] = (
            0.5
            *
            (i + j - 1)
        )

        i = j

    T2 = np.empty(
        N,
        dtype=float
    )

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
            positive_examples[
                r,
                :
            ]
        )

        ty[r, :] = compute_midrank(
            negative_examples[
                r,
                :
            ]
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


    sx = np.cov(
        v01
    )

    sy = np.cov(
        v10
    )


    delong_cov = (
        sx / m
        +
        sy / n
    )


    return (
        aucs,
        delong_cov
    )


def paired_delong_test(
    y_true,
    pred_1,
    pred_2
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    pred_1 = np.asarray(
        pred_1
    ).astype(float)

    pred_2 = np.asarray(
        pred_2
    ).astype(float)


    order = np.argsort(
        -y_true
    )

    label_1_count = int(
        np.sum(
            y_true == 1
        )
    )


    predictions_sorted = np.vstack(
        [
            pred_1,
            pred_2
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
                aucs[0]
                -
                aucs[1]
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


    return {
        "auc_1":
            aucs[0],

        "auc_2":
            aucs[1],

        "delta_auc":
            aucs[0] - aucs[1],

        "z":
            z_value,

        "p":
            p_value
    }


# ============================================================
# 14. Paired DeLong comparisons
#
# Full RF is reference.
# ============================================================

delong_nodr = paired_delong_test(
    y_val,
    full_rf_prob,
    nodr_rf_prob
)


delong_clinical = paired_delong_test(
    y_val,
    full_rf_prob,
    clinical_prob
)


# ============================================================
# 15. Paired bootstrap CI for ΔAUC
# ============================================================

def paired_bootstrap_delta_ci(
    y_true,
    reference_prob,
    comparison_prob,
    n_bootstrap=2000,
    random_state=45
):

    rng = np.random.RandomState(
        random_state
    )

    delta_values = []

    n = len(y_true)


    for _ in range(n_bootstrap):

        indices = rng.choice(
            np.arange(n),
            size=n,
            replace=True
        )

        y_boot = y_true[
            indices
        ]

        if len(
            np.unique(y_boot)
        ) < 2:
            continue


        ref_auc = roc_auc_score(
            y_boot,
            reference_prob[
                indices
            ]
        )

        comp_auc = roc_auc_score(
            y_boot,
            comparison_prob[
                indices
            ]
        )


        delta_values.append(
            ref_auc - comp_auc
        )


    delta_values = np.array(
        delta_values
    )


    lower = np.percentile(
        delta_values,
        2.5
    )

    upper = np.percentile(
        delta_values,
        97.5
    )


    return (
        lower,
        upper
    )


nodr_delta_lower, nodr_delta_upper = (
    paired_bootstrap_delta_ci(
        y_val,
        full_rf_prob,
        nodr_rf_prob,
        N_BOOTSTRAP,
        RANDOM_STATE
    )
)


clinical_delta_lower, clinical_delta_upper = (
    paired_bootstrap_delta_ci(
        y_val,
        full_rf_prob,
        clinical_prob,
        N_BOOTSTRAP,
        RANDOM_STATE
    )
)


# ============================================================
# 16. Print primary results
# ============================================================

print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")


print(
    f"Full 7-feature RF AUC = "
    f"{full_rf_auc:.2f} "
    f"(95% CI, "
    f"{full_ci_lower:.2f}–"
    f"{full_ci_upper:.2f})"
)


print(
    f"RF without DR AUC = "
    f"{nodr_rf_auc:.2f} "
    f"(95% CI, "
    f"{nodr_ci_lower:.2f}–"
    f"{nodr_ci_upper:.2f})"
)


print(
    f"Clinical benchmark AUC = "
    f"{clinical_auc:.2f} "
    f"(95% CI, "
    f"{clinical_ci_lower:.2f}–"
    f"{clinical_ci_upper:.2f})"
)


print("\n------------------------------------------------------------")
print("Full RF vs RF without DR")
print("------------------------------------------------------------")

print(
    f"ΔAUC = "
    f"{full_rf_auc - nodr_rf_auc:.3f}"
)

print(
    f"Bootstrap 95% CI for ΔAUC = "
    f"{nodr_delta_lower:.3f} to "
    f"{nodr_delta_upper:.3f}"
)

if delong_nodr["p"] < 0.001:

    print(
        "Paired DeLong P < 0.001"
    )

else:

    print(
        f"Paired DeLong P = "
        f"{delong_nodr['p']:.3f}"
    )


print("\n------------------------------------------------------------")
print("Full RF vs Clinical Benchmark")
print("------------------------------------------------------------")

print(
    f"ΔAUC = "
    f"{full_rf_auc - clinical_auc:.3f}"
)

print(
    f"Bootstrap 95% CI for ΔAUC = "
    f"{clinical_delta_lower:.3f} to "
    f"{clinical_delta_upper:.3f}"
)

if delong_clinical["p"] < 0.001:

    print(
        "Paired DeLong P < 0.001"
    )

else:

    print(
        f"Paired DeLong P = "
        f"{delong_clinical['p']:.3f}"
    )


# ============================================================
# 17. Sanity check
#
# Full RF MUST reproduce the final internal AUC.
# ============================================================

print("\n============================================================")
print("SANITY CHECK")
print("============================================================")

print(
    "Expected final RF internal AUC ≈ 0.894"
)

print(
    f"Current full RF AUC = "
    f"{full_rf_auc:.4f}"
)

if abs(
    full_rf_auc - 0.8941
) <= 0.005:

    print(
        "Full RF reproduction check: PASS"
    )

else:

    print(
        "Full RF reproduction check: CHECK PIPELINE"
    )


# ============================================================
# 18. ROC curves
# ============================================================

full_fpr, full_tpr, full_thresholds = roc_curve(
    y_val,
    full_rf_prob
)

nodr_fpr, nodr_tpr, nodr_thresholds = roc_curve(
    y_val,
    nodr_rf_prob
)

clinical_fpr, clinical_tpr, clinical_thresholds = roc_curve(
    y_val,
    clinical_prob
)


fig, ax = plt.subplots(
    figsize=(8, 7),
    dpi=300
)


ax.plot(
    full_fpr,
    full_tpr,
    linewidth=2.4,
    label=(
        "7-feature RF "
        f"(AUC = {full_rf_auc:.2f})"
    )
)


ax.plot(
    nodr_fpr,
    nodr_tpr,
    linewidth=2.2,
    label=(
        "RF without DR "
        f"(AUC = {nodr_rf_auc:.2f})"
    )
)


ax.plot(
    clinical_fpr,
    clinical_tpr,
    linewidth=2.2,
    label=(
        "Clinical benchmark "
        f"(AUC = {clinical_auc:.2f})"
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
    "Ablation and Clinical Benchmark Analysis",
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


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ============================================================
# 19. Statistical annotation
# ============================================================

def format_p(p):

    if np.isnan(p):
        return "NA"

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


comparison_text = (
    "vs 7-feature RF:\n"
    f"Without DR: "
    f"ΔAUC = "
    f"{full_rf_auc - nodr_rf_auc:.3f}, "
    f"P = {format_p(delong_nodr['p'])}\n"
    f"Clinical benchmark: "
    f"ΔAUC = "
    f"{full_rf_auc - clinical_auc:.3f}, "
    f"P = {format_p(delong_clinical['p'])}"
)


ax.text(
    0.42,
    0.22,
    comparison_text,
    transform=ax.transAxes,
    fontsize=10.5,
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
    "11-Ablation_Clinical_Benchmark_ROC.pdf",
    format="pdf",
    bbox_inches="tight"
)


plt.savefig(
    "11-Ablation_Clinical_Benchmark_ROC.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. Save patient-level predictions
# ============================================================

prediction_df = pd.DataFrame(
    {
        "True_Label":
            y_val,

        "Full_7Feature_RF":
            full_rf_prob,

        "RF_Without_DR":
            nodr_rf_prob,

        "Clinical_Benchmark_LR":
            clinical_prob
    },
    index=X_val_raw.index
)


prediction_df.to_excel(
    "11-Ablation_Clinical_Benchmark_Predictions.xlsx",
    index=True
)


# ============================================================
# 21. Raw performance table
# ============================================================

performance_df = pd.DataFrame(
    [
        {
            "Model":
                "Full 7-feature RF",

            "Predictors":
                ", ".join(
                    FULL_RF_FEATURES
                ),

            "AUC":
                full_rf_auc,

            "CI_Lower":
                full_ci_lower,

            "CI_Upper":
                full_ci_upper,

            "Delta_AUC_vs_Full_RF":
                0.0,

            "DeLong_P_vs_Full_RF":
                np.nan
        },

        {
            "Model":
                "RF without DR",

            "Predictors":
                ", ".join(
                    NO_DR_FEATURES
                ),

            "AUC":
                nodr_rf_auc,

            "CI_Lower":
                nodr_ci_lower,

            "CI_Upper":
                nodr_ci_upper,

            "Delta_AUC_vs_Full_RF":
                (
                    nodr_rf_auc
                    -
                    full_rf_auc
                ),

            "DeLong_P_vs_Full_RF":
                delong_nodr["p"]
        },

        {
            "Model":
                "Clinical benchmark LR",

            "Predictors":
                ", ".join(
                    CLINICAL_FEATURES
                ),

            "AUC":
                clinical_auc,

            "CI_Lower":
                clinical_ci_lower,

            "CI_Upper":
                clinical_ci_upper,

            "Delta_AUC_vs_Full_RF":
                (
                    clinical_auc
                    -
                    full_rf_auc
                ),

            "DeLong_P_vs_Full_RF":
                delong_clinical["p"]
        }
    ]
)


performance_df.to_excel(
    "11-Ablation_Clinical_Benchmark_Performance.xlsx",
    index=False
)


# ============================================================
# 22. Comparison statistics
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "Comparison":
                "Full RF vs RF without DR",

            "Reference_AUC":
                full_rf_auc,

            "Comparison_AUC":
                nodr_rf_auc,

            "Delta_AUC_Reference_minus_Comparison":
                (
                    full_rf_auc
                    -
                    nodr_rf_auc
                ),

            "Bootstrap_Delta_CI_Lower":
                nodr_delta_lower,

            "Bootstrap_Delta_CI_Upper":
                nodr_delta_upper,

            "DeLong_P":
                delong_nodr["p"]
        },

        {
            "Comparison":
                "Full RF vs Clinical benchmark",

            "Reference_AUC":
                full_rf_auc,

            "Comparison_AUC":
                clinical_auc,

            "Delta_AUC_Reference_minus_Comparison":
                (
                    full_rf_auc
                    -
                    clinical_auc
                ),

            "Bootstrap_Delta_CI_Lower":
                clinical_delta_lower,

            "Bootstrap_Delta_CI_Upper":
                clinical_delta_upper,

            "DeLong_P":
                delong_clinical["p"]
        }
    ]
)


comparison_df.to_excel(
    "11-Ablation_Clinical_Benchmark_Comparisons.xlsx",
    index=False
)


# ============================================================
# 23. ROC source data
# ============================================================

full_roc_df = pd.DataFrame(
    {
        "Model":
            "Full 7-feature RF",

        "FPR":
            full_fpr,

        "TPR":
            full_tpr,

        "Threshold":
            full_thresholds
    }
)


nodr_roc_df = pd.DataFrame(
    {
        "Model":
            "RF without DR",

        "FPR":
            nodr_fpr,

        "TPR":
            nodr_tpr,

        "Threshold":
            nodr_thresholds
    }
)


clinical_roc_df = pd.DataFrame(
    {
        "Model":
            "Clinical benchmark LR",

        "FPR":
            clinical_fpr,

        "TPR":
            clinical_tpr,

        "Threshold":
            clinical_thresholds
    }
)


roc_source_df = pd.concat(
    [
        full_roc_df,
        nodr_roc_df,
        clinical_roc_df
    ],
    ignore_index=True
)


roc_source_df.to_excel(
    "11-Ablation_Clinical_Benchmark_ROC_Source_Data.xlsx",
    index=False
)


# ============================================================
# 24. Manuscript-formatted table
# ============================================================

manuscript_df = pd.DataFrame(
    [
        {
            "Model":
                "Full 7-feature RF",

            "AUC (95% CI)":
                (
                    f"{full_rf_auc:.2f} "
                    f"({full_ci_lower:.2f}–"
                    f"{full_ci_upper:.2f})"
                ),

            "ΔAUC vs full RF":
                "Reference",

            "DeLong P":
                "—"
        },

        {
            "Model":
                "RF without DR",

            "AUC (95% CI)":
                (
                    f"{nodr_rf_auc:.2f} "
                    f"({nodr_ci_lower:.2f}–"
                    f"{nodr_ci_upper:.2f})"
                ),

            "ΔAUC vs full RF":
                (
                    f"{nodr_rf_auc - full_rf_auc:.3f}"
                ),

            "DeLong P":
                format_p(
                    delong_nodr["p"]
                )
        },

        {
            "Model":
                "Clinical benchmark LR",

            "AUC (95% CI)":
                (
                    f"{clinical_auc:.2f} "
                    f"({clinical_ci_lower:.2f}–"
                    f"{clinical_ci_upper:.2f})"
                ),

            "ΔAUC vs full RF":
                (
                    f"{clinical_auc - full_rf_auc:.3f}"
                ),

            "DeLong P":
                format_p(
                    delong_clinical["p"]
                )
        }
    ]
)


manuscript_df.to_excel(
    "11-Ablation_Clinical_Benchmark_Manuscript_Table.xlsx",
    index=False
)


# ============================================================
# 25. Final summary
# ============================================================

print("\n============================================================")
print("STAGE 11 COMPLETED")
print("============================================================")


print("\nFull 7-feature RF:")

print(
    f"AUC = "
    f"{full_rf_auc:.2f} "
    f"(95% CI, "
    f"{full_ci_lower:.2f}–"
    f"{full_ci_upper:.2f})"
)


print("\nRF without DR:")

print(
    f"AUC = "
    f"{nodr_rf_auc:.2f} "
    f"(95% CI, "
    f"{nodr_ci_lower:.2f}–"
    f"{nodr_ci_upper:.2f})"
)

print(
    f"ΔAUC vs full RF = "
    f"{full_rf_auc - nodr_rf_auc:.3f}"
)

print(
    f"DeLong P = "
    f"{format_p(delong_nodr['p'])}"
)


print("\nClinical benchmark:")

print(
    "Predictors = "
    "DR + Duration of DM + "
    f"{ACR_COL} + {EGFR_COL}"
)

print(
    f"AUC = "
    f"{clinical_auc:.2f} "
    f"(95% CI, "
    f"{clinical_ci_lower:.2f}–"
    f"{clinical_ci_upper:.2f})"
)

print(
    f"ΔAUC vs full RF = "
    f"{full_rf_auc - clinical_auc:.3f}"
)

print(
    f"DeLong P = "
    f"{format_p(delong_clinical['p'])}"
)


print("\nSaved files:")

print(
    "11-Ablation_Clinical_Benchmark_ROC.pdf"
)

print(
    "11-Ablation_Clinical_Benchmark_ROC.png"
)

print(
    "11-Ablation_Clinical_Benchmark_Predictions.xlsx"
)

print(
    "11-Ablation_Clinical_Benchmark_Performance.xlsx"
)

print(
    "11-Ablation_Clinical_Benchmark_Comparisons.xlsx"
)

print(
    "11-Ablation_Clinical_Benchmark_ROC_Source_Data.xlsx"
)

print(
    "11-Ablation_Clinical_Benchmark_Manuscript_Table.xlsx"
)
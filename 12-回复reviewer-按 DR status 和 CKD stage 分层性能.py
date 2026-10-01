# ============================================================
# STAGE 12 - FINAL
# Reviewer #3:
# Subgroup performance by DR status and renal function
#
# Final model:
# 7-feature Random Forest
#
# Subgroups retained for final manuscript:
#
# DR status:
#   1. DR absent
#   2. DR present
#
# Renal-function strata:
#   1. eGFR >= 60 mL/min/1.73 m^2
#   2. eGFR 30-59 mL/min/1.73 m^2
#   3. eGFR < 30 mL/min/1.73 m^2
#
# IMPORTANT:
# 1. The final RF model is trained ONCE using the full
#    development cohort.
# 2. The same model is applied to the full internal
#    validation cohort.
# 3. Subgroups are defined AFTER prediction.
# 4. No subgroup-specific model fitting.
# 5. No subgroup-specific threshold optimization.
# 6. Locked DKD classification threshold = 0.45.
# 7. No G1/G2/G3a/G3b/G4/G5 analysis is generated.
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

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. Basic settings
# ============================================================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 45
N_BOOTSTRAP = 2000

DATA_FILE = "test1.xlsx"

TARGET = "Pathology type"

LOCKED_THRESHOLD = 0.45


# ============================================================
# 2. Final 7 predictors
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


CONTINUOUS_FEATURES = [
    "Serum creatinine",
    "TC",
    "Duration of DM",
    "FBG",
    "Urine protein excretion",
    "LDL"
]


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_excel(
    DATA_FILE
)

print("\n============================================================")
print("STAGE 12 - FINAL")
print("Subgroup Performance by DR Status and eGFR")
print("============================================================")

print(
    f"\nData file: "
    f"{DATA_FILE}"
)

print(
    f"Total N = "
    f"{len(df)}"
)


# ============================================================
# 4. Resolve eGFR column
# ============================================================

def resolve_column(
    df,
    candidates,
    label
):

    for col in candidates:

        if col in df.columns:

            return col

    raise ValueError(
        f"\nCould not find {label} column.\n"
        f"Tried: {candidates}\n\n"
        f"Available columns:\n"
        f"{df.columns.tolist()}"
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


print(
    f"eGFR column detected: "
    f"{EGFR_COL}"
)


# ============================================================
# 5. Check required columns
# ============================================================

required_columns = list(
    dict.fromkeys(
        FINAL_FEATURES
        +
        [EGFR_COL]
        +
        [TARGET]
    )
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
# 6. Create EXACT same 80/20 split
#
# Split patient indices so raw eGFR remains available.
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
    f"Internal DKD = "
    f"{np.sum(y_val == 1)}"
)

print(
    f"Internal NDKD = "
    f"{np.sum(y_val == 0)}"
)


# ============================================================
# 7. Primary-model preprocessing
#
# Continuous-variable mean imputation:
# fit on development only.
#
# DR had no missing values and is added directly.
# ============================================================

imputer = SimpleImputer(
    strategy="mean"
)


X_dev_cont = pd.DataFrame(
    imputer.fit_transform(
        df_dev[
            CONTINUOUS_FEATURES
        ]
    ),
    columns=CONTINUOUS_FEATURES,
    index=df_dev.index
)


X_val_cont = pd.DataFrame(
    imputer.transform(
        df_val[
            CONTINUOUS_FEATURES
        ]
    ),
    columns=CONTINUOUS_FEATURES,
    index=df_val.index
)


# ============================================================
# 8. Add DR and restore final feature order
# ============================================================

X_dev_processed = (
    X_dev_cont.copy()
)

X_val_processed = (
    X_val_cont.copy()
)


X_dev_processed[
    "DR"
] = df_dev[
    "DR"
].values


X_val_processed[
    "DR"
] = df_val[
    "DR"
].values


X_dev_processed = X_dev_processed[
    FINAL_FEATURES
]


X_val_processed = X_val_processed[
    FINAL_FEATURES
]


# ============================================================
# 9. Standardization
#
# Same framework as final pipeline.
# ============================================================

scaler = StandardScaler()


X_dev_scaled = scaler.fit_transform(
    X_dev_processed
)


X_val_scaled = scaler.transform(
    X_val_processed
)


# ============================================================
# 10. Train final RF ONCE
# ============================================================

rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE
)


rf_model.fit(
    X_dev_scaled,
    y_dev
)


y_prob = rf_model.predict_proba(
    X_val_scaled
)[:, 1]


overall_auc = roc_auc_score(
    y_val,
    y_prob
)


# ============================================================
# 11. Sanity check
# ============================================================

print("\n============================================================")
print("SANITY CHECK")
print("============================================================")

print(
    "Expected final RF internal AUC ≈ 0.894"
)

print(
    f"Current RF internal AUC = "
    f"{overall_auc:.4f}"
)


if abs(
    overall_auc - 0.8941
) <= 0.005:

    print(
        "RF reproduction check: PASS"
    )

else:

    print(
        "RF reproduction check: CHECK PIPELINE"
    )


# ============================================================
# 12. Create validation prediction dataframe
# ============================================================

validation_df = pd.DataFrame(
    {
        "Original_Index":
            df_val.index,

        "True_Label":
            y_val,

        "Predicted_Probability":
            y_prob,

        "DR":
            df_val[
                "DR"
            ].values,

        "eGFR":
            df_val[
                EGFR_COL
            ].values
    }
)


# ============================================================
# 13. eGFR missingness
#
# Patients with missing eGFR remain available for DR subgroup
# analysis but cannot be assigned to renal-function strata.
# ============================================================

n_missing_egfr = int(
    validation_df[
        "eGFR"
    ].isna().sum()
)


print("\n============================================================")
print("eGFR MISSINGNESS")
print("============================================================")

print(
    f"Validation patients with missing eGFR = "
    f"{n_missing_egfr}"
)


# ============================================================
# 14. Bootstrap AUC CI
# ============================================================

def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_bootstrap=2000,
    random_state=45
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )


    # AUC cannot be estimated with only one class
    if len(
        np.unique(
            y_true
        )
    ) < 2:

        return (
            np.nan,
            np.nan
        )


    rng = np.random.RandomState(
        random_state
    )


    auc_values = []

    n = len(
        y_true
    )


    for _ in range(
        n_bootstrap
    ):

        indices = rng.choice(
            np.arange(n),
            size=n,
            replace=True
        )


        y_boot = y_true[
            indices
        ]


        if len(
            np.unique(
                y_boot
            )
        ) < 2:

            continue


        auc_boot = roc_auc_score(
            y_boot,
            y_prob[
                indices
            ]
        )


        auc_values.append(
            auc_boot
        )


    auc_values = np.array(
        auc_values
    )


    lower = np.percentile(
        auc_values,
        2.5
    )


    upper = np.percentile(
        auc_values,
        97.5
    )


    return (
        lower,
        upper
    )


# ============================================================
# 15. Generic subgroup evaluator
# ============================================================

def evaluate_subgroup(
    subgroup_df,
    subgroup_type,
    subgroup_name,
    bootstrap_seed
):

    n = len(
        subgroup_df
    )


    y_true = subgroup_df[
        "True_Label"
    ].astype(int).values


    y_prob_sub = subgroup_df[
        "Predicted_Probability"
    ].astype(float).values


    n_dkd = int(
        np.sum(
            y_true == 1
        )
    )


    n_ndkd = int(
        np.sum(
            y_true == 0
        )
    )


    # --------------------------------------------------------
    # AUC + bootstrap CI
    # --------------------------------------------------------

    if (
        n_dkd > 0
        and
        n_ndkd > 0
    ):

        auc = roc_auc_score(
            y_true,
            y_prob_sub
        )


        ci_lower, ci_upper = bootstrap_auc_ci(
            y_true,
            y_prob_sub,
            n_bootstrap=N_BOOTSTRAP,
            random_state=bootstrap_seed
        )

    else:

        auc = np.nan
        ci_lower = np.nan
        ci_upper = np.nan


    # --------------------------------------------------------
    # Locked-threshold classification
    # --------------------------------------------------------

    y_pred = (
        y_prob_sub
        >=
        LOCKED_THRESHOLD
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )


    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Stability note
    # --------------------------------------------------------

    if min(
        n_dkd,
        n_ndkd
    ) < 10:

        stability_note = (
            "Small class count; interpret cautiously"
        )

    else:

        stability_note = (
            "Adequate class counts"
        )


    return {
        "Subgroup_Type":
            subgroup_type,

        "Subgroup":
            subgroup_name,

        "N":
            n,

        "DKD_N":
            n_dkd,

        "NDKD_N":
            n_ndkd,

        "AUC":
            auc,

        "CI_Lower":
            ci_lower,

        "CI_Upper":
            ci_upper,

        "Locked_Threshold":
            LOCKED_THRESHOLD,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "PPV":
            ppv,

        "NPV":
            npv,

        "TP":
            tp,

        "FP":
            fp,

        "TN":
            tn,

        "FN":
            fn,

        "Stability_Note":
            stability_note
    }


# ============================================================
# PART A
# DR STATUS SUBGROUPS
# ============================================================

dr_results = []


for i, dr_value in enumerate(
    [0, 1]
):

    subgroup = validation_df[
        validation_df[
            "DR"
        ] == dr_value
    ].copy()


    if dr_value == 0:

        subgroup_name = (
            "DR absent"
        )

    else:

        subgroup_name = (
            "DR present"
        )


    result = evaluate_subgroup(
        subgroup_df=subgroup,
        subgroup_type="DR status",
        subgroup_name=subgroup_name,
        bootstrap_seed=(
            RANDOM_STATE
            +
            i
        )
    )


    dr_results.append(
        result
    )


dr_results_df = pd.DataFrame(
    dr_results
)


# ============================================================
# PART B
# THREE eGFR-BASED RENAL-FUNCTION STRATA ONLY
#
# 1. eGFR >=60
# 2. eGFR 30-59
# 3. eGFR <30
#
# No individual G categories are generated.
# ============================================================

def assign_egfr_stratum(
    x
):

    if pd.isna(
        x
    ):

        return np.nan


    if x >= 60:

        return (
            "eGFR ≥60"
        )


    elif x >= 30:

        return (
            "eGFR 30–59"
        )


    else:

        return (
            "eGFR <30"
        )


validation_df[
    "eGFR_Stratum"
] = validation_df[
    "eGFR"
].apply(
    assign_egfr_stratum
)


egfr_order = [
    "eGFR ≥60",
    "eGFR 30–59",
    "eGFR <30"
]


egfr_results = []


for i, stratum in enumerate(
    egfr_order
):

    subgroup = validation_df[
        validation_df[
            "eGFR_Stratum"
        ] == stratum
    ].copy()


    if len(
        subgroup
    ) == 0:

        continue


    result = evaluate_subgroup(
        subgroup_df=subgroup,
        subgroup_type=(
            "eGFR-based renal-function stratum"
        ),
        subgroup_name=stratum,
        bootstrap_seed=(
            RANDOM_STATE
            +
            100
            +
            i
        )
    )


    egfr_results.append(
        result
    )


egfr_results_df = pd.DataFrame(
    egfr_results
)


# ============================================================
# 16. Print DR subgroup results
# ============================================================

print("\n============================================================")
print("SUBGROUP PERFORMANCE BY DR STATUS")
print("============================================================")


for _, row in dr_results_df.iterrows():

    print(
        f"\n{row['Subgroup']}"
    )


    print(
        f"N = {int(row['N'])}, "
        f"DKD = {int(row['DKD_N'])}, "
        f"NDKD = {int(row['NDKD_N'])}"
    )


    print(
        f"AUC = "
        f"{row['AUC']:.2f} "
        f"(95% CI, "
        f"{row['CI_Lower']:.2f}–"
        f"{row['CI_Upper']:.2f})"
    )


    print(
        f"Sensitivity at 0.45 = "
        f"{row['Sensitivity'] * 100:.1f}%"
    )


    print(
        f"Specificity at 0.45 = "
        f"{row['Specificity'] * 100:.1f}%"
    )


    print(
        f"PPV at 0.45 = "
        f"{row['PPV'] * 100:.1f}%"
    )


    print(
        f"NPV at 0.45 = "
        f"{row['NPV'] * 100:.1f}%"
    )


# ============================================================
# 17. Print eGFR subgroup results
# ============================================================

print("\n============================================================")
print("SUBGROUP PERFORMANCE BY eGFR STRATUM")
print("============================================================")


for _, row in egfr_results_df.iterrows():

    print(
        f"\n{row['Subgroup']}"
    )


    print(
        f"N = {int(row['N'])}, "
        f"DKD = {int(row['DKD_N'])}, "
        f"NDKD = {int(row['NDKD_N'])}"
    )


    print(
        f"AUC = "
        f"{row['AUC']:.2f} "
        f"(95% CI, "
        f"{row['CI_Lower']:.2f}–"
        f"{row['CI_Upper']:.2f})"
    )


    print(
        f"Sensitivity at 0.45 = "
        f"{row['Sensitivity'] * 100:.1f}%"
    )


    print(
        f"Specificity at 0.45 = "
        f"{row['Specificity'] * 100:.1f}%"
    )


    print(
        f"PPV at 0.45 = "
        f"{row['PPV'] * 100:.1f}%"
    )


    print(
        f"NPV at 0.45 = "
        f"{row['NPV'] * 100:.1f}%"
    )


# ============================================================
# 18. Save patient-level validation predictions
# ============================================================

validation_df.to_excel(
    "12-Final_Subgroup_Validation_Predictions.xlsx",
    index=False
)


# ============================================================
# 19. Save raw subgroup results
#
# Only DR + three eGFR strata.
# ============================================================

combined_results_df = pd.concat(
    [
        dr_results_df,
        egfr_results_df
    ],
    ignore_index=True
)


with pd.ExcelWriter(
    "12-Final_Subgroup_Performance.xlsx"
) as writer:

    dr_results_df.to_excel(
        writer,
        sheet_name="DR_status",
        index=False
    )


    egfr_results_df.to_excel(
        writer,
        sheet_name="eGFR_strata",
        index=False
    )


    combined_results_df.to_excel(
        writer,
        sheet_name="Combined",
        index=False
    )


# ============================================================
# 20. Manuscript-formatted table
# ============================================================

def format_auc_ci(
    row
):

    if pd.isna(
        row["AUC"]
    ):

        return (
            "Not estimable"
        )


    return (
        f"{row['AUC']:.2f} "
        f"({row['CI_Lower']:.2f}–"
        f"{row['CI_Upper']:.2f})"
    )


manuscript_df = pd.DataFrame(
    {
        "Subgroup type":
            combined_results_df[
                "Subgroup_Type"
            ],

        "Subgroup":
            combined_results_df[
                "Subgroup"
            ],

        "N":
            combined_results_df[
                "N"
            ].astype(int),

        "DKD":
            combined_results_df[
                "DKD_N"
            ].astype(int),

        "NDKD":
            combined_results_df[
                "NDKD_N"
            ].astype(int),

        "AUC (95% CI)":
            combined_results_df.apply(
                format_auc_ci,
                axis=1
            ),

        "Sensitivity":
            combined_results_df[
                "Sensitivity"
            ].map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "NA"
            ),

        "Specificity":
            combined_results_df[
                "Specificity"
            ].map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "NA"
            ),

        "PPV":
            combined_results_df[
                "PPV"
            ].map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "NA"
            ),

        "NPV":
            combined_results_df[
                "NPV"
            ].map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "NA"
            ),

        "Threshold":
            "0.45"
    }
)


manuscript_df.to_excel(
    "12-Final_Subgroup_Manuscript_Table.xlsx",
    index=False
)


# ============================================================
# 21. Forest plot
#
# Only:
# DR absent
# DR present
# eGFR >=60
# eGFR 30-59
# eGFR <30
# ============================================================

plot_df = combined_results_df[
    combined_results_df[
        "AUC"
    ].notna()
].copy()


plot_df[
    "Plot_Label"
] = plot_df[
    "Subgroup"
]


# Reverse order for top-to-bottom display
plot_df = plot_df.iloc[
    ::-1
].reset_index(
    drop=True
)


y_positions = np.arange(
    len(plot_df)
)


xerr_lower = (
    plot_df[
        "AUC"
    ]
    -
    plot_df[
        "CI_Lower"
    ]
).values


xerr_upper = (
    plot_df[
        "CI_Upper"
    ]
    -
    plot_df[
        "AUC"
    ]
).values


fig, ax = plt.subplots(
    figsize=(8.5, 6.5),
    dpi=300
)


ax.errorbar(
    plot_df[
        "AUC"
    ],
    y_positions,
    xerr=[
        xerr_lower,
        xerr_upper
    ],
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=7
)


# Overall internal-validation AUC
ax.axvline(
    x=overall_auc,
    linestyle="--",
    linewidth=1.3,
    label=(
        f"Overall AUC = "
        f"{overall_auc:.2f}"
    )
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    plot_df[
        "Plot_Label"
    ],
    fontsize=11
)


ax.set_xlabel(
    "AUC (95% CI)",
    fontsize=13,
    fontweight="bold"
)


ax.set_title(
    "Subgroup Performance of the Final 7-Feature Random Forest",
    fontsize=14,
    fontweight="bold",
    pad=14
)


ax.set_xlim(
    0.40,
    1.00
)


ax.grid(
    True,
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25
)


ax.tick_params(
    axis="x",
    labelsize=11
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True
)


plt.tight_layout()


plt.savefig(
    "12-Final_Subgroup_AUC_Forest_Plot.pdf",
    format="pdf",
    bbox_inches="tight"
)


plt.savefig(
    "12-Final_Subgroup_AUC_Forest_Plot.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. Save forest-plot source data
# ============================================================

forest_source_df = combined_results_df[
    [
        "Subgroup_Type",
        "Subgroup",
        "N",
        "DKD_N",
        "NDKD_N",
        "AUC",
        "CI_Lower",
        "CI_Upper"
    ]
].copy()


forest_source_df.to_excel(
    "12-Final_Subgroup_Forest_Plot_Source_Data.xlsx",
    index=False
)


# ============================================================
# 23. Final summary
# ============================================================

print("\n============================================================")
print("STAGE 12 FINAL - COMPLETED")
print("============================================================")


print(
    f"\nOverall internal validation AUC = "
    f"{overall_auc:.2f}"
)


print(
    f"Locked threshold = "
    f"{LOCKED_THRESHOLD:.2f}"
)


print(
    f"Missing eGFR in internal validation = "
    f"{n_missing_egfr}"
)


print("\nFinal subgroup structure:")

print(
    "1. DR absent"
)

print(
    "2. DR present"
)

print(
    "3. eGFR ≥60"
)

print(
    "4. eGFR 30–59"
)

print(
    "5. eGFR <30"
)


print("\nSaved files:")

print(
    "12-Final_Subgroup_Validation_Predictions.xlsx"
)

print(
    "12-Final_Subgroup_Performance.xlsx"
)

print(
    "12-Final_Subgroup_Manuscript_Table.xlsx"
)

print(
    "12-Final_Subgroup_AUC_Forest_Plot.pdf"
)

print(
    "12-Final_Subgroup_AUC_Forest_Plot.png"
)

print(
    "12-Final_Subgroup_Forest_Plot_Source_Data.xlsx"
)
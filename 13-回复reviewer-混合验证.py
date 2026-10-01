# ============================================================
# STAGE 13 - FINAL
# Mixed DKD + NDKD predicted-probability analysis
#
# Reviewer #3 option (b):
# Report how the current model distributes predicted
# probabilities among the mixed DKD + NDKD group.
#
# IMPORTANT:
# 1. Primary model remains pure DKD vs pure NDKD.
# 2. Mixed cases are NEVER used to train/refit the model.
# 3. Locked 7-feature RF and threshold 0.45 are unchanged.
# 4. No accuracy/sensitivity/specificity is calculated for
#    mixed patients because they do not have a unique binary
#    ground-truth class.
# 5. Mixed patients with missing DR are excluded because the
#    primary model had no categorical DR imputation procedure.
# 6. Missing continuous predictors are imputed using means
#    learned from the development cohort, exactly as in the
#    primary pipeline.
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
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Settings
# ============================================================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 45

PURE_DATA_FILE = "test1.xlsx"
MIXED_FILE = "mixed.xlsx"

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
# 3. Load primary pure DKD / pure NDKD dataset
# ============================================================

pure_df = pd.read_excel(
    PURE_DATA_FILE
)


print("\n============================================================")
print("STAGE 13 - FINAL")
print("Mixed DKD + NDKD Probability Distribution Analysis")
print("============================================================")


print(
    f"\nPrimary dataset: {PURE_DATA_FILE}"
)

print(
    f"Pure DKD/NDKD cohort N = {len(pure_df)}"
)


# ============================================================
# 4. Check primary-dataset columns
# ============================================================

required_pure_columns = (
    FINAL_FEATURES
    +
    [TARGET]
)


missing_pure_columns = [
    col
    for col in required_pure_columns
    if col not in pure_df.columns
]


if len(missing_pure_columns) > 0:

    raise ValueError(
        f"Missing columns in primary dataset: "
        f"{missing_pure_columns}"
    )


# ============================================================
# 5. Reproduce exact primary 80/20 split
# ============================================================

all_indices = np.arange(
    len(pure_df)
)


y_all = pure_df[
    TARGET
].astype(int).values


dev_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_all
)


df_dev = pure_df.iloc[
    dev_indices
].copy()


df_val = pure_df.iloc[
    val_indices
].copy()


y_dev = df_dev[
    TARGET
].astype(int).values


y_val = df_val[
    TARGET
].astype(int).values


print("\n============================================================")
print("PRIMARY MODEL DATA SPLIT")
print("============================================================")


print(
    f"Development N = {len(df_dev)}"
)

print(
    f"Internal validation N = {len(df_val)}"
)

print(
    f"Internal DKD = {np.sum(y_val == 1)}"
)

print(
    f"Internal NDKD = {np.sum(y_val == 0)}"
)


# ============================================================
# 6. Fit continuous-variable imputer on development only
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
# 7. Add DR
# ============================================================

X_dev_processed = X_dev_cont.copy()
X_val_processed = X_val_cont.copy()


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
# 8. Standardization
# ============================================================

scaler = StandardScaler()


X_dev_scaled = scaler.fit_transform(
    X_dev_processed
)


X_val_scaled = scaler.transform(
    X_val_processed
)


# ============================================================
# 9. Train locked final RF
# ============================================================

rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE
)


rf_model.fit(
    X_dev_scaled,
    y_dev
)


val_prob = rf_model.predict_proba(
    X_val_scaled
)[:, 1]


val_auc = roc_auc_score(
    y_val,
    val_prob
)


# ============================================================
# 10. Sanity check
# ============================================================

print("\n============================================================")
print("SANITY CHECK")
print("============================================================")


print(
    "Expected final internal-validation AUC ≈ 0.894"
)


print(
    f"Current internal-validation AUC = "
    f"{val_auc:.4f}"
)


if abs(
    val_auc - 0.8941
) <= 0.005:

    print(
        "Primary RF reproduction check: PASS"
    )

else:

    print(
        "Primary RF reproduction check: CHECK PIPELINE"
    )


# ============================================================
# 11. Create pure internal-validation prediction table
# ============================================================

pure_validation_predictions = pd.DataFrame(
    {
        "True_Label":
            y_val,

        "Predicted_Probability_DKD":
            val_prob
    }
)


pure_ndkd_prob = (
    pure_validation_predictions.loc[
        pure_validation_predictions[
            "True_Label"
        ] == 0,
        "Predicted_Probability_DKD"
    ]
    .astype(float)
    .values
)


pure_dkd_prob = (
    pure_validation_predictions.loc[
        pure_validation_predictions[
            "True_Label"
        ] == 1,
        "Predicted_Probability_DKD"
    ]
    .astype(float)
    .values
)


# ============================================================
# 12. Load mixed DKD + NDKD cohort
# ============================================================

mixed_all = pd.read_excel(
    MIXED_FILE
)


print("\n============================================================")
print("MIXED DKD + NDKD COHORT")
print("============================================================")


print(
    f"Mixed cohort file: {MIXED_FILE}"
)

print(
    f"Total mixed cohort N = {len(mixed_all)}"
)


# ============================================================
# 13. Check required predictor columns
# ============================================================

missing_mixed_columns = [
    col
    for col in FINAL_FEATURES
    if col not in mixed_all.columns
]


if len(missing_mixed_columns) > 0:

    raise ValueError(
        f"Missing required predictor columns in mixed cohort: "
        f"{missing_mixed_columns}"
    )


# ============================================================
# 14. Mixed-cohort missingness summary
# ============================================================

mixed_missingness = pd.DataFrame(
    {
        "Variable":
            FINAL_FEATURES,

        "Missing_N":
            [
                mixed_all[
                    col
                ].isna().sum()
                for col in FINAL_FEATURES
            ],

        "Missing_Percent":
            [
                mixed_all[
                    col
                ].isna().mean()
                *
                100
                for col in FINAL_FEATURES
            ]
    }
)


print("\n============================================================")
print("MIXED COHORT MISSINGNESS")
print("============================================================")


print(
    mixed_missingness.to_string(
        index=False
    )
)


# ============================================================
# 15. Exclude mixed cases with missing DR
#
# DR was not imputed in the primary pipeline.
# Therefore, patients lacking DR cannot be passed through the
# locked model without introducing a new preprocessing rule.
# ============================================================

n_mixed_total = len(
    mixed_all
)


n_missing_dr = int(
    mixed_all[
        "DR"
    ].isna().sum()
)


mixed_excluded = mixed_all[
    mixed_all[
        "DR"
    ].isna()
].copy()


mixed_df = mixed_all[
    mixed_all[
        "DR"
    ].notna()
].copy()


n_mixed_analyzed = len(
    mixed_df
)


print("\n============================================================")
print("MIXED-COHORT ANALYSIS POPULATION")
print("============================================================")


print(
    f"Total mixed DKD+NDKD = "
    f"{n_mixed_total}"
)


print(
    f"Excluded because DR missing = "
    f"{n_missing_dr}"
)


print(
    f"Included in probability analysis = "
    f"{n_mixed_analyzed}"
)


# ============================================================
# 16. Continuous imputation for mixed cohort
#
# IMPORTANT:
# Uses the imputer fitted on PRIMARY DEVELOPMENT DATA.
# ============================================================

X_mixed_cont = pd.DataFrame(
    imputer.transform(
        mixed_df[
            CONTINUOUS_FEATURES
        ]
    ),
    columns=CONTINUOUS_FEATURES,
    index=mixed_df.index
)


# ============================================================
# 17. Add observed DR
# ============================================================

X_mixed_processed = X_mixed_cont.copy()


X_mixed_processed[
    "DR"
] = mixed_df[
    "DR"
].values


X_mixed_processed = X_mixed_processed[
    FINAL_FEATURES
]


# ============================================================
# 18. Apply development-fitted scaler
# ============================================================

X_mixed_scaled = scaler.transform(
    X_mixed_processed
)


# ============================================================
# 19. Predict P(DKD) in mixed group
# ============================================================

mixed_prob = rf_model.predict_proba(
    X_mixed_scaled
)[:, 1]


# ============================================================
# 20. Summary statistics
# ============================================================

mixed_mean = np.mean(
    mixed_prob
)


mixed_sd = np.std(
    mixed_prob,
    ddof=1
)


mixed_median = np.median(
    mixed_prob
)


mixed_q1 = np.percentile(
    mixed_prob,
    25
)


mixed_q3 = np.percentile(
    mixed_prob,
    75
)


mixed_min = np.min(
    mixed_prob
)


mixed_max = np.max(
    mixed_prob
)


n_ge_threshold = int(
    np.sum(
        mixed_prob
        >=
        LOCKED_THRESHOLD
    )
)


n_lt_threshold = int(
    np.sum(
        mixed_prob
        <
        LOCKED_THRESHOLD
    )
)


pct_ge_threshold = (
    n_ge_threshold
    /
    n_mixed_analyzed
    *
    100
)


pct_lt_threshold = (
    n_lt_threshold
    /
    n_mixed_analyzed
    *
    100
)


# ============================================================
# 21. Print main reviewer results
# ============================================================

print("\n============================================================")
print("MIXED DKD + NDKD: P(DKD) DISTRIBUTION")
print("============================================================")


print(
    f"Analyzed N = "
    f"{n_mixed_analyzed}"
)


print(
    f"Median P(DKD) = "
    f"{mixed_median:.3f}"
)


print(
    f"IQR = "
    f"{mixed_q1:.3f}–"
    f"{mixed_q3:.3f}"
)


print(
    f"Mean ± SD = "
    f"{mixed_mean:.3f} ± "
    f"{mixed_sd:.3f}"
)


print(
    f"Range = "
    f"{mixed_min:.3f}–"
    f"{mixed_max:.3f}"
)


print(
    f"P(DKD) ≥ {LOCKED_THRESHOLD:.2f}: "
    f"{n_ge_threshold}/"
    f"{n_mixed_analyzed} "
    f"({pct_ge_threshold:.1f}%)"
)


print(
    f"P(DKD) < {LOCKED_THRESHOLD:.2f}: "
    f"{n_lt_threshold}/"
    f"{n_mixed_analyzed} "
    f"({pct_lt_threshold:.1f}%)"
)


# ============================================================
# 22. Three-group descriptive summaries
# ============================================================

def summarize_probabilities(
    group_name,
    probabilities
):

    probabilities = np.asarray(
        probabilities
    )


    return {
        "Group":
            group_name,

        "N":
            len(probabilities),

        "Median":
            np.median(
                probabilities
            ),

        "Q1":
            np.percentile(
                probabilities,
                25
            ),

        "Q3":
            np.percentile(
                probabilities,
                75
            ),

        "Mean":
            np.mean(
                probabilities
            ),

        "SD":
            np.std(
                probabilities,
                ddof=1
            ),

        "Percent_GE_0.45":
            np.mean(
                probabilities
                >=
                LOCKED_THRESHOLD
            )
            *
            100
    }


three_group_summary = pd.DataFrame(
    [
        summarize_probabilities(
            "Pure NDKD",
            pure_ndkd_prob
        ),

        summarize_probabilities(
            "Mixed DKD+NDKD",
            mixed_prob
        ),

        summarize_probabilities(
            "Pure DKD",
            pure_dkd_prob
        )
    ]
)


print("\n============================================================")
print("THREE-GROUP DESCRIPTIVE SUMMARY")
print("============================================================")


for _, row in three_group_summary.iterrows():

    print(
        f"\n{row['Group']}"
    )

    print(
        f"N = "
        f"{int(row['N'])}"
    )

    print(
        f"Median P(DKD) = "
        f"{row['Median']:.3f}"
    )

    print(
        f"IQR = "
        f"{row['Q1']:.3f}–"
        f"{row['Q3']:.3f}"
    )

    print(
        f"P(DKD) ≥0.45 = "
        f"{row['Percent_GE_0.45']:.1f}%"
    )


# ============================================================
# 23. Save all mixed cases with inclusion flag
# ============================================================

mixed_output = mixed_all.copy()


mixed_output[
    "Included_in_probability_analysis"
] = mixed_output[
    "DR"
].notna()


mixed_output[
    "Exclusion_reason"
] = np.where(
    mixed_output[
        "DR"
    ].isna(),
    "Missing DR",
    ""
)


mixed_output[
    "Predicted_Probability_DKD"
] = np.nan


mixed_output.loc[
    mixed_df.index,
    "Predicted_Probability_DKD"
] = mixed_prob


mixed_output[
    "Above_Locked_Threshold_0.45"
] = np.nan


mixed_output.loc[
    mixed_df.index,
    "Above_Locked_Threshold_0.45"
] = (
    mixed_prob
    >=
    LOCKED_THRESHOLD
)


mixed_output.to_excel(
    "13-Final_Mixed_DKD_NDKD_Predictions.xlsx",
    index=False
)


# ============================================================
# 24. Save excluded cases
# ============================================================

mixed_excluded.to_excel(
    "13-Final_Mixed_Excluded_Missing_DR.xlsx",
    index=False
)


# ============================================================
# 25. Save missingness
# ============================================================

mixed_missingness.to_excel(
    "13-Final_Mixed_Missingness.xlsx",
    index=False
)


# ============================================================
# 26. Save main summary
# ============================================================

mixed_summary = pd.DataFrame(
    [
        {
            "Total_Mixed_N":
                n_mixed_total,

            "Missing_DR_Excluded_N":
                n_missing_dr,

            "Analyzed_N":
                n_mixed_analyzed,

            "Median_P_DKD":
                mixed_median,

            "Q1_P_DKD":
                mixed_q1,

            "Q3_P_DKD":
                mixed_q3,

            "Mean_P_DKD":
                mixed_mean,

            "SD_P_DKD":
                mixed_sd,

            "Min_P_DKD":
                mixed_min,

            "Max_P_DKD":
                mixed_max,

            "Locked_Threshold":
                LOCKED_THRESHOLD,

            "N_GE_0.45":
                n_ge_threshold,

            "Percent_GE_0.45":
                pct_ge_threshold,

            "N_LT_0.45":
                n_lt_threshold,

            "Percent_LT_0.45":
                pct_lt_threshold
        }
    ]
)


mixed_summary.to_excel(
    "13-Final_Mixed_Probability_Summary.xlsx",
    index=False
)


# ============================================================
# 27. Save three-group summary
# ============================================================

three_group_summary.to_excel(
    "13-Final_Three_Group_Probability_Summary.xlsx",
    index=False
)


# ============================================================
# 28. Create figure source file
# ============================================================

figure_source = pd.concat(
    [
        pd.DataFrame(
            {
                "Group":
                    "Pure NDKD",

                "Predicted_Probability_DKD":
                    pure_ndkd_prob
            }
        ),

        pd.DataFrame(
            {
                "Group":
                    "Mixed DKD+NDKD",

                "Predicted_Probability_DKD":
                    mixed_prob
            }
        ),

        pd.DataFrame(
            {
                "Group":
                    "Pure DKD",

                "Predicted_Probability_DKD":
                    pure_dkd_prob
            }
        )
    ],
    ignore_index=True
)


figure_source.to_excel(
    "13-Final_Mixed_Probability_Figure_Source.xlsx",
    index=False
)


# ============================================================
# 29. Probability-distribution figure
#
# Pure groups:
# internal validation predictions only
#
# Mixed group:
# locked model predictions
# ============================================================

group_order = [
    "Pure NDKD",
    "Mixed DKD+NDKD",
    "Pure DKD"
]


plot_data = [
    figure_source.loc[
        figure_source[
            "Group"
        ] == group,
        "Predicted_Probability_DKD"
    ].values
    for group in group_order
]


fig, ax = plt.subplots(
    figsize=(8.2, 6.8),
    dpi=300
)


ax.violinplot(
    plot_data,
    positions=[
        1,
        2,
        3
    ],
    widths=0.75,
    showmeans=False,
    showmedians=False,
    showextrema=False
)


ax.boxplot(
    plot_data,
    positions=[
        1,
        2,
        3
    ],
    widths=0.20,
    showfliers=False,
    medianprops={
        "linewidth": 2
    }
)


ax.axhline(
    y=LOCKED_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    label="Locked threshold = 0.45"
)


ax.set_xticks(
    [
        1,
        2,
        3
    ]
)


ax.set_xticklabels(
    group_order,
    fontsize=11
)


ax.set_ylabel(
    "Predicted probability of DKD",
    fontsize=13
)


ax.set_ylim(
    0,
    1.02
)


ax.set_title(
    "Distribution of Predicted DKD Probabilities",
    fontsize=14,
    pad=12
)


ax.grid(
    True,
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.legend(
    loc="upper left",
    fontsize=10,
    frameon=True
)


plt.tight_layout()


plt.savefig(
    "13-Final_Mixed_Probability_Distribution.pdf",
    format="pdf",
    bbox_inches="tight"
)


plt.savefig(
    "13-Final_Mixed_Probability_Distribution.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 30. Manuscript-ready one-row table
# ============================================================

manuscript_summary = pd.DataFrame(
    [
        {
            "Group":
                "Mixed DKD+NDKD",

            "Total N":
                n_mixed_total,

            "Analyzed N":
                n_mixed_analyzed,

            "Excluded for missing DR":
                n_missing_dr,

            "Median P(DKD)":
                f"{mixed_median:.2f}",

            "IQR":
                (
                    f"{mixed_q1:.2f}–"
                    f"{mixed_q3:.2f}"
                ),

            "P(DKD) ≥0.45":
                (
                    f"{pct_ge_threshold:.1f}% "
                    f"({n_ge_threshold}/"
                    f"{n_mixed_analyzed})"
                )
        }
    ]
)


manuscript_summary.to_excel(
    "13-Final_Mixed_Manuscript_Summary.xlsx",
    index=False
)


# ============================================================
# 31. Final output
# ============================================================

print("\n============================================================")
print("STAGE 13 FINAL COMPLETED")
print("============================================================")


print(
    f"\nTotal mixed cohort = "
    f"{n_mixed_total}"
)


print(
    f"Excluded for missing DR = "
    f"{n_missing_dr}"
)


print(
    f"Analyzed mixed cohort = "
    f"{n_mixed_analyzed}"
)


print(
    f"Median P(DKD) = "
    f"{mixed_median:.2f} "
    f"(IQR, "
    f"{mixed_q1:.2f}–"
    f"{mixed_q3:.2f})"
)


print(
    f"P(DKD) ≥0.45 = "
    f"{pct_ge_threshold:.1f}% "
    f"({n_ge_threshold}/"
    f"{n_mixed_analyzed})"
)


print("\nSaved files:")

print(
    "13-Final_Mixed_DKD_NDKD_Predictions.xlsx"
)

print(
    "13-Final_Mixed_Excluded_Missing_DR.xlsx"
)

print(
    "13-Final_Mixed_Missingness.xlsx"
)

print(
    "13-Final_Mixed_Probability_Summary.xlsx"
)

print(
    "13-Final_Three_Group_Probability_Summary.xlsx"
)

print(
    "13-Final_Mixed_Probability_Figure_Source.xlsx"
)

print(
    "13-Final_Mixed_Probability_Distribution.pdf"
)

print(
    "13-Final_Mixed_Probability_Distribution.png"
)

print(
    "13-Final_Mixed_Manuscript_Summary.xlsx"
)
# ============================================================
# STAGE 9
# Final 7-feature Random Forest
#
# External validation:
#   A. Calibration of P(DKD)
#   B. Biopsy-oriented DCA using P(NDKD) = 1 - P(DKD)
#   C. DKD classification threshold analysis
#
# IMPORTANT:
# 1. This script does NOT retrain the model.
# 2. It uses the locked external-validation predictions
#    generated previously.
# 3. Calibration and threshold analysis use DKD as the
#    positive outcome.
# 4. Biopsy-oriented DCA uses NDKD as the positive/event outcome.
# 5. The primary DKD classification threshold = 0.45 and is NOT
#    re-optimized in the external validation cohort.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. Basic settings
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. Input file
# ============================================================

prediction_file = (
    "8-Final_7Feature_RF_External_Validation_Predictions.xlsx"
)

pred_df = pd.read_excel(prediction_file)

print("\n============================================================")
print("STAGE 9")
print("External Calibration + Biopsy-oriented DCA + Threshold Analysis")
print("============================================================")

print(f"\nPrediction file: {prediction_file}")
print(f"External validation N = {len(pred_df)}")


# ============================================================
# 3. Original model outcome
#
# 1 = DKD
# 0 = NDKD
#
# Predicted_Probability = P(DKD)
# ============================================================

y_true_dkd = pred_df[
    "True_Label"
].astype(int).values

y_prob_dkd = pred_df[
    "Predicted_Probability"
].astype(float).values


# ============================================================
# 4. Locked DKD classification threshold
#
# Derived from development 5-fold OOF predictions.
# NEVER re-optimize in external validation.
# ============================================================

LOCKED_DKD_THRESHOLD = 0.45


# ============================================================
# 5. Basic checks
# ============================================================

n_total = len(y_true_dkd)
n_dkd = int(np.sum(y_true_dkd == 1))
n_ndkd = int(np.sum(y_true_dkd == 0))

auc_dkd = roc_auc_score(
    y_true_dkd,
    y_prob_dkd
)

print("\n============================================================")
print("BASIC CHECKS")
print("============================================================")

print(f"N = {n_total}")
print(f"DKD = {n_dkd}")
print(f"NDKD = {n_ndkd}")

print(
    f"P(DKD) range = "
    f"{np.min(y_prob_dkd):.4f} to "
    f"{np.max(y_prob_dkd):.4f}"
)

print(f"AUC = {auc_dkd:.2f}")

print(
    f"Locked DKD classification threshold = "
    f"{LOCKED_DKD_THRESHOLD:.2f}"
)


# ============================================================
# PART A
# CALIBRATION ANALYSIS
#
# Positive outcome:
#   DKD = 1
#
# Probability:
#   P(DKD)
# ============================================================


# ============================================================
# 6. Brier score
# ============================================================

brier = brier_score_loss(
    y_true_dkd,
    y_prob_dkd
)


# ============================================================
# 7. Calibration intercept and slope
# ============================================================

epsilon = 1e-6

y_prob_clipped = np.clip(
    y_prob_dkd,
    epsilon,
    1 - epsilon
)

predicted_logit = logit(
    y_prob_clipped
).reshape(-1, 1)

calibration_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=10000
)

calibration_model.fit(
    predicted_logit,
    y_true_dkd
)

calibration_intercept = (
    calibration_model.intercept_[0]
)

calibration_slope = (
    calibration_model.coef_[0][0]
)


# ============================================================
# 8. Calibration curve
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_true_dkd,
    y_prob_dkd,
    n_bins=10,
    strategy="quantile"
)


# ============================================================
# 9. Print calibration results
# ============================================================

print("\n============================================================")
print("CALIBRATION - DKD PROBABILITY")
print("============================================================")

print(f"Brier score = {brier:.3f}")

print(
    f"Calibration intercept = "
    f"{calibration_intercept:.2f}"
)

print(
    f"Calibration slope = "
    f"{calibration_slope:.2f}"
)


# ============================================================
# 10. Calibration plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 7),
    dpi=300
)

ax.plot(
    prob_pred,
    prob_true,
    marker="o",
    markersize=7,
    linewidth=2.2,
    label="7-feature Random Forest"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Perfect calibration"
)

ax.set_xlabel(
    "Predicted Probability of DKD",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Observed Proportion of DKD",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "External Validation: Calibration",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.xaxis.set_major_formatter(
    FormatStrFormatter("%.1f")
)

ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.1f")
)

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

calibration_text = (
    f"Brier score = {brier:.3f}\n"
    f"Intercept = {calibration_intercept:.2f}\n"
    f"Slope = {calibration_slope:.2f}"
)

ax.text(
    0.05,
    0.95,
    calibration_text,
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
    "9-External_Calibration.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "9-External_Calibration.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. Save calibration source data
# ============================================================

calibration_source_df = pd.DataFrame(
    {
        "Mean_Predicted_Probability_DKD":
            prob_pred,

        "Observed_Proportion_DKD":
            prob_true
    }
)

calibration_source_df.to_excel(
    "9-External_Calibration_Source_Data.xlsx",
    index=False
)


calibration_statistics_df = pd.DataFrame(
    [
        {
            "N":
                n_total,

            "DKD_N":
                n_dkd,

            "NDKD_N":
                n_ndkd,

            "AUC":
                auc_dkd,

            "Brier_Score":
                brier,

            "Calibration_Intercept":
                calibration_intercept,

            "Calibration_Slope":
                calibration_slope
        }
    ]
)

calibration_statistics_df.to_excel(
    "9-External_Calibration_Statistics.xlsx",
    index=False
)


# ============================================================
# PART B
# BIOPSY-ORIENTED DECISION CURVE ANALYSIS
#
# Event:
#   NDKD = 1
#
# Probability:
#   P(NDKD) = 1 - P(DKD)
#
# Interpretation:
# Higher P(NDKD) -> stronger indication to consider biopsy.
# ============================================================


# ============================================================
# 12. Convert to biopsy-oriented outcome
# ============================================================

y_true_ndkd = 1 - y_true_dkd

y_prob_ndkd = 1 - y_prob_dkd

ndkd_prevalence = np.mean(
    y_true_ndkd
)


# ============================================================
# 13. Net benefit function
# ============================================================

def calculate_net_benefit(
    y_true,
    y_prob,
    threshold
):

    predicted_positive = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_positive,
        labels=[0, 1]
    ).ravel()

    n = len(y_true)

    net_benefit = (
        tp / n
        -
        fp / n
        *
        (
            threshold
            /
            (1 - threshold)
        )
    )

    return net_benefit


# ============================================================
# 14. DCA threshold range
# ============================================================

dca_thresholds = np.arange(
    0.01,
    0.81,
    0.01
)


# ============================================================
# 15. Model-guided biopsy
# ============================================================

model_biopsy_nb = []

for threshold in dca_thresholds:

    nb = calculate_net_benefit(
        y_true_ndkd,
        y_prob_ndkd,
        threshold
    )

    model_biopsy_nb.append(nb)


model_biopsy_nb = np.array(
    model_biopsy_nb
)


# ============================================================
# 16. Biopsy-all strategy
# ============================================================

biopsy_all_nb = (
    ndkd_prevalence
    -
    (1 - ndkd_prevalence)
    *
    (
        dca_thresholds
        /
        (1 - dca_thresholds)
    )
)


# ============================================================
# 17. Biopsy-none strategy
# ============================================================

biopsy_none_nb = np.zeros_like(
    dca_thresholds
)


# ============================================================
# 18. DCA information
# ============================================================

print("\n============================================================")
print("BIOPSY-ORIENTED DECISION CURVE ANALYSIS")
print("============================================================")

print("Biopsy-relevant event = NDKD")

print(
    f"NDKD prevalence = "
    f"{ndkd_prevalence * 100:.1f}%"
)

print(
    "DCA probability = P(NDKD) = 1 - P(DKD)"
)

print(
    "No biopsy threshold was optimized "
    "in the external validation cohort."
)


# ============================================================
# 19. DCA figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7),
    dpi=300
)

ax.plot(
    dca_thresholds,
    model_biopsy_nb,
    linewidth=2.4,
    label="Model-guided biopsy"
)

ax.plot(
    dca_thresholds,
    biopsy_all_nb,
    linestyle="--",
    linewidth=1.8,
    label="Biopsy all"
)

ax.plot(
    dca_thresholds,
    biopsy_none_nb,
    linestyle=":",
    linewidth=1.8,
    label="Biopsy none"
)

ax.set_xlabel(
    "Threshold Probability for NDKD",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Net Benefit",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "External Validation: Decision Curve Analysis",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlim(
    0,
    0.80
)

visible_values = np.concatenate(
    [
        model_biopsy_nb,
        biopsy_all_nb,
        biopsy_none_nb
    ]
)

visible_values = visible_values[
    np.isfinite(visible_values)
]

y_min = max(
    np.min(visible_values) - 0.05,
    -0.30
)

y_max = min(
    np.max(visible_values) + 0.05,
    0.70
)

ax.set_ylim(
    y_min,
    y_max
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

ax.legend(
    loc="upper right",
    fontsize=10,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "9-External_Biopsy_DCA.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "9-External_Biopsy_DCA.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. Save DCA source data
# ============================================================

dca_source_df = pd.DataFrame(
    {
        "Threshold_Probability_NDKD":
            dca_thresholds,

        "Model_Guided_Biopsy_Net_Benefit":
            model_biopsy_nb,

        "Biopsy_All_Net_Benefit":
            biopsy_all_nb,

        "Biopsy_None_Net_Benefit":
            biopsy_none_nb
    }
)

dca_source_df.to_excel(
    "9-External_Biopsy_DCA_Source_Data.xlsx",
    index=False
)


# ============================================================
# PART C
# DKD CLASSIFICATION THRESHOLD ANALYSIS
#
# Positive outcome:
#   DKD = 1
#
# Probability:
#   P(DKD)
#
# This analysis is descriptive only.
# No threshold is re-optimized.
# ============================================================


# ============================================================
# 21. Candidate DKD thresholds
# ============================================================

threshold_values = np.round(
    np.arange(0.10, 0.91, 0.05),
    2
)

threshold_values = np.unique(
    np.append(
        threshold_values,
        round(LOCKED_DKD_THRESHOLD, 2)
    )
)

threshold_values = np.sort(threshold_values)


# ============================================================
# 22. Threshold-specific performance
# ============================================================

threshold_results = []

for threshold in threshold_values:

    y_pred_dkd = (
        y_prob_dkd >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_dkd,
        y_pred_dkd,
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

    accuracy = accuracy_score(
        y_true_dkd,
        y_pred_dkd
    )

    f1 = f1_score(
        y_true_dkd,
        y_pred_dkd,
        zero_division=0
    )

    threshold_results.append(
        {
            "Threshold":
                threshold,

            "Locked_Threshold":
                np.isclose(
                    threshold,
                    LOCKED_DKD_THRESHOLD
                ),

            "Sensitivity":
                sensitivity,

            "Specificity":
                specificity,

            "PPV":
                ppv,

            "NPV":
                npv,

            "Accuracy":
                accuracy,

            "F1":
                f1,

            "TP":
                tp,

            "FP":
                fp,

            "TN":
                tn,

            "FN":
                fn
        }
    )


threshold_results_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 23. Locked threshold
# ============================================================

locked_row = threshold_results_df[
    threshold_results_df[
        "Locked_Threshold"
    ]
].iloc[0]


print("\n============================================================")
print("DKD THRESHOLD ANALYSIS")
print("============================================================")

print(
    f"Locked threshold = "
    f"{locked_row['Threshold']:.2f}"
)

print(
    f"Sensitivity = "
    f"{locked_row['Sensitivity'] * 100:.1f}%"
)

print(
    f"Specificity = "
    f"{locked_row['Specificity'] * 100:.1f}%"
)

print(
    f"PPV = "
    f"{locked_row['PPV'] * 100:.1f}%"
)

print(
    f"NPV = "
    f"{locked_row['NPV'] * 100:.1f}%"
)

print(
    f"Accuracy = "
    f"{locked_row['Accuracy'] * 100:.1f}%"
)

print(
    f"F1 = "
    f"{locked_row['F1']:.2f}"
)


# ============================================================
# 24. Save raw threshold table
# ============================================================

threshold_results_df.to_excel(
    "9-External_DKD_Threshold_Analysis.xlsx",
    index=False
)


# ============================================================
# 25. Manuscript-formatted threshold table
# ============================================================

threshold_manuscript_df = pd.DataFrame(
    {
        "Threshold":
            threshold_results_df[
                "Threshold"
            ].map(
                lambda x: f"{x:.2f}"
            ),

        "Sensitivity":
            threshold_results_df[
                "Sensitivity"
            ].map(
                lambda x: f"{x * 100:.1f}%"
            ),

        "Specificity":
            threshold_results_df[
                "Specificity"
            ].map(
                lambda x: f"{x * 100:.1f}%"
            ),

        "PPV":
            threshold_results_df[
                "PPV"
            ].map(
                lambda x: f"{x * 100:.1f}%"
            ),

        "NPV":
            threshold_results_df[
                "NPV"
            ].map(
                lambda x: f"{x * 100:.1f}%"
            ),

        "Accuracy":
            threshold_results_df[
                "Accuracy"
            ].map(
                lambda x: f"{x * 100:.1f}%"
            ),

        "F1":
            threshold_results_df[
                "F1"
            ].map(
                lambda x: f"{x:.2f}"
            ),

        "Primary_Locked_Threshold":
            threshold_results_df[
                "Locked_Threshold"
            ]
    }
)

threshold_manuscript_df.to_excel(
    "9-External_DKD_Threshold_Analysis_Manuscript.xlsx",
    index=False
)


# ============================================================
# 26. Threshold trade-off figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7),
    dpi=300
)

ax.plot(
    threshold_results_df[
        "Threshold"
    ],
    threshold_results_df[
        "Sensitivity"
    ],
    marker="o",
    linewidth=2,
    label="Sensitivity"
)

ax.plot(
    threshold_results_df[
        "Threshold"
    ],
    threshold_results_df[
        "Specificity"
    ],
    marker="s",
    linewidth=2,
    label="Specificity"
)

ax.axvline(
    x=LOCKED_DKD_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    label="Locked threshold = 0.45"
)

ax.set_xlabel(
    "DKD Classification Threshold",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Performance",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "External Validation: Threshold Analysis",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlim(
    0.10,
    0.90
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

ax.legend(
    loc="best",
    fontsize=10,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "9-External_DKD_Threshold_Analysis.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "9-External_DKD_Threshold_Analysis.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 27. Overall raw summary
# ============================================================

stage9_summary_df = pd.DataFrame(
    [
        {
            "Cohort":
                "External validation",

            "N":
                n_total,

            "DKD_N":
                n_dkd,

            "NDKD_N":
                n_ndkd,

            "AUC":
                auc_dkd,

            "Brier_Score":
                brier,

            "Calibration_Intercept":
                calibration_intercept,

            "Calibration_Slope":
                calibration_slope,

            "Locked_DKD_Threshold":
                LOCKED_DKD_THRESHOLD,

            "Sensitivity":
                locked_row["Sensitivity"],

            "Specificity":
                locked_row["Specificity"],

            "PPV":
                locked_row["PPV"],

            "NPV":
                locked_row["NPV"],

            "Accuracy":
                locked_row["Accuracy"],

            "F1":
                locked_row["F1"],

            "DCA_Event":
                "NDKD",

            "DCA_Probability":
                "P(NDKD) = 1 - P(DKD)"
        }
    ]
)

stage9_summary_df.to_excel(
    "9-External_Calibration_DCA_Threshold_Summary.xlsx",
    index=False
)


# ============================================================
# 28. Manuscript-formatted summary
# ============================================================

stage9_manuscript_df = pd.DataFrame(
    [
        {
            "Cohort":
                "External validation",

            "N":
                n_total,

            "DKD":
                n_dkd,

            "NDKD":
                n_ndkd,

            "AUC":
                f"{auc_dkd:.2f}",

            "Brier score":
                f"{brier:.3f}",

            "Calibration intercept":
                f"{calibration_intercept:.2f}",

            "Calibration slope":
                f"{calibration_slope:.2f}",

            "Locked threshold":
                f"{LOCKED_DKD_THRESHOLD:.2f}",

            "Sensitivity":
                f"{locked_row['Sensitivity'] * 100:.1f}%",

            "Specificity":
                f"{locked_row['Specificity'] * 100:.1f}%",

            "PPV":
                f"{locked_row['PPV'] * 100:.1f}%",

            "NPV":
                f"{locked_row['NPV'] * 100:.1f}%",

            "Accuracy":
                f"{locked_row['Accuracy'] * 100:.1f}%",

            "F1":
                f"{locked_row['F1']:.2f}"
        }
    ]
)

stage9_manuscript_df.to_excel(
    "9-External_Manuscript_Summary.xlsx",
    index=False
)


# ============================================================
# 29. Final output
# ============================================================

print("\n============================================================")
print("STAGE 9 COMPLETED")
print("============================================================")

print("\nExternal validation:")
print(f"N = {n_total}")
print(f"DKD = {n_dkd}")
print(f"NDKD = {n_ndkd}")

print("\nDiscrimination:")
print(f"AUC = {auc_dkd:.2f}")

print("\nCalibration:")
print(
    f"Brier score = "
    f"{brier:.3f}"
)
print(
    f"Calibration intercept = "
    f"{calibration_intercept:.2f}"
)
print(
    f"Calibration slope = "
    f"{calibration_slope:.2f}"
)

print("\nPrimary DKD classification threshold:")
print(
    f"Threshold = "
    f"{LOCKED_DKD_THRESHOLD:.2f}"
)
print(
    f"Sensitivity = "
    f"{locked_row['Sensitivity'] * 100:.1f}%"
)
print(
    f"Specificity = "
    f"{locked_row['Specificity'] * 100:.1f}%"
)
print(
    f"PPV = "
    f"{locked_row['PPV'] * 100:.1f}%"
)
print(
    f"NPV = "
    f"{locked_row['NPV'] * 100:.1f}%"
)
print(
    f"Accuracy = "
    f"{locked_row['Accuracy'] * 100:.1f}%"
)
print(
    f"F1 = "
    f"{locked_row['F1']:.2f}"
)

print("\nBiopsy-oriented DCA:")
print("Event = NDKD")
print("Probability = P(NDKD) = 1 - P(DKD)")
print("Comparator 1 = Biopsy all")
print("Comparator 2 = Biopsy none")

print("\nSaved files:")
print("9-External_Calibration.pdf")
print("9-External_Calibration.png")
print("9-External_Calibration_Source_Data.xlsx")
print("9-External_Calibration_Statistics.xlsx")

print("9-External_Biopsy_DCA.pdf")
print("9-External_Biopsy_DCA.png")
print("9-External_Biopsy_DCA_Source_Data.xlsx")

print("9-External_DKD_Threshold_Analysis.xlsx")
print("9-External_DKD_Threshold_Analysis_Manuscript.xlsx")
print("9-External_DKD_Threshold_Analysis.pdf")
print("9-External_DKD_Threshold_Analysis.png")

print("9-External_Calibration_DCA_Threshold_Summary.xlsx")
print("9-External_Manuscript_Summary.xlsx")
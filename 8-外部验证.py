# ============================================================
# STAGE 8
# External Validation of the Locked 7-Feature RF Model
#
# Final model:
# 7-feature Random Forest
#
# Final predictors:
# 1. Serum creatinine
# 2. DR
# 3. TC
# 4. Duration of DM
# 5. FBG
# 6. Urine protein excretion
# 7. LDL
#
# Locked development-derived OOF threshold:
# 0.45
#
# DISPLAY FORMAT FOR MANUSCRIPT:
# AUC / 95% CI: 2 decimals
# Threshold: 2 decimals
# Sensitivity / Specificity / PPV / NPV / Accuracy: percentage, 1 decimal
# F1: 2 decimals
#
# IMPORTANT:
# Excel/source data retain full numerical precision.
# ============================================================


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. 基本设置
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(45)
random.seed(45)


# ============================================================
# 2. 文件名
# ============================================================

DEVELOPMENT_FILE = "test1.xlsx"

EXTERNAL_FILE = "验证队列.xlsx"


# ============================================================
# 3. 最终7个变量
# ============================================================

FINAL_FEATURES = [
    'Serum creatinine',
    'DR',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]


FINAL_CONTINUOUS_FEATURES = [
    'Serum creatinine',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]


TARGET = 'Pathology type'


# ============================================================
# 4. 锁定threshold
#
# 来自development cohort 5-fold OOF
# 实际计算值保持0.4500
# 论文展示为0.45
# ============================================================

LOCKED_THRESHOLD = 0.4500


# ============================================================
# 5. 读取原始建模数据
# ============================================================

df = pd.read_excel(
    DEVELOPMENT_FILE
)


print("\n============================================================")
print("EXTERNAL VALIDATION")
print("Locked 7-Feature Random Forest")
print("============================================================")


print(
    f"\nDevelopment dataset loaded: "
    f"{len(df)} patients"
)


# ============================================================
# 6. 读取17个原始变量
#
# 保持与前面train_test_split完全一致
# ============================================================

all_feature_names = [
    'DR',
    'Duration of DM',
    'HbA1c',
    'Serum creatinine',
    'TC',
    'Urine protein excretion',
    'FBG',
    'BMI',
    'Age',
    'SBP',
    'LDL',
    'TG',
    'ACR',
    'DBP',
    'HDL',
    'Duration of DN',
    'Sex'
]


X = df[
    all_feature_names
].copy()


y = df[
    TARGET
].copy()


# ============================================================
# 7. 重现完全相同的80/20划分
#
# 最终RF仍然只使用原来的80% development cohort训练
# 20% internal validation不重新加入训练
# ============================================================

X_dev, X_internal, y_dev, y_internal = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=45,
    stratify=y
)


print(
    f"Development cohort used for final model: "
    f"{len(X_dev)}"
)


print(
    f"Internal validation cohort excluded from training: "
    f"{len(X_internal)}"
)


# ============================================================
# 8. Development cohort mean imputation
#
# 与Stage 5完全一致
# ============================================================

mean_imputer = SimpleImputer(
    strategy='mean'
)


X_dev_continuous = pd.DataFrame(

    mean_imputer.fit_transform(
        X_dev[
            FINAL_CONTINUOUS_FEATURES
        ]
    ),

    columns=FINAL_CONTINUOUS_FEATURES,

    index=X_dev.index
)


# ============================================================
# 9. 拼接DR
# ============================================================

X_dev_processed = pd.concat(
    [
        X_dev_continuous,

        X_dev[
            ['DR']
        ]
    ],

    axis=1
)


X_dev_processed = X_dev_processed[
    FINAL_FEATURES
]


# ============================================================
# 10. StandardScaler
#
# 与Stage 5保持一致
# ============================================================

scaler = StandardScaler()


X_dev_scaled = pd.DataFrame(

    scaler.fit_transform(
        X_dev_processed
    ),

    columns=FINAL_FEATURES,

    index=X_dev_processed.index
)


# ============================================================
# 11. 训练锁定的最终7-feature RF
#
# 不进行任何调参
# ============================================================

final_rf = RandomForestClassifier(
    random_state=45
)


final_rf.fit(
    X_dev_scaled,
    y_dev
)


print(
    "\nFinal locked RF model trained successfully."
)


# ============================================================
# 12. 保存最终model + preprocessing objects
# ============================================================

joblib.dump(
    final_rf,
    'Final_7Feature_RF_Model.joblib'
)


joblib.dump(
    mean_imputer,
    'Final_7Feature_RF_MeanImputer.joblib'
)


joblib.dump(
    scaler,
    'Final_7Feature_RF_Scaler.joblib'
)


print(
    "Final model, imputer, and scaler saved."
)


# ============================================================
# 13. 读取external validation cohort
# ============================================================

external_df = pd.read_excel(
    EXTERNAL_FILE
)


print("\n========================================")
print("External validation cohort")
print("========================================")


print(
    f"External cohort N = "
    f"{len(external_df)}"
)


# ============================================================
# 14. 检查external cohort必须包含的列
# ============================================================

required_columns = (
    FINAL_FEATURES
    +
    [TARGET]
)


missing_columns = [
    column
    for column in required_columns
    if column not in external_df.columns
]


if len(
    missing_columns
) > 0:

    raise ValueError(
        f"External dataset is missing columns: "
        f"{missing_columns}"
    )


# ============================================================
# 15. External cohort缺失情况
# ============================================================

external_missing_summary = pd.DataFrame(
    {

        'Feature':
            FINAL_FEATURES,

        'Missing_N':
            [
                external_df[
                    feature
                ].isna().sum()

                for feature in FINAL_FEATURES
            ],

        'Missing_Percent':
            [
                external_df[
                    feature
                ].isna().mean()
                *
                100

                for feature in FINAL_FEATURES
            ]
    }
)


print(
    "\nExternal cohort missing data:"
)


# 这里只控制屏幕显示
# Excel仍然保存完整精度
external_missing_print = (
    external_missing_summary.copy()
)


external_missing_print[
    'Missing_Percent'
] = external_missing_print[
    'Missing_Percent'
].round(1)


print(
    external_missing_print.to_string(
        index=False
    )
)


# ============================================================
# 16. External cohort preprocessing
#
# IMPORTANT:
# 只能transform
# 不能fit
# ============================================================

X_external_continuous = pd.DataFrame(

    mean_imputer.transform(
        external_df[
            FINAL_CONTINUOUS_FEATURES
        ]
    ),

    columns=FINAL_CONTINUOUS_FEATURES,

    index=external_df.index
)


# ============================================================
# 17. 拼接external DR
# ============================================================

X_external_processed = pd.concat(
    [
        X_external_continuous,

        external_df[
            ['DR']
        ]
    ],

    axis=1
)


X_external_processed = X_external_processed[
    FINAL_FEATURES
]


# ============================================================
# 18. 使用development scaler transform
#
# 绝对不能在external cohort重新fit scaler
# ============================================================

X_external_scaled = pd.DataFrame(

    scaler.transform(
        X_external_processed
    ),

    columns=FINAL_FEATURES,

    index=X_external_processed.index
)


# ============================================================
# 19. External true labels
# ============================================================

y_external = external_df[
    TARGET
].astype(int).values


print(
    f"\nPositive cases = "
    f"{np.sum(y_external == 1)}"
)


print(
    f"Negative cases = "
    f"{np.sum(y_external == 0)}"
)


# ============================================================
# 20. External predicted probabilities
# ============================================================

external_probability = final_rf.predict_proba(
    X_external_scaled
)[:, 1]


# ============================================================
# 21. External AUC
# ============================================================

external_auc = roc_auc_score(
    y_external,
    external_probability
)


print("\n========================================")
print("External discrimination")
print("========================================")


# 论文展示格式：AUC保留2位
print(
    f"AUC = "
    f"{external_auc:.2f}"
)


# ============================================================
# 22. Bootstrap 95% CI
#
# 2000 bootstrap replicates
#
# 对patient index进行bootstrap，
# 同时抽取label和prediction，
# 保持paired structure
# ============================================================

N_BOOTSTRAPS = 2000


rng = np.random.RandomState(
    45
)


bootstrap_aucs = []


n_external = len(
    y_external
)


for i in range(
    N_BOOTSTRAPS
):

    indices = rng.randint(
        0,
        n_external,
        n_external
    )


    y_boot = y_external[
        indices
    ]


    prob_boot = external_probability[
        indices
    ]


    # Bootstrap sample必须同时包含两个类别
    if np.unique(
        y_boot
    ).size < 2:

        continue


    auc_boot = roc_auc_score(
        y_boot,
        prob_boot
    )


    bootstrap_aucs.append(
        auc_boot
    )


bootstrap_aucs = np.array(
    bootstrap_aucs
)


auc_lower = np.percentile(
    bootstrap_aucs,
    2.5
)


auc_upper = np.percentile(
    bootstrap_aucs,
    97.5
)


# 论文展示格式：CI保留2位
print(
    f"95% CI = "
    f"{auc_lower:.2f}–"
    f"{auc_upper:.2f}"
)


print(
    f"Valid bootstrap samples = "
    f"{len(bootstrap_aucs)}"
)


# ============================================================
# 23. 使用锁定0.45 threshold分类
#
# External cohort绝对不能重新寻找Youden threshold
# ============================================================

external_prediction = (
    external_probability
    >=
    LOCKED_THRESHOLD
).astype(int)


# ============================================================
# 24. Confusion matrix
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_external,
    external_prediction,
    labels=[0, 1]
).ravel()


# ============================================================
# 25. Classification metrics
# ============================================================

sensitivity = (
    tp /
    (tp + fn)
)


specificity = (
    tn /
    (tn + fp)
)


ppv = (
    tp /
    (tp + fp)
)


npv = (
    tn /
    (tn + fn)
)


accuracy = accuracy_score(
    y_external,
    external_prediction
)


f1 = f1_score(
    y_external,
    external_prediction
)


print("\n========================================")
print("External performance at locked threshold")
print("========================================")


# Threshold：2位小数
print(
    f"Locked threshold = "
    f"{LOCKED_THRESHOLD:.2f}"
)


# 临床性能指标：
# 百分比形式，保留1位小数
print(
    f"Sensitivity = "
    f"{sensitivity * 100:.1f}%"
)


print(
    f"Specificity = "
    f"{specificity * 100:.1f}%"
)


print(
    f"PPV = "
    f"{ppv * 100:.1f}%"
)


print(
    f"NPV = "
    f"{npv * 100:.1f}%"
)


print(
    f"Accuracy = "
    f"{accuracy * 100:.1f}%"
)


# F1：保留2位小数
print(
    f"F1 = "
    f"{f1:.2f}"
)


print(
    f"TP = {tp}"
)


print(
    f"FP = {fp}"
)


print(
    f"TN = {tn}"
)


print(
    f"FN = {fn}"
)


# ============================================================
# 26. ROC curve
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_external,
    external_probability
)


fig, ax = plt.subplots(
    figsize=(7.5, 7),
    dpi=300
)


# Figure中AUC和95% CI统一保留2位小数
ax.plot(
    fpr,
    tpr,
    linewidth=2.3,
    label=(
        f'7-feature RF '
        f'(AUC = {external_auc:.2f}, '
        f'95% CI {auc_lower:.2f}–{auc_upper:.2f})'
    )
)


ax.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    linewidth=1.3,
    label='Reference'
)


ax.set_xlabel(
    '1 − Specificity',
    fontsize=14,
    fontweight='bold'
)


ax.set_ylabel(
    'Sensitivity',
    fontsize=14,
    fontweight='bold'
)


ax.set_title(
    'External Validation of the 7-Feature Random Forest Model',
    fontsize=15,
    fontweight='bold',
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


ax.xaxis.set_major_formatter(
    FormatStrFormatter('%.1f')
)


ax.yaxis.set_major_formatter(
    FormatStrFormatter('%.1f')
)


ax.tick_params(
    axis='both',
    labelsize=11
)


ax.grid(
    True,
    linestyle='--',
    linewidth=0.7,
    alpha=0.25
)


ax.spines[
    'top'
].set_visible(False)


ax.spines[
    'right'
].set_visible(False)


ax.legend(
    loc='lower right',
    fontsize=10,
    frameon=True
)


plt.tight_layout()


# ============================================================
# 27. 保存ROC
# ============================================================

plt.savefig(
    '8-Final_7Feature_RF_External_Validation_ROC.pdf',
    format='pdf',
    bbox_inches='tight'
)


plt.savefig(
    '8-Final_7Feature_RF_External_Validation_ROC.png',
    format='png',
    dpi=600,
    bbox_inches='tight'
)


plt.show()


# ============================================================
# 28. 保存每个external patient的预测结果
#
# IMPORTANT:
# Excel保存完整精度，不进行round()
# ============================================================

external_predictions_df = pd.DataFrame(
    {

        'Index':
            external_df.index,

        'True_Label':
            y_external,

        'Predicted_Probability':
            external_probability,

        'Locked_Threshold':
            LOCKED_THRESHOLD,

        'Predicted_Label':
            external_prediction
    }
)


external_predictions_df.to_excel(
    '8-Final_7Feature_RF_External_Validation_Predictions.xlsx',
    index=False
)


# ============================================================
# 29. 保存performance summary
#
# IMPORTANT:
# 这里仍然保存原始完整精度
# 不进行round()
# ============================================================

external_performance_df = pd.DataFrame(
    [
        {

            'Model':
                '7-feature Random Forest',

            'External_N':
                len(y_external),

            'Positive_N':
                int(
                    np.sum(
                        y_external == 1
                    )
                ),

            'Negative_N':
                int(
                    np.sum(
                        y_external == 0
                    )
                ),

            'AUC':
                external_auc,

            'AUC_95CI_Lower':
                auc_lower,

            'AUC_95CI_Upper':
                auc_upper,

            'Locked_Threshold':
                LOCKED_THRESHOLD,

            'Sensitivity':
                sensitivity,

            'Specificity':
                specificity,

            'PPV':
                ppv,

            'NPV':
                npv,

            'Accuracy':
                accuracy,

            'F1':
                f1,

            'TP':
                tp,

            'FP':
                fp,

            'TN':
                tn,

            'FN':
                fn
        }
    ]
)


external_performance_df.to_excel(
    '8-Final_7Feature_RF_External_Validation_Performance.xlsx',
    index=False
)


# ============================================================
# 30. 额外保存一份Manuscript格式的Performance Table
#
# 这一份专门方便复制进论文：
# AUC/CI/F1 = 2位
# Threshold = 2位
# 其他指标 = 百分比1位
# ============================================================

external_manuscript_df = pd.DataFrame(
    [
        {

            'Cohort':
                'External validation',

            'N':
                len(y_external),

            'DKD_N':
                int(
                    np.sum(
                        y_external == 1
                    )
                ),

            'NDKD_N':
                int(
                    np.sum(
                        y_external == 0
                    )
                ),

            'AUC':
                f"{external_auc:.2f}",

            '95% CI':
                f"{auc_lower:.2f}–{auc_upper:.2f}",

            'Threshold':
                f"{LOCKED_THRESHOLD:.2f}",

            'Sensitivity':
                f"{sensitivity * 100:.1f}%",

            'Specificity':
                f"{specificity * 100:.1f}%",

            'PPV':
                f"{ppv * 100:.1f}%",

            'NPV':
                f"{npv * 100:.1f}%",

            'Accuracy':
                f"{accuracy * 100:.1f}%",

            'F1':
                f"{f1:.2f}"
        }
    ]
)


external_manuscript_df.to_excel(
    '8-Final_7Feature_RF_External_Validation_Manuscript_Table.xlsx',
    index=False
)


# ============================================================
# 31. 保存ROC source data
#
# 完整精度
# ============================================================

roc_source_df = pd.DataFrame(
    {

        'FPR':
            fpr,

        'TPR':
            tpr,

        'ROC_Threshold':
            roc_thresholds
    }
)


roc_source_df.to_excel(
    '8-Final_7Feature_RF_External_Validation_ROC_Source_Data.xlsx',
    index=False
)


# ============================================================
# 32. 保存external missing-data summary
#
# 完整精度
# ============================================================

external_missing_summary.to_excel(
    '8-Final_7Feature_RF_External_Missing_Data_Summary.xlsx',
    index=False
)


# ============================================================
# 33. 最终输出
#
# 全部按照论文统一格式显示
# ============================================================

print("\n============================================================")
print("STAGE 8 EXTERNAL VALIDATION COMPLETED")
print("============================================================")


print(
    f"\nExternal validation N = "
    f"{len(y_external)}"
)


print(
    f"Positive N = "
    f"{np.sum(y_external == 1)}"
)


print(
    f"Negative N = "
    f"{np.sum(y_external == 0)}"
)


# AUC + CI：2位小数
print(
    f"\nAUC = "
    f"{external_auc:.2f} "
    f"(95% CI "
    f"{auc_lower:.2f}–"
    f"{auc_upper:.2f})"
)


# Threshold：2位小数
print(
    f"\nLocked threshold = "
    f"{LOCKED_THRESHOLD:.2f}"
)


# 临床指标：百分比1位小数
print(
    f"Sensitivity = "
    f"{sensitivity * 100:.1f}%"
)


print(
    f"Specificity = "
    f"{specificity * 100:.1f}%"
)


print(
    f"PPV = "
    f"{ppv * 100:.1f}%"
)


print(
    f"NPV = "
    f"{npv * 100:.1f}%"
)


print(
    f"Accuracy = "
    f"{accuracy * 100:.1f}%"
)


# F1：2位小数
print(
    f"F1 = "
    f"{f1:.2f}"
)


print(
    f"Confusion matrix: "
    f"TP={tp}, FP={fp}, "
    f"TN={tn}, FN={fn}"
)


print("\nSaved files:")


print(
    "8-Final_7Feature_RF_External_Validation_ROC.pdf"
)


print(
    "8-Final_7Feature_RF_External_Validation_ROC.png"
)


print(
    "8-Final_7Feature_RF_External_Validation_Predictions.xlsx"
)


print(
    "8-Final_7Feature_RF_External_Validation_Performance.xlsx"
)


print(
    "8-Final_7Feature_RF_External_Validation_Manuscript_Table.xlsx"
)


print(
    "8-Final_7Feature_RF_External_Validation_ROC_Source_Data.xlsx"
)


print(
    "8-Final_7Feature_RF_External_Missing_Data_Summary.xlsx"
)
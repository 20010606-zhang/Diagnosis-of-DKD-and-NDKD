# ============================================================
# STAGE 5
# Final 7-Feature Random Forest
# Internal Validation
#
# DISPLAY FORMAT FOR MANUSCRIPT:
# AUC / 95% CI: 2 decimals
# Threshold: 2 decimals
# Sensitivity / Specificity / PPV / NPV / Accuracy:
#     percentage, 1 decimal
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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(45)
random.seed(45)


# ============================================================
# 2. 读取数据
# ============================================================

df = pd.read_excel(
    "test1.xlsx"
)


# ============================================================
# 3. 与前面完全一致的17个原始变量
# ============================================================

feature_names = [
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


target_name = 'Pathology type'


X = df[
    feature_names
].copy()


y = df[
    target_name
].copy()


# ============================================================
# 4. 最终锁定的7-feature RF
# ============================================================

FINAL_N = 7


FINAL_FEATURES = [
    'Serum creatinine',
    'DR',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]


# ============================================================
# 5. 锁定的development OOF threshold
#
# 实际计算保持0.4500
# 论文显示为0.45
# ============================================================

OOF_THRESHOLD = 0.4500


print("\n========================================")
print("Final model specification")
print("========================================")


print(
    f"Number of features: "
    f"{FINAL_N}"
)


# 显示为2位小数
print(
    f"Locked OOF threshold: "
    f"{OOF_THRESHOLD:.2f}"
)


print("\nFinal features:")


for i, feature in enumerate(
    FINAL_FEATURES,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 6. 连续变量
#
# DR无缺失
# 其余6个连续变量采用development mean imputation
# ============================================================

FINAL_CONTINUOUS_FEATURES = [
    'Serum creatinine',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]


# ============================================================
# 7. 与前面完全一致的80/20划分
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=45,
    stratify=y
)


print("\n========================================")
print("Dataset split")
print("========================================")


print(
    f"Development cohort: "
    f"{len(X_dev)}"
)


print(
    f"Internal validation cohort: "
    f"{len(X_test)}"
)


print(
    f"Positive cases in validation cohort: "
    f"{int((y_test == 1).sum())}"
)


print(
    f"Negative cases in validation cohort: "
    f"{int((y_test == 0).sum())}"
)


# ============================================================
# 8. Mean imputation
#
# imputer只在development cohort中fit
# internal validation只transform
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


X_test_continuous = pd.DataFrame(

    mean_imputer.transform(
        X_test[
            FINAL_CONTINUOUS_FEATURES
        ]
    ),

    columns=FINAL_CONTINUOUS_FEATURES,

    index=X_test.index
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


X_test_processed = pd.concat(

    [
        X_test_continuous,

        X_test[
            ['DR']
        ]
    ],

    axis=1
)


# ============================================================
# 10. 恢复最终7个变量顺序
# ============================================================

X_dev_processed = X_dev_processed[
    FINAL_FEATURES
]


X_test_processed = X_test_processed[
    FINAL_FEATURES
]


# ============================================================
# 11. StandardScaler
#
# scaler只在development cohort中fit
# internal validation只transform
# ============================================================

scaler = StandardScaler()


X_dev_scaled = pd.DataFrame(

    scaler.fit_transform(
        X_dev_processed
    ),

    columns=FINAL_FEATURES,

    index=X_dev_processed.index
)


X_test_scaled = pd.DataFrame(

    scaler.transform(
        X_test_processed
    ),

    columns=FINAL_FEATURES,

    index=X_test_processed.index
)


# ============================================================
# 12. 最终Random Forest
#
# sklearn default RF
# random_state = 45
# ============================================================

final_rf = RandomForestClassifier(
    random_state=45
)


final_rf.fit(
    X_dev_scaled,
    y_dev
)


# ============================================================
# 13. Internal validation probability
# ============================================================

y_prob = final_rf.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 14. AUC
# ============================================================

validation_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n========================================")
print("Internal validation AUC")
print("========================================")


# AUC显示2位小数
print(
    f"AUC = "
    f"{validation_auc:.2f}"
)


# ============================================================
# 15. Bootstrap 95% CI for AUC
#
# 2000次bootstrap
# 仅用于估计AUC置信区间
# ============================================================

N_BOOTSTRAPS = 2000


rng = np.random.RandomState(
    45
)


bootstrap_aucs = []


y_test_array = np.asarray(
    y_test
)


y_prob_array = np.asarray(
    y_prob
)


for i in range(
    N_BOOTSTRAPS
):

    bootstrap_indices = rng.randint(
        0,
        len(y_test_array),
        len(y_test_array)
    )


    y_boot = y_test_array[
        bootstrap_indices
    ]


    prob_boot = y_prob_array[
        bootstrap_indices
    ]


    # bootstrap样本必须同时包含两个类别
    if np.unique(
        y_boot
    ).size < 2:

        continue


    bootstrap_auc = roc_auc_score(
        y_boot,
        prob_boot
    )


    bootstrap_aucs.append(
        bootstrap_auc
    )


bootstrap_aucs = np.array(
    bootstrap_aucs
)


auc_ci_lower = np.percentile(
    bootstrap_aucs,
    2.5
)


auc_ci_upper = np.percentile(
    bootstrap_aucs,
    97.5
)


# CI显示2位小数
print(
    f"95% CI = "
    f"{auc_ci_lower:.2f}–"
    f"{auc_ci_upper:.2f}"
)


print(
    f"Valid bootstrap samples = "
    f"{len(bootstrap_aucs)}"
)


# ============================================================
# 16. 使用锁定OOF threshold进行分类
#
# 不重新计算Youden threshold
# ============================================================

y_pred = (
    y_prob >= OOF_THRESHOLD
).astype(int)


# ============================================================
# 17. Confusion matrix
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()


# ============================================================
# 18. Classification metrics
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
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


# ============================================================
# 19. 输出最终internal validation结果
#
# 统一论文格式
# ============================================================

print("\n========================================")
print("Final 7-feature RF")
print("Internal validation performance")
print("========================================")


# AUC：2位
print(
    f"AUC = "
    f"{validation_auc:.2f}"
)


# CI：2位
print(
    f"95% CI = "
    f"{auc_ci_lower:.2f}–"
    f"{auc_ci_upper:.2f}"
)


# Threshold：2位
print(
    f"Locked threshold = "
    f"{OOF_THRESHOLD:.2f}"
)


# Sensitivity等：百分比1位
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


# F1：2位
print(
    f"F1 score = "
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
# 20. ROC curve
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)


fig, ax = plt.subplots(
    figsize=(7.5, 7),
    dpi=300
)


# ROC图中的AUC和CI均显示2位小数
ax.plot(

    fpr,

    tpr,

    linewidth=2.2,

    label=(
        f'7-feature RF '
        f'(AUC = {validation_auc:.2f}, '
        f'95% CI {auc_ci_lower:.2f}–{auc_ci_upper:.2f})'
    )
)


# Reference diagonal
ax.plot(

    [0, 1],

    [0, 1],

    linestyle='--',

    linewidth=1.2,

    label='Reference'
)


# ============================================================
# 坐标轴
# ============================================================

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
    'Internal Validation of the 7-Feature Random Forest Model',
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
    FormatStrFormatter(
        '%.1f'
    )
)


ax.yaxis.set_major_formatter(
    FormatStrFormatter(
        '%.1f'
    )
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
# 21. 保存ROC
# ============================================================

plt.savefig(
    '5-Final_7Feature_RF_Internal_Validation_ROC.pdf',
    format='pdf',
    bbox_inches='tight'
)


plt.savefig(
    '5-Final_7Feature_RF_Internal_Validation_ROC.png',
    format='png',
    dpi=600,
    bbox_inches='tight'
)


plt.show()


# ============================================================
# 22. 保存每个患者的预测结果
#
# 保留完整精度
# ============================================================

prediction_df = pd.DataFrame(
    {

        'Index':
            X_test.index,

        'True_Label':
            y_test.values,

        'Predicted_Probability':
            y_prob,

        'Locked_Threshold':
            OOF_THRESHOLD,

        'Predicted_Label':
            y_pred
    }
)


prediction_df.to_excel(
    '5-Final_7Feature_RF_Internal_Validation_Predictions.xlsx',
    index=False
)


# ============================================================
# 23. 保存performance summary
#
# 原始统计结果保留完整精度
# ============================================================

performance_df = pd.DataFrame(
    [
        {

            'Model':
                '7-feature Random Forest',

            'N':
                len(y_test),

            'Positive_N':
                int(
                    (y_test == 1).sum()
                ),

            'Negative_N':
                int(
                    (y_test == 0).sum()
                ),

            'AUC':
                validation_auc,

            'AUC_95CI_Lower':
                auc_ci_lower,

            'AUC_95CI_Upper':
                auc_ci_upper,

            'OOF_Locked_Threshold':
                OOF_THRESHOLD,

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


performance_df.to_excel(
    '5-Final_7Feature_RF_Internal_Validation_Performance.xlsx',
    index=False
)


# ============================================================
# 24. 保存Manuscript格式的Performance Table
#
# 这一份可以直接用于整理论文
# ============================================================

manuscript_df = pd.DataFrame(
    [
        {

            'Cohort':
                'Internal validation',

            'N':
                len(y_test),

            'DKD_N':
                int(
                    (y_test == 1).sum()
                ),

            'NDKD_N':
                int(
                    (y_test == 0).sum()
                ),

            'AUC':
                f"{validation_auc:.2f}",

            '95% CI':
                f"{auc_ci_lower:.2f}–{auc_ci_upper:.2f}",

            'Threshold':
                f"{OOF_THRESHOLD:.2f}",

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


manuscript_df.to_excel(
    '5-Final_7Feature_RF_Internal_Validation_Manuscript_Table.xlsx',
    index=False
)


# ============================================================
# 25. 保存ROC source data
#
# 保留完整精度
# ============================================================

roc_source_df = pd.DataFrame(
    {

        'FPR':
            fpr,

        'TPR':
            tpr,

        'ROC_Threshold':
            thresholds
    }
)


roc_source_df.to_excel(
    '5-Final_7Feature_RF_Internal_Validation_ROC_Source_Data.xlsx',
    index=False
)


# ============================================================
# 26. 保存最终模型变量
# ============================================================

feature_df = pd.DataFrame(
    {

        'Rank':
            range(
                1,
                FINAL_N + 1
            ),

        'Feature':
            FINAL_FEATURES
    }
)


feature_df.to_excel(
    '5-Final_7Feature_RF_Selected_Features.xlsx',
    index=False
)


# ============================================================
# 27. 最终输出
#
# 全部按照论文统一格式
# ============================================================

print("\n============================================================")
print("STAGE 5 COMPLETED")
print("============================================================")


print(
    f"\nFinal model: "
    f"{FINAL_N}-feature Random Forest"
)


print(
    f"Internal validation N = "
    f"{len(y_test)}"
)


# AUC / CI：2位
print(
    f"AUC = "
    f"{validation_auc:.2f} "
    f"(95% CI "
    f"{auc_ci_lower:.2f}–"
    f"{auc_ci_upper:.2f})"
)


# Threshold：2位
print(
    f"Locked development OOF threshold = "
    f"{OOF_THRESHOLD:.2f}"
)


# 临床指标：百分比1位
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


# F1：2位
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
    "5-Final_7Feature_RF_Internal_Validation_ROC.pdf"
)


print(
    "5-Final_7Feature_RF_Internal_Validation_ROC.png"
)


print(
    "5-Final_7Feature_RF_Internal_Validation_Predictions.xlsx"
)


print(
    "5-Final_7Feature_RF_Internal_Validation_Performance.xlsx"
)


print(
    "5-Final_7Feature_RF_Internal_Validation_Manuscript_Table.xlsx"
)


print(
    "5-Final_7Feature_RF_Internal_Validation_ROC_Source_Data.xlsx"
)


print(
    "5-Final_7Feature_RF_Selected_Features.xlsx"
)
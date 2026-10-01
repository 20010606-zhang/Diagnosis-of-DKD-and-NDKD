import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import warnings
import random

# ------------------------------------------------------------
# IterativeImputer是experimental功能
# 必须先enable
# ------------------------------------------------------------

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score
)


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
# 5. 最终7个变量中的连续变量
#
# DR无缺失，因此不进入MICE
# ============================================================

MICE_CONTINUOUS_FEATURES = [
    'Serum creatinine',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]


# ============================================================
# 6. 锁定的development OOF threshold
#
# 来自第三段
# 不允许在MICE sensitivity analysis重新优化
# ============================================================

OOF_THRESHOLD = 0.4500


# ============================================================
# 7. Primary analysis结果
#
# 来自第五段mean-imputation final model
# 仅用于最后比较
# ============================================================

PRIMARY_AUC = 0.8941


# ============================================================
# 8. MICE设置
#
# 5个multiply imputed datasets
# 不同random seed产生不同posterior draws
# ============================================================

MICE_SEEDS = [
    45,
    145,
    245,
    345,
    445
]


N_IMPUTATIONS = len(
    MICE_SEEDS
)


print("\n============================================================")
print("MICE SENSITIVITY ANALYSIS")
print("============================================================")


print(
    f"\nFinal model: "
    f"{FINAL_N}-feature Random Forest"
)


print(
    f"Number of imputations: "
    f"{N_IMPUTATIONS}"
)


print(
    f"Locked OOF threshold: "
    f"{OOF_THRESHOLD:.4f}"
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
# 9. 检查最终7个变量的缺失情况
# ============================================================

missing_summary = pd.DataFrame(
    {

        'Feature':
            FINAL_FEATURES,

        'Missing_N':
            [
                df[
                    feature
                ].isna().sum()

                for feature in FINAL_FEATURES
            ],

        'Missing_Percent':
            [
                (
                    df[
                        feature
                    ].isna().mean()
                    *
                    100
                )

                for feature in FINAL_FEATURES
            ]
    }
)


print("\n========================================")
print("Missing data in final 7 features")
print("========================================")


missing_print = missing_summary.copy()


missing_print[
    'Missing_Percent'
] = missing_print[
    'Missing_Percent'
].round(3)


print(
    missing_print.to_string(
        index=False
    )
)


# ============================================================
# 10. 与前面完全一致的80/20划分
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
# 11. 保存每次MICE的结果
# ============================================================

mice_results = []


# 保存5次internal validation predicted probabilities
mice_test_probabilities = []


# ============================================================
# 12. 开始5次MICE
# ============================================================

for imputation_number, seed in enumerate(
    MICE_SEEDS,
    start=1
):


    print("\n")
    print(
        "============================================================"
    )

    print(
        f"MICE imputation "
        f"{imputation_number}/{N_IMPUTATIONS}"
    )

    print(
        f"Random seed = "
        f"{seed}"
    )

    print(
        "============================================================"
    )


    # ========================================================
    # 13. 建立MICE imputer
    #
    # sample_posterior=True：
    # 允许不同imputation产生不同的合理插补值
    #
    # 只在development cohort中fit
    # ========================================================

    mice_imputer = IterativeImputer(

        max_iter=20,

        sample_posterior=True,

        random_state=seed,

        initial_strategy='mean',

        skip_complete=True
    )


    # ========================================================
    # 14. Development cohort：
    # fit + transform
    # ========================================================

    X_dev_mice_continuous = pd.DataFrame(

        mice_imputer.fit_transform(

            X_dev[
                MICE_CONTINUOUS_FEATURES
            ]
        ),

        columns=MICE_CONTINUOUS_FEATURES,

        index=X_dev.index
    )


    # ========================================================
    # 15. Internal validation：
    # 只transform
    # ========================================================

    X_test_mice_continuous = pd.DataFrame(

        mice_imputer.transform(

            X_test[
                MICE_CONTINUOUS_FEATURES
            ]
        ),

        columns=MICE_CONTINUOUS_FEATURES,

        index=X_test.index
    )


    # ========================================================
    # 16. 拼接DR
    #
    # DR无缺失，因此保留原始值
    # ========================================================

    X_dev_mice = pd.concat(

        [
            X_dev_mice_continuous,

            X_dev[
                ['DR']
            ]
        ],

        axis=1
    )


    X_test_mice = pd.concat(

        [
            X_test_mice_continuous,

            X_test[
                ['DR']
            ]
        ],

        axis=1
    )


    # ========================================================
    # 17. 恢复最终7个变量顺序
    # ========================================================

    X_dev_mice = X_dev_mice[
        FINAL_FEATURES
    ]


    X_test_mice = X_test_mice[
        FINAL_FEATURES
    ]


    # ========================================================
    # 18. StandardScaler
    #
    # 与第三、第五段完全一致
    # 每个imputed dataset分别在development fit
    # ========================================================

    scaler = StandardScaler()


    X_dev_scaled = pd.DataFrame(

        scaler.fit_transform(
            X_dev_mice
        ),

        columns=FINAL_FEATURES,

        index=X_dev_mice.index
    )


    X_test_scaled = pd.DataFrame(

        scaler.transform(
            X_test_mice
        ),

        columns=FINAL_FEATURES,

        index=X_test_mice.index
    )


    # ========================================================
    # 19. Random Forest
    #
    # 与primary analysis完全相同
    # 不重新调参
    # ========================================================

    rf_model = RandomForestClassifier(
        random_state=45
    )


    rf_model.fit(
        X_dev_scaled,
        y_dev
    )


    # ========================================================
    # 20. Internal validation probability
    # ========================================================

    y_prob_mice = rf_model.predict_proba(
        X_test_scaled
    )[:, 1]


    # 保存预测概率
    mice_test_probabilities.append(
        y_prob_mice
    )


    # ========================================================
    # 21. 当前imputation AUC
    # ========================================================

    mice_auc = roc_auc_score(
        y_test,
        y_prob_mice
    )


    print(
        f"AUC = "
        f"{mice_auc:.6f}"
    )


    # ========================================================
    # 22. 当前imputation固定threshold分类结果
    # ========================================================

    y_pred_mice = (
        y_prob_mice >= OOF_THRESHOLD
    ).astype(int)


    tn_i, fp_i, fn_i, tp_i = confusion_matrix(
        y_test,
        y_pred_mice,
        labels=[0, 1]
    ).ravel()


    sensitivity_i = (
        tp_i /
        (tp_i + fn_i)
    )


    specificity_i = (
        tn_i /
        (tn_i + fp_i)
    )


    ppv_i = (
        tp_i /
        (tp_i + fp_i)
    )


    npv_i = (
        tn_i /
        (tn_i + fn_i)
    )


    accuracy_i = accuracy_score(
        y_test,
        y_pred_mice
    )


    f1_i = f1_score(
        y_test,
        y_pred_mice
    )


    # ========================================================
    # 23. 保存当前imputation结果
    # ========================================================

    mice_results.append(
        {

            'Imputation':
                imputation_number,

            'Seed':
                seed,

            'AUC':
                mice_auc,

            'Locked_Threshold':
                OOF_THRESHOLD,

            'Sensitivity':
                sensitivity_i,

            'Specificity':
                specificity_i,

            'PPV':
                ppv_i,

            'NPV':
                npv_i,

            'Accuracy':
                accuracy_i,

            'F1':
                f1_i,

            'TP':
                tp_i,

            'FP':
                fp_i,

            'TN':
                tn_i,

            'FN':
                fn_i
        }
    )


# ============================================================
# 24. 转换5次MICE结果
# ============================================================

mice_results_df = pd.DataFrame(
    mice_results
)


# ============================================================
# 25. 5次MICE AUC summary
# ============================================================

mean_mice_auc = mice_results_df[
    'AUC'
].mean()


sd_mice_auc = mice_results_df[
    'AUC'
].std(
    ddof=1
)


min_mice_auc = mice_results_df[
    'AUC'
].min()


max_mice_auc = mice_results_df[
    'AUC'
].max()


print("\n============================================================")
print("MICE AUC SUMMARY")
print("============================================================")


print(
    f"Mean AUC = "
    f"{mean_mice_auc:.6f}"
)


print(
    f"SD = "
    f"{sd_mice_auc:.6f}"
)


print(
    f"Range = "
    f"{min_mice_auc:.6f} – "
    f"{max_mice_auc:.6f}"
)


# ============================================================
# 26. 平均5个imputed models的预测概率
#
# 注意：
# 这是prediction averaging
# 不是Rubin's rules pooled AUC
# ============================================================

mice_probability_matrix = np.vstack(
    mice_test_probabilities
)


mean_mice_probability = np.mean(
    mice_probability_matrix,
    axis=0
)


# ============================================================
# 27. Averaged-prediction AUC
# ============================================================

averaged_mice_auc = roc_auc_score(
    y_test,
    mean_mice_probability
)


print("\n============================================================")
print("AVERAGED MICE PREDICTION")
print("============================================================")


print(
    f"AUC = "
    f"{averaged_mice_auc:.6f}"
)


# ============================================================
# 28. 固定0.4500 threshold
# ============================================================

mean_mice_pred = (
    mean_mice_probability >= OOF_THRESHOLD
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    mean_mice_pred,
    labels=[0, 1]
).ravel()


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
    mean_mice_pred
)


f1 = f1_score(
    y_test,
    mean_mice_pred
)


print(
    f"Locked threshold = "
    f"{OOF_THRESHOLD:.4f}"
)


print(
    f"Sensitivity = "
    f"{sensitivity:.4f}"
)


print(
    f"Specificity = "
    f"{specificity:.4f}"
)


print(
    f"PPV = "
    f"{ppv:.4f}"
)


print(
    f"NPV = "
    f"{npv:.4f}"
)


print(
    f"Accuracy = "
    f"{accuracy:.4f}"
)


print(
    f"F1 = "
    f"{f1:.4f}"
)


print(
    f"Confusion matrix: "
    f"TP={tp}, FP={fp}, "
    f"TN={tn}, FN={fn}"
)


# ============================================================
# 29. 与primary mean-imputation model比较
# ============================================================

delta_auc = (
    averaged_mice_auc
    -
    PRIMARY_AUC
)


print("\n============================================================")
print("PRIMARY vs MICE")
print("============================================================")


print(
    f"Primary mean-imputation AUC = "
    f"{PRIMARY_AUC:.4f}"
)


print(
    f"MICE averaged-prediction AUC = "
    f"{averaged_mice_auc:.4f}"
)


print(
    f"Delta AUC "
    f"(MICE - Primary) = "
    f"{delta_auc:+.4f}"
)


# ============================================================
# 30. 保存每次MICE结果
# ============================================================

mice_results_df.to_excel(
    '6-Final_7Feature_RF_MICE_5Imputations_Performance.xlsx',
    index=False
)


# ============================================================
# 31. 保存missing data summary
# ============================================================

missing_summary.to_excel(
    '6-Final_7Feature_RF_Missing_Data_Summary.xlsx',
    index=False
)


# ============================================================
# 32. 保存5次预测概率 + averaged probability
# ============================================================

prediction_df = pd.DataFrame(
    {

        'Index':
            X_test.index,

        'True_Label':
            y_test.values
    }
)


for i in range(
    N_IMPUTATIONS
):

    prediction_df[
        f'MICE_{i + 1}_Probability'
    ] = mice_probability_matrix[
        i,
        :
    ]


prediction_df[
    'Mean_MICE_Probability'
] = mean_mice_probability


prediction_df[
    'Locked_Threshold'
] = OOF_THRESHOLD


prediction_df[
    'Predicted_Label'
] = mean_mice_pred


prediction_df.to_excel(
    '6-Final_7Feature_RF_MICE_Predictions.xlsx',
    index=False
)


# ============================================================
# 33. 保存summary
# ============================================================

summary_df = pd.DataFrame(
    [
        {

            'Model':
                'Primary mean imputation',

            'AUC':
                PRIMARY_AUC,

            'Mean_5_Imputation_AUC':
                np.nan,

            'SD_5_Imputation_AUC':
                np.nan,

            'Min_AUC':
                np.nan,

            'Max_AUC':
                np.nan,

            'Locked_Threshold':
                OOF_THRESHOLD,

            'Sensitivity':
                0.7862,

            'Specificity':
                0.8900,

            'PPV':
                0.8382,

            'NPV':
                0.8517,

            'Accuracy':
                0.8464,

            'F1':
                0.8114
        },

        {

            'Model':
                'MICE averaged prediction',

            'AUC':
                averaged_mice_auc,

            'Mean_5_Imputation_AUC':
                mean_mice_auc,

            'SD_5_Imputation_AUC':
                sd_mice_auc,

            'Min_AUC':
                min_mice_auc,

            'Max_AUC':
                max_mice_auc,

            'Locked_Threshold':
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
                f1
        }
    ]
)


summary_df.to_excel(
    '6-Final_7Feature_RF_MICE_Sensitivity_Summary.xlsx',
    index=False
)


# ============================================================
# 34. 最终输出
# ============================================================

print("\n============================================================")
print("STAGE 6 COMPLETED")
print("============================================================")


print(
    f"\nPrimary mean-imputation AUC = "
    f"{PRIMARY_AUC:.4f}"
)


print(
    f"\nFive MICE AUCs:"
)


for i, row in mice_results_df.iterrows():

    print(
        f"Imputation {int(row['Imputation'])}: "
        f"{row['AUC']:.4f}"
    )


print(
    f"\nMean MICE AUC = "
    f"{mean_mice_auc:.4f} "
    f"± {sd_mice_auc:.4f}"
)


print(
    f"MICE AUC range = "
    f"{min_mice_auc:.4f}–"
    f"{max_mice_auc:.4f}"
)


print(
    f"\nAveraged-prediction MICE AUC = "
    f"{averaged_mice_auc:.4f}"
)


print(
    f"Delta AUC vs primary = "
    f"{delta_auc:+.4f}"
)


print(
    f"\nLocked threshold = "
    f"{OOF_THRESHOLD:.4f}"
)


print(
    f"Sensitivity = "
    f"{sensitivity:.4f}"
)


print(
    f"Specificity = "
    f"{specificity:.4f}"
)


print(
    f"PPV = "
    f"{ppv:.4f}"
)


print(
    f"NPV = "
    f"{npv:.4f}"
)


print(
    f"Accuracy = "
    f"{accuracy:.4f}"
)


print(
    f"F1 = "
    f"{f1:.4f}"
)


print(
    f"Confusion matrix: "
    f"TP={tp}, FP={fp}, "
    f"TN={tn}, FN={fn}"
)


print("\nSaved files:")


print(
    "6-Final_7Feature_RF_MICE_5Imputations_Performance.xlsx"
)


print(
    "6-Final_7Feature_RF_Missing_Data_Summary.xlsx"
)


print(
    "6-Final_7Feature_RF_MICE_Predictions.xlsx"
)


print(
    "6-Final_7Feature_RF_MICE_Sensitivity_Summary.xlsx"
)
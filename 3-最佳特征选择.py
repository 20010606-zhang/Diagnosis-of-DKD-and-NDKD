import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
# 3. 17个原始变量
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
# 4. 第二段RF-RFE得到的固定排序
# ============================================================

ranked_features = [
    'Serum creatinine',
    'DR',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL',
    'HbA1c',
    'BMI',
    'SBP',
    'Age',
    'TG',
    'ACR',
    'HDL',
    'DBP',
    'Duration of DN',
    'Sex'
]


# ============================================================
# 5. 最终模型特征数
#
# 第四段模型精简分析后：
# 13-feature RF作为reference model
#
# 7-feature RF满足：
# |Delta AUC| <= 0.01
# paired DeLong P > 0.05
#
# 且为满足上述条件的最小特征集
# ============================================================

FINAL_N = 7


final_features = ranked_features[
    :FINAL_N
]


print("\n========================================")
print("Final parsimonious model")
print("========================================")

print(
    f"Number of features: "
    f"{FINAL_N}"
)

print("\nSelected features:")

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 6. 连续变量
#
# DR和Sex缺失均为0，因此无需mode imputation
# ============================================================

mean_columns = [
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
    'Duration of DN'
]


# ============================================================
# 7. 与第二、第四段完全一致的80/20划分
#
# 80% = development cohort
# 20% = held-out internal validation cohort
#
# 第三段完全不使用20% validation cohort
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
    f"Held-out internal validation cohort: "
    f"{len(X_test)}"
)


# ============================================================
# 8. 5-fold stratified CV
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=45
)


# ============================================================
# 9. 六种模型名称
# ============================================================

model_names = [
    'Random Forest',
    'Decision Tree',
    'LightGBM',
    'XGBoost',
    'SVM',
    'Logistic Regression'
]


# ============================================================
# 10. 保存1–17变量的5-fold AUC
# ============================================================

results = []


# ============================================================
# 11. 保存最终7-feature RF的OOF probability
#
# OOF predictions仅来自development cohort
# 每个样本的预测来自未包含该样本的fold-training model
# ============================================================

oof_prob_final = pd.Series(
    index=X_dev.index,
    dtype=float
)


oof_fold_final = pd.Series(
    index=X_dev.index,
    dtype=int
)


# ============================================================
# 12. 1–17变量循环
# ============================================================

for n_features in range(
    1,
    len(ranked_features) + 1
):


    selected_features = ranked_features[
        :n_features
    ]


    print("\n")

    print(
        "========================================"
    )

    print(
        f"Number of features: "
        f"{n_features}"
    )

    print(
        selected_features
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # 保存当前变量数量下
    # 六个模型各自的5-fold AUC
    # --------------------------------------------------------

    fold_aucs = {

        model_name: []

        for model_name in model_names
    }


    # ========================================================
    # 13. 5-fold CV
    # ========================================================

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(

        skf.split(
            X_dev,
            y_dev
        ),

        start=1
    ):


        # ----------------------------------------------------
        # 当前fold training / validation
        # ----------------------------------------------------

        X_fold_train = X_dev.iloc[
            train_idx
        ].copy()


        X_fold_val = X_dev.iloc[
            val_idx
        ].copy()


        y_fold_train = y_dev.iloc[
            train_idx
        ].copy()


        y_fold_val = y_dev.iloc[
            val_idx
        ].copy()


        # ====================================================
        # 14. Mean imputation
        #
        # 只在当前fold training fit
        # validation只transform
        # ====================================================

        mean_imputer = SimpleImputer(
            strategy='mean'
        )


        X_train_mean = pd.DataFrame(

            mean_imputer.fit_transform(

                X_fold_train[
                    mean_columns
                ]
            ),

            columns=mean_columns,

            index=X_fold_train.index
        )


        X_val_mean = pd.DataFrame(

            mean_imputer.transform(

                X_fold_val[
                    mean_columns
                ]
            ),

            columns=mean_columns,

            index=X_fold_val.index
        )


        # ====================================================
        # 15. 拼接Sex和DR
        # ====================================================

        X_train_processed = pd.concat(

            [
                X_train_mean,

                X_fold_train[
                    ['Sex', 'DR']
                ]
            ],

            axis=1
        )


        X_val_processed = pd.concat(

            [
                X_val_mean,

                X_fold_val[
                    ['Sex', 'DR']
                ]
            ],

            axis=1
        )


        # ----------------------------------------------------
        # 恢复17变量顺序
        # ----------------------------------------------------

        X_train_processed = (

            X_train_processed[
                feature_names
            ]
        )


        X_val_processed = (

            X_val_processed[
                feature_names
            ]
        )


        # ====================================================
        # 16. StandardScaler
        #
        # scaler只在当前fold training fit
        # validation只transform
        #
        # 为保持六种模型使用完全相同的数据处理流程，
        # 此处继续保留StandardScaler
        # ====================================================

        scaler = StandardScaler()


        X_train_scaled = pd.DataFrame(

            scaler.fit_transform(
                X_train_processed
            ),

            columns=feature_names,

            index=X_train_processed.index
        )


        X_val_scaled = pd.DataFrame(

            scaler.transform(
                X_val_processed
            ),

            columns=feature_names,

            index=X_val_processed.index
        )


        # ====================================================
        # 17. 选择当前前n个RF-RFE变量
        # ====================================================

        X_train_selected = (

            X_train_scaled[
                selected_features
            ]
        )


        X_val_selected = (

            X_val_scaled[
                selected_features
            ]
        )


        # ====================================================
        # 18. 每个fold重新建立六种模型
        # ====================================================

        models = {


            'Random Forest':

                RandomForestClassifier(
                    random_state=45
                ),


            'Decision Tree':

                DecisionTreeClassifier(
                    random_state=45
                ),


            'LightGBM':

                LGBMClassifier(
                    random_state=45,
                    verbose=-1
                ),


            'XGBoost':

                XGBClassifier(
                    eval_metric='logloss',
                    random_state=45
                ),


            'SVM':

                SVC(
                    probability=True,
                    kernel='linear',
                    random_state=45
                ),


            'Logistic Regression':

                LogisticRegression(
                    random_state=45,
                    max_iter=1000
                )
        }


        # ====================================================
        # 19. 训练并计算当前fold AUC
        # ====================================================

        for model_name, model in models.items():


            model.fit(
                X_train_selected,
                y_fold_train
            )


            y_prob = model.predict_proba(
                X_val_selected
            )[:, 1]


            auc_value = roc_auc_score(
                y_fold_val,
                y_prob
            )


            fold_aucs[
                model_name
            ].append(
                auc_value
            )


            # =================================================
            # 保存最终7-feature RF的OOF probability
            # =================================================

            if (
                n_features == FINAL_N
                and
                model_name == 'Random Forest'
            ):

                oof_prob_final.loc[
                    X_fold_val.index
                ] = y_prob


                oof_fold_final.loc[
                    X_fold_val.index
                ] = fold_number


    # ========================================================
    # 20. 汇总当前变量数量的mean AUC ± SD
    # ========================================================

    for model_name in model_names:


        auc_array = np.array(

            fold_aucs[
                model_name
            ]
        )


        mean_auc = np.mean(
            auc_array
        )


        sd_auc = np.std(
            auc_array,
            ddof=1
        )


        results.append(
            {

                'Number_of_Features':
                    n_features,

                'Model':
                    model_name,

                'Mean_AUC':
                    mean_auc,

                'SD_AUC':
                    sd_auc,

                'Selected_Features':
                    ', '.join(
                        selected_features
                    )
            }
        )


# ============================================================
# 21. 转换结果
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 22. 单独提取RF结果
# ============================================================

rf_results_df = results_df[

    results_df[
        'Model'
    ] == 'Random Forest'

].copy()


print("\n========================================")
print("Random Forest 5-fold CV results")
print("========================================")


rf_print_df = rf_results_df.copy()


rf_print_df[
    'Mean_AUC'
] = rf_print_df[
    'Mean_AUC'
].round(4)


rf_print_df[
    'SD_AUC'
] = rf_print_df[
    'SD_AUC'
].round(4)


print(

    rf_print_df.to_string(
        index=False
    )
)


# ============================================================
# 23. 检查最终7-feature RF OOF predictions是否完整
# ============================================================

print("\n========================================")

print(
    f"{FINAL_N}-feature RF OOF prediction check"
)

print(
    "========================================"
)


print(
    "Development sample size:",
    len(X_dev)
)


print(
    "Number of OOF probabilities:",
    oof_prob_final.notna().sum()
)


print(
    "Missing OOF probabilities:",
    oof_prob_final.isna().sum()
)


if oof_prob_final.isna().sum() > 0:

    raise ValueError(
        "Some development samples "
        "do not have OOF predictions."
    )


# ============================================================
# 24. 计算最终7-feature RF pooled OOF AUC
# ============================================================

oof_auc_final = roc_auc_score(

    y_dev.loc[
        oof_prob_final.index
    ],

    oof_prob_final.values
)


print(

    f"\n{FINAL_N}-feature RF pooled OOF AUC = "
    f"{oof_auc_final:.4f}"
)


# ============================================================
# 25. 使用OOF probability计算Youden threshold
#
# threshold完全来自development cohort
# 不使用held-out internal validation cohort
# ============================================================

fpr_oof, tpr_oof, thresholds_oof = roc_curve(

    y_dev.loc[
        oof_prob_final.index
    ],

    oof_prob_final.values
)


youden_index = (
    tpr_oof
    -
    fpr_oof
)


best_index = np.argmax(
    youden_index
)


oof_threshold_final = (

    thresholds_oof[
        best_index
    ]
)


oof_sensitivity = (

    tpr_oof[
        best_index
    ]
)


oof_specificity = (

    1
    -
    fpr_oof[
        best_index
    ]
)


print("\n========================================")

print(
    f"{FINAL_N}-feature RF OOF-derived threshold"
)

print(
    "========================================"
)


print(
    f"OOF threshold = "
    f"{oof_threshold_final:.4f}"
)


print(
    f"OOF sensitivity = "
    f"{oof_sensitivity:.4f}"
)


print(
    f"OOF specificity = "
    f"{oof_specificity:.4f}"
)


# ============================================================
# 26. 保存OOF prediction
# ============================================================

oof_prediction_df = pd.DataFrame(
    {

        'Index':
            X_dev.index,

        'True_Label':
            y_dev.loc[
                X_dev.index
            ].values,

        'Fold':
            oof_fold_final.loc[
                X_dev.index
            ].values,

        f'OOF_Probability_{FINAL_N}_Feature_RF':
            oof_prob_final.loc[
                X_dev.index
            ].values
    }
)


# ============================================================
# 27. 保存OOF threshold
# ============================================================

oof_threshold_df = pd.DataFrame(
    [
        {

            'Model':
                f'{FINAL_N}-feature RF',

            'OOF_AUC':
                oof_auc_final,

            'OOF_Youden_Threshold':
                oof_threshold_final,

            'OOF_Sensitivity_at_Threshold':
                oof_sensitivity,

            'OOF_Specificity_at_Threshold':
                oof_specificity
        }
    ]
)


# ============================================================
# 28. 保存Excel
#
# 保留完整精度
# ============================================================

results_df.to_excel(
    '3-RF_RFE_Development_5FoldCV_1-17变量_6模型.xlsx',
    index=False
)


rf_results_df.to_excel(
    '3-RF_RFE_Development_5FoldCV_RF结果.xlsx',
    index=False
)


oof_prediction_df.to_excel(
    f'3-{FINAL_N}Feature_RF_OOF_Predictions.xlsx',
    index=False
)


oof_threshold_df.to_excel(
    f'3-{FINAL_N}Feature_RF_OOF_Threshold.xlsx',
    index=False
)


# ============================================================
# 29. MAIN FIGURE
#
# RF 1–17 feature models
# 红星正式标记最终7-feature RF
# ============================================================

rf_plot_df = (

    rf_results_df

    .sort_values(
        'Number_of_Features'
    )

    .copy()
)


# ------------------------------------------------------------
# 最终7-feature model
# ------------------------------------------------------------

rf_final_row = rf_plot_df[

    rf_plot_df[
        'Number_of_Features'
    ] == FINAL_N

].iloc[0]


rf_final_mean = float(

    rf_final_row[
        'Mean_AUC'
    ]
)


rf_final_sd = float(

    rf_final_row[
        'SD_AUC'
    ]
)


# ============================================================
# 创建Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6.5),
    dpi=300
)


# ------------------------------------------------------------
# RF曲线 + SD error bars
# ------------------------------------------------------------

ax.errorbar(

    rf_plot_df[
        'Number_of_Features'
    ],

    rf_plot_df[
        'Mean_AUC'
    ],

    yerr=rf_plot_df[
        'SD_AUC'
    ],

    fmt='o-',

    color='#1565C0',

    ecolor='#1565C0',

    linewidth=2.0,

    markersize=6,

    capsize=4,

    capthick=1.2,

    elinewidth=1.2,

    label='Random Forest (mean ± SD)'
)


# ------------------------------------------------------------
# 最终7-feature model
# ------------------------------------------------------------

ax.scatter(

    FINAL_N,

    rf_final_mean,

    s=220,

    marker='*',

    color='#E21A1A',

    edgecolor='#E21A1A',

    linewidth=0.8,

    zorder=10,

    label=f'{FINAL_N}-feature model'
)


# ------------------------------------------------------------
# 最终模型竖线
# ------------------------------------------------------------

ax.axvline(

    x=FINAL_N,

    color='#E21A1A',

    linestyle='--',

    linewidth=1.5,

    alpha=0.90,

    zorder=1
)


# ------------------------------------------------------------
# Annotation
# ------------------------------------------------------------

annotation_text = (

    f"{FINAL_N} features\n"

    f"AUC = "
    f"{rf_final_mean:.2f} ± "
    f"{rf_final_sd:.2f}"
)


ax.annotate(

    annotation_text,

    xy=(
        FINAL_N,
        rf_final_mean
    ),

    xytext=(
        FINAL_N + 0.7,
        0.955
    ),

    fontsize=11,

    ha='left',

    va='bottom',

    color='black',

    bbox=dict(

        boxstyle='round,pad=0.4',

        facecolor='#FFF4F4',

        edgecolor='#E21A1A',

        linewidth=1.0
    ),

    arrowprops=dict(

        arrowstyle='-|>',

        color='#E21A1A',

        linewidth=1.3,

        shrinkA=0,

        shrinkB=8
    )
)


# ============================================================
# 坐标轴
# ============================================================

ax.set_xlabel(
    'Number of Features',
    fontsize=15,
    fontweight='bold'
)


ax.set_ylabel(
    'Mean AUC (5-fold CV)',
    fontsize=15,
    fontweight='bold'
)


ax.set_title(
    'Random Forest Performance by Number of Features',
    fontsize=17,
    fontweight='bold',
    pad=14
)


ax.set_xticks(
    range(
        1,
        18
    )
)


ax.tick_params(
    axis='x',
    labelsize=11
)


ax.tick_params(
    axis='y',
    labelsize=11
)


ax.set_ylim(
    0.50,
    1.00
)


ax.yaxis.set_major_formatter(

    FormatStrFormatter(
        '%.2f'
    )
)


ax.grid(

    True,

    linestyle='--',

    linewidth=0.7,

    alpha=0.30
)


ax.spines[
    'top'
].set_visible(False)


ax.spines[
    'right'
].set_visible(False)


ax.spines[
    'left'
].set_linewidth(1.2)


ax.spines[
    'bottom'
].set_linewidth(1.2)


ax.legend(

    loc='lower right',

    fontsize=10,

    frameon=True,

    framealpha=0.95
)


plt.tight_layout()


# ============================================================
# 保存主Figure
# ============================================================

plt.savefig(

    '3-Main_RF_1-17_Features_MeanAUC_SD.pdf',

    format='pdf',

    bbox_inches='tight'
)


plt.savefig(

    '3-Main_RF_1-17_Features_MeanAUC_SD.png',

    format='png',

    dpi=600,

    bbox_inches='tight'
)


plt.show()


# ============================================================
# 30. SUPPLEMENTARY FIGURE
#
# Six algorithms
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7),
    dpi=300
)


for model_name in model_names:


    temp_df = results_df[

        results_df[
            'Model'
        ] == model_name

    ].sort_values(
        'Number_of_Features'
    )


    ax.plot(

        temp_df[
            'Number_of_Features'
        ],

        temp_df[
            'Mean_AUC'
        ],

        marker='o',

        linewidth=1.5,

        markersize=4,

        label=model_name
    )


ax.set_xlabel(
    'Number of Features',
    fontsize=14
)


ax.set_ylabel(
    'Mean AUC (5-fold CV)',
    fontsize=14
)


ax.set_title(
    'Performance of Six Algorithms by Number of Features',
    fontsize=14
)


ax.set_xticks(
    range(
        1,
        18
    )
)


ax.tick_params(
    axis='x',
    labelsize=10
)


ax.tick_params(
    axis='y',
    labelsize=11
)


ax.yaxis.set_major_formatter(

    FormatStrFormatter(
        '%.2f'
    )
)


ax.legend(
    loc='lower right',
    fontsize=9
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


plt.tight_layout()


plt.savefig(

    '3-Supplementary_6Models_1-17_Features.pdf',

    format='pdf',

    bbox_inches='tight'
)


plt.savefig(

    '3-Supplementary_6Models_1-17_Features.png',

    format='png',

    dpi=600,

    bbox_inches='tight'
)


plt.show()


# ============================================================
# 31. RF Figure source data
# ============================================================

rf_figure_source_df = rf_plot_df[
    [
        'Number_of_Features',
        'Mean_AUC',
        'SD_AUC',
        'Selected_Features'
    ]
].copy()


# ------------------------------------------------------------
# 正式标记最终7-feature model
# ------------------------------------------------------------

rf_figure_source_df[
    'Selected_Final_Model'
] = np.where(

    rf_figure_source_df[
        'Number_of_Features'
    ] == FINAL_N,

    'Yes',

    'No'
)


rf_figure_source_df.to_excel(

    '3-Main_RF_Figure_Source_Data.xlsx',

    index=False
)


# ============================================================
# 32. 六模型Figure source data
# ============================================================

six_models_figure_source_df = (

    results_df[
        [
            'Number_of_Features',
            'Model',
            'Mean_AUC',
            'SD_AUC',
            'Selected_Features'
        ]
    ]

    .sort_values(
        [
            'Model',
            'Number_of_Features'
        ]
    )

    .copy()
)


six_models_figure_source_df.to_excel(

    '3-Supplementary_6Models_Figure_Source_Data.xlsx',

    index=False
)


# ============================================================
# 33. 最终输出
# ============================================================

print("\n============================================================")
print("STAGE 3 COMPLETED")
print("============================================================")


print(
    f"\nFinal parsimonious model: "
    f"{FINAL_N}-feature Random Forest"
)


print(
    "\nFinal selected features:"
)


for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


print(
    f"\nDevelopment pooled OOF AUC = "
    f"{oof_auc_final:.4f}"
)


print(
    f"OOF-derived Youden threshold = "
    f"{oof_threshold_final:.4f}"
)


print(
    f"OOF sensitivity at threshold = "
    f"{oof_sensitivity:.4f}"
)


print(
    f"OOF specificity at threshold = "
    f"{oof_specificity:.4f}"
)


print("\nSaved files:")


print(
    "3-RF_RFE_Development_5FoldCV_1-17变量_6模型.xlsx"
)


print(
    "3-RF_RFE_Development_5FoldCV_RF结果.xlsx"
)


print(
    f"3-{FINAL_N}Feature_RF_OOF_Predictions.xlsx"
)


print(
    f"3-{FINAL_N}Feature_RF_OOF_Threshold.xlsx"
)


print(
    "3-Main_RF_1-17_Features_MeanAUC_SD.pdf"
)


print(
    "3-Main_RF_1-17_Features_MeanAUC_SD.png"
)


print(
    "3-Main_RF_Figure_Source_Data.xlsx"
)


print(
    "3-Supplementary_6Models_1-17_Features.pdf"
)


print(
    "3-Supplementary_6Models_1-17_Features.png"
)


print(
    "3-Supplementary_6Models_Figure_Source_Data.xlsx"
)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import warnings
import random

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scipy.stats import norm


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
# 5. 第三段中Mean CV AUC最高的特征数
#
# 第三段结果：
# 13 features
# Mean CV AUC = 0.905055
#
# 因此手动固定13作为reference model
# ============================================================

REFERENCE_N = 13


reference_features = ranked_features[
    :REFERENCE_N
]


print("\n========================================")
print("Reference model")
print("========================================")

print(
    f"Reference feature number: "
    f"{REFERENCE_N}"
)

print("\nReference features:")

for i, feature in enumerate(
    reference_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 6. 连续变量
#
# DR和Sex无缺失
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
# 7. 与第二、第三段完全一致的80/20划分
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
    f"Held-out test cohort: "
    f"{len(X_test)}"
)

print(
    f"Test positive: "
    f"{int((y_test == 1).sum())}"
)

print(
    f"Test negative: "
    f"{int((y_test == 0).sum())}"
)


# ============================================================
# 8. Mean imputation
#
# 只在development cohort中fit
# held-out test只transform
# ============================================================

mean_imputer = SimpleImputer(
    strategy='mean'
)


X_dev_mean = pd.DataFrame(
    mean_imputer.fit_transform(
        X_dev[
            mean_columns
        ]
    ),
    columns=mean_columns,
    index=X_dev.index
)


X_test_mean = pd.DataFrame(
    mean_imputer.transform(
        X_test[
            mean_columns
        ]
    ),
    columns=mean_columns,
    index=X_test.index
)


# ============================================================
# 9. 拼接Sex和DR
# ============================================================

X_dev_processed = pd.concat(
    [
        X_dev_mean,
        X_dev[
            ['Sex', 'DR']
        ]
    ],
    axis=1
)


X_test_processed = pd.concat(
    [
        X_test_mean,
        X_test[
            ['Sex', 'DR']
        ]
    ],
    axis=1
)


# 恢复原始17变量顺序
X_dev_processed = (
    X_dev_processed[
        feature_names
    ]
)


X_test_processed = (
    X_test_processed[
        feature_names
    ]
)


# ============================================================
# 10. 训练1–13 feature RF
#
# 每个模型：
# development cohort训练
# held-out test cohort预测
#
# 保存所有held-out probabilities
# ============================================================

prediction_dict = {}

auc_dict = {}


print("\n========================================")
print("Held-out RF AUC")
print("========================================")


for n_features in range(
    1,
    REFERENCE_N + 1
):


    selected_features = ranked_features[
        :n_features
    ]


    # --------------------------------------------------------
    # Random Forest
    #
    # 与前面保持完全相同：
    # default parameters
    # random_state = 45
    # --------------------------------------------------------

    model = RandomForestClassifier(
        random_state=45
    )


    model.fit(
        X_dev_processed[
            selected_features
        ],
        y_dev
    )


    # --------------------------------------------------------
    # held-out probability
    # --------------------------------------------------------

    y_prob = model.predict_proba(
        X_test_processed[
            selected_features
        ]
    )[:, 1]


    prediction_dict[
        n_features
    ] = y_prob


    # --------------------------------------------------------
    # held-out AUC
    # --------------------------------------------------------

    auc_value = roc_auc_score(
        y_test,
        y_prob
    )


    auc_dict[
        n_features
    ] = auc_value


    print(
        f"{n_features:2d} features: "
        f"AUC = {auc_value:.6f}"
    )


# ============================================================
# 11. DeLong test functions
#
# 用于同一个held-out cohort中的
# correlated ROC curves
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
            and
            Z[j] == Z[i]
        ):

            j += 1


        T[i:j] = (
            0.5
            *
            (
                i
                +
                j
                -
                1
            )
        )


        i = j


    T2 = np.empty(
        N,
        dtype=float
    )


    T2[J] = T


    return T2 + 1


# ============================================================

def fast_delong(
    predictions_sorted_transposed,
    label_1_count
):


    m = label_1_count

    n = (
        predictions_sorted_transposed.shape[1]
        -
        m
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
        [
            k,
            m
        ]
    )


    ty = np.empty(
        [
            k,
            n
        ]
    )


    tz = np.empty(
        [
            k,
            m + n
        ]
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
        tz[
            :,
            :m
        ].sum(
            axis=1
        )
        /
        m
        /
        n
        -
        (
            m + 1.0
        )
        /
        2.0
        /
        n
    )


    v01 = (
        tz[
            :,
            :m
        ]
        -
        tx
    ) / n


    v10 = (
        1.0
        -
        (
            tz[
                :,
                m:
            ]
            -
            ty
        )
        /
        m
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


# ============================================================

def calc_pvalue(
    aucs,
    sigma
):


    contrast = np.array(
        [
            [1, -1]
        ]
    )


    variance = float(
        contrast
        @
        sigma
        @
        contrast.T
    )


    if variance <= 0:

        return np.nan


    z_value = (
        abs(
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
        (
            1
            -
            norm.cdf(
                z_value
            )
        )
    )


    return float(
        p_value
    )


# ============================================================

def delong_roc_test(
    ground_truth,
    prediction_reference,
    prediction_comparator
):


    ground_truth = np.asarray(
        ground_truth
    )


    prediction_reference = np.asarray(
        prediction_reference
    )


    prediction_comparator = np.asarray(
        prediction_comparator
    )


    # --------------------------------------------------------
    # DeLong要求positive samples排列在前
    # --------------------------------------------------------

    order = np.argsort(
        -ground_truth
    )


    label_1_count = int(
        np.sum(
            ground_truth
        )
    )


    predictions_sorted_transposed = np.vstack(
        [
            prediction_reference,
            prediction_comparator
        ]
    )[
        :,
        order
    ]


    aucs, covariance = fast_delong(
        predictions_sorted_transposed,
        label_1_count
    )


    p_value = calc_pvalue(
        aucs,
        covariance
    )


    return (
        float(
            aucs[0]
        ),

        float(
            aucs[1]
        ),

        p_value
    )


# ============================================================
# 12. 13-feature reference probability
# ============================================================

reference_probability = (
    prediction_dict[
        REFERENCE_N
    ]
)


reference_auc = (
    auc_dict[
        REFERENCE_N
    ]
)


print("\n========================================")
print("Reference held-out performance")
print("========================================")

print(
    f"{REFERENCE_N}-feature RF "
    f"AUC = {reference_auc:.6f}"
)


# ============================================================
# 13. 1–12 feature RF分别与13-feature RF进行paired DeLong
# ============================================================

delong_results = []


for n_features in range(
    1,
    REFERENCE_N
):


    comparator_probability = (
        prediction_dict[
            n_features
        ]
    )


    auc_reference, \
    auc_comparator, \
    p_value = delong_roc_test(

        y_test.values,

        reference_probability,

        comparator_probability
    )


    # --------------------------------------------------------
    # Delta AUC定义：
    #
    # reference AUC - reduced model AUC
    #
    # 正值 = reference更高
    # --------------------------------------------------------

    delta_auc = (
        auc_reference
        -
        auc_comparator
    )


    delong_results.append(
        {

            'Reference_Model':
                f'{REFERENCE_N}-feature RF',

            'Comparator_Model':
                f'{n_features}-feature RF',

            'Reference_AUC':
                auc_reference,

            'Comparator_AUC':
                auc_comparator,

            'Delta_AUC':
                delta_auc,

            'Absolute_Delta_AUC':
                abs(
                    delta_auc
                ),

            'DeLong_P':
                p_value,

            'Comparator_Features':
                ', '.join(
                    ranked_features[
                        :n_features
                    ]
                )
        }
    )


delong_df = pd.DataFrame(
    delong_results
)


# ============================================================
# 14. 打印完整DeLong结果
# ============================================================

print("\n")
print("============================================================")
print("PAIRED DELONG TEST")
print(
    f"Reference model = "
    f"{REFERENCE_N}-feature RF"
)
print("============================================================")


for _, row in delong_df.iterrows():


    print(
        f"{row['Comparator_Model']:15s} "
        f"vs "
        f"{row['Reference_Model']:15s} | "
        f"AUC = "
        f"{row['Comparator_AUC']:.6f} "
        f"vs "
        f"{row['Reference_AUC']:.6f} | "
        f"Delta = "
        f"{row['Delta_AUC']:.6f} | "
        f"P = "
        f"{row['DeLong_P']:.6f}"
    )


# ============================================================
# 15. 增加两个描述性判断列
#
# 注意：
# 这里只是方便我们下一步看结果
# 暂时不自动选择最终模型
# ============================================================

delong_df[
    'P_gt_0.05'
] = (
    delong_df[
        'DeLong_P'
    ]
    >
    0.05
)


delong_df[
    'Abs_Delta_AUC_le_0.01'
] = (
    delong_df[
        'Absolute_Delta_AUC'
    ]
    <=
    0.01
)


delong_df[
    'Both'
] = (
    delong_df[
        'P_gt_0.05'
    ]
    &
    delong_df[
        'Abs_Delta_AUC_le_0.01'
    ]
)


# ============================================================
# 16. 单独打印parsimony相关结果
#
# 暂时只是看：
# P > 0.05
# Delta AUC <= 0.01
#
# 不自动选择
# ============================================================

print("\n")
print("============================================================")
print("PARSIMONY-RELATED RESULTS")
print("============================================================")


for _, row in delong_df.iterrows():


    print(
        f"{row['Comparator_Model']:15s} | "
        f"AUC = "
        f"{row['Comparator_AUC']:.4f} | "
        f"|Delta AUC| = "
        f"{row['Absolute_Delta_AUC']:.4f} | "
        f"P = "
        f"{row['DeLong_P']:.4f} | "
        f"P>0.05 = "
        f"{row['P_gt_0.05']} | "
        f"|Delta|<=0.01 = "
        f"{row['Abs_Delta_AUC_le_0.01']}"
    )


# ============================================================
# 17. 保存所有held-out probabilities
#
# 后面如果要做：
# ROC
# calibration
# DCA
# threshold
# 都可以直接使用
# ============================================================

prediction_df = pd.DataFrame(
    {
        'Index':
            X_test.index,

        'True_Label':
            y_test.loc[
                X_test.index
            ].values
    }
)


for n_features in range(
    1,
    REFERENCE_N + 1
):


    prediction_df[
        f'Probability_{n_features}_Feature_RF'
    ] = prediction_dict[
        n_features
    ]


# ============================================================
# 18. 保存1–13 held-out AUC
# ============================================================

auc_results = []


for n_features in range(
    1,
    REFERENCE_N + 1
):


    auc_results.append(
        {

            'Number_of_Features':
                n_features,

            'Heldout_AUC':
                auc_dict[
                    n_features
                ],

            'Features':
                ', '.join(
                    ranked_features[
                        :n_features
                    ]
                )
        }
    )


auc_df = pd.DataFrame(
    auc_results
)


# ============================================================
# 19. 保存Excel
#
# 不round，保留完整精度
# ============================================================

delong_df.to_excel(
    '4-Delong_1-12_vs_13Feature_RF.xlsx',
    index=False
)


auc_df.to_excel(
    '4-Heldout_AUC_1-13Feature_RF.xlsx',
    index=False
)


prediction_df.to_excel(
    '4-Heldout_Predictions_1-13Feature_RF.xlsx',
    index=False
)


# ============================================================
# 20. 最终输出
# ============================================================

print("\n")
print("============================================================")
print("STAGE 4 COMPLETED")
print("============================================================")


print(
    f"Reference model: "
    f"{REFERENCE_N}-feature RF"
)


print(
    f"Reference held-out AUC: "
    f"{reference_auc:.6f}"
)


print("\nSaved files:")

print(
    "4-Delong_1-12_vs_13Feature_RF.xlsx"
)

print(
    "4-Heldout_AUC_1-13Feature_RF.xlsx"
)

print(
    "4-Heldout_Predictions_1-13Feature_RF.xlsx"
)


print("\nNo final feature number has been selected yet.")
# ============================================================
# 21. ROC + DeLong Figure
#
# 展示：
# 5-, 6-, 7-, 8-, 13-feature RF
#
# 13-feature RF = reference model
# ============================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)


# ============================================================
# 21A. 要展示的模型
# ============================================================

PLOT_FEATURES = [
    5,
    6,
    7,
    8,
    13
]


# ============================================================
# 21B. Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 7.5),
    dpi=300
)


# ============================================================
# 21C. ROC curves
# ============================================================

for n_features in PLOT_FEATURES:

    y_prob = prediction_dict[
        n_features
    ]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc_value = roc_auc_score(
        y_test,
        y_prob
    )


    # --------------------------------------------------------
    # reference model线稍微加粗
    # --------------------------------------------------------

    if n_features == REFERENCE_N:

        ax.plot(
            fpr,
            tpr,
            linewidth=2.8,
            label=(
                f'{n_features} features '
                f'(reference), '
                f'AUC = {auc_value:.3f}'
            )
        )

    else:

        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            label=(
                f'{n_features} features, '
                f'AUC = {auc_value:.3f}'
            )
        )


# ============================================================
# 21D. Chance line
# ============================================================

ax.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    linewidth=1.2,
    color='gray',
    label='Chance'
)


# ============================================================
# 21E. DeLong结果文本
#
# 从前面已经计算好的delong_df中自动读取
# 不手动输入P值
# ============================================================

delong_text_lines = [
    f'DeLong test vs {REFERENCE_N}-feature reference'
]


for n_features in [
    5,
    6,
    7,
    8
]:

    row = delong_df[
        delong_df[
            'Comparator_Model'
        ] == f'{n_features}-feature RF'
    ].iloc[0]


    delta_auc = float(
        row[
            'Delta_AUC'
        ]
    )

    p_value = float(
        row[
            'DeLong_P'
        ]
    )


    delong_text_lines.append(
        f'{n_features} vs {REFERENCE_N}: '
        f'ΔAUC = {delta_auc:.3f}, '
        f'P = {p_value:.3f}'
    )


delong_text = '\n'.join(
    delong_text_lines
)


# ============================================================
# 21F. DeLong结果框
# ============================================================

ax.text(

    0.97,
    0.05,

    delong_text,

    transform=ax.transAxes,

    fontsize=10.5,

    horizontalalignment='right',

    verticalalignment='bottom',

    linespacing=1.45,

    bbox=dict(
        boxstyle='round,pad=0.55',
        facecolor='white',
        edgecolor='black',
        linewidth=1.0,
        alpha=0.95
    )
)


# ============================================================
# 21G. 坐标轴
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
    'ROC Comparison of Reduced Random Forest Models',
    fontsize=16,
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


ax.tick_params(
    axis='both',
    labelsize=11
)


# ============================================================
# 21H. Grid
# ============================================================

ax.grid(
    True,
    linestyle='--',
    linewidth=0.6,
    alpha=0.25
)


# ============================================================
# 21I. 去除顶部和右侧边框
# ============================================================

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


# ============================================================
# 21J. Legend
# ============================================================

ax.legend(
    loc='lower right',
    bbox_to_anchor=(
        1.0,
        0.40
    ),
    fontsize=9.5,
    frameon=True,
    framealpha=0.95
)


plt.tight_layout()


# ============================================================
# 21K. 保存Figure
# ============================================================

plt.savefig(
    '4-ROC_DeLong_Reduced_RF_vs_13Feature.pdf',
    format='pdf',
    bbox_inches='tight'
)


plt.savefig(
    '4-ROC_DeLong_Reduced_RF_vs_13Feature.png',
    format='png',
    dpi=600,
    bbox_inches='tight'
)


plt.show()


# ============================================================
# 21L. 保存这张图对应的source data
# ============================================================

roc_source_list = []


for n_features in PLOT_FEATURES:

    y_prob = prediction_dict[
        n_features
    ]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )


    temp_df = pd.DataFrame(
        {
            'Model':
                f'{n_features}-feature RF',

            'FPR':
                fpr,

            'TPR':
                tpr,

            'Threshold':
                thresholds
        }
    )


    roc_source_list.append(
        temp_df
    )


roc_source_df = pd.concat(
    roc_source_list,
    ignore_index=True
)


roc_source_df.to_excel(
    '4-ROC_DeLong_Figure_Source_Data.xlsx',
    index=False
)


# ============================================================
# 21M. 保存Figure中使用的DeLong结果
# ============================================================

figure_delong_df = delong_df[
    delong_df[
        'Comparator_Model'
    ].isin(
        [
            '5-feature RF',
            '6-feature RF',
            '7-feature RF',
            '8-feature RF'
        ]
    )
].copy()


figure_delong_df.to_excel(
    '4-ROC_DeLong_Figure_Statistics.xlsx',
    index=False
)


print("\n========================================")
print("ROC + DeLong Figure saved")
print("========================================")

print(
    "4-ROC_DeLong_Reduced_RF_vs_13Feature.pdf"
)

print(
    "4-ROC_DeLong_Reduced_RF_vs_13Feature.png"
)

print(
    "4-ROC_DeLong_Figure_Source_Data.xlsx"
)

print(
    "4-ROC_DeLong_Figure_Statistics.xlsx"
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import warnings
import shap


# ============================================================
# 1. 基本设置
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置全局字体为 Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
})

# 全局随机种子
np.random.seed(45)
random.seed(45)

RANDOM_STATE = 45


# ============================================================
# 2. 最终7个特征
# ============================================================

feature_names = [
    'Serum creatinine',
    'DR',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]

# 连续变量：均值插补
mean_columns = [
    'Serum creatinine',
    'TC',
    'Duration of DM',
    'FBG',
    'Urine protein excretion',
    'LDL'
]

target_name = 'Pathology type'


# ============================================================
# 3. 读取数据
# ============================================================

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise


print("\n============================================================")
print("FINAL 7-FEATURE RF SHAP ANALYSIS")
print("============================================================")

print(f"总样本量: {len(df)}")
print("最终特征:")
for feature in feature_names:
    print(f"  - {feature}")


# ============================================================
# 4. 检查必要变量
# ============================================================

required_columns = feature_names + [target_name]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) > 0:
    raise ValueError(
        f"数据中缺少以下变量: {missing_columns}"
    )


# DR在最终数据中应无缺失
n_missing_dr = df['DR'].isna().sum()

print(f"\nDR缺失值数量: {n_missing_dr}")

if n_missing_dr > 0:
    raise ValueError(
        "DR存在缺失值。最终主分析流程未对DR进行插补，"
        "请先检查原始数据。"
    )


# ============================================================
# 5. 提取原始特征和结局
# ============================================================

X_raw = df[feature_names].copy()
y = df[target_name].astype(int).copy()


# ============================================================
# 6. 先划分 development / internal validation
#
# 重要：
# 必须先split，再拟合imputer，避免信息泄漏
# ============================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")

print(f"Development N = {len(X_train_raw)}")
print(f"Internal validation N = {len(X_test_raw)}")

print(f"Development DKD = {(y_train == 1).sum()}")
print(f"Development NDKD = {(y_train == 0).sum()}")

print(f"Internal DKD = {(y_test == 1).sum()}")
print(f"Internal NDKD = {(y_test == 0).sum()}")


# ============================================================
# 7. 缺失值填补
#
# 仅使用development cohort拟合均值
# ============================================================

mean_imputer = SimpleImputer(
    strategy='mean'
)


# Development cohort
X_train_mean = pd.DataFrame(
    mean_imputer.fit_transform(
        X_train_raw[mean_columns]
    ),
    columns=mean_columns,
    index=X_train_raw.index
)


# Internal validation cohort
X_test_mean = pd.DataFrame(
    mean_imputer.transform(
        X_test_raw[mean_columns]
    ),
    columns=mean_columns,
    index=X_test_raw.index
)


# ============================================================
# 8. 加回DR，并恢复最终特征顺序
# ============================================================

X_train = X_train_mean.copy()
X_test = X_test_mean.copy()


X_train['DR'] = X_train_raw['DR'].values
X_test['DR'] = X_test_raw['DR'].values


X_train = X_train[feature_names]
X_test = X_test[feature_names]


# ============================================================
# 9. 保存处理后的数据
# ============================================================

train_output = X_train.copy()
train_output[target_name] = y_train.values
train_output['Dataset'] = 'Development'

test_output = X_test.copy()
test_output[target_name] = y_test.values
test_output['Dataset'] = 'Internal validation'

data_with_target = pd.concat(
    [
        train_output,
        test_output
    ],
    axis=0
)

data_with_target.to_csv(
    'your_data_2.csv',
    index=False
)


# ============================================================
# 10. 创建最终随机森林分类器
#
# 与最终模型一致：
# RandomForestClassifier(random_state=45)
# 其余参数使用默认值
# ============================================================

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)

rf_classifier.fit(
    X_train,
    y_train
)


# ============================================================
# 11. 检查模型是否复现最终结果
# ============================================================

y_test_prob = rf_classifier.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    y_test_prob
)


print("\n============================================================")
print("SANITY CHECK")
print("============================================================")

print("Expected internal-validation AUC ≈ 0.894")
print(f"Current internal-validation AUC = {auc:.4f}")

if abs(auc - 0.8941) <= 0.005:
    print("Final RF reproduction check: PASS")
else:
    print("Final RF reproduction check: CHECK PIPELINE")


# ============================================================
# 12. 计算SHAP值
# ============================================================

explainer = shap.TreeExplainer(
    rf_classifier
)

raw_shap_values = explainer.shap_values(
    X_test
)


# ============================================================
# 13. 提取DKD正类（class = 1）的SHAP值
#
# 兼容不同SHAP版本
# ============================================================

if isinstance(raw_shap_values, list):

    # 旧版本SHAP：
    # [class0, class1]
    shap_values = np.asarray(
        raw_shap_values[1]
    )

else:

    raw_shap_values = np.asarray(
        raw_shap_values
    )

    if raw_shap_values.ndim == 3:

        # 新版本：
        # (samples, features, classes)
        shap_values = raw_shap_values[
            :,
            :,
            1
        ]

    elif raw_shap_values.ndim == 2:

        shap_values = raw_shap_values

    else:

        raise ValueError(
            f"无法识别SHAP输出格式: "
            f"{raw_shap_values.shape}"
        )


# ============================================================
# 14. 打印SHAP值信息
# ============================================================

print("\n============================================================")
print("SHAP INFORMATION")
print("============================================================")

print(
    "shap_values 的类型:",
    type(shap_values)
)

print(
    "shap_values 的形状:",
    shap_values.shape
)

print(
    f"SHAP值特征数量: "
    f"{shap_values.shape[1]}"
)


# ============================================================
# 15. SHAP全局重要性
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)


shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Absolute_SHAP': mean_abs_shap
})


shap_importance_df = (
    shap_importance_df
    .sort_values(
        'Mean_Absolute_SHAP',
        ascending=False
    )
    .reset_index(drop=True)
)


shap_importance_df['Rank'] = np.arange(
    1,
    len(shap_importance_df) + 1
)


shap_importance_df = shap_importance_df[
    [
        'Rank',
        'Feature',
        'Mean_Absolute_SHAP'
    ]
]


print("\n============================================================")
print("GLOBAL SHAP IMPORTANCE")
print("============================================================")

print(
    shap_importance_df.to_string(
        index=False
    )
)


shap_importance_df.to_excel(
    'shap_global_importance.xlsx',
    index=False
)


# ============================================================
# 16. 通用字体设置函数
# ============================================================

def set_font_for_all_text(obj):

    if hasattr(obj, 'get_text'):

        try:
            obj.set_fontfamily(
                'Times New Roman'
            )
        except Exception:
            pass

    if hasattr(obj, 'get_children'):

        for child in obj.get_children():

            set_font_for_all_text(
                child
            )


# ============================================================
# 17. 绘制并保存SHAP摘要图（蜜蜂图）
# ============================================================

def save_shap_summary_plot(
    shap_values,
    X_test,
    filename='shap_summary_plot.pdf'
):

    print(
        f"\n正在保存SHAP摘要图到 "
        f"{filename}..."
    )


    plt.close('all')


    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type='dot',
        max_display=len(feature_names),
        show=False
    )


    fig = plt.gcf()

    fig.set_size_inches(
        8,
        6
    )


    ax = plt.gca()


    # 所有文字设置为Times New Roman
    set_font_for_all_text(
        fig
    )


    # 坐标轴
    if ax.get_xlabel():

        ax.set_xlabel(
            ax.get_xlabel(),
            fontfamily='Times New Roman',
            fontsize=8
        )


    if ax.get_ylabel():

        ax.set_ylabel(
            ax.get_ylabel(),
            fontfamily='Times New Roman',
            fontsize=8
        )


    # 刻度字体
    for tick in ax.get_xticklabels():

        tick.set_fontfamily(
            'Times New Roman'
        )

        tick.set_fontsize(
            8
        )


    for tick in ax.get_yticklabels():

        tick.set_fontfamily(
            'Times New Roman'
        )

        tick.set_fontsize(
            8
        )


    # SHAP summary_plot通常会增加颜色条axes
    # 对figure中的所有axes统一字体
    for current_ax in fig.axes:

        for tick in current_ax.get_xticklabels():

            tick.set_fontfamily(
                'Times New Roman'
            )

            tick.set_fontsize(
                8
            )


        for tick in current_ax.get_yticklabels():

            tick.set_fontfamily(
                'Times New Roman'
            )

            tick.set_fontsize(
                8
            )


        if current_ax.get_xlabel():

            current_ax.xaxis.label.set_fontfamily(
                'Times New Roman'
            )


        if current_ax.get_ylabel():

            current_ax.yaxis.label.set_fontfamily(
                'Times New Roman'
            )


    plt.tight_layout()

    fig.canvas.draw()


    fig.savefig(
        filename,
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )


    plt.close(
        fig
    )


    print(
        f"SHAP摘要图已保存为 "
        f"{filename}"
    )


# ============================================================
# 18. 绘制并保存SHAP条形图
# ============================================================

def save_shap_bar_plot(
    shap_values,
    X_test,
    filename='shap_summary_bar_plot.pdf'
):

    print(
        f"\n正在保存SHAP条形图到 "
        f"{filename}..."
    )


    plt.close('all')


    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type='bar',
        max_display=len(feature_names),
        show=False
    )


    fig = plt.gcf()

    fig.set_size_inches(
        8,
        6
    )


    ax = plt.gca()


    # 所有字体Times New Roman
    set_font_for_all_text(
        fig
    )


    if ax.get_xlabel():

        ax.set_xlabel(
            ax.get_xlabel(),
            fontfamily='Times New Roman',
            fontsize=8
        )


    if ax.get_ylabel():

        ax.set_ylabel(
            ax.get_ylabel(),
            fontfamily='Times New Roman',
            fontsize=8
        )


    for tick in ax.get_xticklabels():

        tick.set_fontfamily(
            'Times New Roman'
        )

        tick.set_fontsize(
            8
        )


    for tick in ax.get_yticklabels():

        tick.set_fontfamily(
            'Times New Roman'
        )

        tick.set_fontsize(
            8
        )


    plt.tight_layout()

    fig.canvas.draw()


    fig.savefig(
        filename,
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )


    plt.close(
        fig
    )


    print(
        f"SHAP条形图已保存为 "
        f"{filename}"
    )


# ============================================================
# 19. 保存SHAP源数据
# ============================================================

shap_source_df = X_test.copy()


for i, feature in enumerate(
    feature_names
):

    shap_source_df[
        f'SHAP_{feature}'
    ] = shap_values[
        :,
        i
    ]


shap_source_df[
    'True_Label'
] = y_test.values


shap_source_df[
    'Predicted_P_DKD'
] = y_test_prob


shap_source_df.to_excel(
    'shap_summary_source_data.xlsx',
    index=False
)


# ============================================================
# 20. 调用函数
# ============================================================

save_shap_summary_plot(
    shap_values,
    X_test,
    'shap_summary_plot.pdf'
)


save_shap_bar_plot(
    shap_values,
    X_test,
    'shap_summary_bar_plot.pdf'
)


# ============================================================
# 21. 完成信息
# ============================================================

print("\n============================================================")
print("SHAP ANALYSIS COMPLETED")
print("============================================================")

print(
    f"Internal-validation AUC = "
    f"{auc:.4f}"
)

print("\n已保存文件:")

print(
    "1. shap_summary_plot.pdf"
)

print(
    "2. shap_summary_bar_plot.pdf"
)

print(
    "3. shap_global_importance.xlsx"
)

print(
    "4. shap_summary_source_data.xlsx"
)

print(
    "5. your_data_2.csv"
)

print(
    "\n所有SHAP图均保存为PDF矢量图，"
    "字体为Times New Roman。"
)
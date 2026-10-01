import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
import random

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置全局字体和符号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 设置全局随机种子
np.random.seed(45)
random.seed(45)

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = [
    'DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
    'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
    'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex'
]

target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 需要进行均值填充的数值型特征
mean_columns = [
    'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
    'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
    'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN'
]


X_train_all, X_validation, y_train_all, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45,
    stratify=y
)

# 模型定义
models = [
    RandomForestClassifier(random_state=45),
    DecisionTreeClassifier(random_state=45),
    LGBMClassifier(random_state=45, verbose=-1),
    XGBClassifier(eval_metric='logloss', random_state=45),
    SVC(probability=True, random_state=45, kernel='linear'),
    LogisticRegression(random_state=45, max_iter=1000)
]

model_names = ["RF", "DT", "LightGBM", "XGBoost", "SVM", "LR"]

n_iterations = 10

colors = ['b', 'g', 'r', 'c', 'm', 'y']

# 初始化存储ROC指标的字典
train_all_fpr = {name: [] for name in model_names}
train_all_tpr = {name: [] for name in model_names}
train_all_auc = {name: [] for name in model_names}

val_all_fpr = {name: [] for name in model_names}
val_all_tpr = {name: [] for name in model_names}
val_all_auc = {name: [] for name in model_names}

# 存储每种模型每次迭代的其他指标
all_metrics = {
    name: {'AUC': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [], 'Accuracy': [], 'F1-score': []} for
    name in model_names}



def calculate_metrics(model, X_test, y_test):

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return (
        auc_score,
        sensitivity,
        specificity,
        ppv,
        npv,
        accuracy,
        f1
    )


# 创建固定种子序列
split_seeds = [45 + i for i in range(n_iterations)]


# ============================================================
# 训练模型并收集指标
# ============================================================

for i in range(n_iterations):

    # 使用固定种子进行划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.2,
        random_state=split_seeds[i],
        stratify=y_train_all
    )

    # ========================================================
    # 修改2：
    # 均值填充器只在当前X_train上进行fit
    # ========================================================

    mean_imputer = SimpleImputer(strategy='mean')

    # 当前训练集：fit + transform
    X_train_mean = pd.DataFrame(
        mean_imputer.fit_transform(X_train[mean_columns]),
        columns=mean_columns,
        index=X_train.index
    )

    # 当前测试集：只transform
    X_test_mean = pd.DataFrame(
        mean_imputer.transform(X_test[mean_columns]),
        columns=mean_columns,
        index=X_test.index
    )

    # 固定验证集：只transform
    X_validation_mean = pd.DataFrame(
        mean_imputer.transform(X_validation[mean_columns]),
        columns=mean_columns,
        index=X_validation.index
    )

    # ========================================================
    # 拼接Sex和DR
    # 保持与你原始代码完全相同的处理方式
    # ========================================================

    X_train_processed = pd.concat(
        [
            X_train_mean,
            X_train[['Sex', 'DR']]
        ],
        axis=1
    )

    X_test_processed = pd.concat(
        [
            X_test_mean,
            X_test[['Sex', 'DR']]
        ],
        axis=1
    )

    X_validation_processed = pd.concat(
        [
            X_validation_mean,
            X_validation[['Sex', 'DR']]
        ],
        axis=1
    )

    # 确保列顺序与feature_names一致
    X_train_processed = X_train_processed[feature_names]
    X_test_processed = X_test_processed[feature_names]
    X_validation_processed = X_validation_processed[feature_names]

    # ========================================================
    # 修改3：
    # StandardScaler只在当前X_train上进行fit
    # ========================================================

    scaler = StandardScaler()

    # 当前训练集：fit + transform
    X_train_processed = scaler.fit_transform(
        X_train_processed
    )

    # 当前测试集：只transform
    X_test_processed = scaler.transform(
        X_test_processed
    )

    # 固定验证集：只transform
    X_validation_processed = scaler.transform(
        X_validation_processed
    )

    # ========================================================
    # 后面的模型训练逻辑保持不变
    # ========================================================

    for model, name in zip(models, model_names):

        # 特别处理SVM的内存问题
        if name == "SVM" and X_train_processed.shape[0] > 1000:
            model.set_params(cache_size=200)

        model.fit(
            X_train_processed,
            y_train
        )

        # 计算ROC相关指标
        y_pred_proba_test = model.predict_proba(
            X_test_processed
        )[:, 1]

        fpr_test, tpr_test, thresholds_test = roc_curve(
            y_test,
            y_pred_proba_test
        )

        roc_auc_test = auc(
            fpr_test,
            tpr_test
        )

        train_all_fpr[name].append(
            fpr_test
        )

        train_all_tpr[name].append(
            tpr_test
        )

        train_all_auc[name].append(
            roc_auc_test
        )

        # 验证集ROC指标
        y_pred_proba_val = model.predict_proba(
            X_validation_processed
        )[:, 1]

        fpr_val, tpr_val, thresholds_val = roc_curve(
            y_validation,
            y_pred_proba_val
        )

        roc_auc_val = auc(
            fpr_val,
            tpr_val
        )

        val_all_fpr[name].append(
            fpr_val
        )

        val_all_tpr[name].append(
            tpr_val
        )

        val_all_auc[name].append(
            roc_auc_val
        )

        # 计算其他性能指标
        (
            auc_score,
            sensitivity,
            specificity,
            ppv,
            npv,
            accuracy,
            f1
        ) = calculate_metrics(
            model,
            X_test_processed,
            y_test
        )

        all_metrics[name]['AUC'].append(
            auc_score
        )

        all_metrics[name]['Sensitivity'].append(
            sensitivity
        )

        all_metrics[name]['Specificity'].append(
            specificity
        )

        all_metrics[name]['PPV'].append(
            ppv
        )

        all_metrics[name]['NPV'].append(
            npv
        )

        all_metrics[name]['Accuracy'].append(
            accuracy
        )

        all_metrics[name]['F1-score'].append(
            f1
        )


# ============================================================
# 绘制ROC曲线
# 以下代码保持原来的分析逻辑
# ============================================================

plt.figure(figsize=(8, 6))

for i, name in enumerate(model_names):

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []

    for j in range(n_iterations):

        tpr = np.interp(
            mean_fpr,
            train_all_fpr[name][j],
            train_all_tpr[name][j]
        )

        tpr[0] = 0.0

        tprs.append(tpr)

    mean_tpr = np.mean(
        tprs,
        axis=0
    )

    mean_tpr[-1] = 1.0

    mean_auc = np.mean(
        train_all_auc[name]
    )

    std_auc = np.std(
        train_all_auc[name]
    )

    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
        color=colors[i],
        linewidth=1.5
    )


plt.plot(
    [0, 1],
    [0, 1],
    'k--',
    label='Random Guess (AUC = 0.50)',
    linewidth=1.5
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel(
    'False Positive Rate',
    fontsize=8
)

plt.ylabel(
    'True Positive Rate',
    fontsize=8
)

plt.title(
    'ROC Curves of Different Models',
    fontsize=10
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.grid(
    True,
    alpha=0.5
)

# 去除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig(
    '1-6种模型全部指标ROC曲线.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.tight_layout()

plt.show()


# ============================================================
# 计算每个模型各项指标的平均值和标准差并保存
# ============================================================

results = []

for name in model_names:

    mean_auc = np.mean(
        all_metrics[name]['AUC']
    )

    std_auc = np.std(
        all_metrics[name]['AUC']
    )

    mean_sensitivity = np.mean(
        all_metrics[name]['Sensitivity']
    )

    std_sensitivity = np.std(
        all_metrics[name]['Sensitivity']
    )

    mean_specificity = np.mean(
        all_metrics[name]['Specificity']
    )

    std_specificity = np.std(
        all_metrics[name]['Specificity']
    )

    mean_ppv = np.mean(
        all_metrics[name]['PPV']
    )

    std_ppv = np.std(
        all_metrics[name]['PPV']
    )

    mean_npv = np.mean(
        all_metrics[name]['NPV']
    )

    std_npv = np.std(
        all_metrics[name]['NPV']
    )

    mean_accuracy = np.mean(
        all_metrics[name]['Accuracy']
    )

    std_accuracy = np.std(
        all_metrics[name]['Accuracy']
    )

    mean_f1 = np.mean(
        all_metrics[name]['F1-score']
    )

    std_f1 = np.std(
        all_metrics[name]['F1-score']
    )

    # 保存平均值和标准差
    results.append([
        name,
        f"{mean_auc:.2f} ± {std_auc:.2f}",
        f"{mean_sensitivity:.2f} ± {std_sensitivity:.2f}",
        f"{mean_specificity:.2f} ± {std_specificity:.2f}",
        f"{mean_ppv:.2f} ± {std_ppv:.2f}",
        f"{mean_npv:.2f} ± {std_npv:.2f}",
        f"{mean_accuracy:.2f} ± {std_accuracy:.2f}",
        f"{mean_f1:.2f} ± {std_f1:.2f}"
    ])


# 创建DataFrame并保存到Excel
results_df = pd.DataFrame(
    results,
    columns=[
        'Model',
        'AUC',
        'Sensitivity',
        'Specificity',
        'PPV',
        'NPV',
        'Accuracy',
        'F1-score'
    ]
)


with pd.ExcelWriter(
    '1-6种模型性能比较.xlsx'
) as writer:

    results_df.to_excel(
        writer,
        sheet_name='Model_Metrics',
        index=False
    )


print("代码执行完成，结果已保存！")
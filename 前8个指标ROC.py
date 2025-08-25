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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
import time
import random

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略XGBoost的用户警告

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

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 仅对数值型特征进行均值填充
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                'Urine protein excretion', 'FBG', 'BMI']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)

# 拼接特征列
X = pd.concat([X_mean, X[['DR']]], axis=1)
X = X[feature_names]  # 确保列顺序与feature_names一致

# 对数据进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和验证集
X_train_all, X_validation, y_train_all, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

# 更新模型定义，移除XGBoost的use_label_encoder参数
models = [
    RandomForestClassifier(random_state=45),
    DecisionTreeClassifier(random_state=45),
    LGBMClassifier(random_state=45, verbose=-1),
    XGBClassifier(eval_metric='logloss', random_state=45),
    SVC(probability=True, random_state=45, kernel='linear')
]
model_names = ["RF", "DT", "LightGBM", "XGBoost", "SVM"]
n_iterations = 10
colors = ['b', 'g', 'r', 'c', 'm']

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

    return auc_score, sensitivity, specificity, ppv, npv, accuracy, f1


# 创建固定种子序列
split_seeds = [45 + i for i in range(n_iterations)]

# 训练模型并收集指标
for i in range(n_iterations):
    # 使用固定种子进行划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_all, y_train_all,
        test_size=0.2,
        random_state=split_seeds[i],  # 使用固定种子
        stratify=y_train_all
    )

    for model, name in zip(models, model_names):
        # 特别处理SVM的内存问题
        if name == "SVM" and X_train.shape[0] > 1000:
            # 对大数据集使用较小的缓存大小
            model.set_params(cache_size=200)

        model.fit(X_train, y_train)

        # 计算ROC相关指标
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
        roc_auc_test = auc(fpr_test, tpr_test)

        train_all_fpr[name].append(fpr_test)
        train_all_tpr[name].append(tpr_test)
        train_all_auc[name].append(roc_auc_test)

        # 验证集ROC指标
        y_pred_proba_val = model.predict_proba(X_validation)[:, 1]
        fpr_val, tpr_val, thresholds_val = roc_curve(y_validation, y_pred_proba_val)
        roc_auc_val = auc(fpr_val, tpr_val)

        val_all_fpr[name].append(fpr_val)
        val_all_tpr[name].append(tpr_val)
        val_all_auc[name].append(roc_auc_val)

        # 计算其他性能指标
        auc_score, sensitivity, specificity, ppv, npv, accuracy, f1 = calculate_metrics(model, X_test, y_test)
        all_metrics[name]['AUC'].append(auc_score)
        all_metrics[name]['Sensitivity'].append(sensitivity)
        all_metrics[name]['Specificity'].append(specificity)
        all_metrics[name]['PPV'].append(ppv)
        all_metrics[name]['NPV'].append(npv)
        all_metrics[name]['Accuracy'].append(accuracy)
        all_metrics[name]['F1-score'].append(f1)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for i, name in enumerate(model_names):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for j in range(n_iterations):
        tpr = np.interp(mean_fpr, train_all_fpr[name][j], train_all_tpr[name][j])
        tpr[0] = 0.0
        tprs.append(tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(train_all_auc[name])
    std_auc = np.std(train_all_auc[name])
    plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', color=colors[i], linewidth=1.5)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)', linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves of Different Models', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.5)
# 去除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('5种模型前8指标ROC曲线', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


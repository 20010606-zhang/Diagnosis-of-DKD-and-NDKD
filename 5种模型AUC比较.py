import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
import time
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI',
                 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 数值型特征均值填充
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
                'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
X = pd.concat([X_mean, X[['Sex', 'DR']]], axis=1)[feature_names]

# 标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和验证集
X_train_all, X_validation, y_train_all, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = [
    RandomForestClassifier(random_state=42),
    DecisionTreeClassifier(random_state=42),
    LGBMClassifier(random_state=42, verbose=-1),
    XGBClassifier(eval_metric='logloss', random_state=42),
    LogisticRegression(random_state=42, max_iter=500000)
]
model_names = ["RF", "DT", "LightGBM", "XGBoost", "LR"]
n_iterations = 10
colors = ['b', 'g', 'r', 'c', 'm']

# 新增存储置信区间的变量
train_ci_auc = {name: [] for name in model_names}  # 训练集AUC的置信区间
val_ci_auc = {name: [] for name in model_names}  # 验证集AUC的置信区间

# 其他原有变量初始化
train_all_fpr = {name: [] for name in model_names}
train_all_tpr = {name: [] for name in model_names}
train_all_auc = {name: [] for name in model_names}
val_all_fpr = {name: [] for name in model_names}
val_all_tpr = {name: [] for name in model_names}
val_all_auc = {name: [] for name in model_names}

internal_all_precision = {name: [] for name in model_names}
internal_all_recall = {name: [] for name in model_names}
internal_all_auprc = {name: [] for name in model_names}
external_all_precision = {name: [] for name in model_names}
external_all_recall = {name: [] for name in model_names}
external_all_auprc = {name: [] for name in model_names}

all_metrics = {
    name: {'AUC': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [], 'Accuracy': [], 'F1-score': []} for
    name in model_names
}


def calculate_metrics(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return auc_score, sensitivity, specificity, ppv, npv, accuracy, f1


# 新增：计算AUC置信区间的函数
def calculate_auc_ci(auc, n_pos, n_neg):
    """
    计算AUC的95%置信区间
    使用Hanley和McNeil方法，处理AUC接近1或0的情况
    """
    # 计算AUC的方差
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)

    # 计算标准误
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))

    # 计算95%置信区间
    z = norm.ppf(0.975)
    ci_lower = max(0, auc - z * se)
    ci_upper = min(1, auc + z * se)

    return ci_lower, ci_upper


for _ in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=None)

    # 计算阳性和阴性样本数量
    n_pos_train = np.sum(y_train == 1)
    n_neg_train = np.sum(y_train == 0)
    n_pos_val = np.sum(y_validation == 1)
    n_neg_val = np.sum(y_validation == 0)

    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)

        # 计算训练集指标及置信区间
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_pred_proba_train)

        # 使用新的置信区间计算方法
        ci_lower_train, ci_upper_train = calculate_auc_ci(auc_train, n_pos_train, n_neg_train)
        train_ci_auc[name].append((ci_lower_train, ci_upper_train))

        # 计算验证集指标及置信区间
        y_pred_proba_val = model.predict_proba(X_validation)[:, 1]
        auc_val = roc_auc_score(y_validation, y_pred_proba_val)

        # 使用新的置信区间计算方法
        ci_lower_val, ci_upper_val = calculate_auc_ci(auc_val, n_pos_val, n_neg_val)
        val_ci_auc[name].append((ci_lower_val, ci_upper_val))

        # 计算ROC曲线数据
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
        roc_auc_test = auc(fpr_test, tpr_test)

        train_all_fpr[name].append(fpr_test)
        train_all_tpr[name].append(tpr_test)
        train_all_auc[name].append(roc_auc_test)

        # 计算验证集ROC曲线数据
        fpr_val, tpr_val, thresholds_val = roc_curve(y_validation, y_pred_proba_val)
        roc_auc_val = auc(fpr_val, tpr_val)

        val_all_fpr[name].append(fpr_val)
        val_all_tpr[name].append(tpr_val)
        val_all_auc[name].append(roc_auc_val)

        # 计算PR曲线数据
        y_pred_proba_internal = model.predict_proba(X_test)[:, 1]
        precision_internal, recall_internal, _ = precision_recall_curve(y_test, y_pred_proba_internal)
        auprc_internal = auc(recall_internal, precision_internal)

        internal_all_precision[name].append(precision_internal)
        internal_all_recall[name].append(recall_internal)
        internal_all_auprc[name].append(auprc_internal)

        y_pred_proba_external = model.predict_proba(X_validation)[:, 1]
        precision_external, recall_external, _ = precision_recall_curve(y_validation, y_pred_proba_external)
        auprc_external = auc(recall_external, precision_external)

        external_all_precision[name].append(precision_external)
        external_all_recall[name].append(recall_external)
        external_all_auprc[name].append(auprc_external)

        # 计算其他指标
        auc_score, sensitivity, specificity, ppv, npv, accuracy, f1 = calculate_metrics(model, X_test, y_test)
        all_metrics[name]['AUC'].append(auc_score)
        all_metrics[name]['Sensitivity'].append(sensitivity)
        all_metrics[name]['Specificity'].append(specificity)
        all_metrics[name]['PPV'].append(ppv)
        all_metrics[name]['NPV'].append(npv)
        all_metrics[name]['Accuracy'].append(accuracy)
        all_metrics[name]['F1-score'].append(f1)

# 绘制ROC曲线时添加置信区间注释
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

    # 计算置信区间均值
    ci_lower_mean = np.mean([ci[0] for ci in train_ci_auc[name]])
    ci_upper_mean = np.mean([ci[1] for ci in train_ci_auc[name]])

    plt.plot(mean_fpr, mean_tpr,
             label=f'{name} (AUC = {mean_auc:.2f}\n95% CI: {ci_lower_mean:.2f}-{ci_upper_mean:.2f})', color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('1-ROC曲线.png', dpi=300)
plt.show()

# 输出验证集AUC的置信区间到控制台
print("验证集AUC的95%置信区间：")
for name in model_names:
    ci_values = val_ci_auc[name]
    mean_lower = np.mean([ci[0] for ci in ci_values])
    mean_upper = np.mean([ci[1] for ci in ci_values])
    print(f"{name}: {mean_lower:.2f} - {mean_upper:.2f}")

# 绘制ROC箱型图
plt.figure(figsize=(10, 6))
train_auc_data = [train_all_auc[name] for name in model_names]
bp = plt.boxplot(train_auc_data, labels=model_names)
for i, patch in enumerate(bp['boxes']):
    patch.set_color(colors[i])
for i, median in enumerate(bp['medians']):
    median.set_color(colors[i])
plt.ylabel('AUC')
plt.grid(True)
plt.savefig('1-ROC-箱型图.png', dpi=300)
plt.show()

# 绘制PR曲线
plt.figure(figsize=(10, 8))
for i, name in enumerate(model_names):
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    for j in range(n_iterations):
        precision = np.interp(mean_recall, internal_all_recall[name][j][::-1], internal_all_precision[name][j][::-1])
        precisions.append(precision)
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(internal_all_auprc[name])

    plt.plot(mean_recall, mean_precision, label=f'{name} (AUPRC = {mean_auprc:.2f})', color=colors[i])

random_guess_internal = len(y_test[y_test == 1]) / len(y_test)
plt.axhline(y=random_guess_internal, color='k', linestyle='--',
            label=f'Random Guess (AUPRC = {random_guess_internal:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.grid(True)

plt.savefig('1-PR曲线.png', dpi=300)
plt.show()

# 绘制PR箱型图
plt.figure(figsize=(10, 6))
internal_auprc_data = [internal_all_auprc[name] for name in model_names]
bp = plt.boxplot(internal_auprc_data, labels=model_names)
for i, patch in enumerate(bp['boxes']):
    patch.set_color(colors[i])
for i, median in enumerate(bp['medians']):
    median.set_color(colors[i])

plt.ylabel('AUPRC')
plt.grid(True)

plt.savefig('1-PR-箱型图.png', dpi=300)
plt.show()

# 计算每个模型各项指标的平均值
results = []
for name in model_names:
    mean_auc = np.mean(all_metrics[name]['AUC'])
    mean_sensitivity = np.mean(all_metrics[name]['Sensitivity'])
    mean_specificity = np.mean(all_metrics[name]['Specificity'])
    mean_ppv = np.mean(all_metrics[name]['PPV'])
    mean_npv = np.mean(all_metrics[name]['NPV'])
    mean_accuracy = np.mean(all_metrics[name]['Accuracy'])
    mean_f1 = np.mean(all_metrics[name]['F1-score'])
    results.append([name, mean_auc, mean_sensitivity, mean_specificity, mean_ppv, mean_npv, mean_accuracy, mean_f1])

# 创建DataFrame
results_df = pd.DataFrame(results,
                          columns=['Model', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1-score'])

# 保存到Excel文件
with pd.ExcelWriter('5种模型性能比较.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Model_Metrics', index=False)

# 保存验证集AUC置信区间到Excel
ci_results = []
for name in model_names:
    ci_values = val_ci_auc[name]
    mean_lower = np.mean([ci[0] for ci in ci_values])
    mean_upper = np.mean([ci[1] for ci in ci_values])
    ci_results.append([name, mean_lower, mean_upper])

ci_df = pd.DataFrame(ci_results, columns=['Model', '95% CI Lower', '95% CI Upper'])
with pd.ExcelWriter('5种模型性能比较.xlsx', mode='a', engine='openpyxl') as writer:
    ci_df.to_excel(writer, sheet_name='AUC_Confidence_Intervals', index=False)

print("所有模型评估完成，结果已保存至Excel文件。")
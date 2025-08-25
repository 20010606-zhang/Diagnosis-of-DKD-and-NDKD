import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import joblib
import random  # 补充导入random模块

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # 补充忽略用户警告

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 补充全局随机种子设置
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
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)

# 拼接特征列并保持顺序
X = pd.concat([X_mean, X[['DR']]], axis=1)
X = X[feature_names]  # 确保列顺序与feature_names一致

# 将目标变量合并到特征矩阵中并保存
data_with_target = pd.concat([X, y], axis=1)
data_with_target.to_csv('your_data.csv', index=False)

# 划分训练集和测试集（将random_state从42改为45）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

# 创建随机森林分类器（将random_state从42改为45）
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=45)

# 定义绘制ROC曲线并保存的函数
def plot_and_save_roc_cv(X, y, cv, model, filename):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[train], y.iloc[train])
        y_pred_proba = model.predict_proba(X.iloc[test])[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[test], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.2f})')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f"{cv.n_splits}-fold Cross - Validated ROC Curve")
    ax.legend(loc="lower right")

    plt.savefig(filename, dpi=300)
    plt.close()

# 5倍交叉验证（补充随机种子，确保划分结果可复现）
cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=45)

# 用于存储每个折叠的评估指标
auc_scores = []
sensitivity_scores = []
specificity_scores = []
ppv_scores = []
npv_scores = []
accuracy_scores = []
f1_scores = []

for i, (train, test) in enumerate(cv_5.split(X_train, y_train)):
    rf_classifier.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = rf_classifier.predict(X_train.iloc[test])
    y_pred_proba = rf_classifier.predict_proba(X_train.iloc[test])[:, 1]

    # 计算AUC
    fpr, tpr, _ = roc_curve(y_train.iloc[test], y_pred_proba)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_train.iloc[test], y_pred).ravel()

    # 计算各指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_train.iloc[test], y_pred)
    f1 = f1_score(y_train.iloc[test], y_pred)

    # 存储指标
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    ppv_scores.append(ppv)
    npv_scores.append(npv)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

# 计算均值和标准差
metrics = {
    'AUC': auc_scores,
    'Sensitivity': sensitivity_scores,
    'Specificity': specificity_scores,
    'PPV': ppv_scores,
    'NPV': npv_scores,
    'Accuracy': accuracy_scores,
    'F1 - score': f1_scores
}

results = []
for metric, scores in metrics.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results.append([metric, mean_score, std_score])

# 创建并保存结果DataFrame
results_df = pd.DataFrame(results, columns=['Metric', 'Mean', 'Std'])
results_df.to_excel('cv5_cross_validation_metrics.xlsx', index=False)

# 绘制ROC曲线
plot_and_save_roc_cv(X_train, y_train, cv_5, rf_classifier, '5_fold_roc_curve.png')

# 保存模型
joblib.dump(rf_classifier, 'random_forest_model.joblib')

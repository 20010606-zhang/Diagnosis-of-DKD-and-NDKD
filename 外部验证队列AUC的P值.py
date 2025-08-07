import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import joblib
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI',
                 'LDL', 'SBP']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 仅对数值型特征进行均值填充（移除Sex列的填充）
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'LDL',
                'SBP']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)

# 直接拼接已编码的'DR'列
X = pd.concat([X_mean, X[['DR']]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# 定义绘制 ROC 曲线并保存的函数
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


# 5 倍交叉验证
cv_5 = StratifiedKFold(n_splits=5)

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

    # 计算 AUC
    fpr, tpr, _ = roc_curve(y_train.iloc[test], y_pred_proba)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_train.iloc[test], y_pred).ravel()

    # 计算 Sensitivity
    sensitivity = tp / (tp + fn)
    sensitivity_scores.append(sensitivity)

    # 计算 Specificity
    specificity = tn / (tn + fp)
    specificity_scores.append(specificity)

    # 计算 PPV
    ppv = tp / (tp + fp)
    ppv_scores.append(ppv)

    # 计算 NPV
    npv = tn / (tn + fn)
    npv_scores.append(npv)

    # 计算 Accuracy
    accuracy = accuracy_score(y_train.iloc[test], y_pred)
    accuracy_scores.append(accuracy)

    # 计算 F1 - score
    f1 = f1_score(y_train.iloc[test], y_pred)
    f1_scores.append(f1)

# 计算均值和标准差
metrics = {
    'AUC': auc_scores,
}


# 定义 DeLong 检验函数
def delong_roc_test(y_true1, y_score1, y_true2, y_score2):
    def auc(X, Y):
        return np.mean([1 if x > y else 0.5 if x == y else 0 for x, y in zip(X, Y)])

    def count_pairs(X, Y):
        return len(X) * len(Y)

    def get_sigma(auc_val, X, Y):
        def kernel(X, Y, i, j):
            return 1 if (X[i] > Y[j]) else 0.5 if (X[i] == Y[j]) else 0

        n = len(X)
        m = len(Y)
        var_auc = 0
        for i in range(n):
            for j in range(m):
                var_auc += (kernel(X, Y, i, j) - auc_val) ** 2
        return var_auc / (n * m)

    auc1 = auc(y_score1[y_true1 == 1], y_score1[y_true1 == 0])
    auc2 = auc(y_score2[y_true2 == 1], y_score2[y_true2 == 0])

    sigma1 = get_sigma(auc1, y_score1[y_true1 == 1], y_score1[y_true1 == 0])
    sigma2 = get_sigma(auc2, y_score2[y_true2 == 1], y_score2[y_true2 == 0])

    z = (auc1 - auc2) / np.sqrt(sigma1 + sigma2)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return p


try:
    # 保存模型
    joblib.dump(rf_classifier, 'random_forest_model.joblib')

    # 加载模型
    loaded_model = joblib.load('random_forest_model.joblib')
    print("模型加载成功")

    # 读取外部验证队列数据
    external_validation_data = pd.read_excel("验证队列.xlsx")
    print("外部验证队列数据读取成功")

    # 提取特征
    feature_columns = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG',
                       'BMI', 'LDL', 'SBP']

    # 确保外部验证队列数据的特征列与训练数据一致
    external_validation_X = external_validation_data[feature_columns]
    # 检查是否存在缺失值并进行填充，这里使用与训练数据相同的均值填充器
    external_validation_X_mean = pd.DataFrame(mean_imputer.transform(external_validation_X[mean_columns]), columns=mean_columns)
    external_validation_X = pd.concat([external_validation_X_mean, external_validation_X[['DR']]], axis=1)

    external_validation_true_labels = external_validation_data['Pathology type']

    # 提取内部验证队列特征和标签
    internal_validation_X = X_test
    internal_validation_true_labels = y_test

    # 获取内部验证队列预测概率
    internal_validation_y_pred_proba = loaded_model.predict_proba(internal_validation_X)[:, 1]

    # 计算内部验证队列的 AUC
    from sklearn.metrics import roc_auc_score
    internal_auc = roc_auc_score(internal_validation_true_labels, internal_validation_y_pred_proba)
    print(f"内部验证队列的 AUC: {internal_auc}")

    # 获取外部验证队列预测概率
    external_validation_y_pred_proba = loaded_model.predict_proba(external_validation_X)[:, 1]

    # 计算外部验证队列的 AUC
    external_auc = roc_auc_score(external_validation_true_labels, external_validation_y_pred_proba)
    print(f"外部验证队列的 AUC: {external_auc}")

    # 计算 ΔAUC
    delta_auc = abs(internal_auc - external_auc)
    print(f"ΔAUC 的值: {delta_auc}")

    # 进行 DeLong 检验
    p_value = delong_roc_test(internal_validation_true_labels, internal_validation_y_pred_proba,
                              external_validation_true_labels, external_validation_y_pred_proba)
    print(f"DeLong 检验的 p 值: {p_value}")
    if p_value < 0.05:
        print("内部和外部验证队列的 ROC 曲线下面积存在显著差异。")
    else:
        print("内部和外部验证队列的 ROC 曲线下面积不存在显著差异。")

except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
except KeyError as e:
    print(f"数据中不存在名为 {e} 的列，请检查列名是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")
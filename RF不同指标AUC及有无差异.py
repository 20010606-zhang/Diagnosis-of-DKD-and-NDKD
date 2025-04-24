import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 仅对数值型特征进行均值填充（移除Sex列的填充）
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
                'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)

# 直接拼接已编码的'Sex'列
X = pd.concat([X_mean, X[['Sex', 'DR']]], axis=1)
X = X[feature_names]  # 确保列顺序与feature_names一致

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 定义随机森林模型
model = RandomForestClassifier(random_state=42)

# 执行RFE
feature_count = len(X_train.columns)
step = 1
rfe = RFE(estimator=model, n_features_to_select=1, step=step)
rfe.fit(X_train, y_train)

rankings = rfe.ranking_
sorted_indices = sorted(range(len(rankings)), key=lambda k: rankings[k])

# 要绘制AUC曲线的特征数量
selected_feature_counts = [3, 9, 10, 17]

# 存储预测概率和AUC值
y_preds = []
aucs = []

plt.figure(figsize=(12, 8))

# 自定义图例名称
legend_names = ["RF models with 3 Features", "RF models with 9 Features", "RF models with 10 Features", "RF models with 17 Features"]

for i, num_features in enumerate(selected_feature_counts):
    selected_features_train = X_train.iloc[:, sorted_indices[:num_features]]
    selected_features_test = X_test.iloc[:, sorted_indices[:num_features]]

    model.fit(selected_features_train, y_train)
    y_pred_proba = model.predict_proba(selected_features_test)[:, 1]
    y_preds.append(y_pred_proba)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # 使用自定义图例名称
    plt.plot(fpr, tpr, label=f'{legend_names[i]} (AUC = {roc_auc:.2f})', linewidth=1.5)

# 绘制对角线（随机猜测）
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)


# 定义DeLong's test相关函数
def compute_midrank(x):
    """
    计算数组x的中间排名
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def compute_midrank_weight(x, sample_weight):
    """
    计算带权重的中间排名
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    快速计算DeLong的协方差矩阵
    """
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    计算无权重的DeLong协方差矩阵
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    计算带权重的DeLong协方差矩阵
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    positive_weights = sample_weight[:m]
    negative_weights = sample_weight[m:]
    positive_weight_sum = positive_weights.sum()
    negative_weight_sum = negative_weights.sum()
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], positive_weights)
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], negative_weights)
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    aucs = (np.sum(tz[:, :m] * positive_weights[np.newaxis, :], axis=1) / positive_weight_sum / negative_weight_sum -
            positive_weight_sum * (positive_weight_sum + 1.0) / 2.0 / positive_weight_sum / negative_weight_sum)
    v01 = (tz[:, :m] - tx[:, :]) / negative_weight_sum
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / positive_weight_sum
    sx = np.cov(v01, aweights=positive_weights)
    sy = np.cov(v10, aweights=negative_weights)
    delongcov = sx / positive_weight_sum + sy / negative_weight_sum
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """
    计算p值
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 1 - norm.cdf(z)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]
    return order, label_1_count, ordered_sample_weight


def delong_roc_test(ground_truth, predictions_one, predictions_two, sample_weight=None):
    """
    执行DeLong's test
    """
    from scipy.stats import norm
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(ground_truth, sample_weight)
    predictions_sorted_transposed = np.vstack((predictions_one[order], predictions_two[order]))
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    p = calc_pvalue(aucs, delongcov)
    return p[0][0]


# 比较并显示 ΔAUC 和 P 值
for i in range(len(selected_feature_counts) - 1):
    delta_auc = aucs[-1] - aucs[i]
    p_value = delong_roc_test(y_test.values, y_preds[-1], y_preds[i])
    plt.text(0.55, 0.3 - i * 0.05, f'ΔAUC ({legend_names[i]} vs {legend_names[-1]}): {delta_auc:.3f}, p={p_value:.3f}',
             fontsize=10)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves for Different Feature Counts', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig("ROC不同指标及AUC有无差异.png", format='png', bbox_inches='tight')
plt.tight_layout()
plt.show()

# 获取真实标签
y_true = y_test.values

# 获取4个模型的预测概率
y_pred1, y_pred2, y_pred3, y_pred4 = y_preds

print("y_true:", y_true)
print("y_pred1:", y_pred1)
print("y_pred2:", y_pred2)
print("y_pred3:", y_pred3)
print("y_pred4:", y_pred4)
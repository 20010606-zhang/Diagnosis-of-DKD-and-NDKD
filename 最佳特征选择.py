import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import warnings
import time
import random

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 设置全局随机种子
np.random.seed(45)
random.seed(45)

try:
    df = pd.read_excel('test1.xlsx')
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
                 'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

# 数值型特征均值填充
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
                'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)

X = pd.concat([X_mean, X[['Sex', 'DR']]], axis=1)
X = X[feature_names]

# 划分数据集（随机种子45）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)


def perform_rfe_single_model(model, X_train, y_train, X_test, y_test, model_name):
    feature_count = len(X_train.columns)
    step = 1

    if model_name == 'SVM':
        rfe = RFE(estimator=model, n_features_to_select=1, step=step,
                  importance_getter=lambda clf: np.abs(clf.coef_[0]))
    else:
        rfe = RFE(estimator=model, n_features_to_select=1, step=step)

    rfe.fit(X_train, y_train)

    rankings = rfe.ranking_
    sorted_indices = sorted(range(len(rankings)), key=lambda k: rankings[k])

    scores = []
    feature_counts = []
    for i in range(feature_count, 0, -step):  # 从最大特征数递减到1
        selected_features_train = X_train.iloc[:, sorted_indices[:i]]
        selected_features_test = X_test.iloc[:, sorted_indices[:i]]

        model.fit(selected_features_train, y_train)

        try:
            y_pred = model.predict_proba(selected_features_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
        except Exception as e:
            print(f"计算 {model_name} 的 AUC 时出错: {e}")
            score = np.nan

        scores.append(score)
        feature_counts.append(i)  # 记录当前特征数量（包含1、2、...、最大特征数）

    results_df = pd.DataFrame({
        'Number_of_Features': feature_counts,
        model_name: scores
    })
    return results_df


# 模型定义（随机种子45）
models = {
    'RF': RandomForestClassifier(random_state=45),
    'DT': DecisionTreeClassifier(random_state=45),
    'LightGBM': LGBMClassifier(random_state=45, verbose=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=45),
    'SVM': SVC(probability=True, random_state=45, kernel='linear')
}

results_df = None

for model_name, model in models.items():
    print(f"运行 {model_name} 的 RFE 特征选择...")
    start_time = time.time()
    try:
        model_results = perform_rfe_single_model(model, X_train, y_train, X_test, y_test, model_name)
        if results_df is None:
            results_df = model_results
        else:
            results_df = results_df.merge(model_results, on='Number_of_Features', how='outer')
    except Exception as e:
        print(f"{model_name} 运行出错: {e}")
    end_time = time.time()
    print(f"{model_name} 耗时: {end_time - start_time:.2f} 秒")

if results_df is not None:
    results_df.reset_index(drop=True, inplace=True)
    print("\n特征数量与AUC分数对应表:\n", results_df)

    # 关键修改：直接使用所有特征数量（从1开始）绘图，不做过滤
    # 按特征数量升序排列（从1到最大特征数），使曲线从左到右更直观
    plot_df = results_df.sort_values(by='Number_of_Features', ascending=True).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    for column in plot_df.columns[1:]:  # 遍历所有模型
        plt.plot(
            plot_df['Number_of_Features'],  # x轴：特征数量（1,2,...,最大）
            plot_df[column],
            label=column,
            marker='o',
            linewidth=1.5
        )

    optimal_features = 8  # 可根据实际结果调整最佳特征数
    plt.axvline(
        x=optimal_features,
        color='black',
        linestyle='--',
        label='Optimal Features'
    )

    plt.title('Feature Reduction', fontsize=16)
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Area Under Curve(AUC)', fontsize=14)
    plt.xticks(ticks=plot_df['Number_of_Features'], fontsize=10)  # x轴显示所有特征数量（包括1、2）
    plt.yticks(fontsize=12)
    plt.legend(title='模型', fontsize=12, loc='best')
    plt.grid(axis='y', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig('特征选择_从1个特征开始.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

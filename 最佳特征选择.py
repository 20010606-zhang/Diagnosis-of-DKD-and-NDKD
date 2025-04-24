import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP',
                 'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex']
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


def perform_rfe_single_model(model, X_train, y_train, X_test, y_test, model_name):
    feature_count = len(X_train.columns)
    step = 1
    rfe = RFE(estimator=model, n_features_to_select=1, step=step)
    rfe.fit(X_train, y_train)

    rankings = rfe.ranking_
    sorted_indices = sorted(range(len(rankings)), key=lambda k: rankings[k])

    scores = []
    feature_counts = []
    for i in range(feature_count, 0, -step):
        selected_features_train = X_train.iloc[:, sorted_indices[:i]]
        selected_features_test = X_test.iloc[:, sorted_indices[:i]]

        model.fit(selected_features_train, y_train)

        try:
            y_pred = model.predict_proba(selected_features_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
        except Exception as e:
            print(f"在计算 {model_name} 模型的 ROC AUC 分数时出现错误: {e}")
            score = np.nan

        scores.append(score)
        feature_counts.append(i)

    results_df = pd.DataFrame({
        "Number_of_Features": feature_counts[::-1],
        model_name: scores[::-1]
    })
    return results_df


models = {
    "RF": RandomForestClassifier(random_state=42),
    "DT": DecisionTreeClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LR": LogisticRegression(random_state=42, max_iter=500000)
}

results_df = None

for model_name, model in models.items():
    print(f"Running RFE for {model_name}...")
    start_time = time.time()
    try:
        model_results = perform_rfe_single_model(model, X_train, y_train, X_test, y_test, model_name)
        if results_df is None:
            results_df = model_results
        else:
            results_df = results_df.merge(model_results, on="Number_of_Features", how="outer")
    except Exception as e:
        print(f"在运行 {model_name} 模型的 RFE 时出现错误: {e}")
    end_time = time.time()
    print(f"{model_name} 模型训练时间: {end_time - start_time:.2f} 秒")

if results_df is not None:
    results_df.reset_index(drop=True, inplace=True)
    print(results_df)

    step_size = 1
    initial_value = 1
    filtered_results_df = results_df[
                              results_df["Number_of_Features"] >= initial_value
                              ].iloc[::step_size, :].reset_index(drop=True)
    filtered_results_df.sort_values(by="Number_of_Features", ascending=False, inplace=True)

    plt.figure(figsize=(12, 8))
    for column in filtered_results_df.columns[1:]:
        plt.plot(
            filtered_results_df["Number_of_Features"],
            filtered_results_df[column],
            label=column,
            marker='o',
            linewidth=1.5
        )

    optimal_features = 10  # 最佳特征数量
    plt.axvline(
        x=optimal_features,
        color='black',
        linestyle='--',
        label='Optimal Features'
    )

    plt.title('Feature Reduction (Step=1, Initial=1)', fontsize=16)
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Area Under the ROC Curve (AUC)', fontsize=14)
    plt.xticks(
        ticks=filtered_results_df["Number_of_Features"],
        fontsize=10
    )
    plt.yticks(fontsize=12)
    plt.legend(title="Models", fontsize=12, loc="best")
    plt.grid(axis='y', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig("最佳特征选择.png", format='png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

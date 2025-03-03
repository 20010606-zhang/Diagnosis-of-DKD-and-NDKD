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

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', '2hPBG', 'Duration of DN', 'Sex']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', '2hPBG', 'Duration of DN']
median_columns = ['DR', 'Sex']

mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')

X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
X_median = pd.DataFrame(median_imputer.fit_transform(X[median_columns]), columns=median_columns)

X = pd.concat([X_mean, X_median], axis=1)
X = X[feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def perform_rfe_single_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    对单个模型执行 RFE 并计算不同特征子集的 ROC AUC 分数
    :param model: 模型对象
    :param X_train: 训练集特征
    :param y_train: 训练集目标变量
    :param X_test: 测试集特征
    :param y_test: 测试集目标变量
    :param model_name: 模型名称
    :return: 包含特征数量和对应 ROC AUC 分数的 DataFrame
    """
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
    "LR": LogisticRegression(random_state=42, max_iter=20000)
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

    # 筛选出满足步长为1、初始值为1的行
    step_size = 1  # 步长设置为1
    initial_value = 1  # 初始值设置为1
    # 筛选满足条件的数据行：特征数量 >= 初始值，并按步长取样
    filtered_results_df = results_df[
        results_df["Number_of_Features"] >= initial_value  # 筛选特征数量大于等于初始值的行
    ].iloc[::step_size, :].reset_index(drop=True)  # 按步长取样，并重置索引
    # 按 "Number_of_Features" 列从大到小排序
    filtered_results_df.sort_values(by="Number_of_Features", ascending=False, inplace=True)

    # 绘制图形
    plt.figure(figsize=(12, 8))  # 设置画布大小
    for column in filtered_results_df.columns[1:]:  # 遍历所有模型的列（从第2列开始）
        plt.plot(
            filtered_results_df["Number_of_Features"],  # X轴：特征数量
            filtered_results_df[column],  # Y轴：对应模型的AUC分数
            label=column,  # 设置图例为模型名称
            marker='o',  # 在曲线上标记点
            linewidth=1.5  # 设置线条宽度
        )

    # 绘制最佳特征数量的垂直虚线
    optimal_features = 10  # 最佳特征数量
    plt.axvline(
        x=optimal_features,  # 垂直线的位置
        color='black',  # 设置线的颜色为黑色
        linestyle='--',  # 设置线型为虚线
        label='Optimal Features'  # 图例说明
    )

    # 设置图表标题和坐标轴标签
    plt.title('Feature Reduction (Step=1, Initial=1)', fontsize=16)  # 图表标题
    plt.xlabel('Number of Features', fontsize=14)  # X轴标签
    plt.ylabel('Area Under the ROC Curve (AUC)', fontsize=14)  # Y轴标签
    # 设置X轴的刻度值和字体大小
    plt.xticks(
        ticks=filtered_results_df["Number_of_Features"],  # 设置刻度值为筛选后的特征数量
        fontsize=10  # 设置字体大小
    )
    plt.yticks(fontsize=12)  # 设置Y轴字体大小
    plt.legend(title="Models", fontsize=12, loc="best")  # 图例标题、字体大小及位置
    plt.grid(axis='y', alpha=0.5)  # 添加Y轴方向的网格线，并设置透明度
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右侧边框
    plt.savefig("最佳特征选择.png", format='png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import shap
import os
from IPython.display import display, HTML
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time
from selenium.common.exceptions import WebDriverException  # 导入正确的异常类

# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置 matplotlib 字体和负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 定义特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
target_name = 'Pathology type'

# 读取数据
try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

# 提取特征和目标变量
X = df[feature_names]
y = df[target_name]

# 打印特征和目标变量信息
print("特征名称:", feature_names)
print("目标变量的唯一值:", y.unique())

# 数据预处理：缺失值填充
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
X = pd.concat([X_mean, X[['DR']]], axis=1)[feature_names]

# 保存处理后的数据
data_with_target = pd.concat([X, y], axis=1)
data_with_target.to_csv('your_data_2.csv', index=False)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练数据特征数量: {X_train.shape[1]}")
print(f"测试数据特征数量: {X_test.shape[1]}")

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 计算 SHAP 值
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test)
# 处理二分类问题的 shap_values
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# 打印 SHAP 值信息
print("shap_values 的类型:", type(shap_values))
if isinstance(shap_values, list):
    print("shap_values 的长度:", len(shap_values))
    for i, val in enumerate(shap_values):
        print(f"shap_values 第 {i} 个元素的形状:", val.shape)
else:
    print("shap_values 的形状:", shap_values.shape)
print(f"SHAP 值特征数量: {shap_values[0].shape[0]}")

# 绘制并保存 SHAP 图
def save_shap_plot(plot_func, filename, *args, **kwargs):
    plt.figure(figsize=(8, 10))  # 这里宽度设为8，高度设为10，可根据实际调整
    plot_func(*args, **kwargs)
    plt.savefig(filename, dpi=300)
    plt.close()

save_shap_plot(shap.summary_plot,'shap_summary_plot.png', shap_values, X_test)
save_shap_plot(shap.summary_plot,'shap_summary_bar_plot.png', shap_values, X_test, plot_type='bar')

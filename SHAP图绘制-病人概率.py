import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import shap
import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置 matplotlib 字体和负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path, feature_names, target_name):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        raise
    X = df[feature_names]
    y = df[target_name]
    print("特征名称:", feature_names)
    print("目标变量的唯一值:", y.unique())

    # 数据预处理：缺失值填充
    mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
    mean_imputer = SimpleImputer(strategy='mean')
    X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
    X = pd.concat([X_mean, X[['DR']]], axis=1)[feature_names]

    # 保存处理后的数据
    data_with_target = pd.concat([X, y], axis=1)
    data_with_target.to_csv('your_data.csv', index=False)

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"训练数据特征数量: {X_train.shape[1]}")
    print(f"测试数据特征数量: {X_test.shape[1]}")

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    return rf_classifier, X_test

def calculate_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # 处理二分类问题的 shap_values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return explainer, shap_values

def convert_html_to_image(html_file, image_file):
    try:
        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service)
        driver.get('file://' + os.path.abspath(html_file))
        # 等待页面加载，使用显式等待
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        driver.save_screenshot(image_file)
        driver.quit()
    except ImportError:
        print("请安装 selenium 和 webdriver_manager 以将 HTML 转换为图片。")
    except Exception as e:
        print(f"转换 HTML 为图片时出错: {e}")

# 定义特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
target_name = 'Pathology type'

# 加载并预处理数据
X, y = load_and_preprocess_data("test1.xlsx", feature_names, target_name)

# 训练模型
rf_classifier, X_test = train_model(X, y)

# 计算 SHAP 值
explainer, shap_values = calculate_shap_values(rf_classifier, X_test)

sample_index = 5
sample_shap_values = shap_values[sample_index].reshape(1, -1)
sample_features = X_test.iloc[sample_index].values.reshape(1, -1)
force_plot = shap.force_plot(explainer.expected_value[1], sample_shap_values, sample_features, feature_names=feature_names)
shap.save_html('shap_force_plot.html', force_plot)

# 绘制并保存瀑布图
plt.figure(figsize=(8, 20))  # 适当增大高度
plt.subplots_adjust(left=0.3)
shap.plots.waterfall(shap.Explanation(values=sample_shap_values[0],
                                      base_values=explainer.expected_value[1],
                                      data=sample_features[0],
                                      feature_names=feature_names))
plt.savefig('shap_waterfall_plot.png', dpi=300)
plt.close()
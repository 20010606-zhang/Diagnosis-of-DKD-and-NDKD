import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import shap
import warnings

# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning)

def load_and_preprocess_data(file_path, feature_names, target_name):
    """
    加载并预处理数据
    :param file_path: 数据文件路径
    :param feature_names: 特征名称列表
    :param target_name: 目标变量名称
    :return: 处理后的特征矩阵 X 和目标向量 y
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("文件未找到，请检查文件路径。")
    X = df[feature_names]
    y = df[target_name]
    # 数据预处理：缺失值填充
    mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP']
    median_columns = ['DR']
    mean_imputer = SimpleImputer(strategy='mean')
    median_imputer = SimpleImputer(strategy='median')
    X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
    X_median = pd.DataFrame(median_imputer.fit_transform(X[median_columns]), columns=median_columns)
    X = pd.concat([X_mean, X_median], axis=1)[feature_names]

    # 保存处理后的数据
    data_with_target = pd.concat([X, y], axis=1)
    data_with_target.to_csv('your_data.csv', index=False)

    return X, y

def train_model(X, y):
    """
    训练随机森林分类器模型
    :param X: 特征矩阵
    :param y: 目标向量
    :return: 训练好的模型和测试集特征矩阵
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier, X_test

def calculate_shap_values(model, X):
    """
    计算 SHAP 值
    :param model: 训练好的模型
    :param X: 特征矩阵
    :return: SHAP 解释器和 SHAP 值
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"计算得到的 SHAP 值形状: {np.array(shap_values).shape}")
    print(f"输入特征的形状: {X.shape}")
    # 处理二分类问题的 shap_values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # 对于二分类问题，通常取正类的 SHAP 值
        shap_values = shap_values[1]
    return explainer, shap_values
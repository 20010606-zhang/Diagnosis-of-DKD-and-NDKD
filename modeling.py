import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import joblib

def load_and_preprocess_data(file_path, feature_names, target_name):
    """
    从 Excel 文件中加载数据，并分离特征和目标变量
    :param file_path: Excel 文件路径
    :param feature_names: 特征名称列表
    :param target_name: 目标变量名称
    :return: 特征矩阵 X 和目标向量 y
    """
    data = pd.read_excel(file_path)
    X = data[feature_names]
    y = data[target_name]
    return X, y

def train_model(X, y):
    """
    划分训练集和测试集，然后使用随机森林分类器进行训练
    :param X: 特征矩阵
    :param y: 目标向量
    :return: 训练好的模型和测试集特征矩阵
    """
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y)
    return rf_classifier, X_test

def calculate_shap_values(model, X_test):
    """
    计算 SHAP 值
    :param model: 训练好的模型
    :param X_test: 测试集特征矩阵
    :return: SHAP 解释器
    """
    explainer = shap.TreeExplainer(model)
    return explainer

def build_and_save_output(file_path, feature_names, target_name, output_path):
    """
    构建模型、计算 SHAP 解释器，并将模型和解释器保存到指定文件
    :param file_path: Excel 文件路径
    :param feature_names: 特征名称列表
    :param target_name: 目标变量名称
    :param output_path: 输出文件路径
    """
    X, y = load_and_preprocess_data(file_path, feature_names, target_name)
    model, X_test = train_model(X, y)
    explainer = calculate_shap_values(model, X_test)
    joblib.dump([model, explainer, X_test], output_path)
    print(f"输出已保存到 {output_path}")


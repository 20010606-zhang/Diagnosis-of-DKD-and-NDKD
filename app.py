import streamlit as st
import matplotlib
import os
import requests
from io import BytesIO
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 手动设置数据文件路径为公开 URL
data_file_path = "https://raw.githubusercontent.com/20010606-zhang/DKD/master/test1.xlsx"

# 设置 matplotlib 字体和负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 定义特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP']
target_name = 'Pathology type'

# Streamlit 应用标题
st.title("SHAP 模型可视化应用")

# 定义基础路径为当前脚本所在目录
base_path = os.getcwd()

# 检查基础路径是否存在，如果不存在则创建
if not os.path.exists(base_path):
    os.makedirs(base_path)

# 定义输出文件路径
output_path = os.path.join(base_path, 'model_output.joblib')

try:
    response = requests.get(data_file_path)
    response.raise_for_status()
    data = pd.read_excel(BytesIO(response.content))
    X = data[feature_names]
    y = data[target_name]

    def load_and_preprocess_data(X, y):
        return X, y

    def train_model(X, y):
        """
        划分训练集和测试集，然后使用随机森林分类器进行训练
        :param X: 特征矩阵
        :param y: 目标向量
        :return: 训练好的模型和测试集特征矩阵
        """
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_train, y_train)
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

    def build_and_save_output(X, y, feature_names, target_name, output_path):
        """
        构建模型、计算 SHAP 解释器，并将模型和解释器保存到指定文件
        :param X: 特征矩阵
        :param y: 目标向量
        :param feature_names: 特征名称列表
        :param target_name: 目标变量名称
        :param output_path: 输出文件路径
        """
        model, X_test = train_model(X, y)
        explainer = calculate_shap_values(model, X_test)
        joblib.dump([model, explainer, X_test], output_path)
        print(f"输出已保存到 {output_path}")

    # 检查输出文件是否存在，如果不存在则生成
    if not os.path.exists(output_path):
        build_and_save_output(X, y, feature_names, target_name, output_path)

    # 加载输出文件
    model, explainer, X_test = joblib.load(output_path)

    st.write("特征名称:", feature_names)
    st.write("训练数据特征数量: ", X_test.shape[1])

    # 添加输入组件让用户输入指标值
    st.subheader("请输入指标值")
    input_features = [st.number_input(f"输入 {feature} 的值", step=0.01) for feature in feature_names]

    if st.button("查看结果"):
        input_features = np.array([input_features])
        # 计算输入样本的 SHAP 值
        shap_values = explainer.shap_values(input_features)

        # 检查 shap_values 的结构
        if isinstance(shap_values, list):
            # 对于多分类问题，shap_values 可能是列表
            if len(shap_values) > 1:
                # 选择正类（索引为 1）的 SHAP 值
                sample_shap_values = shap_values[1].flatten()
            else:
                sample_shap_values = shap_values[0].flatten()
        elif shap_values.ndim == 3:
            # 对于二分类问题且 shap_values 是三维数组的情况
            sample_shap_values = shap_values[:, :, 1].flatten()
        else:
            sample_shap_values = shap_values.flatten()

        sample_features = input_features[0]

        # 检查长度是否一致
        print(f"SHAP 值长度: {len(sample_shap_values)}")
        print(f"特征值长度: {len(sample_features)}")

        if len(sample_shap_values) != len(sample_features):
            st.error(f"SHAP 值长度 ({len(sample_shap_values)}) 与特征值长度 ({len(sample_features)}) 不匹配！")
        else:
            try:
                # 绘制并保存力图
                force_plot = shap.force_plot(
                    explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0],
                    sample_shap_values, sample_features, feature_names=feature_names)
                force_image_file = os.path.join(base_path, 'shap_force_plot.png')
                shap.save_html(force_image_file, force_plot)  # 这里实际上是保存 HTML 文件

                # 绘制并保存瀑布图
                plt.figure(figsize=(8, 20))
                plt.subplots_adjust(left=0.3)
                shap.plots.waterfall(shap.Explanation(values=sample_shap_values,
                                                      base_values=explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0],
                                                      data=sample_features,
                                                      feature_names=feature_names))
                waterfall_image_file = os.path.join(base_path, 'shap_waterfall_plot.png')
                plt.savefig(waterfall_image_file, dpi=300)
                plt.close()

                # 显示瀑布图
                if os.path.exists(waterfall_image_file):
                    st.image(waterfall_image_file, caption='SHAP 瀑布图')
                else:
                    st.error("SHAP 瀑布图文件未找到。")
                # 显示力图
                if os.path.exists(force_image_file):
                    import streamlit.components.v1 as components
                    with open(force_image_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=400)
                else:
                    st.error("SHAP 力图文件未找到。")
            except Exception as e:
                st.error(f"生成可视化图表时出错: {e}")
except requests.RequestException as e:
    st.error(f"下载文件时出错: {e}")
except Exception as e:
    st.error(f"发生未知错误: {e}")
    #streamlit run "D:\Users\17927\Desktop\mechine study\app.py"
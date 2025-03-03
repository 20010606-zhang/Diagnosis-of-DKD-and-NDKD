import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from modeling import load_and_preprocess_data, train_model, calculate_shap_values

# 设置 matplotlib 字体和负号显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def convert_html_to_image(html_file, image_file):
    """
    将 HTML 文件转换为图片
    :param html_file: HTML 文件路径
    :param image_file: 输出图片文件路径
    """
    try:
        edge_options = Options()
        edge_options.add_argument('--ignore-certificate-errors')
        edge_options.add_argument('--allow-running-insecure-content')
        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=edge_options)
        print(f"尝试打开 HTML 文件: {html_file}")
        driver.get('file://' + os.path.abspath(html_file))

        # 增加等待时间，等待页面上的特定元素加载完成
        try:
            # 假设页面上有一个 id 为 'shap-plot' 的元素，可根据实际情况修改
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, 'shap-plot'))
            )
        except Exception as e:
            print(f"等待特定元素加载时出错: {e}")

        print("等待页面加载...")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        print(f"尝试保存截图到: {image_file}")
        driver.save_screenshot(image_file)
        if os.path.exists(image_file):
            print("截图保存成功。")
        else:
            print("截图保存失败，文件未找到。")
    except ImportError:
        st.error("请安装 selenium 和 webdriver_manager 以将 HTML 转换为图片。")
    except Exception as e:
        st.error(f"转换 HTML 为图片时出错: {e}")
        print(f"转换 HTML 为图片时出错: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()


# 定义特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP']
target_name = 'Pathology type'

# Streamlit 应用标题
st.title("SHAP 模型可视化应用")

# 定义基础路径
base_path = r"D:\Users\17927\Desktop\mechine study"

# 检查基础路径是否存在，如果不存在则创建
if not os.path.exists(base_path):
    os.makedirs(base_path)

try:
    # 加载并预处理数据
    X, y = load_and_preprocess_data(os.path.join(base_path, "test1.xlsx"), feature_names, target_name)
    st.write("特征名称:", feature_names)
    st.write("目标变量的唯一值:", y.unique())
    # 训练模型
    rf_classifier, X_test = train_model(X, y)
    st.write(f"训练数据特征数量: {X_test.shape[1]}")
    # 计算 SHAP 值
    explainer, _ = calculate_shap_values(rf_classifier, X_test)

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
                force_plot = shap.force_plot(
                    explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0],
                    sample_shap_values, sample_features, feature_names=feature_names)
                # 保存 HTML 文件
                html_file = os.path.join(base_path, 'shap_force_plot.html')
                shap.save_html(html_file, force_plot)

                # 检查 HTML 文件是否存在
                if not os.path.exists(html_file):
                    st.error("生成的 HTML 文件不存在。")
                else:
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
                    # 将 HTML 转换为图片
                    force_image_file = os.path.join(base_path, 'shap_force_plot.png')
                    convert_html_to_image(html_file, force_image_file)
                    # 显示瀑布图
                    if os.path.exists(waterfall_image_file):
                        st.image(waterfall_image_file, caption='SHAP 瀑布图')
                    else:
                        st.error("SHAP 瀑布图文件未找到。")
                    # 显示力图（以图片形式）
                    if os.path.exists(force_image_file):
                        st.image(force_image_file, caption='SHAP 力图')
                    else:
                        st.error("SHAP 力图文件未找到。")
            except Exception as e:
                st.error(f"生成可视化图表时出错: {e}")

except FileNotFoundError as e:
    st.error(str(e))
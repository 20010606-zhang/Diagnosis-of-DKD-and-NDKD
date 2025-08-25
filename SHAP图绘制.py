import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import shap

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置全局字体为Times New Roman（核心修改：通过matplotlib全局设置）
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
    'axes.labelsize': 12,  # 坐标轴标签字体大小
    'axes.titlesize': 14,  # 标题字体大小
    'xtick.labelsize': 10,  # x轴刻度字体大小
    'ytick.labelsize': 10,  # y轴刻度字体大小
    'legend.fontsize': 10  # 图例字体大小
})

# 设置全局随机种子
np.random.seed(45)
random.seed(45)

# 定义特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                 'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
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

# 数据预处理：缺失值填充
mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC',
                'Urine protein excretion', 'FBG', 'BMI', 'LDL', 'SBP']
mean_imputer = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
X = pd.concat([X_mean, X[['DR']]], axis=1)[feature_names]

# 保存处理后的数据
data_with_target = pd.concat([X, y], axis=1)
data_with_target.to_csv('your_data_2.csv', index=False)

# 划分训练集和测试集（随机种子45）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

# 创建随机森林分类器（随机种子45）
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=45)
rf_classifier.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test)

# 处理二分类问题的SHAP值
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values = shap_values[1]  # 取正类的SHAP值
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# 打印SHAP值信息
print("shap_values 的类型:", type(shap_values))
if isinstance(shap_values, list):
    print("shap_values 的长度:", len(shap_values))
    for i, val in enumerate(shap_values):
        print(f"shap_values 第 {i} 个元素的形状:", val.shape)
else:
    print("shap_values 的形状:", shap_values.shape)
if shap_values.ndim >= 2:
    print(f"SHAP 值特征数量: {shap_values[0].shape[0]}")
else:
    print(f"SHAP 值特征数量: {shap_values.shape[0]}")


# 定义SHAP图保存函数（修正字体设置方式）
def save_shap_plot(plot_func, filename, *args, **kwargs):
    plt.figure(figsize=(8, 10))

    # 只传递SHAP支持的参数，移除plot_font
    kwargs['feature_names'] = feature_names

    # 绘制SHAP图
    plot_func(*args, **kwargs)

    # 强制设置所有文本元素的字体为Times New Roman
    # 获取当前图中的所有文本元素并修改字体
    for text in plt.gca().findobj(plt.Text):
        try:
            text.set_fontfamily('Times New Roman')
        except:
            continue

    # 额外确保坐标轴标签字体
    plt.xlabel(plt.gca().get_xlabel(), fontfamily='Times New Roman')
    plt.ylabel(plt.gca().get_ylabel(), fontfamily='Times New Roman')
    plt.title(plt.gca().get_title(), fontfamily='Times New Roman')

    # 保存图片（确保标签完整）
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# 绘制并保存SHAP摘要图（蜜蜂图）和条形图
save_shap_plot(shap.summary_plot, 'shap_summary_plot.png', shap_values, X_test)
save_shap_plot(shap.summary_plot, 'shap_summary_bar_plot.png', shap_values, X_test, plot_type='bar')

print("SHAP图已保存，所有字体已设置为Times New Roman！")
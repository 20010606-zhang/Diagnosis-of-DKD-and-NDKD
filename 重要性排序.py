import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import random  # 补充导入random模块

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 补充全局随机种子设置（确保所有随机过程一致）
np.random.seed(45)
random.seed(45)

# 读取数据
try:
    df = pd.read_excel('test1.xlsx')
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

# 划分特征和目标变量
feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', 'Duration of DN', 'Sex']
target_name = 'Pathology type'
X = df[feature_names]
y = df[target_name]

# 划分训练集和测试集（将random_state从42改为45）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=y)

# 创建随机森林分类器（将random_state从42改为45）
rf_clf = RandomForestClassifier(random_state=45)

# 使用RFE进行特征选择
# 选择全部特征
rfe = RFE(estimator=rf_clf, n_features_to_select=len(feature_names))
rfe.fit(X_train, y_train)

# 获取被选中的特征（由于选择了全部特征，此处等价于原始特征）
selected_features = X.columns[rfe.support_]
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# 在选择的特征上重新训练随机森林模型（已设置随机种子45）
rf_clf.fit(X_train_selected, y_train)

# 获取特征重要性
feature_importances = rf_clf.feature_importances_

# 构建特征重要性排名
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 绘制特征重要性排名图
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=40)  # 旋转标签（0为水平）
plt.gca().invert_yaxis()  # 使重要性最高的特征在上方

# 在每个条形上添加重要性数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.savefig("重要性排序.png", format='png', bbox_inches='tight', dpi=300)  # 补充dpi参数，使图片更清晰
plt.show()
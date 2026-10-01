import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import random

# 设置图片字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 设置全局随机种子
np.random.seed(45)
random.seed(45)

# ============================================================
# 1. 读取数据
# ============================================================

try:
    df = pd.read_excel('test1.xlsx')
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise


# ============================================================
# 2. 定义特征和目标变量
# ============================================================

feature_names = [
    'DR',
    'Duration of DM',
    'HbA1c',
    'Serum creatinine',
    'TC',
    'Urine protein excretion',
    'FBG',
    'BMI',
    'Age',
    'SBP',
    'LDL',
    'TG',
    'ACR',
    'DBP',
    'HDL',
    'Duration of DN',
    'Sex'
]

target_name = 'Pathology type'

X = df[feature_names].copy()
y = df[target_name].copy()


# ============================================================
# 3. 定义需要均值插补的连续变量
# ============================================================

mean_columns = [
    'Duration of DM',
    'HbA1c',
    'Serum creatinine',
    'TC',
    'Urine protein excretion',
    'FBG',
    'BMI',
    'Age',
    'SBP',
    'LDL',
    'TG',
    'ACR',
    'DBP',
    'HDL',
    'Duration of DN'
]


# ============================================================
# 4. 先划分训练集和测试集
#    在插补之前划分，避免数据泄漏
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45,
    stratify=y
)


# ============================================================
# 5. 缺失值处理
#    Imputer只在训练集上fit
# ============================================================

mean_imputer = SimpleImputer(strategy='mean')

# 训练集：fit + transform
X_train_mean = pd.DataFrame(
    mean_imputer.fit_transform(X_train[mean_columns]),
    columns=mean_columns,
    index=X_train.index
)

# 测试集：只transform
X_test_mean = pd.DataFrame(
    mean_imputer.transform(X_test[mean_columns]),
    columns=mean_columns,
    index=X_test.index
)


# ============================================================
# 6. 拼接Sex和DR
#    保持原始代码中的处理方式
# ============================================================

X_train_processed = pd.concat(
    [
        X_train_mean,
        X_train[['Sex', 'DR']]
    ],
    axis=1
)

X_test_processed = pd.concat(
    [
        X_test_mean,
        X_test[['Sex', 'DR']]
    ],
    axis=1
)

# 保证变量顺序与原始feature_names一致
X_train_processed = X_train_processed[feature_names]
X_test_processed = X_test_processed[feature_names]


# ============================================================
# 7. 创建随机森林分类器
# ============================================================

rf_clf = RandomForestClassifier(
    random_state=45
)


# ============================================================
# 8. 使用真正的RFE进行特征排序
#
# n_features_to_select=1：
# 从17个变量开始，每轮删除一个最不重要变量，
# 最终保留1个变量。
#
# 因此rfe.ranking_可以得到完整的1-17排名。
# ============================================================

rfe = RFE(
    estimator=rf_clf,
    n_features_to_select=1,
    step=1
)

# RFE只在训练集上进行
rfe.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 9. 获取RFE排名
# ============================================================

rfe_ranking = rfe.ranking_

rfe_ranking_df = pd.DataFrame({
    'Feature': feature_names,
    'RFE_Rank': rfe_ranking
})

# 按RFE排名从1到17排序
rfe_ranking_df = rfe_ranking_df.sort_values(
    by='RFE_Rank',
    ascending=True
).reset_index(drop=True)


# ============================================================
# 10. 为了绘制RF特征重要性图，
#     使用全部17个变量重新训练RF
#
# 注意：
# RF Importance用于图形展示；
# 真正用于第三段逐步纳入变量的顺序是RFE_Rank。
# ============================================================

rf_full = RandomForestClassifier(
    random_state=45
)

rf_full.fit(
    X_train_processed,
    y_train
)

feature_importances = rf_full.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': feature_importances
})


# ============================================================
# 11. 将RF Importance合并到RFE结果中
# ============================================================

final_ranking_df = rfe_ranking_df.merge(
    importance_df,
    on='Feature',
    how='left'
)

# 再次确保按照RFE排名排序
final_ranking_df = final_ranking_df.sort_values(
    by='RFE_Rank',
    ascending=True
).reset_index(drop=True)


# ============================================================
# 12. 输出RFE排名
# ============================================================

print("\nRF-RFE特征排序结果：")
print(final_ranking_df)


# ============================================================
# 13. 保存RFE排名
# ============================================================

final_ranking_df.to_excel(
    '2-RF_RFE特征排序.xlsx',
    index=False
)


# ============================================================
# 14. 绘制RF Feature Importance
#
# 图按照RFE排名排列，
# 但横坐标仍然是RF Importance。
# ============================================================

plt.figure(figsize=(8, 6))

bars = plt.barh(
    final_ranking_df['Feature'],
    final_ranking_df['RF_Importance'],
    color='skyblue'
)

plt.xlabel(
    'Importance',
    fontsize=14
)

plt.ylabel(
    'Feature',
    fontsize=14
)

plt.title(
    'Feature Importance',
    fontsize=14
)

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=12,
    rotation=0
)

# Rank 1显示在最上方
plt.gca().invert_yaxis()


# ============================================================
# 15. 在条形上添加Importance数值
# ============================================================

for bar in bars:

    width = bar.get_width()

    plt.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.2f}',
        ha='left',
        va='center',
        fontsize=10
    )


plt.tight_layout(
    pad=1.0
)

plt.savefig(
    "2-重要性排序.pdf",
    format='pdf',
    dpi=300
)

plt.show()

print("\n代码执行完成！")
print("RF-RFE排名已保存至：2-RF_RFE特征排序.xlsx")
print("特征重要性图已保存至：2-重要性排序.pdf")
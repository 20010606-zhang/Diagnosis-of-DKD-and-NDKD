import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 文件
df = pd.read_excel('test1.xlsx')

# 创建斯皮尔曼相关系数矩阵
corr = df.corr(method='spearman')

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8), dpi=1200)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=-1, vmax=1)
scatter_handles = []

# 循环绘制气泡图和数值
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i > j:  # 对角线左下部，只显示气泡
            color = cmap(norm(corr.iloc[i, j]))  # 根据相关系数获取颜色
            scatter = ax.scatter(i, j, s=np.abs(corr.iloc[i, j]) * 1000, color=color, alpha=0.75)
            scatter_handles.append(scatter)  # 保存scatter对象用于颜色条
        elif i < j:  # 对角线右上部分，只显示数值
            color = cmap(norm(corr.iloc[i, j]))  # 数值的颜色同样基于相关系数
            ax.text(i, j, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color=color, fontsize=10)
        else:  # 对角线部分，显示空白
            ax.scatter(i, j, s=1, color='white')

# 设置坐标轴标签
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=10)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 仅用于显示颜色条
fig.colorbar(sm, ax=ax, label='Correlation Coefficient')

# 添加标题和布局调整
plt.tight_layout()
# 保存图形为 PNG 图片文件
plt.savefig("斯皮尔曼相关.png", format='png', bbox_inches='tight')
# 显示图形
plt.show()
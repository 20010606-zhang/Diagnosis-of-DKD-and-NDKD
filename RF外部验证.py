import joblib
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

# 设置字体为Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

try:
    # 加载模型
    loaded_model = joblib.load('random_forest_model.joblib')
    print("模型加载成功")

    # 读取新数据
    new_data = pd.read_excel("验证队列.xlsx")
    print("数据读取成功")

    # 提取特征
    feature_columns = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG',
                       'BMI']
    new_X = new_data[feature_columns]

    # 进行预测
    predictions = loaded_model.predict(new_X)
    print("预测结果：", predictions)

    # 保存预测结果到文件
    result_df = pd.DataFrame({'Predictions': predictions})
    result_df.to_excel("外部验证_validation_predictions.xlsx", index=False)
    print("预测结果已保存到 外部验证_validation_predictions.xlsx")

    # 获取预测概率
    if hasattr(loaded_model, "predict_proba"):
        # 假设是二分类问题，取正类的概率
        y_pred_proba = loaded_model.predict_proba(new_X)[:, 1]
    else:
        raise ValueError("模型不支持 predict_proba 方法，无法计算 AUC。")

    # 提取真实标签，将 'label' 替换为实际的列名
    true_labels = new_data['Pathology type']

    # 计算 AUC
    auc = roc_auc_score(true_labels, y_pred_proba)
    print(f"验证队列的 AUC: {auc:.3f}")

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(true_labels, y_pred_proba)
    # 进行 Bootstrap 抽样来估计置信区间
    n_bootstraps = 1000
    bootstrapped_aucs = []
    bootstrapped_tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for _ in range(n_bootstraps):
        y_true_bs, y_pred_proba_bs = resample(true_labels, y_pred_proba)
        auc_bs = roc_auc_score(y_true_bs, y_pred_proba_bs)
        fpr_bs, tpr_bs, _ = roc_curve(y_true_bs, y_pred_proba_bs)
        tpr_bs = np.interp(base_fpr, fpr_bs, tpr_bs)
        tpr_bs[0] = 0.0
        bootstrapped_aucs.append(auc_bs)
        bootstrapped_tprs.append(tpr_bs)

    bootstrapped_tprs = np.array(bootstrapped_tprs)
    tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
    auc_lower = np.percentile(bootstrapped_aucs, 2.5)
    auc_upper = np.percentile(bootstrapped_aucs, 97.5)

    # 输出95%置信区间
    print(f"AUC的95%置信区间: [{auc_lower:.2f}, {auc_upper:.2f}]")

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', linewidth=2)
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='95% CI')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)

    # 调整刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 保存图片
    plt.savefig('外部验证_roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC 曲线已保存为 外部验证_roc_curve.png")

    # 显示图形（可选）
    plt.show()


except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
except KeyError as e:
    print(f"数据中不存在名为 {e} 的列，请检查列名是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")


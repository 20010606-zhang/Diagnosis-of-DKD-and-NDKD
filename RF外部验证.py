import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc as pr_auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

try:
    # 加载模型
    loaded_model = joblib.load('random_forest_model.joblib')
    print("模型加载成功")

    # 读取新数据
    new_data = pd.read_excel("验证队列.xlsx")
    print("数据读取成功")

    # 提取特征
    feature_columns = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG',
                       'BMI', 'LDL', 'SBP']
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
    print(f"验证队列的 AUC: {auc}")

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

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2, label='95% CI')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic Curve', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)

    # 保存图片
    plt.savefig('外部验证_roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC 曲线已保存为 外部验证_roc_curve.png")

    # 显示图形（可选）
    plt.show()

    # 计算 PR 曲线
    precision, recall, _ = precision_recall_curve(true_labels, y_pred_proba)
    pr_auc_score = pr_auc(recall, precision)

    # 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc_score:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, pad=20)
    plt.legend(loc="upper right", fontsize=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    # 保存 PR 曲线图片
    plt.savefig('外部验证_pr_curve.png', dpi=300, bbox_inches='tight')
    print("PR 曲线已保存为 外部验证_pr_curve.png")

    # 显示 PR 曲线图形（可选）
    plt.show()

    # 计算 DCA 曲线
    thresholds = np.linspace(0, 1, 100)
    net_benefits = []
    all_treat = []
    none_treat = []
    for t in thresholds:
        if t == 1:
            continue  # 避免除以零的错误
        y_pred_threshold = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, y_pred_threshold).ravel()
        n = len(true_labels)
        net_benefit = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(net_benefit)
        all_treat.append((sum(true_labels) / n) - ((n - sum(true_labels)) / n) * (t / (1 - t)))
        none_treat.append(0)

    # 绘制 DCA 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds[:-1], net_benefits, label='Model')
    plt.plot(thresholds[:-1], all_treat, label='Treat all', linestyle='--')
    plt.plot(thresholds[:-1], none_treat, label='Treat none', linestyle='--')
    plt.xlabel('Threshold probability', fontsize=12)
    plt.ylabel('Net benefit', fontsize=12)
    plt.title('Decision Curve Analysis', fontsize=14, pad=20)
    plt.legend(loc="upper right", fontsize=10)
    plt.xlim([0, 1])
    plt.ylim([min(net_benefits + all_treat + none_treat) - 0.02, max(net_benefits + all_treat + none_treat) + 0.02])

    # 保存 DCA 曲线图片
    plt.savefig('外部验证_dca_curve.png', dpi=300, bbox_inches='tight')
    print("DCA 曲线已保存为 外部验证_dca_curve.png")

    # 显示 DCA 曲线图形（可选）
    plt.show()

except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
except KeyError as e:
    print(f"数据中不存在名为 {e} 的列，请检查列名是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

try:
    # 加载模型
    loaded_model = joblib.load('random_forest_model.joblib')
    print("模型加载成功")

    # 读取新数据
    new_data = pd.read_excel("验证队列.xlsx")
    print("数据读取成功")

    # 提取特征
    feature_columns = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP']
    new_X = new_data[feature_columns]

    # 定义不同填充策略的列
    mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP']
    median_columns = ['DR']

    # 创建填充器
    mean_imputer = SimpleImputer(strategy='mean')
    median_imputer = SimpleImputer(strategy='median')

    # 对不同列进行填充
    X_mean = pd.DataFrame(mean_imputer.fit_transform(new_X[mean_columns]), columns=mean_columns)
    X_median = pd.DataFrame(median_imputer.fit_transform(new_X[median_columns]), columns=median_columns)

    # 合并填充后的结果
    new_X_imputed = pd.concat([X_mean, X_median], axis=1)

    # 确保列顺序和原数据一致
    new_X_imputed = new_X_imputed[feature_columns]

    # 进行预测
    predictions = loaded_model.predict(new_X_imputed)
    print("预测结果：", predictions)

    # 保存预测结果到文件
    result_df = pd.DataFrame({'Predictions': predictions})
    result_df.to_excel("validation_predictions.xlsx", index=False)
    print("预测结果已保存到 validation_predictions.xlsx")

    # 获取预测概率
    if hasattr(loaded_model, "predict_proba"):
        # 假设是二分类问题，取正类的概率
        y_pred_proba = loaded_model.predict_proba(new_X_imputed)[:, 1]
    else:
        raise ValueError("模型不支持 predict_proba 方法，无法计算 AUC。")

    # 提取真实标签，将 'label' 替换为实际的列名
    true_labels = new_data['Pathology type']

    # 计算 AUC
    auc = roc_auc_score(true_labels, y_pred_proba)
    print(f"验证队列的 AUC: {auc}")

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(true_labels, y_pred_proba)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")

    # 保存图片
    plt.savefig('roc_curve.png')
    print("ROC 曲线已保存为 roc_curve.png")

    # 显示图形（可选）
    plt.show()

except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
except KeyError as e:
    print(f"数据中不存在名为 {e} 的列，请检查列名是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")
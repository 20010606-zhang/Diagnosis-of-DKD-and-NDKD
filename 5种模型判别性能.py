import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel("test1.xlsx")
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    raise

feature_names = ['DR', 'Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', '2hPBG', 'Duration of DN', 'Sex']
target_name = 'Pathology type'

X = df[feature_names]
y = df[target_name]

mean_columns = ['Duration of DM', 'HbA1c', 'Serum creatinine', 'TC', 'Urine protein excretion', 'FBG', 'BMI', 'Age', 'SBP', 'LDL', 'TG', 'ACR', 'DBP', 'HDL', '2hPBG', 'Duration of DN']
median_columns = ['DR', 'Sex']

mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')

X_mean = pd.DataFrame(mean_imputer.fit_transform(X[mean_columns]), columns=mean_columns)
X_median = pd.DataFrame(median_imputer.fit_transform(X[median_columns]), columns=median_columns)

X = pd.concat([X_mean, X_median], axis=1)
X = X[feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def calculate_metrics(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return auc, sensitivity, specificity, ppv, npv, accuracy, f1


models = {
    "RF": RandomForestClassifier(random_state=42),
    "DT": DecisionTreeClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LR": LogisticRegression(random_state=42, max_iter=20000)
}

metrics_df = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()
    try:
        model.fit(X_train, y_train)

        # 计算模型判别性能
        auc, sensitivity, specificity, ppv, npv, accuracy, f1 = calculate_metrics(model, X_test, y_test)
        metrics_df.append([model_name, auc, sensitivity, specificity, ppv, npv, accuracy, f1])
    except Exception as e:
        print(f"在训练 {model_name} 模型时出现错误: {e}")
    end_time = time.time()
    print(f"{model_name} 模型训练时间: {end_time - start_time:.2f} 秒")

# 创建包含判别性能的 DataFrame
metrics_df = pd.DataFrame(metrics_df, columns=['Model', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1-score'])

# 保存到 Excel 文件
with pd.ExcelWriter('5种模型性能比较.xlsx') as writer:
    metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
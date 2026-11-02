
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

import joblib


# ==============================
# 1. 读取数据
# ==============================

DATA_PATH = Path("diabetes_binary_health_indicators_BRFSS2015.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

print("数据集大小：", X.shape)
print("目标变量分布：")
print(y.value_counts())


# ==============================
# 2. 划分训练集、验证集和测试集
# ==============================

# 首先划分80%训练部分和20%测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 再从80%训练部分中划分10%作为验证集
# 最终比例：72%训练集、8%验证集、20%测试集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=2026,
    stratify=y_train_full
)

print("\n===== 数据集划分结果 =====")
print("训练集：", X_train.shape)
print("验证集：", X_val.shape)
print("测试集：", X_test.shape)


# ==============================
# 3. 特征标准化
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==============================
# 4. 训练逻辑回归模型
# ==============================

print("\n===== 训练逻辑回归模型 =====")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

print("逻辑回归训练完成。")
print("实际迭代次数：", lr_model.n_iter_)


# ==============================
# 5. 验证集评价
# ==============================

y_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]

def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "AUC": roc_auc_score(y_true, y_prob)
    }


print("\n===== 验证集默认阈值0.5评价 =====")
val_default = evaluate_model(y_val, y_val_prob, 0.5)

for key, value in val_default.items():
    print(f"{key}: {value:.4f}")


# ==============================
# 6. 验证集寻找最佳F1阈值
# ==============================

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    result = evaluate_model(y_val, y_val_prob, threshold)

    if result["F1"] > best_f1:
        best_f1 = result["F1"]
        best_threshold = threshold

print("\n===== 验证集最佳阈值 =====")
print("最佳阈值：", round(best_threshold, 2))
print("最佳F1：", round(best_f1, 4))


# ==============================
# 7. 测试集最终评价
# ==============================

y_test_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n===== 测试集默认阈值0.5评价 =====")
test_default = evaluate_model(y_test, y_test_prob, 0.5)

for key, value in test_default.items():
    print(f"{key}: {value:.4f}")


print("\n===== 测试集验证集最优阈值评价 =====")
test_best = evaluate_model(
    y_test,
    y_test_prob,
    best_threshold
)

for key, value in test_best.items():
    print(f"{key}: {value:.4f}")


# ==============================
# 8. 保存模型与实验结果
# ==============================

OUTPUT_DIR = Path("logistic_regression_results")
OUTPUT_DIR.mkdir(exist_ok=True)

joblib.dump(lr_model, OUTPUT_DIR / "lr_model.pkl")
joblib.dump(scaler, OUTPUT_DIR / "lr_scaler.pkl")

pd.DataFrame([
    {"Dataset": "Test_Default_0.5", **test_default},
    {"Dataset": "Test_Best_Threshold", **test_best}
]).to_csv(
    OUTPUT_DIR / "lr_test_results.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "y_true": y_test.to_numpy(),
    "y_prob": y_test_prob
}).to_csv(
    OUTPUT_DIR / "lr_test_probabilities.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n逻辑回归模型及实验结果已保存。")

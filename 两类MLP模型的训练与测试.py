from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ==================================================
# 1. 读取数据
# ==================================================

DATA_PATH = "diabetes_binary_health_indicators_BRFSS2015.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

print("数据集总样本数：", len(df))
print("原始类别分布：")
print(y.value_counts())


# ==================================================
# 2. 划分训练集、验证集、测试集
# ==================================================

# 第一步：80%训练数据 + 20%测试数据
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 第二步：从80%训练数据中划出10%作为验证集
# 最终比例：72%训练 + 8%验证 + 20%测试
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=2026,
    stratify=y_train_full
)

print("\n数据划分结果：")
print("训练集：", len(X_train))
print("验证集：", len(X_val))
print("测试集：", len(X_test))

print("\n训练集类别分布：")
print(y_train.value_counts())

print("\n验证集类别分布：")
print(y_val.value_counts())

print("\n测试集类别分布：")
print(y_test.value_counts())


# ==================================================
# 3. 数据标准化
# ==================================================

# 只使用训练集拟合标准化器
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 4. 定义统一的 MLP 模型参数
# ==================================================

mlp_params = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 300,
    "random_state": 42,
    "early_stopping": False
}


# ==================================================
# 5. 训练原始 MLP 模型
# ==================================================

print("\n正在训练原始 MLP 模型……")
mlp_original = MLPClassifier(**mlp_params)

mlp_original.fit(
    X_train_scaled,
    y_train
)
print("原始 MLP 训练完成！")


# ==================================================
# 6. 对训练集进行随机过采样
# ==================================================

print("\n正在进行随机过采样……")

X_train_array = np.asarray(X_train_scaled)
y_train_array = np.asarray(y_train)

# 获取负类和正类样本的索引
negative_indices = np.where(y_train_array == 0)[0]
positive_indices = np.where(y_train_array == 1)[0]

# 将正类数量增加到负类数量的50%
target_positive_count = int(len(negative_indices) * 0.5)

additional_count = target_positive_count - len(positive_indices)

if additional_count > 0:

    rng = np.random.RandomState(42)

    # 有放回抽取正类样本
    additional_indices = rng.choice(
        positive_indices,
        size=additional_count,
        replace=True
    )

    # 合并原始训练数据和新增正类样本
    X_train_resampled = np.vstack([
        X_train_array,
        X_train_array[additional_indices]
    ])

    y_train_resampled = np.concatenate([
        y_train_array,
        y_train_array[additional_indices]
    ])

else:
    X_train_resampled = X_train_array.copy()
    y_train_resampled = y_train_array.copy()


# 随机打乱过采样后的训练数据
shuffle_rng = np.random.RandomState(42)

shuffle_indices = shuffle_rng.permutation(
    len(y_train_resampled)
)

X_train_resampled = X_train_resampled[shuffle_indices]
y_train_resampled = y_train_resampled[shuffle_indices]

print("过采样前类别分布：")
print(pd.Series(y_train_array).value_counts())

print("\n过采样后类别分布：")
print(pd.Series(y_train_resampled).value_counts())


# ==================================================
# 7. 训练过采样 MLP 模型
# ==================================================

print("\n正在训练过采样 MLP 模型……")

mlp_resampled = MLPClassifier(**mlp_params)

mlp_resampled.fit(
    X_train_resampled,
    y_train_resampled
)

print("过采样 MLP 训练完成！")


# ==================================================
# 8. 定义模型评价函数
# ==================================================


def evaluate_model(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "Specificity": specificity,
        "AUC": roc_auc_score(y_true, y_prob)
    }


# ==================================================
# 9. 在验证集上比较两个模型
# ==================================================

original_val_prob = mlp_original.predict_proba(
    X_val_scaled
)[:, 1]

resampled_val_prob = mlp_resampled.predict_proba(
    X_val_scaled
)[:, 1]


# 默认阈值0.5
original_val_pred = (
    original_val_prob >= 0.5
).astype(int)

resampled_val_pred = (
    resampled_val_prob >= 0.5
).astype(int)


print("\n========== 验证集评价结果（阈值0.5） ==========")

original_val_result = evaluate_model(
    y_val,
    original_val_pred,
    original_val_prob
)

resampled_val_result = evaluate_model(
    y_val,
    resampled_val_pred,
    resampled_val_prob
)

print("\n原始 MLP：")
print(pd.Series(original_val_result).round(4))

print("\n过采样 MLP：")
print(pd.Series(resampled_val_result).round(4))


# ==================================================
# 10. 根据验证集选择各自的最佳F1阈值
# ==================================================

def find_best_threshold(y_true, y_prob):

    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


original_best_threshold, original_best_f1 = find_best_threshold(
    y_val,
    original_val_prob
)

resampled_best_threshold, resampled_best_f1 = find_best_threshold(
    y_val,
    resampled_val_prob
)

print("\n========== 验证集最佳阈值 ==========")

print(
    f"原始 MLP 最佳阈值：{original_best_threshold:.2f}"
)

print(
    f"原始 MLP 验证集最佳 F1：{original_best_f1:.4f}"
)

print(
    f"过采样 MLP 最佳阈值：{resampled_best_threshold:.2f}"
)

print(
    f"过采样 MLP 验证集最佳 F1：{resampled_best_f1:.4f}"
)

print("\n========== 模型训练情况 ==========")

print("原始 MLP 实际迭代次数：", mlp_original.n_iter_)
print("原始 MLP 最终损失：", mlp_original.loss_)

print("过采样 MLP 实际迭代次数：", mlp_resampled.n_iter_)
print("过采样 MLP 最终损失：", mlp_resampled.loss_)

# ==================================================
# 11. 在测试集上进行最终评价
# ==================================================

original_test_prob = mlp_original.predict_proba(
    X_test_scaled
)[:, 1]

resampled_test_prob = mlp_resampled.predict_proba(
    X_test_scaled
)[:, 1]


# 默认阈值0.5的测试结果
original_test_pred_05 = (
    original_test_prob >= 0.5
).astype(int)

resampled_test_pred_05 = (
    resampled_test_prob >= 0.5
).astype(int)


# 使用验证集确定的阈值进行测试
original_test_pred_best = (
    original_test_prob >= original_best_threshold
).astype(int)

resampled_test_pred_best = (
    resampled_test_prob >= resampled_best_threshold
).astype(int)


test_results = []

for model_name, threshold_type, y_pred, y_prob in [

    ("原始 MLP", "默认阈值0.5",
     original_test_pred_05, original_test_prob),

    ("过采样 MLP", "默认阈值0.5",
     resampled_test_pred_05, resampled_test_prob),

    ("原始 MLP", "验证集选定阈值",
     original_test_pred_best, original_test_prob),

    ("过采样 MLP", "验证集选定阈值",
     resampled_test_pred_best, resampled_test_prob)

]:

    result = evaluate_model(
        y_test,
        y_pred,
        y_prob
    )

    result["Model"] = model_name
    result["Threshold_Type"] = threshold_type

    test_results.append(result)


test_results_df = pd.DataFrame(test_results)

print("\n========== 测试集最终评价结果 ==========")

print(
    test_results_df[
        ["Model", "Threshold_Type",
         "Accuracy", "Precision", "Recall",
         "F1", "Specificity", "AUC"]
    ].round(4).to_string(index=False)
)


# ==================================================
# 12. 保存实验结果和模型
# ==================================================

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# 保存测试集评价结果
test_results_df.to_csv(
    "results/mlp_test_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存验证集概率
val_probabilities = pd.DataFrame({
    "y_true": np.asarray(y_val),
    "original_probability": original_val_prob,
    "oversampled_probability": resampled_val_prob
})

val_probabilities.to_csv(
    "results/mlp_validation_probabilities.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存测试集概率
test_probabilities = pd.DataFrame({
    "y_true": np.asarray(y_test),
    "original_probability": original_test_prob,
    "oversampled_probability": resampled_test_prob
})

test_probabilities.to_csv(
    "results/mlp_test_probabilities.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存模型
joblib.dump(
    mlp_original,
    "models/mlp_model_original.pkl"
)

joblib.dump(
    mlp_resampled,
    "models/mlp_model_oversampled.pkl"
)

joblib.dump(
    scaler,
    "models/scaler_comparison.pkl"
)

# 保存阈值信息
threshold_info = pd.DataFrame({
    "Model": ["原始 MLP", "过采样 MLP"],
    "Best_Threshold": [
        original_best_threshold,
        resampled_best_threshold
    ],
    "Validation_Best_F1": [
        original_best_f1,
        resampled_best_f1
    ]
})

threshold_info.to_csv(
    "results/mlp_best_thresholds.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n所有实验完成，结果已保存！")
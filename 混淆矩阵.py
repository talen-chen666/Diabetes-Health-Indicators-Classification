
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==================================================
# 1. 读取三个模型的测试集预测结果
# ==================================================

lr = pd.read_csv(
    "logistic_regression_results/lr_test_probabilities.csv"
)

mlp = pd.read_csv(
    "results/mlp_test_probabilities.csv"
)

# ==================================================
# 2. 检查测试集真实标签是否一致
# ==================================================

y_test = lr["y_true"].to_numpy()

assert (y_test == mlp["y_true"].to_numpy()).all(), \
    "两个文件中的测试集真实标签不一致！"

print("测试集真实标签检查通过！")

# ==================================================
# 3. 提取三个模型的预测概率和最佳阈值
# ==================================================

models = {
    "Logistic Regression": (
        lr["y_prob"].to_numpy(), 0.21
    ),
    "Original MLP": (
        mlp["original_probability"].to_numpy(), 0.21
    ),
    "Oversampled MLP": (
        mlp["oversampled_probability"].to_numpy(), 0.47
    )
}

# ==================================================
# 4. 分别绘制默认阈值和优化阈值下的混淆矩阵
# ==================================================

for threshold_type in ["default", "optimized"]:

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, (y_prob, best_threshold)) in zip(
        axes, models.items()
    ):

        # 默认阈值0.5；优化阈值采用验证集确定的结果
        threshold = (
            0.5 if threshold_type == "default"
            else best_threshold
        )

        # 根据阈值生成预测类别
        y_pred = (y_prob >= threshold).astype(int)

        # 计算混淆矩阵
        cm = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        # 绘制混淆矩阵
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax
        )

        ax.set_title(
            f"{name}\nThreshold = {threshold:.2f}"
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # 总标题
    title = (
        "Confusion Matrices (Default Threshold = 0.5)"
        if threshold_type == "default"
        else "Confusion Matrices (Optimized Threshold)"
    )

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()

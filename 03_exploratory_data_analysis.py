
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 读取数据
file_path = Path("diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.read_csv(file_path)

# 2. 设置图表字体，尽量兼容中文
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False


# 3. 分析目标变量分布
print("===== 目标变量分布 =====")

target_count = df["Diabetes_binary"].value_counts().sort_index()
target_ratio = df["Diabetes_binary"].value_counts(
    normalize=True
).sort_index()

print("类别数量：")
print(target_count)

print("\n类别比例：")
print(target_ratio.round(4))

# 绘制目标变量分布图
fig, ax = plt.subplots(figsize=(7, 5))
target_count.plot(kind="bar", ax=ax)

ax.set_title("Distribution of Diabetes Target")
ax.set_xlabel("Diabetes_binary")
ax.set_ylabel("Number of Samples")
ax.set_xticks(range(len(target_count)))
ax.set_xticklabels(target_count.index, rotation=0)

# 在柱子上方标注样本数量
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

ax.set_ylim(0, target_count.max() * 1.12)

fig.tight_layout()
plt.show()


# 4. 分析高血压与目标变量的关系
print("\n===== 不同高血压状态的目标类别比例 =====")

bp_ratio = pd.crosstab(
    df["HighBP"],
    df["Diabetes_binary"],
    normalize="index"
)

print(bp_ratio.round(4))

# 绘制高血压与糖尿病目标类别的比例图
fig, ax = plt.subplots(figsize=(7, 5))
bp_ratio.plot(kind="bar", ax=ax)
ax.set_title("Diabetes Target Proportion by HighBP")
ax.set_xlabel("HighBP (0 = No, 1 = Yes)")
ax.set_ylabel("Proportion")
ax.set_xticks(range(len(bp_ratio)))
ax.set_xticklabels(bp_ratio.index, rotation=0)
ax.legend(title="Diabetes_binary")

# 在每根柱子上标注百分比
for container in ax.containers:
    labels = [
        f"{bar.get_height():.1%}" if bar.get_height() > 0 else ""
        for bar in container
    ]
    ax.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=9
    )

ax.set_ylim(0, 1.12)

fig.tight_layout()
plt.show()


# 5. 分析高胆固醇与目标变量的关系
print("\n===== 不同高胆固醇状态的目标类别比例 =====")

chol_ratio = pd.crosstab(
    df["HighChol"],
    df["Diabetes_binary"],
    normalize="index"
)

print(chol_ratio.round(4))

# 绘制高胆固醇与糖尿病目标类别的比例图
fig, ax = plt.subplots(figsize=(7, 5))
chol_ratio.plot(kind="bar", ax=ax)
ax.set_title("Diabetes Target Proportion by HighChol")
ax.set_xlabel("HighChol (0 = No, 1 = Yes)")
ax.set_ylabel("Proportion")
ax.set_xticks(range(len(chol_ratio)))
ax.set_xticklabels(chol_ratio.index, rotation=0)
ax.legend(title="Diabetes_binary")

# 在每根柱子上标注百分比
for container in ax.containers:
    labels = [
        f"{bar.get_height():.1%}" if bar.get_height() > 0 else ""
        for bar in container
    ]
    ax.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=9
    )

ax.set_ylim(0, 1.12)

fig.tight_layout()
plt.show()


# 6. 分析 BMI 分布
print("\n===== BMI 描述性统计 =====")
print(df["BMI"].describe())

fig, ax = plt.subplots(figsize=(7, 5))

counts, bin_edges, patches = ax.hist(
    df["BMI"],
    bins=40,
    edgecolor="black"
)

ax.set_title("BMI Distribution")
ax.set_xlabel("BMI")
ax.set_ylabel("Frequency")

# 只标注频数最高的 5 个区间，避免图表过于拥挤
top_n = 5
top_indices = counts.argsort()[-top_n:]

for i in top_indices:
    if counts[i] > 0:
        ax.annotate(
            f"{int(counts[i])}",
            xy=(
                patches[i].get_x() + patches[i].get_width() / 2,
                counts[i]
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

# 为顶部标签留出空间
ax.set_ylim(0, max(counts) * 1.12)

fig.tight_layout()
plt.show()

import pandas as pd
from pathlib import Path

# 1. 设置数据文件路径
file_path = Path("diabetes_binary_health_indicators_BRFSS2015.csv")

# 2. 检查文件是否存在
print("===== 文件检查 =====")
print("数据文件是否存在：", file_path.exists())

# 如果文件不存在，则停止运行并提示
if not file_path.exists():
    raise FileNotFoundError(
        f"找不到数据文件：{file_path.resolve()}\n"
        "请检查文件名和文件所在位置。"
    )

# 3. 读取 CSV 文件
df = pd.read_csv(file_path)

# 4. 查看数据集前5行
print("\n===== 数据集前5行 =====")
print(df.head())

# 5. 查看数据集规模
print("\n===== 数据集规模 =====")
print("样本数量：", df.shape[0])
print("字段数量：", df.shape[1])

# 6. 查看字段名称
print("\n===== 字段名称 =====")
print(df.columns.tolist())

# 7. 查看字段类型和非空值数量
print("\n===== 数据基本信息 =====")
df.info()

# 8. 统计各字段缺失值
print("\n===== 缺失值统计 =====")
print(df.isnull().sum())

# 9. 查看目标变量的类别数量
print("\n===== 目标变量类别数量 =====")
print(df["Diabetes_binary"].value_counts(dropna=False).sort_index())

# 10. 查看目标变量的类别比例
print("\n===== 目标变量类别比例 =====")
print(
    df["Diabetes_binary"]
    .value_counts(normalize=True, dropna=False)
    .sort_index()
)
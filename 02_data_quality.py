import pandas as pd
from pathlib import Path

# 1. 读取数据
file_path = Path("diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.read_csv(file_path)

# 2. 检查完全重复的记录
print("===== 重复记录检查 =====")
duplicate_count = df.duplicated().sum()         #统计完全重复的行数
print("完全重复的记录数量：", duplicate_count)
print("重复记录占比：", duplicate_count / len(df))

# 3. 检查目标变量的实际取值
print("\n===== 目标变量取值检查 =====")
print(df["Diabetes_binary"].value_counts(dropna=False).sort_index())

# 4. 检查每个字段有多少种不同取值
print("\n===== 各字段不同取值数量 =====")
print(df.nunique().sort_values())

# 5. 查看各字段的取值范围和描述性统计
print("\n===== 数值字段描述性统计 =====")
print(df.describe().T)              #汇总数值字段的统计量，并将字段转为行展示

# 6. 检查若干关键字段的具体取值
check_columns = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "Education",
    "Income"
]

print("\n===== 关键字段取值检查 =====")

for column in check_columns:
    print(f"\n{column} 的取值及频数：")
    print(df[column].value_counts().sort_index())
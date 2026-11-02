
import numpy as np
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

models = ['逻辑回归', '原始MLP', '过采样MLP']

metrics = ['准确率', '精确率', '召回率', 'F1分数', '特异度']

# ==================================
# 1. 默认阈值 0.5 下的测试集结果
# ==================================

default_results = {
    '逻辑回归': [0.8622, 0.5182, 0.1590, 0.2433,0.9761],
    '原始MLP': [0.8612,0.5057, 0.1760, 0.2611, 0.9722],
    '过采样MLP': [0.8116, 0.3765,0.5376, 0.4429, 0.8559]
}

# ==================================
# 2. 最优阈值下的测试集结果
# ==================================

optimal_results = {
    '逻辑回归': [0.7977, 0.3650, 0.6104, 0.4568, 0.8281],
    '原始MLP': [0.7911, 0.3551, 0.6118, 0.4494, 0.8202],
    '过采样MLP': [0.7889, 0.3503, 0.6023, 0.4430, 0.8191]
}

# ==================================
# 3. 绘图函数
# ==================================

def draw_chart(results, title, filename):

    if any(
        value is None
        for values in results.values()
        for value in values
    ):
        raise ValueError('请先填入核实后的测试集指标！')

    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):

        values = results[model]

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=model
        )

        ax.bar_label(
            bars,
            fmt='%.3f',
            padding=3,
            fontsize=9
        )

    ax.set_title(title, fontsize=15)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)

    ax.set_ylim(0, 1.10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ==================================
# 4. 生成两张图
# ==================================

draw_chart(
    default_results,
    '三个模型在默认阈值下的性能比较',
    '图1_默认阈值性能比较.png'
)

draw_chart(
    optimal_results,
    '三个模型在最优阈值下的性能比较',
    '图2_最优阈值性能比较.png'
)

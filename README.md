1. Project Overview
项目简介
本项目基于糖尿病相关健康指标数据集（Diabetes Health Indicators Dataset），利用传统机器学习与多层感知机（Multilayer Perceptron, MLP）模型，对调查数据中的糖尿病相关状态进行分类研究。

项目重点关注健康调查数据中的类别不平衡问题，通过构建逻辑回归基线模型、原始数据 MLP 模型以及过采样 MLP 模型，对比不同建模方法在分类任务中的表现。

此外，项目进一步开展分类阈值分析，观察不同阈值下模型精确率、召回率及 F1-score 等评价指标的变化，探讨模型在不同分类决策要求下的表现。
项目目标
- 完成健康调查数据的基本探索性分析与预处理。
- 构建逻辑回归模型，作为传统机器学习基线。
- 构建多层感知机模型，探索神经网络在该分类任务中的应用。
- 针对类别不平衡问题，尝试随机过采样方法。
- 对比不同模型的分类性能，并分析模型阈值变化对分类结果的影响。
2. Dataset
数据集介绍
本项目使用 Diabetes Health Indicators 数据集，数据来源于美国行为风险因素监测系统（Behavioral Risk Factor Surveillance System，BRFSS）相关调查数据。

该数据集包含个人健康状况、生活方式及相关健康指标等信息，可用于开展糖尿病相关状态的统计分析与分类研究。
项目	内容
数据集	Diabetes Health Indicators
数据来源	BRFSS 相关调查数据
数据类型	结构化健康调查数据
研究任务	二分类任务
目标变量	Diabetes_binary

目标变量
Diabetes_binary 为项目的目标变量，用于表示数据中定义的糖尿病相关状态。
- 0：非糖尿病相关类别
- 1：糖尿病相关类别
这里的标签含义应以你下载的数据集说明为准，不能将其直接描述为临床确诊结果。
3. Data Exploration and Preprocessing
数据探索
在模型训练之前，对原始数据进行基本的数据质量检查和探索性分析，主要包括：
- 数据规模与字段信息检查。
- 各变量的数据类型检查。
- 缺失值检查。
- 重复记录检查。
- 目标变量类别分布分析。
- 主要特征的分布与基本统计描述。
数据预处理
根据实际实验流程，主要涉及以下处理：
1. 检查并处理数据中的重复记录。
2. 检查数据是否存在缺失值。
3. 分离特征变量与目标变量。
4. 划分训练集、验证集和测试集（以实际代码为准）。
5. 对特征进行标准化处理。
6. 在训练数据中进行随机过采样实验，以缓解类别不平衡问题。
注意： 过采样应仅作用于训练数据，避免验证集和测试集的信息泄漏。标准化参数也应仅由训练数据拟合，再应用于其他数据集。
4. Model Construction
本项目主要构建以下模型：
4.1 Logistic Regression
使用逻辑回归建立基线分类模型，作为传统机器学习方法的参照。

通过逻辑回归模型初步评估健康指标与目标分类任务之间的关系，为后续神经网络模型的性能比较提供基础。
4.2 Original MLP
基于原始训练数据构建多层感知机分类模型。

模型采用全连接神经网络结构，通过隐藏层学习输入特征与目标变量之间的非线性关系。

当前实验使用的隐藏层结构为：
hidden_layer_sizes = (64, 32)



4.3 MLP with Random Oversampling
针对数据集中正负类别数量不均衡的问题，在训练集上实施随机过采样。

通过增加少数类别样本在训练数据中的占比，改善模型对少数类别的学习能力，并进一步与原始 MLP 模型进行对比。
5. Experimental Design
为了比较不同建模方法的分类效果，本项目设计了以下实验：
实验	模型	主要目的
Experiment 1	Logistic Regression	建立传统机器学习基线
Experiment 2	Original MLP	评估原始数据上的神经网络表现
Experiment 3	Oversampled MLP	分析过采样对少数类别识别的影响
Experiment 4	Threshold Analysis	分析分类阈值对评价指标的影响

实验主要围绕类别不平衡问题展开，重点关注模型对少数类别的识别能力，而不只是整体准确率。
6. Model Evaluation
本项目采用多种分类评价指标对模型进行综合评价。
指标	英文名称	评价意义
准确率	Accuracy	整体预测正确的比例
精确率	Precision	被预测为正类的样本中实际正类的比例
召回率	Recall	实际正类样本中被正确识别的比例
F1-score	F1-score	精确率与召回率的综合指标
ROC-AUC	ROC-AUC	模型区分正负类别的能力

可视化分析
项目可展示以下实验结果：
- 目标变量类别分布图。
- 模型混淆矩阵。
- ROC 曲线。
- 不同分类阈值下的 Precision、Recall 和 F1-score 变化曲线。
- 不同模型评价指标对比图。
7. Threshold Analysis
在二分类任务中，模型通常根据预测概率和预设分类阈值确定最终类别。

默认分类阈值通常为 0.5，但在类别不平衡的情况下，默认阈值未必能够满足少数类别识别的需求。

因此，本项目进一步调整分类阈值，观察不同阈值下 Precision、Recall 和 F1-score 的变化，分析分类决策标准对模型结果的影响。
8. Project Structure
文件或目录	说明
data/	数据集说明及数据存放位置
notebooks/	探索性数据分析
src/	数据处理、模型训练与评价代码
results/figures/	实验结果可视化图表
results/tables/	模型评价结果
requirements.txt	Python 依赖环境
README.md	项目介绍与运行说明

9. Limitations
本项目仍存在一定局限性：
- 数据来源于健康调查，数据质量与调查设计可能影响分析结果。
- 类别不平衡问题可能影响模型对少数类别的识别能力。
- 过采样方法可能增加少数类别样本的重复程度，不能完全解决类别分布带来的问题。
- 当前模型主要用于课程实验与方法探索，不能替代临床诊断或专业医疗判断。
- 模型在其他人群或不同数据来源中的泛化能力仍需进一步验证。
10. Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook（如果实际使用）
11. References
- Behavioral Risk Factor Surveillance System (BRFSS).
- Diabetes Health Indicators Dataset.
- Scikit-learn Documentation: https://scikit-learn.org/

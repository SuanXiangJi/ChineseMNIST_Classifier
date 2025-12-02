# ChineseMNIST-Classifier

## 概述

本工程为中文手写数字（Chinese-MNIST）分类器，实现了多种MLP（多层感知器）变体和CNN（卷积神经网络）模型，用于识别15个类别的中文手写数字。

主目标：**可复现、可扩展、便于学术研究和论文撰写**。

## 环境设置

### 使用Conda创建环境

1. 创建并激活环境：

   ```bash
   conda create -n chinesemnist python=3.10
   conda activate chinesemnist
   ```

2. 安装依赖：

   ```bash
   pip install tensorflow matplotlib numpy pandas scikit-learn
   ```

## 使用说明

### 训练模型

#### 训练CNN模型

```bash
python train_cnn.py
```

#### 训练MLP基础模型

```bash
python train_mlp.py
```

#### 训练MLP变体

```bash
# MLP v1 (GELU + LayerNormalization)
python train_mlp_v1.py

# MLP v2 (Residual Connections + GELU + LayerNormalization)
python train_mlp_v2.py

# Ablation Studies
python train_mlp_ablation_gelu_only.py
python train_mlp_ablation_layernorm_only.py
```

### 评估模型

#### 评估CNN模型

```bash
python eval_cnn.py
```

#### 评估MLP模型

```bash
python eval_mlp.py
```

## 代码文件功能

### 核心配置文件

- **config.py**：定义项目配置，包括数据集路径、图像尺寸、训练参数等

### 核心模块（src/）

- **src/dataset.py**：数据加载和预处理，使用TensorFlow的image_dataset_from_directory加载灰度图像
- **src/model.py**：模型定义，包含多种MLP变体和CNN模型：
  - build_mlp()：基础MLP模型
  - build_cnn()：简单CNN模型
  - build_mlp_v1_gelu_ln()：带GELU激活和LayerNormalization的MLP
  - build_mlp_v2_residual()：带残差连接的MLP
  - build_mlp_ablation_gelu_only()：仅使用GELU激活的MLP（消融实验）
  - build_mlp_ablation_layernorm_only()：仅使用LayerNormalization的MLP（消融实验）
- **src/trainer.py**：模型训练逻辑，包括训练循环、日志记录和模型保存
- **src/evaluator.py**：模型评估逻辑，包括准确率计算、混淆矩阵生成等

### 训练脚本

- **train_cnn.py**：训练CNN模型
- **train_mlp.py**：训练基础MLP模型
- **train_mlp_v1.py**：训练MLP v1模型
- **train_mlp_v2.py**：训练MLP v2模型
- **train_mlp_ablation_gelu_only.py**：训练仅使用GELU的MLP（消融实验）
- **train_mlp_ablation_layernorm_only.py**：训练仅使用LayerNormalization的MLP（消融实验）

### 评估脚本

- **eval_cnn.py**：评估CNN模型
- **eval_mlp.py**：评估MLP模型

### 主脚本

- **main_train.py**：主训练脚本（示例）
- **main_eval.py**：主评估脚本（示例）

### 工具脚本

- **utils.py**：通用工具函数，如目录创建等

## 实验结果

### 训练收敛曲线

#### MLP与CNN收敛对比

![Convergence Curve](/experiments/figures/convergence_curve.png)

该图展示了MLP和CNN模型在训练过程中的准确率变化情况。可以看出，CNN模型收敛速度更快，最终准确率也更高。

#### 消融实验收敛对比

![Ablation Convergence](/experiments/figures/ablation_convergence.png)

该图展示了不同MLP变体在训练过程中的准确率变化情况，包括基础MLP、MLP v1、MLP v2以及消融实验模型。

### 最终准确率对比

#### 不同模型最终准确率

![Final Accuracy](/experiments/figures/ablation_final_accuracy.png)

该图展示了不同模型的最终测试准确率对比，包括：

- MLP（基础模型）
- MLP v1（GELU + LayerNormalization）
- MLP v2（Residual + GELU + LayerNormalization）
- MLP Ablation GELU Only
- MLP Ablation LayerNorm Only
- CNN（对比模型）

### 过拟合差距

![Overfitting Gap](/experiments/figures/overfitting_gap.png)

该图展示了不同模型的训练准确率与测试准确率之间的差距，反映了模型的过拟合程度。

### 准确率与参数数量关系

![Accuracy vs Params](/experiments/figures/accuracy_vs_params.png)

该图展示了不同模型的准确率与参数数量之间的关系，帮助分析模型复杂度与性能的权衡。

### 混淆矩阵

#### 所有模型混淆矩阵对比

![Confusion Matrices](/experiments/figures/confusion_matrices.png)

该图展示了所有模型的混淆矩阵对比，直观反映了模型在每个类别上的分类表现。

#### CNN模型混淆矩阵

![CNN Confusion Matrix](/experiments/figures/confusion_matrix_cnn.png)

该图展示了CNN模型的混淆矩阵，详细反映了CNN在每个类别上的分类表现。

#### MLP模型混淆矩阵

![MLP Confusion Matrix](/experiments/figures/confusion_matrix_mlp.png)

该图展示了基础MLP模型的混淆矩阵，详细反映了MLP在每个类别上的分类表现。

### 模型对比表格

| 模型名称                      | 训练准确率 | 测试准确率 | 参数数量 | 过拟合差距 |
| ----------------------------- | ---------- | ---------- | -------- | ---------- |
| MLP (基础)                    | 88.6%      | 82.2%      | 4.9M     | 6.4%       |
| MLP v1 (GELU + LN)            | 97.3%      | 87.0%      | 4.9M     | 10.3%      |
| MLP v2 (Residual + GELU + LN) | 98.9%      | 76.3%      | 10.9M    | 22.6%      |
| MLP Ablation GELU Only        | 88.4%      | 82.5%      | 4.9M     | 5.9%       |
| MLP Ablation LayerNorm Only   | 96.4%      | 87.0%      | 4.9M     | 9.4%       |
| CNN                           | 99.1%      | 98.1%      | 2.2M     | 1.0%       |

## 实验结论

1. **CNN模型表现最佳**：在所有模型中，CNN模型的测试准确率最高（98.1%），过拟合程度最低（仅1.0%），是性能最好的模型。

2. **MLP变体性能差异**：
   - MLP v1（GELU + LayerNormalization）比基础MLP提高了4.8%的测试准确率
   - MLP v2（添加残差连接）虽然训练准确率最高（98.9%），但测试准确率反而下降，过拟合严重（22.6%）
   - 残差连接的引入可能导致了MLP v2的过拟合问题

3. **消融实验结果**：
   - 仅使用GELU激活比基础MLP提高了0.3%的测试准确率
   - 仅使用LayerNormalization比基础MLP提高了4.8%的测试准确率
   - LayerNormalization对MLP性能提升的贡献更大

4. **模型复杂度与性能权衡**：
   - MLP v2参数数量最多（10.9M），但过拟合严重，测试准确率最低
   - CNN模型参数数量适中（2.2M），但性能最优
   - 基础MLP模型参数数量较少（4.9M），过拟合程度相对较低

5. **MLP模型的局限性**：
   - 所有MLP模型的测试准确率都低于CNN模型
   - MLP模型更容易发生过拟合，尤其是在增加模型复杂度后
   - 对于图像分类任务，CNN模型在特征提取方面具有天然优势

## 扩展说明

1. **添加新模型**：可以在`src/model.py`中添加新的模型定义，然后创建对应的训练脚本。

2. **调整超参数**：可以在`config.py`中修改训练参数，如批量大小、学习率、训练轮数等。

3. **添加新的评估指标**：可以在`src/evaluator.py`中添加新的评估指标和可视化方法。

4. **使用不同的数据集**：可以修改`config.py`中的数据集路径，使用其他类似的图像分类数据集。

## 引用

如果您在学术研究中使用了本项目，请引用以下相关工作：

- Chinese-MNIST数据集：[https://www.kaggle.com/datasets/gpreda/chinese-mnist](https://www.kaggle.com/datasets/gpreda/chinese-mnist)
- MLP相关技术：
  - GELU激活函数：Hendrycks & Gimpel (2016)
  - LayerNormalization：Ba et al. (2016)
  - 残差连接：He et al. (2016)

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

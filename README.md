# 基于归一化分类与困难负类约束的手写汉字识别

CASIA-HWDB1.1 手写汉字识别课程实验的配套代码、配置、结果记录和模型权重说明。

本项目比较归一化分类基线与在退化视图上增加困难负类间隔约束的方法，包含 500 类探索、三个训练种子的重复实验、3755 类扩展和辅助权重分析。结果展示不同方法的识别取舍，不宣称组合方法在所有条件下优于基线。

## 目录

- [方法](#方法)
- [主要结果](#主要结果)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [预训练权重](#预训练权重)
- [训练](#训练)
- [评估](#评估)
- [单张图像识别](#单张图像识别)
- [实验记录与复现边界](#实验记录与复现边界)

## 方法

两个方法共享 ResNet18 主干：输入为单通道 64×64 图像，首层改为 3×3、步长 1 的卷积，移除 max-pooling，输出覆盖 3755 个汉字。

**Baseline：归一化分类。** 对图像特征和分类器权重分别进行 L2 归一化，使用缩放系数为 30 的余弦分数计算交叉熵；分类头不使用偏置。每个源图像形成干净和随机退化两个视图，两个视图共同参与分类训练。

**提出方法：归一化分类 + Top1 困难负类约束。** 在上述基线上，只针对退化视图选择余弦分数最高的错误类别，加入以下辅助损失：

```text
L = L_CE + λ(epoch) × mean(max(0, 0.2 − (cos_y − cos_hard_negative)))
λ(epoch) = 0（第1轮），0.5（第2轮），1（第3至10轮）
```

预测时两个方法使用相同的归一化余弦分类头，不需要真实标签，也不添加训练间隔。`train.py` 调用保存在 `experiments/archive/src/` 中的实际实验实现；归档代码保留原样，新的项目入口负责处理项目相对路径。

## 主要结果

### 3755 类：主实验

以下结果使用种子 42、FP32、第 10 轮检查点、**原始 BN**，在 20 名验证书写者的 74,760 张图像上评估。所有准确率单位为 %。

| 方法 | 干净图像 | 6种训练内退化平均 | 8种训练外退化平均 | 低分辨率20 | 强模糊σ=2.0 |
|---|---:|---:|---:|---:|---:|
| 归一化分类 Baseline | 97.8170 | 97.6231 | 97.0526 | 93.6236 | 96.9395 |
| 归一化分类 + Top1 | 97.6940 | 97.6384 | 97.0817 | 93.6851 | 96.9797 |

组合方法的退化平均提高 0.0291 个百分点，干净准确率下降 0.1231 个百分点。逐书写者配对分析中，干净图像净新增 92 个错误，15/20 名书写者下降。完整结果包含在归档的 `full3755_normalized_hinge_20261012_231818_134276_seed42/` 与对应审计目录中。

### 500 类：三个种子重复实验

以下为种子 42、43、44 的均值 ± 样本标准差；三个种子使用相同的验证集。标准差不代表置信区间。

| 方法 | 干净图像 | 8种训练外退化平均 | 低分辨率20 | 强模糊σ=2.0 |
|---|---:|---:|---:|---:|
| 归一化分类 | 99.2326 ± 0.0453 | 98.7773 ± 0.0260 | 96.8567 ± 0.2093 | 98.6998 ± 0.0837 |
| 归一化分类 + Top1 | 99.2561 ± 0.0628 | 98.8765 ± 0.0088 | 97.0946 ± 0.1561 | 98.8573 ± 0.0878 |
| 归一化分类 + Top3 | 99.2695 ± 0.0614 | 98.8506 ± 0.0315 | 96.9907 ± 0.1075 | 98.7970 ± 0.0912 |

500 类结果中的小幅收益未完整保留到 3755 类。仓库还记录了普通增强、固定间隔、CosFace、BN 校准及权重 0.25/0.5 的比较结果。

## 环境安装

实际训练环境：Linux、Python 3.10.13、PyTorch 2.2.1、torchvision 0.17.1、CUDA 11.8，单张服务器 RTX 4090。正式全类别比较使用 FP32。

```bash
git clone https://github.com/ShuDaonan-3/HWDB-Handwritten-Character-Recognition.git
cd HWDB-Handwritten-Character-Recognition
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'numpy>=1.23,<2' 'Pillow>=9.1,<12'
```

数据预处理和单张识别可以使用 CPU；完整训练入口要求 CUDA。已发生过 FP16 前向非有限值故障，因此本项目默认采用 FP32。故障诊断记录保存在归档中。

## 数据准备

从 CASIA-HWDB 数据发布方获取 HWDB1.1 的 GNT 文件，并遵守数据集的使用条款。仓库不重新分发原始数据或论文 PDF。

```text
data/raw/hwdb1.1/
├── train/  # 1001-f.gnt ... 1240-f.gnt，共240名书写者
└── test/   # 1241-f.gnt ... 1300-f.gnt，共60名书写者
```

```bash
python prepare_data.py
```

预处理：GB2312 一级汉字，共 3755 类；等比例将最长边缩放至 56 像素，居中填充为白底 64×64，像素归一化到 [-1,1]。用于单张预测的图片应为已裁切的单个汉字，黑字白底，不包含额外的大面积留白。

| 划分 | 书写者 | 图像数 | 使用情况 |
|---|---:|---:|---|
| train | 200 | 748,158 | 训练 |
| confusion | 20 | 74,840 | 本次主实验未使用 |
| val | 20 | 74,760 | 方法选择与本文结果 |
| 官方 test | 60 | 223,991 | 本次实验未评估 |

书写者划分和 500 类子集均由固定种子 42 产生，见 `configs/`。本仓库的验证结果不能直接当作官方测试结果，也不能与采用 HWDB1.0+1.1/ICDAR2013 协议的文献数值直接比较。

## 预训练权重

| 模型 | 预期文件 | BN | 轮次 | 种子 |
|---|---|---|---:|---:|
| Baseline | `weights/baseline_normalized_softmax.pt` | 原始 | 10 | 42 |
| 提出方法 | `weights/proposed_normalized_hinge_top1.pt` | 原始 | 10 | 42 |

两份文件均从对应实验的 `last.pt` 原样导出，包含 `model`、`config`、`epoch` 和 `val_accuracy`。完整来源、大小和 SHA256 见 [权重清单](weights/manifest.json)。权重分片保存在 [weights/](weights/) 中。下载仓库后执行下列命令，即可自动合并 ZIP、核对 SHA256 并解压得到两份 `.pt`；如果只下载了代码，脚本会从本仓库补齐分片。

```bash
python download_weights.py
```

分片仅用于绕过上传接口大小限制，恢复后的 `.pt` 与服务器原文件逐字节一致。

SHA256：

```text
d5191ae33b086a77f86e594ba422969be560190c52c817419bcdbd129c1f5a7d  baseline_normalized_softmax.pt
78fec6a71558cc792001226e2a6d6a28786c4951f407b4a956ebe2c6b024487f  proposed_normalized_hinge_top1.pt
```

两个权重共享同一个归一化分类模型结构。辅助损失仅在训练中使用。加载公开检查点前应核对来源；脚本采用 PyTorch 2.2.1 的检查点格式加载本项目可信权重。

## 训练

```bash
# 将可用的物理GPU编号替换为自己的编号；程序内部使用cuda:0
CUDA_VISIBLE_DEVICES=0 python train.py --method baseline

# 用上一步输出的norm_softmax目录替换下列路径
CUDA_VISIBLE_DEVICES=0 python train.py --method proposed \
  --baseline-run outputs/baseline_时间/norm_softmax
```

设置为 AdamW、初始学习率 0.001、权重衰减 0.0001、10轮余弦退火、源图像 batch size 128、拼接视图 batch size 256，保留末尾不足一批的数据。提出方法第1轮仅使用分类损失，脚本检查它是否重现对应基线第1轮的验证结果。

训练输出包含逐轮指标、原始检查点、校准检查点、两种 BN 状态下的退化评估。**所有主结果统一采用原始 BN**，不能为不同方法挑选更有利的 BN 状态。

## 评估

准备好数据索引后，可以用同一入口评估两份权重：

```bash
python evaluate.py --checkpoint weights/baseline_normalized_softmax.pt \
  --suite unseen --output outputs/baseline_unseen.csv
python evaluate.py --checkpoint weights/proposed_normalized_hinge_top1.pt \
  --suite unseen --output outputs/proposed_unseen.csv
```

`--suite` 可选 `clean`、`seen`、`unseen`。`unseen` 指未用于训练的退化参数/组合，**不代表未曾用于方法选择的独立测试集**。8种条件为模糊σ=0.75/1.25/2.0，低分辨率40/28/20，以及降至32后叠加σ=0.5/1.0模糊。平均值不包含干净图像。

## 单张图像识别

```bash
python predict.py --checkpoint weights/baseline_normalized_softmax.pt --image character.png
python predict.py --checkpoint weights/proposed_normalized_hinge_top1.pt --image character.png
```

默认使用 CPU，输出 Top5 汉字、类别编号及 softmax 分数。它用于单字符分类，不提供整页检测、行切分或连续文本识别。softmax 分数不作为经过校准的置信概率。

## 实验记录与复现边界

无需GPU即可核对代码语法、类别顺序、权重分片校验值及离线恢复流程：

```bash
python -m unittest discover -s tests -v
```

```text
├── README.md
├── requirements.txt
├── prepare_data.py / train.py / evaluate.py / predict.py
├── project.py                 # 归档实现加载与权重模型构建
├── configs/                   # 固定类别、书写者划分和实验协议
├── download_weights.py        # 权重分片恢复、下载与校验
├── weights/                   # 两份主实验权重分片、来源与校验值
└── experiments/archive/
    ├── src/                   # 原始实验、消融与诊断脚本
    └── outputs/               # 实际训练指标、汇总表、协议及源码快照
```

历史归档保留了实验时的 `/workspace/shouxieti` 路径以便追溯，不要求直接执行历史入口；请使用仓库根目录的新入口。`data/raw`、缓存、原始图像和大型预测数组不随代码发布；当前权重包只含上述两个主实验模型，不含每个消融和种子的权重。

原实验代码已在服务器运行；新增便携入口在发布整理阶段进行语法和结构核查，尚未在本地重新进行完整GPU训练。不同硬件和计算库可能导致数值差异。结论依据为本仓库记录的实际验证实验，尚无官方测试集性能结论。

本项目用于课程实验与复现学习。方法实现和报告撰写可根据课程要求说明所使用的 AI 辅助。ResNet18 和归一化/间隔分类采用已有思想，不能将本项目组合直接表述为已验证的新颖算法。

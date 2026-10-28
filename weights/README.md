# 主实验权重

此目录包含两份 3755 类主实验模型的 ZIP 分片，均为第10轮、种子42、FP32训练的**原始BN**检查点。

在仓库根目录执行：

```bash
python download_weights.py
```

如果通过 GitHub 的 Code → Download ZIP 下载整个仓库，全部分片已在本地，恢复权重不需要联网；如果只复制了代码和清单，脚本会从本仓库下载缺少的分片。

恢复后的文件：

- `baseline_normalized_softmax.pt`：归一化分类基线。
- `proposed_normalized_hinge_top1.pt`：归一化分类 + 退化视图Top1困难负类间隔约束。

脚本先验证每个分片，再验证合并ZIP和解压后的`.pt`校验值。来源、精确字节数和 SHA256 见 [manifest.json](manifest.json)。不要把`.part00`等分片直接传给`torch.load`。

这两份检查点对应根目录README的3755类主结果，不包括其他消融实验或500类三个种子的全部权重。结构一致，区别来自训练损失；预测时均使用无偏置、缩放系数30的归一化余弦分类头。

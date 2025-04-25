# 光谱-图像恒星分类 (Spectrum-Image Star Classification)

本项目旨在进行论文项目复现，利用深度多模态网络，通过结合恒星的光谱数据和测光图像数据，对 M 型恒星进行分类。
复现参照仓库：https://github.com/Gajiln/Spectrum-Image-Star-Classification

## 项目结构

```
.
├── checkpoints/         # 保存训练好的模型检查点 (.pth 文件)
├── images/              # 存储恒星的测光图像 (按子类分类, 如 M0/, M1/, ...)
├── plots/               # 保存训练过程中的指标图 (.png 文件)
├── spectra/             # 存储恒星的光谱数据 (.fits 文件, 按子类分类)
├── objects.csv          # 包含恒星目标信息和标签的 CSV 文件
├── model.py             # 定义多模态融合模型 (FusionModel)，本代码来自https://github.com/Gajiln/Spectrum-Image-Star-Classification
├── loss_sic.py          # 实现光谱-图像对比损失 (SIC Loss)，本代码来自https://github.com/Gajiln/Spectrum-Image-Star-Classification
├── train.py             # 训练脚本，包含数据加载、训练、验证、评估和历史记录
├── training_history.json # 记录每次训练的配置、逐轮指标和最终结果
├── README.md            # 本说明文件
└──download_sdss_data.py  #用于在SDSS DR16数据库中根据objects.csv爬取恒星数据的文件

```

## 依赖项

主要的 Python 依赖项如下：

*   `torch`: PyTorch 深度学习框架
*   `torchvision`: PyTorch 的计算机视觉库
*   `pandas`: 用于数据处理，特别是读取 CSV 文件
*   `numpy`: 用于数值计算
*   `Pillow`: 用于图像处理
*   `tqdm`: 用于显示进度条
*   `matplotlib`: 用于绘制训练指标图
*   `astropy`: 用于读取 FITS 格式的光谱文件
*   `scikit-learn`: 用于数据分割、标签编码和评估指标计算

建议使用 `pip` 或 `conda` 创建虚拟环境并安装这些依赖项。例如，使用 pip：

```bash
pip install torch torchvision pandas numpy Pillow tqdm matplotlib astropy scikit-learn
```

*注意：请根据你的系统和 CUDA 版本安装合适的 PyTorch 版本。*

## 数据准备

1.  **CSV 文件**: 确保根目录下有 `objects.csv` 文件，包含 `name`, `subclass`, `plate`, `mjd`, `fiberID` 列。`subclass` 应包含 M 型恒星的子类（例如 'M0', 'M1', ..., 'M4'）。
2.  **光谱数据**: 将 FITS 光谱文件放入 `spectra/` 目录下，并根据 `objects.csv` 中的 `subclass` 分类存放。例如，M0 型恒星的光谱应放在 `spectra/M0/` 目录下。文件名格式应为 `spec-pppp-mmmmm-ffff.fits`（p=plate, m=mjd, f=fiberID）。
3.  **图像数据**: 将 JPG 图像文件放入 `images/` 目录下，同样根据 `subclass` 分类存放。文件名格式应为 `{name}_image.jpg`，其中 `{name}` 对应 `objects.csv` 中的 `name` 列。

*脚本在启动时会检查 `objects.csv` 中列出的每个样本对应的光谱和图像文件是否存在，只使用文件齐全的样本进行训练。*

## 模型架构

模型 (`FusionModel` in `model.py`) 采用多模态融合策略：

1.  **图像编码器**: 使用预训练的 ResNet-152 提取图像的全局特征和序列特征。
2.  **光谱编码器**: 使用一系列 1D 卷积层处理光谱数据，提取全局特征和序列特征。
3.  **融合模块**: 将图像和光谱的序列特征与一个可学习的 `[CLS]` token 拼接，并添加 token 类型嵌入和位置嵌入。
4.  **Transformer Encoder**: 使用标准的 PyTorch Transformer Encoder 处理融合后的序列。
5.  **分类头**: 使用 `[CLS]` token 的 Transformer 输出进行最终的分类。
6.  **对比损失**: 同时计算图像和光谱全局特征之间的对比损失（InfoNCE 形式），与分类损失结合，以促进模态对齐。

## 训练与评估

执行以下命令开始训练：

```bash
python train.py
```

脚本将：

1.  加载数据并进行预处理。
2.  分割训练集和验证集。
3.  初始化模型、优化器和损失函数。
4.  执行训练循环，并在每个轮次后进行验证。
5.  打印每轮的损失、准确率、精确率、召回率和 F1 分数。
6.  将验证 F1 分数最高的模型保存到 `checkpoints/` 目录下（文件名包含训练 ID）。
7.  训练结束后，加载最佳模型在验证集上进行最终评估，并打印详细的分类报告。
8.  将本次训练的配置、逐轮指标、最终结果和分类报告追加到 `training_history.json` 文件中。
9.  生成包含训练过程指标（损失、准确率、精确率、召回率、F1）的 PNG 图表，保存到 `plots/` 目录下（文件名包含训练 ID）。

## 输出

*   **模型**: 训练好的模型检查点保存在 `checkpoints/` 目录。
*   **历史记录**: 详细的训练历史保存在 `training_history.json` 文件中。
*   **图表**: 训练过程的可视化图表保存在 `plots/` 目录。

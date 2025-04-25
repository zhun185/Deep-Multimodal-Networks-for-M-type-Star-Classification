import os
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib # 新增

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

# 从你的文件中导入模型和损失函数
from model import FusionModel
from loss_sic import loss_sic

# --- Matplotlib 中文显示配置 --- #
# 尝试使用 SimHei 字体，如果找不到则不报错，依赖系统默认
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
    print("已尝试设置 Matplotlib 字体为 SimHei 以支持中文显示。")
except Exception as e:
    print(f"设置 Matplotlib 中文字体失败: {e}. 中文可能无法正常显示。")

# --- 配置 --- #
CSV_FILE = 'objects.csv'  # CSV 文件名
IMAGE_DIR = 'images'      # 保存图像的根目录
SPECTRA_DIR = 'spectra'    # 保存光谱的根目录
CHECKPOINT_DIR = 'checkpoints' # 保存模型检查点的目录
PLOT_DIR = 'plots' # 新增：保存绘图的目录
HISTORY_FILE = 'training_history.json' # 新增：训练历史记录文件
NUM_EPOCHS = 20 # 训练轮数，根据需要调整
BATCH_SIZE = 32 # 批次大小，根据 GPU 内存调整
LEARNING_RATE = 1e-4 # 学习率
VALIDATION_SPLIT = 0.2 # 验证集比例 (20%)
RANDOM_SEED = 42 # 随机种子，用于可复现的分割
SPEC_LENGTH = 3522 # 光谱向量截断后的期望长度

# --- 设备设置 --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检查是否有可用 GPU
print(f"Using device: {device}")

# --- 创建目录 --- #
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(PLOT_DIR): # 新增
    os.makedirs(PLOT_DIR)      # 新增

# --- 生成唯一的训练 ID --- #
training_id = datetime.now().strftime("%Y%m%d_%H%M%S") # 新增
print(f"Training ID: {training_id}") # 新增

# --- 训练历史记录和绘图函数 --- #

# 保存训练历史记录
def save_training_history(history_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {HISTORY_FILE} 包含无效的 JSON。将创建一个新的历史文件。")
            history = [] # 如果文件无效，则开始新的历史记录

    history.append(history_data)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"训练历史已保存到 {HISTORY_FILE}")

# 绘制训练指标图
def plot_training_history(history, plot_filename):
    epochs = range(1, len(history['epoch_metrics']['train_loss']) + 1)

    plt.figure(figsize=(18, 12)) # 调整画布大小

    # 绘制损失
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['epoch_metrics']['train_loss'], 'bo-', label='训练损失')
    plt.plot(epochs, history['epoch_metrics']['val_loss'], 'ro-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['epoch_metrics']['val_accuracy'], 'go-', label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)

    # 绘制 F1 分数
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['epoch_metrics']['val_f1'], 'mo-', label='验证 F1 分数 (加权)')
    plt.title('验证 F1 分数 (加权)')
    plt.xlabel('轮次')
    plt.ylabel('F1 分数')
    plt.legend()
    plt.grid(True)

    # 绘制精确率
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['epoch_metrics']['val_precision'], 'co-', label='验证精确率 (加权)')
    plt.title('验证精确率 (加权)')
    plt.xlabel('轮次')
    plt.ylabel('精确率')
    plt.legend()
    plt.grid(True)

    # 绘制召回率
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['epoch_metrics']['val_recall'], 'yo-', label='验证召回率 (加权)')
    plt.title('验证召回率 (加权)')
    plt.xlabel('轮次')
    plt.ylabel('召回率')
    plt.legend()
    plt.grid(True)

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出标题空间
    plt.suptitle(f'训练历史记录 (ID: {history["config"]["training_id"]})', fontsize=16) # 修改
    plt.savefig(plot_filename)
    print(f"训练指标图已保存到 {plot_filename}")
    plt.close() # 关闭图形，防止显示


# --- 自定义数据集 --- #
class StarDataset(Dataset):
    def __init__(self, dataframe, image_dir, spectra_dir, spec_transform=None, img_transform=None, label_encoder=None):
        """初始化数据集"
        Args:
            dataframe (pd.DataFrame): 包含文件信息和标签的 DataFrame。
            image_dir (str): 图像文件所在的根目录。
            spectra_dir (str): 光谱文件所在的根目录。
            spec_transform (callable, optional): 应用于光谱的可选转换。
            img_transform (callable, optional): 应用于图像的可选转换。
            label_encoder (LabelEncoder): 用于将子类名转换为数字标签的编码器。
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.spectra_dir = spectra_dir
        self.spec_transform = spec_transform
        self.img_transform = img_transform
        self.label_encoder = label_encoder

    def __len__(self):
        """返回数据集的大小"""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx] # 获取对应行的数据
        spec_id = str(row['name']) # 确保 name 是字符串类型用于构建文件名
        subclass = row['subclass'] # M 型子类 (标签)
        plate = int(row['plate']) # 确保为整数
        mjd = int(row['mjd'])     # 确保为整数
        fiberID = int(row['fiberID']) # 确保为整数

        # --- 加载光谱 --- #
        spec_filename = f"spec-{plate:04d}-{mjd:5d}-{fiberID:04d}.fits" # 构建光谱文件名
        spec_path = os.path.join(self.spectra_dir, subclass, spec_filename) # 构建完整路径
        spectrum = None
        try:
            with fits.open(spec_path) as hdul:
                # 假设通量数据在 HDU 1 (SDSS spec-*.fits 文件的常见结构)
                # 如果你的 FITS 结构不同，需要调整 HDU 索引或数据键名
                flux = hdul[1].data['flux'].astype(np.float32)

                # 确保光谱长度符合要求 (可选，取决于 FITS 文件内容)
                if len(flux) != SPEC_LENGTH:
                     # 处理长度不匹配 (例如：填充、截断或报错)
                     # 目前，如果需要，会打印警告并进行填充/截断
                     # 这部分可能需要根据实际 FITS 数据进行调整
                     #print(f"警告: 光谱文件 {spec_filename} 长度为 {len(flux)}, 期望长度为 {SPEC_LENGTH}。需要跳过或调整大小。")
                     # 简单的截断/填充示例 (谨慎使用):
                     if len(flux) > SPEC_LENGTH:
                         flux = flux[:SPEC_LENGTH] # 截断
                     else:
                         flux = np.pad(flux, (0, SPEC_LENGTH - len(flux)), 'constant') # 填充


                if self.spec_transform: # 如果定义了光谱转换
                    spectrum = self.spec_transform(flux)
                else:
                    spectrum = torch.from_numpy(flux).float() # 基础的 NumPy 到 Tensor 转换

        except FileNotFoundError:
            #print(f"错误: 光谱文件未找到: {spec_path}") # 减少打印，因为文件检查应该已经过滤
            # 返回 None 或抛出错误，以便在 DataLoader 的 collate_fn 中处理缺失数据
            return None # 或者采取其他处理方式
        except Exception as e:
            print(f"加载光谱 {spec_path} 时出错: {e}")
            return None

        # --- 加载图像 --- #
        img_filename = f"{spec_id}_image.jpg" # 构建图像文件名
        img_path = os.path.join(self.image_dir, subclass, img_filename) # 构建完整路径
        image = None
        try:
            image = Image.open(img_path).convert('RGB') # 打开图像并确保为 RGB 格式
            if self.img_transform: # 如果定义了图像转换
                image = self.img_transform(image)
        except FileNotFoundError:
            #print(f"错误: 图像文件未找到: {img_path}") # 减少打印
            return None # 处理缺失数据
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            return None

        # --- 获取标签 --- #
        label = self.label_encoder.transform([subclass])[0] # 使用 label_encoder 将子类名转换为数字
        label = torch.tensor(label, dtype=torch.long) # 转换为 Tensor

        return spectrum, image, label # 返回光谱、图像和标签

# --- 光谱转换 (Min-Max 缩放) --- #
class MinMaxScalerTransform:
    def __call__(self, spectrum_np):
        # 为缩放器调整形状: (样本数, 特征数)
        spectrum_reshaped = spectrum_np.reshape(-1, 1)
        scaler = MinMaxScaler() # 对每个光谱单独进行缩放
        scaled_spectrum = scaler.fit_transform(spectrum_reshaped).flatten()
        return torch.from_numpy(scaled_spectrum).float()

# --- 图像转换 --- #
# 使用标准的 ImageNet 均值和标准差进行归一化
image_transform = transforms.Compose([
    transforms.Resize((128, 128)), # 确保图像大小一致
    transforms.ToTensor(),        # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化
])

spectrum_transform = MinMaxScalerTransform() # 实例化光谱转换器

# --- 加载数据和准备标签 --- #
print("加载 CSV 数据...")
try:
    df = pd.read_csv(CSV_FILE)
    # 过滤掉 subclass 可能为 'Unknown' 或无效的行 (如果下载脚本产生了这些)
    valid_subclasses = ['M0', 'M1', 'M2', 'M3', 'M4']
    df = df[df['subclass'].isin(valid_subclasses)].copy() # 确保只使用期望的类别

    if df.empty:
         print(f"错误: 在 {CSV_FILE} 中未找到子类为 {valid_subclasses} 的有效数据。请确保 CSV 已填充且子类名称正确。")
         exit()

    # --- 文件存在性检查 --- #
    print("检查本地文件是否存在...")
    initial_count = len(df)

    def check_files(row):
        try:
            # 强制转换，以防数据类型问题
            spec_id = str(int(row['name']))
            subclass = str(row['subclass'])
            plate = int(row['plate'])
            mjd = int(row['mjd'])
            fiberID = int(row['fiberID'])

            spec_filename = f"spec-{plate:04d}-{mjd:5d}-{fiberID:04d}.fits"
            spec_path = os.path.join(SPECTRA_DIR, subclass, spec_filename)

            img_filename = f"{spec_id}_image.jpg"
            img_path = os.path.join(IMAGE_DIR, subclass, img_filename)

            return os.path.exists(spec_path) and os.path.exists(img_path)
        except Exception as e:
            # 如果行数据有问题（例如无法转换为整数），也认为文件不存在
            # print(f"检查文件时出错 (行索引 {row.name}): {e}") # 减少打印
            return False

    # 应用检查函数并过滤 DataFrame
    file_exists_mask = df.apply(check_files, axis=1)
    df_filtered = df[file_exists_mask].copy()
    final_count = len(df_filtered)

    print(f"文件检查完成。找到 {final_count} 个具有光谱和图像文件的有效样本 (共 {initial_count} 行)。")

    if df_filtered.empty:
         print(f"错误: 未找到任何同时存在光谱和图像文件的有效样本。请检查下载的数据。")
         exit()

    # 使用过滤后的 DataFrame df_filtered 进行后续操作
    df = df_filtered # 将过滤后的 df 重新赋值给 df，以便后续代码使用

    # 编码标签 (在过滤后的数据上进行)
    label_encoder = LabelEncoder() # 实例化标签编码器
    label_encoder.fit(valid_subclasses) # 基于期望的类别进行拟合
    df['label'] = label_encoder.transform(df['subclass']) # 创建数字标签列
    num_classes = len(label_encoder.classes_) # 获取类别数量
    print(f"找到 {num_classes} 个类别: {label_encoder.classes_}")
    # 获取类别名称映射
    class_names = label_encoder.classes_.tolist() # 新增

except FileNotFoundError:
    print(f"错误: {CSV_FILE} 未找到。")
    exit()
except KeyError as e:
     print(f"错误: 在 {CSV_FILE} 中未找到列 '{e}'。请确保 CSV 包含 'name', 'subclass', 'plate', 'mjd', 'fiberID'。")
     exit()
except Exception as e:
    print(f"加载、处理 CSV 或检查文件时出错: {e}")
    exit()

# --- 分割数据 --- #
print("分割数据...")
train_df, val_df = train_test_split(
    df, # 使用过滤后的 DataFrame
    test_size=VALIDATION_SPLIT, # 验证集比例
    random_state=RANDOM_SEED,   # 随机种子
    stratify=df['label']         # 按标签分层抽样，保持类别分布
)

print(f"训练样本数: {len(train_df)}")
print(f"验证样本数: {len(val_df)}")

# --- 创建数据集和数据加载器 --- #
train_dataset = StarDataset(train_df, IMAGE_DIR, SPECTRA_DIR,
                            spec_transform=spectrum_transform,
                            img_transform=image_transform,
                            label_encoder=label_encoder)
val_dataset = StarDataset(val_df, IMAGE_DIR, SPECTRA_DIR,
                          spec_transform=spectrum_transform,
                          img_transform=image_transform,
                          label_encoder=label_encoder)

# 自定义 collate_fn 以处理数据集中可能出现的 None 值 (加载错误)
def collate_fn(batch):
    # 注意：由于我们在上面已经过滤了不存在的文件，理论上这里不应再收到 None
    # 但保留此函数以处理 StarDataset.__getitem__ 中可能出现的其他加载错误
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None
    spectra, images, labels = zip(*batch)
    # 确保所有返回的张量都是正确的类型和形状
    try:
        spectra = torch.stack(spectra)
        images = torch.stack(images)
        labels = torch.stack(labels)
    except RuntimeError as e:
        print(f"错误: 在批次处理中堆叠张量时出错: {e}")
        print(f"检查批次中的数据项...")
        # 可以在这里添加更多调试信息，例如打印每个项的形状
        # for i, item in enumerate(zip(spectra, images, labels)):
        #     print(f"Item {i}: spec shape={item[0].shape}, img shape={item[1].shape}, label shape={item[2].shape}")
        return None, None, None # 返回 None 以跳过此批次
    return spectra, images, labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn) # num_workers=0 有时可提高 Windows 兼容性
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

# --- 初始化模型、损失函数、优化器 --- #
print("初始化模型...")
model = FusionModel(num_classes=num_classes).to(device) # 传入 num_classes
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 使用 Adam 优化器

# --- 记录初始配置 --- #
config_data = {
    "training_id": training_id,
    "timestamp": datetime.now().isoformat(),
    "csv_file": CSV_FILE,
    "image_dir": IMAGE_DIR,
    "spectra_dir": SPECTRA_DIR,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "validation_split": VALIDATION_SPLIT,
    "random_seed": RANDOM_SEED,
    "spec_length": SPEC_LENGTH,
    "device": str(device),
    "num_classes": num_classes,
    "class_names": class_names, # 新增
    "model_architecture": model.__class__.__name__, # 新增
    "optimizer": optimizer.__class__.__name__ # 新增
}
print("训练配置:", json.dumps(config_data, indent=2)) # 打印配置信息

# --- 初始化训练历史列表 --- #
epoch_metrics = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": []
}

# --- 训练循环 --- #
print("开始训练...")
best_val_f1 = 0.0 # 用于跟踪最佳 F1 分数
best_epoch = -1 # 新增：跟踪最佳轮次

start_train_time = time.time() # 新增：记录总训练开始时间

for epoch in range(NUM_EPOCHS):
    model.train() # 设置模型为训练模式
    train_loss = 0.0
    start_epoch_time = time.time() # 记录每轮开始时间

    # 使用 tqdm 显示进度条
    train_iterator = tqdm(train_loader, desc=f"轮次 {epoch+1}/{NUM_EPOCHS} [训练]")

    for batch_idx, batch in enumerate(train_iterator):
        # 如果 collate_fn 返回 None (例如，批次中所有项都有错误或堆叠失败)，则跳过
        if batch is None or batch[0] is None:
            print(f"警告: 跳过空的或无效的训练批次 {batch_idx}")
            continue

        spectra, images, targets = batch # 解包批次数据
        spectra, images, targets = spectra.to(device), images.to(device), targets.to(device) # 移动数据到设备

        optimizer.zero_grad() # 清空梯度

        # 前向传播
        align_logits, fuse_logits = model(spectra, images)

        # 计算损失
        loss = loss_sic(align_logits, fuse_logits, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # 累加训练损失

        # 更新 tqdm 进度条的后缀信息
        train_iterator.set_postfix(loss=loss.item())

    # 检查 train_loader 是否为空
    if not train_loader:
        print("错误: 训练数据加载器为空，无法计算平均训练损失。")
        avg_train_loss = 0.0 # 或者其他适当的值
    else:
        avg_train_loss = train_loss / len(train_loader) # 计算平均训练损失
    epoch_metrics["train_loss"].append(avg_train_loss) # 记录训练损失

    # --- 验证循环 --- #
    model.eval() # 设置模型为评估模式
    val_loss = 0.0
    all_targets = [] # 存储所有真实标签
    all_predictions = [] # 存储所有预测结果

    val_iterator = tqdm(val_loader, desc=f"轮次 {epoch+1}/{NUM_EPOCHS} [验证]")

    with torch.no_grad(): # 禁用梯度计算
        for batch in val_iterator:
             # 如果 collate_fn 返回 None，则跳过
            if batch is None or batch[0] is None:
                 print(f"警告: 跳过空的或无效的验证批次")
                 continue

            spectra, images, targets = batch
            spectra, images, targets = spectra.to(device), images.to(device), targets.to(device)

            align_logits, fuse_logits = model(spectra, images) # 前向传播
            loss = loss_sic(align_logits, fuse_logits, targets) # 计算验证损失
            val_loss += loss.item() # 累加验证损失

            # 获取预测结果
            predictions = torch.argmax(fuse_logits, dim=1)
            all_targets.extend(targets.cpu().numpy()) # 收集真实标签
            all_predictions.extend(predictions.cpu().numpy()) # 收集预测结果

            val_iterator.set_postfix(loss=loss.item()) # 更新进度条

    # 检查 val_loader 和 all_targets 是否为空
    if not val_loader:
        print("错误: 验证数据加载器为空，无法计算验证指标。")
        avg_val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
    elif not all_targets: # 如果所有验证批次都被跳过
        print("警告: 没有有效的验证样本来计算指标。")
        avg_val_loss = val_loss # 可能非零，取决于是否有无效批次贡献损失
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
    else:
        avg_val_loss = val_loss / len(val_loader) # 计算平均验证损失
        val_accuracy = accuracy_score(all_targets, all_predictions) # 计算准确率
        # 计算加权精确率、召回率、F1 分数 (适用于多分类和类别不平衡)
        # zero_division=0 避免在某个类别没有预测样本时产生警告或错误
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )

    # 记录验证指标
    epoch_metrics["val_loss"].append(avg_val_loss)
    epoch_metrics["val_accuracy"].append(val_accuracy)
    epoch_metrics["val_precision"].append(val_precision)
    epoch_metrics["val_recall"].append(val_recall)
    epoch_metrics["val_f1"].append(val_f1)

    epoch_time = time.time() - start_epoch_time # 计算此轮耗时

    # 更新打印信息以包含 Precision 和 Recall
    print(f"轮次 {epoch+1}/{NUM_EPOCHS} | 耗时: {epoch_time:.2f}s | \
          训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | \
          验证准确率: {val_accuracy:.4f} | 验证 Prec: {val_precision:.4f} | \
          验证 Rec: {val_recall:.4f} | 验证 F1: {val_f1:.4f}") # 修改

    # --- 保存最佳模型 --- #
    # 如果当前验证 F1 分数优于之前的最佳分数
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1 # 记录最佳轮次
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model_{training_id}.pth') # 文件名包含训练 ID
        torch.save(model.state_dict(), checkpoint_path) # 保存模型的状态字典
        print(f"** 已保存最佳模型 (轮次 {epoch+1}) 到 {checkpoint_path}，F1 分数: {best_val_f1:.4f} **")

# --- 训练结束 --- #
end_train_time = time.time() # 新增
total_training_time = end_train_time - start_train_time # 新增
print("训练完成。")
print(f"总训练耗时: {total_training_time:.2f}s")
print(f"最佳验证 F1 分数: {best_val_f1:.4f} (在轮次 {best_epoch})")

# --- 使用最佳模型进行最终评估 --- #
final_metrics = {}
final_classification_report = ""
best_model_path = os.path.join(CHECKPOINT_DIR, f'best_model_{training_id}.pth')

if os.path.exists(best_model_path):
    print(f"加载最佳模型 {best_model_path} 进行最终评估...")
    # 重新创建模型实例以加载状态字典
    best_model = FusionModel(num_classes=num_classes).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval() # 设置为评估模式

    all_targets_final = []
    all_predictions_final = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="最终评估"):
            if batch is None or batch[0] is None:
                continue
            spectra, images, targets = batch
            spectra, images, targets = spectra.to(device), images.to(device), targets.to(device)
            _, fuse_logits = best_model(spectra, images)
            predictions = torch.argmax(fuse_logits, dim=1)
            all_targets_final.extend(targets.cpu().numpy())
            all_predictions_final.extend(predictions.cpu().numpy())

    if all_targets_final: # 确保有评估结果
        final_accuracy = accuracy_score(all_targets_final, all_predictions_final)
        final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
            all_targets_final, all_predictions_final, average='weighted', zero_division=0
        )
        # 生成分类报告
        final_classification_report = classification_report(
            all_targets_final, all_predictions_final, target_names=class_names, zero_division=0
        )

        final_metrics = {
            "best_epoch": best_epoch,
            "final_val_accuracy": final_accuracy,
            "final_val_precision_weighted": final_precision,
            "final_val_recall_weighted": final_recall,
            "final_val_f1_weighted": final_f1
        }

        print("--- 最终评估结果 (使用最佳模型) ---")
        print(f"准确率: {final_accuracy:.4f}")
        print(f"精确率 (加权): {final_precision:.4f}")
        print(f"召回率 (加权): {final_recall:.4f}")
        print(f"F1 分数 (加权): {final_f1:.4f}")
        print("\n分类报告:\n", final_classification_report)
    else:
        print("警告: 未能使用最佳模型进行最终评估（没有有效数据）。")

else:
    print(f"错误: 未找到最佳模型文件 {best_model_path}，无法进行最终评估。")


# --- 保存训练历史和绘图 --- #
training_history_data = {
    "config": config_data,
    "epoch_metrics": epoch_metrics,
    "final_metrics": final_metrics,
    "classification_report": final_classification_report,
    "total_training_time_seconds": total_training_time # 新增
}

# 保存历史到 JSON 文件
save_training_history(training_history_data)

# 绘制并保存指标图
plot_filename = os.path.join(PLOT_DIR, f'training_plot_{training_id}.png')
plot_training_history(training_history_data, plot_filename)

print("脚本执行完毕。") 
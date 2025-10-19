# %% [markdown]
# # 阶段 1：自监督预训练 (SSL)
# 
# 我们将使用对比学习 (SimCLR-style) 来预训练 EEGConformer。
# 我们会加载除 contrastChangeDetection (我们的目标任务) 之外的所有任务
# 作为无标签的源数据。

# %% 1. Imports and setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_target_channels
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import EEGConformer
from pathlib import Path
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import copy

# 识别设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% 2. 定义常量和数据目录
DATA_DIR = Path('/ssd1/zhanghongbo04/002/project/AI-competition/202510EEG_Foundation_Challenge/startkit/data')
# DATA_DIR.mkdir(parents=True, exist_ok=True) # 假设目录已存在

SFREQ = 100
EPOCH_LEN_S = 2.0
WINDOW_SIZE_SAMPLES = int(EPOCH_LEN_S * SFREQ)

# 自监督训练参数
PRETRAIN_EPOCHS = 100  # 真实训练应该更多, e.g., 200-500
PRETRAIN_LR = 1E-3
PRETRAIN_BATCH_SIZE = 128 # 越大越好，取决于你的显存
TEMPERATURE = 0.1     # 对比损失的温度系数
PROJECTION_DIM = 128  # 投影头的输出维度

# %% 3. 加载所有 *源任务* 数据
# 这是你提供的文件列表中除目标任务外的所有任务
all_source_tasks = [
    'DespicableMe',
    'DiaryOfAWimpyKid',
    'FunwithFractals',
    'RestingState',
    'seqLearning8target',
    'surroundSupp',
    'symbolSearch',
    'ThePresent',
    'contrastChangeDetection'
]
source_tasks = ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'RestingState', 'ThePresent']
print(f"Loading {len(source_tasks)} source tasks for pre-training...")

# 注意:
# 1. 'mini=True' 是为了快速演示。真实比赛中必须使用 'mini=False'
# 2. 我们使用 'R5' release
all_source_datasets = []
for task in source_tasks:
    try:
        dataset = EEGChallengeDataset(
            task=task,
            release="R5",
            cache_dir=DATA_DIR,
            mini=False  # !! 设置为 False 来获取完整数据集
        )
        all_source_datasets.append(dataset)
        print(f"Loaded {task} (mini=False) with {len(dataset.datasets)} recordings.")
    except Exception as e:
        print(f"Could not load task {task}: {e}")

# 将所有源数据集合并为一个大数据集
pretrain_dataset = BaseConcatDataset(all_source_datasets)
print(f"Total recordings for pre-training: {len(pretrain_dataset.datasets)}")

# %% 4. 创建无标签的预训练窗口
# 我们不需要任何事件锚点，只是在连续数据上创建滑动窗口
# 我们使用 create_windows_from_target_channels 并设置 mapping=None 来表示无监督
print("Creating unsupervised windows from all source tasks...")
pretrain_windows = create_fixed_length_windows(
    pretrain_dataset,
    window_size_samples=WINDOW_SIZE_SAMPLES,
    window_stride_samples=int(WINDOW_SIZE_SAMPLES * 0.5), # 50% 重叠
    drop_last_window=True,  # 丢弃末尾不够一个窗口的样本，确保所有窗口大小一致
    preload=True,
    mapping=None, # 关键：表示这是一个无监督数据集 (y=None)
    n_jobs=4      # 使用多核处理
)

print(f"Total windows for pre-training: {len(pretrain_windows)}")

# %% 5. 定义数据增强和 Collate Function
# 简单的 EEG 增强
def augment(X, noise_level=0.1, channel_dropout=0.2):
    """
    X: [Batch, Channels, Time]
    """
    # 1. 添加高斯噪声
    noise = torch.randn_like(X) * noise_level
    X = X + noise
    
    # 2. 随机通道丢弃 (Channel Dropout)
    if channel_dropout > 0:
        B, C, T = X.shape
        # 为每个样本和通道创建一个 mask
        mask = (torch.rand(B, C, 1, device=X.device) > channel_dropout).float()
        X = X * mask
        
    return X

def collate_fn_contrastive(batch):
    """
    自定义 collate_fn
    - batch 是一个 list of (X, y, i) tuples
    - y 和 i 在这里被忽略
    - 我们只提取 X，堆叠它们，然后应用两次增强
    - 返回一个 [2*B, C, T] 的张量
    """
    # 1. 从 batch 中提取 X (item[0])
    # 确保它们是 torch tensor
    X_list = [torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
    X = torch.stack(X_list)
    
    # 2. 创建两个不同的增强视图
    X_i = augment(X)
    X_j = augment(X)
    
    # 3. 将它们连接成一个 [2*B, C, T] 的张量
    return torch.cat([X_i, X_j], dim=0)

# %% 6. 创建预训练 DataLoader
pretrain_loader = DataLoader(
    pretrain_windows,
    batch_size=PRETRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True, # !! 必须：对比损失需要固定的 batch size
    collate_fn=collate_fn_contrastive
)

# %% 7. 定义对比损失 (NT-Xent Loss)
class NTXentLoss(nn.Module):
    def __init__(self, temperature, batch_size):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        
        # 掩码，用于移除对角线上的“正样本” (i.e., (z_i, z_i))
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(self.device)

    def forward(self, z):
        """
        z: [2*B, ProjectionDim] - 包含了所有视图的投影
        """
        B = z.shape[0] // 2
        
        # 动态检查 batch size，以防最后一个 batch 较小 (尽管我们用了 drop_last)
        if B != self.batch_size:
            print(f"Warning: Batch size mismatch. Expected {self.batch_size}, got {B}")
            # 动态创建掩码
            current_mask = (~torch.eye(B * 2, B * 2, dtype=torch.bool)).float().to(self.device)
        else:
            current_mask = self.mask

        # 归一化投影
        z_norm = F.normalize(z, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = (z_norm @ z_norm.T) / self.temperature
        
        # 创建标签 (正样本对)
        # z_i 和 z_j 是正样本对
        # z_i 在 [0..B-1], z_j 在 [B..2B-1]
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(B)
        ]).to(self.device)
        
        # (2B, 2B) -> (2B, 2B-1)
        logits = sim_matrix[current_mask.bool()].view(2 * B, -1)
        
        # 计算 CrossEntropyLoss
        loss = F.cross_entropy(logits, labels)
        return loss

# %% 8. 定义带投影头的预训练模型
class PretrainEEGConformer(nn.Module):
    def __init__(self, n_chans, n_times, sfreq, projection_dim):
        super().__init__()
        # 1. 加载骨干网络
        self.backbone = EEGConformer(
            n_chans=n_chans,
            n_outputs=1, # 随便设置一个，我们马上会替换掉它
            n_times=n_times,
            sfreq=sfreq,
        )
        
        # 2. 获取骨干的输出维度
        embedding_size = self.backbone.fc.in_features
        
        # 3. 替换掉骨干的最后一层 (fc)
        self.backbone.fc = nn.Identity()
        
        # 4. 创建新的投影头 (Projector)
        self.projector = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, projection_dim)
        )
        
    def forward(self, x):
        # x: [2*B, C, T]
        features = self.backbone(x)  # [2*B, embedding_size]
        projections = self.projector(features) # [2*B, projection_dim]
        return projections

# %% 9. 初始化模型、损失和优化器
# 从窗口数据中自动获取 n_chans 和 n_times
n_chans = pretrain_windows[0][0].shape[0]
n_times = pretrain_windows[0][0].shape[1]
print(f"Detected input shape: {n_chans} channels, {n_times} time points")

pretrain_model = PretrainEEGConformer(
    n_chans=n_chans,
    n_times=n_times,
    sfreq=SFREQ,
    projection_dim=PROJECTION_DIM
).to(device)

loss_fn = NTXentLoss(temperature=TEMPERATURE, batch_size=PRETRAIN_BATCH_SIZE)
optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

# %% 10. 预训练循环
print("Starting self-supervised pre-training...")

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    pretrain_model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(pretrain_loader), total=len(pretrain_loader))
    
    for batch_idx, X_augmented in progress_bar:
        # X_augmented 已经是 [2*B, C, T] 并且在 collate_fn 中被创建
        X_augmented = X_augmented.to(device).float()
        
        optimizer.zero_grad(set_to_none=True)
        
        # 获取投影
        projections = pretrain_model(X_augmented)
        
        # 计算对比损失
        loss = loss_fn(projections)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_description(
            f"Pre-train Epoch {epoch}, Batch {batch_idx + 1}/{len(pretrain_loader)}, "
            f"Loss: {loss.item():.6f}"
        )
        
    scheduler.step()
    avg_loss = total_loss / len(pretrain_loader)
    print(f"--- Epoch {epoch}/{PRETRAIN_EPOCHS} ---")
    print(f"Average Pre-training Loss: {avg_loss:.6f}")

print("Pre-training finished.")

# %% 11. 保存*骨干网络*的权重
# 这才是阶段 2 (微调) 需要的文件！
# 我们丢弃投影头 (self.projector)，只保存 self.backbone
save_path = "pretrain_backbone_weights.pt"
torch.save(pretrain_model.backbone.state_dict(), save_path)
print(f"Backbone model weights saved to '{save_path}'")
# %% [markdown]
# # 阶段 2：在 CCD 任务上进行监督微调 (Supervised Fine-Tuning)
# 
# 1. 加载第一阶段预训练好的骨干网络权重（patch_embedding 和 transformer 部分）。
# 2. 加载 CCD 任务数据集。
# 3. 构建一个新的 EEGConformer 模型用于微调。
# 4. 将预训练的 patch_embedding 和 transformer 权重加载到新模型中。
# 5. 保持新的 fc 和 final_layer（用于回归）为随机初始化或根据需要调整。
# 6. 在 CCD 数据集上进行监督微调。

# %% 1. Imports and setup (Assuming necessary imports are already done)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.models import EEGConformer
from pathlib import Path
from eegdash.dataset import EEGChallengeDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from typing import Optional
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import copy
import numpy as np
from eegdash.hbn.windows import ( # Assuming these are needed from the starter kit
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Identify device (assuming device is already set)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for fine-tuning: {device}")

# %% 2. 定义常量和数据目录 (Same as starter kit, adjust paths as needed)
DATA_DIR = Path('/ssd2/zhanghongbo04/data') # Adjust path to your data location
# DATA_DIR.mkdir(parents=True, exist_ok=True) # Assuming directory exists

SFREQ = 100
EPOCH_LEN_S = 2.0
WINDOW_SIZE_SAMPLES = int(EPOCH_LEN_S * SFREQ)
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

# 微调参数 (通常比预训练的学习率小)
FINETUNE_EPOCHS = 100 # e.g., 100-200 or more, depending on performance
FINETUNE_LR = 1E-4 # e.g., 1E-4 or 5E-5, smaller than pretrain LR
FINETUNE_BATCH_SIZE = 128 # Same as pretrain, adjust if needed
VALID_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 2025
EARLY_STOPPING_PATIENCE = 20 # e.g., 10-20

# %% 3. 加载预训练的骨干网络权重 (patch_embedding 和 transformer 部分)
PRETRAINED_BACKBONE_PATH = "pretrain_backbone_weights.pt" # Path to weights saved in Phase 1

# 加载 CCD 任务数据以获取 n_chans 和 n_times (类似预训练部分)
ccd_task = 'contrastChangeDetection'
print(f"Loading target task for fine-tuning: {ccd_task}...")
dataset_ccd_raw = EEGChallengeDataset(
    task=ccd_task,
    release="R1", # Use the same release as pretrain, or adjust if needed
    cache_dir=DATA_DIR,
    mini=True # !! Set to False for full dataset
)
# Prepare the dataset for windowing (similar to starter kit)
transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
        require_stimulus=True, require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd_raw, transformation_offline, n_jobs=1) # Adjust n_jobs as needed

ANCHOR = "stimulus_anchor"
dataset_ccd_raw = keep_only_recordings_with(ANCHOR, dataset_ccd_raw)

# Create single-interval windows (similar to starter kit)
single_windows_ccd = create_windows_from_events(
    dataset_ccd_raw,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),                 # +0.5 s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),   # +2.5 s
    window_size_samples=WINDOW_SIZE_SAMPLES, # 200 samples
    window_stride_samples=SFREQ, # 100 samples for overlapping
    preload=True,
)

single_windows_ccd = add_extras_columns(
    single_windows_ccd,
    dataset_ccd_raw,
    desc=ANCHOR,
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
)

# Get n_chans and n_times from the loaded windows
n_chans_finetune = single_windows_ccd[0][0].shape[0] # Number of channels
n_times_finetune = single_windows_ccd[0][0].shape[1] # Number of time points
print(f"Detected input shape for fine-tuning: {n_chans_finetune} channels, {n_times_finetune} time points")

# %% 4. 构建用于微调的完整模型
# 创建一个新的 EEGConformer 实例, 配置为回归任务
model = EEGConformer(
    n_chans=n_chans_finetune,
    n_outputs=1, # Output 1 value for regression (response time)
    n_times=n_times_finetune,
    sfreq=SFREQ,
    final_fc_length="auto" # Calculate final_fc_length based on new model structure
)

# --- 加载预训练的骨干网络权重 ---
# 1. 加载预训练的 state_dict
pretrained_state_dict = torch.load(PRETRAINED_BACKBONE_PATH, map_location=device)
print(f"Loaded pre-trained state_dict keys: {pretrained_state_dict.keys()}")

# 2. 准备当前模型的 state_dict
model_state_dict = model.state_dict()
print(f"Current model state_dict keys: {model_state_dict.keys()}")

# 3. 过滤出预训练权重中属于 patch_embedding 和 transformer 的部分
filtered_pretrained_weights = {}
for key, value in pretrained_state_dict.items():
    # EEGConformer 的主要骨干部分是 patch_embedding 和 transformer
    if key.startswith('patch_embedding.') or key.startswith('transformer.'):
        # 检查当前模型是否也有这个键
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            filtered_pretrained_weights[key] = value
        else:
            print(f"Warning: Pretrained key '{key}' not found in model or shape mismatch. Skipping.")

print(f"Filtered keys to load: {filtered_pretrained_weights.keys()}")

# 4. 加载过滤后的权重
model.load_state_dict(filtered_pretrained_weights, strict=False) # Use strict=False to ignore missing keys (fc, final_layer)
print(f"Loaded pre-trained 'patch_embedding' and 'transformer' weights from '{PRETRAINED_BACKBONE_PATH}'.")

# 5. Reinitialize the final regression layer to ensure it's suitable for regression with 1 output
# EEGConformer 在初始化时已经根据 n_outputs=1 设置了 final_layer
# 但为了确保其权重是随机初始化的（而不是加载了可能不匹配的预训练权重），我们重新创建它
# 然而，上面的 load_state_dict(strict=False) 不会加载 final_layer，所以它保持随机初始化
# 或者，我们可以明确地重新初始化它
# 注意：final_layer 的输入维度是 model.fc.hidden_channels，这在 model 初始化时也已确定
model.final_layer = nn.Linear(model.fc.hidden_channels, 1) # 重新初始化最终层用于回归
print("Reinitialized 'final_layer' for regression (1 output).")

# --- 可选：冻结骨干网络参数 ---
# FREEZE_BACKBONE = True # Set to True to freeze, False to allow fine-tuning
# if FREEZE_BACKBONE:
#     for name, param in model.named_parameters():
#         if name.startswith('patch_embedding.') or name.startswith('transformer.'):
#             param.requires_grad = False
#     print("Frozen backbone parameters (patch_embedding, transformer).")
# else:
#     print("Allowing fine-tuning of backbone parameters (patch_embedding, transformer).")
# -----------------------------------

print("Model architecture after loading pre-trained backbone and reinitializing head:")
print(model)

model.to(device)

# %% 5. 加载并准备 CCD 数据集进行微调 (Using the starter kit logic)
# Split data into train/valid/test
meta_information = single_windows_ccd.get_metadata()
subjects = meta_information["subject"].unique()
# Remove problematic subjects if any (as in starter kit)
# sub_rm = ["NDARWV769JM7", "NDARME789TD2", ...] # Define if needed
# subjects = [s for s in subjects if s not in sub_rm]

train_subj, valid_test_subject = train_test_split(
    subjects, test_size=(VALID_FRAC + TEST_FRAC), random_state=check_random_state(SEED), shuffle=True
)
valid_subj, test_subj = train_test_split(
    valid_test_subject, test_size=TEST_FRAC/(VALID_FRAC + TEST_FRAC), random_state=check_random_state(SEED + 1), shuffle=True
)

# Create splits
subject_split = single_windows_ccd.split("subject")
train_set = BaseConcatDataset([subject_split[s] for s in subject_split if s in train_subj])
valid_set = BaseConcatDataset([subject_split[s] for s in subject_split if s in valid_subj])
test_set = BaseConcatDataset([subject_split[s] for s in subject_split if s in test_subj])

print("Number of examples in each split for fine-tuning")
print(f"Train:\t{len(train_set)}")
print(f"Valid:\t{len(valid_set)}")
print(f"Test:\t{len(test_set)}")

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4) # Adjust num_workers
valid_loader = DataLoader(valid_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4) # Usually not needed for final eval script

# %% 6. 定义训练函数 (Same as starter kit)
def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
):
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Fine-tune Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse

@torch.no_grad()
def valid_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats
    )

    for batch_idx, batch in iterator:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
    return avg_loss, rmse

# %% 7. 开始微调循环
optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1E-5) # Adjust weight decay if needed
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS - 1)
loss_fn = torch.nn.MSELoss() # Use MSE for regression

patience = EARLY_STOPPING_PATIENCE
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

print("Starting supervised fine-tuning on CCD task...")
for epoch in range(1, FINETUNE_EPOCHS + 1):
    print(f"Fine-tune Epoch {epoch}/{FINETUNE_EPOCHS}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, device
    )
    val_loss, val_rmse = valid_model(valid_loader, model, loss_fn, device) # Use valid_loader, not test_loader

    print(
        f"Train RMSE: {train_rmse:.6f}, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Val RMSE: {val_rmse:.6f}, "
        f"Average Val Loss: {val_loss:.6f}"
    )

    if val_rmse < best_rmse - min_delta:
        best_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
        print(f"  -> New best model found at epoch {epoch} with Val RMSE {val_rmse:.6f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best model state from epoch {best_epoch}.")

# %% 8. 保存微调后的完整模型
FINETUNED_MODEL_PATH = "weights_challenge_1.pt"
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved as '{FINETUNED_MODEL_PATH}'")

# %% 9. (可选) 在测试集上进行最终评估
# 注意：在竞赛中，测试集标签通常是不可见的。
# 评估脚本会加载你的模型并在其私有测试集上运行。
# 这里仅作为示例展示如何评估。
print("Evaluating on test set (if labels are available for validation)...")
model.eval()
final_test_loss, final_test_rmse = valid_model(test_loader, model, loss_fn, device)
print(f"Final Test RMSE (on provided test split): {final_test_rmse:.6f}")
print(f"Final Test Loss (on provided test split): {final_test_loss:.6f}")

print("Fine-tuning process completed.")
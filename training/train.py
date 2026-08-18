import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from dataset import get_train_val_split, CustomDataset
from cGAN import Generator, Discriminator
import time
from datetime import datetime, timedelta
import os
import csv
import json

SEED = 9
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def format_time(seconds):
    """Format seconds as hours:minutes:seconds."""
    return str(timedelta(seconds=int(seconds)))


# Unified timestamp (used for all outputs of this training run)
training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create the timestamped weight-save directory
weight_dir = f"weights_{training_timestamp}"
os.makedirs('logs', exist_ok=True)
os.makedirs('weight', exist_ok=True)  # kept for compatibility with legacy code (currently unused)
os.makedirs(weight_dir, exist_ok=True)

# Initialize model and optimizers
device = "cuda"
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Kaiming Uniform initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

generator.apply(init_weights)
discriminator.evaluator.apply(init_weights)
print("✅ 使用 Kaiming Uniform 初始化权重")

# Optimizers
optimizer_G = Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
optimizer_D = Adam(discriminator.evaluator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Data loading
X_train, y_train, X_val, y_val = get_train_val_split(
    "dataset.csv",
    y_mean_file="y_mean.npy",
    y_std_file="y_std.npy"
)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=12500, shuffle=False, pin_memory=True)

# Training configuration
epochs = 100000
best_val_loss = float('inf')

# Define training parameters (for saving)
training_config = {
    "SEED": SEED,
    "epochs": epochs,
    "dropout": 0.9,
    "batch_size_train": 50000,
    "batch_size_val": 12500,
    "generator": "3 resblock-256",
    "Evaluator": "1 resblock-256",
    "lr_G_initial": 1e-3,
    "lr_D_initial": 2e-4,
    "lr_decay_start_epoch": 50000,
    "lr_decay_total_epochs": 50000,
    "reg_loss_alpha_rampup_epochs": 20000,
    "generator_latent_dim": 2,
    "device": device,
    "optimizer_G_betas": [0.5, 0.999],
    "optimizer_D_betas": [0.5, 0.999],
    "loss_D_real": "hinge (relu(1 - real_score))",
    "loss_D_fake": "hinge (relu(1 + fake_score))",
    "loss_G_adv": "hinge (-mean(fake_score))",
    "regressor_loss": "MSELoss",
    "dataset": "dataset.csv",
    "y_mean_file": "y_mean.npy",
    "y_std_file": "y_std.npy",
    "weight_dir": weight_dir,
    "csv_log_file": f"logs/training_log_{training_timestamp}.csv",
    "config_file": f"logs/config_{training_timestamp}.json",
    "training_start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# CSV log file path
csv_log_file = f"logs/training_log_{training_timestamp}.csv"

# Write the CSV header
with open(csv_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Epoch", "Time", "ElapsedTime", "RemainingTime", "EpochTime",
        "D_Loss", "D_Loss_Real", "D_Loss_Fake", "Real_Score", "Fake_Score",
        "G_Loss", "G_Adv_Loss", "G_Reg_Loss", "Reg_Loss_Weight",
        "Val_D_Loss", "Val_D_Real", "Val_D_Fake",
        "Val_G_Loss", "Val_G_Adv", "Val_G_Reg",
        "LR_G", "LR_D"
    ])

# Save the training configuration
config_file = f"logs/config_{training_timestamp}.json"
with open(config_file, 'w') as f:
    json.dump(training_config, f, indent=4)

# Set models to training mode
discriminator.train()
generator.train()

total_start_time = time.time()
epoch_times = []

print(f"🚀 开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔖 本次训练时间戳: {training_timestamp}")
print(f"📁 权重保存目录: {weight_dir}")
print(f"📊 CSV 日志文件: {csv_log_file}")
print(f"📝 配置文件: {config_file}")
print(f"预计总轮次: {epochs}")
print(f"学习率衰减起始: 50000 轮")
print(f"回归损失权重 α 在前 20000 轮线性增长至 1.0")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    # Dynamically adjust the learning rate (linear decay to 0 starting at epoch 50000)
    current_epoch = epoch
    lr_G = 1e-3 * (1 - max(0, (current_epoch - 50000) / 50000))
    lr_D = 2e-4 * (1 - max(0, (current_epoch - 50000) / 50000))

    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr_G
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr_D

    # Train for one epoch
    for i, (real_thickness, real_lab) in enumerate(train_loader):
        real_thickness = real_thickness.to(device)
        real_lab = real_lab.to(device)

        # ---------------------
        # Update the discriminator (Evaluator)
        # ---------------------
        optimizer_D.zero_grad()

        z_E = torch.randn(real_lab.size(0), 2, device=device)
        fake_thickness_E = generator(z_E, real_lab)

        real_score, _ = discriminator(real_thickness)
        fake_score_E, _ = discriminator(fake_thickness_E)

        loss_real = torch.mean(torch.relu(1. - real_score))
        loss_fake = torch.mean(torch.relu(1. + fake_score_E))
        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        # Update the generator
        # ---------------------
        optimizer_G.zero_grad()

        z_G = torch.randn(real_lab.size(0), 2, device=device)
        fake_thickness_G = generator(z_G, real_lab)

        fake_score_G, _ = discriminator(fake_thickness_G)
        adv_loss = -torch.mean(fake_score_G)

        predicted_lab = discriminator.lab_regressor(fake_thickness_G)
        reg_loss = nn.MSELoss()(predicted_lab, real_lab)

        alpha = min(1.0, current_epoch / 20000)
        reg_loss_weighted = alpha * reg_loss

        total_loss = adv_loss + reg_loss_weighted

        total_loss.backward()
        optimizer_G.step()

    # Timing statistics
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = np.mean(epoch_times[-100:]) if len(epoch_times) >= 100 else np.mean(epoch_times)
    elapsed_time = time.time() - total_start_time
    remaining_time = avg_epoch_time * (epochs - epoch)

    # Validation (run every epoch)
    with torch.no_grad():
        val_loss_D = 0.0
        val_loss_real_total = 0.0
        val_loss_fake_total = 0.0

        val_loss_G = 0.0
        val_loss_G_adv_total = 0.0
        val_loss_G_reg_total = 0.0

        generator.eval()
        discriminator.eval()

        for val_thickness, val_lab in val_loader:
            val_thickness = val_thickness.to(device)
            val_lab = val_lab.to(device)

            # Discriminator validation
            real_score_val, _ = discriminator(val_thickness)
            z_E_val = torch.randn(val_lab.size(0), 2, device=device)
            fake_thickness_E_val = generator(z_E_val, val_lab)
            fake_score_E_val, _ = discriminator(fake_thickness_E_val)

            loss_real_val = torch.mean(torch.relu(1. - real_score_val))
            loss_fake_val = torch.mean(torch.relu(1. + fake_score_E_val))
            loss_D_val = loss_real_val + loss_fake_val

            val_loss_D += loss_D_val.item()
            val_loss_real_total += loss_real_val.item()
            val_loss_fake_total += loss_fake_val.item()

            # Generator validation
            z_G_val = torch.randn(val_lab.size(0), 2, device=device)
            fake_thickness_G_val = generator(z_G_val, val_lab)
            fake_score_G_val, _ = discriminator(fake_thickness_G_val)

            adv_loss_val = -torch.mean(fake_score_G_val)
            predicted_lab_val = discriminator.lab_regressor(fake_thickness_G_val)
            reg_loss_val = nn.MSELoss()(predicted_lab_val, val_lab)
            alpha_val = min(1.0, float(current_epoch) / 20000)
            reg_loss_val_weighted = alpha_val * reg_loss_val

            total_loss_G_val = adv_loss_val + reg_loss_val_weighted

            val_loss_G += total_loss_G_val.item()
            val_loss_G_adv_total += adv_loss_val.item()
            val_loss_G_reg_total += reg_loss_val_weighted.item()

        # Average validation losses
        val_loss_D_avg = val_loss_D / len(val_loader)
        val_loss_real_avg = val_loss_real_total / len(val_loader)
        val_loss_fake_avg = val_loss_fake_total / len(val_loader)

        val_loss_G_avg = val_loss_G / len(val_loader)
        val_loss_G_adv_avg = val_loss_G_adv_total / len(val_loader)
        val_loss_G_reg_avg = val_loss_G_reg_total / len(val_loader)

        generator.train()
        discriminator.train()

        # Write to the CSV log
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                datetime.now().strftime('%H:%M:%S'),
                format_time(elapsed_time),
                format_time(remaining_time),
                f"{epoch_time:.2f}",
                f"{loss_D.item():.6f}",
                f"{loss_real.item():.6f}",
                f"{loss_fake.item():.6f}",
                f"{real_score.mean().item():.6f}",
                f"{fake_score_E.mean().item():.6f}",
                f"{total_loss.item():.6f}",
                f"{adv_loss.item():.6f}",
                f"{reg_loss_weighted.item():.6f}",
                f"{alpha:.6f}",
                f"{val_loss_D_avg:.6f}",
                f"{val_loss_real_avg:.6f}",
                f"{val_loss_fake_avg:.6f}",
                f"{val_loss_G_avg:.6f}",
                f"{val_loss_G_adv_avg:.6f}",
                f"{val_loss_G_reg_avg:.6f}",
                f"{lr_G:.6f}",
                f"{lr_D:.6f}"
            ])

        # Print information
        current_time = datetime.now().strftime('%H:%M:%S')
        d_loss_info = (f"D: {loss_D.item():.4f} "
                       f"(real: {loss_real.item():.4f} "
                       f"fake: {loss_fake.item():.4f} "
                       f"scores: R{real_score.mean().item():.3f}/F{fake_score_E.mean().item():.3f})")
        g_loss_info = (f"G: {total_loss.item():.4f} "
                       f"(adv: {adv_loss.item():.4f} "
                       f"reg: {reg_loss_weighted.item():.4f} "
                       f"α: {alpha:.3f})")
        val_d_info = (f"Val_D: {val_loss_D_avg:.4f} "
                      f"(real: {val_loss_real_avg:.4f} "
                      f"fake: {val_loss_fake_avg:.4f})")
        val_g_info = (f"Val_G: {val_loss_G_avg:.4f} "
                      f"(adv: {val_loss_G_adv_avg:.4f} "
                      f"reg: {val_loss_G_reg_avg:.4f})")

        print(f"[{current_time}] Epoch {epoch:6d}/{epochs} | "
              f"Time: {format_time(elapsed_time)}<-{format_time(remaining_time)} | "
              f"Epoch: {epoch_time:.2f}s\n"
              f"    Train: {d_loss_info} | {g_loss_info}\n"
              f"    Val:   {val_d_info} | {val_g_info}\n"
              f"    LR: G{lr_G:.2e}/D{lr_D:.2e}")

        # Save the best model (timestamped)
        if val_loss_G_avg < best_val_loss:
            best_val_loss = val_loss_G_avg
            torch.save(generator.state_dict(), f'{weight_dir}/best_generator_{training_timestamp}.pth')
            torch.save(discriminator.state_dict(), f'{weight_dir}/best_discriminator_{training_timestamp}.pth')
            print(f"    💾 保存最佳模型 - Val G Loss: {best_val_loss:.4f}")

    # Periodically save checkpoints (timestamped)
    if epoch % 5000 == 0:
        torch.save(generator.state_dict(), f"{weight_dir}/generator_epoch{epoch}_{training_timestamp}.pth")
        torch.save(discriminator.state_dict(), f"{weight_dir}/discriminator_epoch{epoch}_{training_timestamp}.pth")
        print(f"    💾 保存检查点 - Epoch {epoch}")

# Training finished
total_time = time.time() - total_start_time
print(f"✅ 训练完成! 总时间: {format_time(total_time)}")
print(f"🏁 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 CSV 日志文件: {csv_log_file}")
print(f"📁 权重目录: {weight_dir}")

# Append the training summary (CSV is unsuitable for a trailing comment line, so write it to the config instead)
training_config["training_end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
training_config["total_training_time"] = format_time(total_time)
training_config["best_val_G_loss"] = best_val_loss

with open(config_file, 'w') as f:
    json.dump(training_config, f, indent=4)

print(f"✅ 训练配置已更新并保存至: {config_file}")
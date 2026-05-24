"""
最简 CNN Demo —— 用 PyTorch 从零实现
任务：MNIST 手写数字分类（10 类）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 卷积 → 激活 → 池化 基础块
# ============================================================
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


# ============================================================
# 2. 完整 CNN
# ============================================================
class CNN(nn.Module):
    """
    结构：
      ConvBlock(1 → 16)   → [B, 16, 14, 14]
      ConvBlock(16 → 32)  → [B, 32, 7,  7]
      ConvBlock(32 → 64)  → [B, 64, 3,  3]
      GlobalAvgPool       → [B, 64]
      Linear(64 → 10)     → [B, 10]
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 64, 1, 1] → [B, 64]
        return self.head(x)


# ============================================================
# 3. 训练 & 评估
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += logits.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 数据加载 ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值 & 标准差
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # ---- 模型 ----
    model = CNN(n_classes=10).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.2%}"
        )

    # ---- 单张推理演示 ----
    model.eval()
    x, y = next(iter(test_loader))
    x, y = x[:8].to(device), y[:8]
    with torch.no_grad():
        preds = model(x).argmax(1).cpu()
    print(f"\nSample predictions: {preds.tolist()}")
    print(f"Ground truth:       {y.tolist()}")


if __name__ == "__main__":
    main()

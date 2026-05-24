"""
最简 GNN Demo —— 用 PyTorch 从零实现 GCN（图卷积网络）
任务：Zachary's Karate Club 空手道俱乐部 —— 节点分类（34 人分属 2 个派系）

图结构：34 个节点（会员），78 条边（社交关系）
目标：根据社交网络预测每个人属于哪个派系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Karate Club 图数据（硬编码，无需外部依赖）
# ============================================================
def load_karate_club():
    """
    Zachary's Karate Club —— 图神经网络领域的 "MNIST"
    34 个节点，78 条无向边，2 类标签

    返回: 邻接矩阵 A [34, 34], 特征 X [34, 34] (one-hot 单位阵), 标签 y [34]
    """
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
        (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
        (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
        (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
        (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
        (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
        (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
        (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
        (30,32),(30,33),(31,32),(31,33),(32,33),
    ]
    n_nodes = 34
    A = torch.zeros(n_nodes, n_nodes)
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0  # 无向边
    X = torch.eye(n_nodes)       # 特征 = one-hot 身份（GNN 经典初始特征）
    y = torch.tensor([
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,
        1,1,1,1,1,1,1,1,1,1,1,1
    ])
    return A, X, y


# ============================================================
# 2. GCN 层（从零实现）
# ============================================================
class GCNLayer(nn.Module):
    """
    H' = σ(D̃^{-1/2} Ã D̃^{-1/2} · H · W)

    核心步骤：
      1. Ã = A + I           （加自环，让节点也聚合自己的特征）
      2. D̃ = Σ Ã 按行求和     （度矩阵）
      3. 归一化邻接矩阵 = D̃^{-1/2} · Ã · D̃^{-1/2}
      4. 线性变换 H·W 后传播
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 1) 加自环
        A_tilde = A + torch.eye(A.size(0), device=A.device)

        # 2) 计算 D̃^{-1/2}
        deg = A_tilde.sum(dim=1)            # [N] 每个节点的度
        deg_inv_sqrt = deg.pow(-0.5)         # [N] 度的 -1/2 次方
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # 3) 对称归一化: D^{-1/2} · A_tilde · D^{-1/2}
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt  # [N, N]

        # 4) 图卷积: 聚合邻居 → 线性变换 → 激活
        support = X @ self.W                         # [N, out_dim]
        out = A_norm @ support + self.b               # [N, out_dim]
        return out


# ============================================================
# 3. 两层 GCN 模型
# ============================================================
class GCN(nn.Module):
    """
    GCNLayer(in → hidden) → ReLU → Dropout → GCNLayer(hidden → n_classes)
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(X, A))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, A)


# ============================================================
# 4. 训练
# ============================================================
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    return pred.eq(y[mask]).float().mean().item()


def main():
    A, X, y = load_karate_club()
    N = A.size(0)

    # 半监督设定：每类只用 2 个节点训练，其余测试
    torch.manual_seed(42)
    train_mask = torch.zeros(N, dtype=torch.bool)
    for c in [0, 1]:
        idx = (y == c).nonzero(as_tuple=True)[0]
        picked = idx[torch.randperm(len(idx))[:2]]
        train_mask[picked] = True
    test_mask = ~train_mask

    print(f"Nodes: {N}, Edges: {int(A.sum()/2)}")
    print(f"Classes: 2, Train nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")

    model = GCN(in_dim=N, hidden_dim=16, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        logits = model(X, A)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 40 == 0:
            acc = accuracy(logits, y, test_mask)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Acc: {acc:.2%}")

    # ---- 最终评估 ----
    model.eval()
    with torch.no_grad():
        logits = model(X, A)
        preds = logits.argmax(dim=1)
        final_acc = preds[test_mask].eq(y[test_mask]).float().mean().item()

    print(f"\nFinal Test Accuracy: {final_acc:.2%}")
    print(f"Predictions: {preds.tolist()}")
    print(f"GroundTruth: {y.tolist()}")
    print(f"Correct: {(preds == y).sum().item()}/34")


if __name__ == "__main__":
    main()

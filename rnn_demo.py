"""
最简 RNN Demo —— 用 PyTorch 从零实现
任务：字符级语言模型（Char-RNN）—— 给定前文，预测下一个字符
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 从零实现 RNN 单元（展示循环公式）
# ============================================================
class RNNCell(nn.Module):
    """
    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)

    这是最基础的 RNN 单元，每一步：
      1. 当前输入 x_t 和上一时刻隐藏状态 h_{t-1} 分别乘以权重
      2. 相加后过 tanh 激活，得到新的隐藏状态 h_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        # x_t: [B, input_size], h_prev: [B, hidden_size]
        return torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))


# ============================================================
# 2. 完整 RNN 模型
# ============================================================
class CharRNN(nn.Module):
    """
    Embedding → RNN（逐时间步循环）→ Linear → 预测下一个字符

    输入: 一串字符索引  [B, seq_len]
    输出: 每个位置的下一字符预测  [B, seq_len, vocab_size]
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell = RNNCell(embed_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """
        x: [B, seq_len]  字符 ID 序列
        返回: logits [B, seq_len, vocab_size], 最终隐藏状态 h
        """
        B, T = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)

        emb = self.embed(x)  # [B, T, embed_dim]

        outputs = []
        for t in range(T):
            h = self.rnn_cell(emb[:, t, :], h)  # [B, hidden_size]
            outputs.append(self.output(h))       # [B, vocab_size]

        logits = torch.stack(outputs, dim=1)     # [B, T, vocab_size]
        return logits, h

    def generate(self, start_chars: list[int], length: int, temperature: float = 0.8):
        """从起始字符自回归生成文本"""
        generated = list(start_chars)
        x = torch.tensor([start_chars])
        h = torch.zeros(1, self.hidden_size)

        for _ in range(length):
            emb = self.embed(x[:, -1:])  # 只取最后一个字符 [1, 1, embed_dim]
            for t in range(emb.size(1)):
                h = self.rnn_cell(emb[:, t, :], h)
                logits = self.output(h)
            # 温度采样
            probs = F.softmax(logits.squeeze(0) / temperature, dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            generated.append(next_char)
            x = torch.cat([x, torch.tensor([[next_char]])], dim=1)

        return generated


# ============================================================
# 3. 数据准备
# ============================================================
def build_dataset(text: str, seq_len: int, batch_size: int):
    """将文本转为字符级训练数据"""
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}

    encoded = torch.tensor([c2i[c] for c in text], dtype=torch.long)

    # 构造 (输入序列, 目标序列) 对，目标 = 输入右移一位
    num_seqs = (len(encoded) - 1) // seq_len
    inputs = encoded[: num_seqs * seq_len].view(-1, seq_len)
    targets = encoded[1 : num_seqs * seq_len + 1].view(-1, seq_len)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, vocab_size, c2i, i2c


# ============================================================
# 4. 训练
# ============================================================
def main():
    # 莎士比亚十四行诗片段（公开域文本）
    text = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date:
    Sometime too hot the eye of heaven shines,
    And often is his gold complexion dimmed,
    And every fair from fair sometime declines,
    By chance, or nature's changing course untrimmed:
    But thy eternal summer shall not fade,
    Nor lose possession of that fair thou ow'st,
    Nor shall death brag thou wand'rest in his shade,
    When in eternal lines to time thou grow'st:
       So long as men can breathe or eyes can see,
       So long lives this, and this gives life to thee.
    """
    text = text.lower().replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")

    SEQ_LEN = 30
    BATCH_SIZE = 64
    EMBED_DIM = 64
    HIDDEN_SIZE = 128
    EPOCHS = 150
    LR = 3e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, vocab_size, c2i, i2c = build_dataset(text, SEQ_LEN, BATCH_SIZE)
    print(f"Vocab size: {vocab_size}, Chars: {''.join(sorted(c2i.keys()))}")

    model = CharRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)  # [B, seq_len, vocab_size]
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        if epoch == 1 or epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss / n:.4f}")

    # ---- 文本生成 ----
    model.eval()
    with torch.no_grad():
        prefix = "shall i"
        start = [c2i[c] for c in prefix]
        gen_ids = model.generate(start, length=200, temperature=0.7)
        generated = "".join(i2c[i] for i in gen_ids)
        print(f"\n--- Generation (seed: '{prefix}') ---")
        print(generated)


if __name__ == "__main__":
    main()

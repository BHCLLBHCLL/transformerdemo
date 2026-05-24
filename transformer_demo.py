"""
最简 Transformer Demo —— 用 PyTorch 从零实现
任务：序列复刻（Seq2Seq Copy Task）
输入一串数字 → 输出相同的一串数字
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 位置编码（Positional Encoding）
# ============================================================
class PositionalEncoding(nn.Module):
    """用正弦/余弦函数给每个位置赋予唯一的向量表示"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位用 cos
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


# ============================================================
# 2. 多头自注意力（Multi-Head Self-Attention）
# ============================================================
class MultiHeadAttention(nn.Module):
    """Q/K/V 线性投影 → 拆成多个头 → 各自做 Scaled Dot-Product Attention → 拼接"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.size(0)

        # 线性投影并拆成多头: [B, seq, d_model] → [B, n_heads, seq, d_k]
        def project_and_split(x: torch.Tensor, w: nn.Linear):
            return w(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(query, self.W_q)
        K = project_and_split(key, self.W_k)
        V = project_and_split(value, self.W_v)

        # Scaled Dot-Product Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_heads, seq_q, seq_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = attn_weights @ V  # [B, n_heads, seq_q, d_k]

        # 合并多头: [B, n_heads, seq, d_k] → [B, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.W_o(context)


# ============================================================
# 3. 前馈网络（Position-wise Feed-Forward）
# ============================================================
class FeedForward(nn.Module):
    """两层全连接，中间用 ReLU —— 对每个位置独立应用"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4. Encoder 层
# ============================================================
class EncoderLayer(nn.Module):
    """Self-Attention → Add&Norm → FeedForward → Add&Norm"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-Attention + 残差连接 + LayerNorm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-Forward + 残差连接 + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ============================================================
# 5. Decoder 层
# ============================================================
class DecoderLayer(nn.Module):
    """Masked Self-Attn → Cross-Attn → FeedForward，各自带 Add&Norm"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) 带因果遮罩的自注意力（只看当前及之前的位置）
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # 2) 交叉注意力（Q 来自 decoder，K/V 来自 encoder）
        cross_out = self.cross_attn(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        # 3) 前馈网络
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# ============================================================
# 6. 完整 Transformer
# ============================================================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.dropout(self.src_embed(src) * math.sqrt(self.d_model))
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.dropout(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.pos_enc(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_out = self.encode(src, src_mask)
        return self.decode(tgt, encoder_out, src_mask, tgt_mask)


# ============================================================
# 7. 工具函数
# ============================================================
def generate_causal_mask(sz: int) -> torch.Tensor:
    """下三角矩阵 —— 确保位置 i 只能看到 ≤i 的位置"""
    return torch.tril(torch.ones(sz, sz)).unsqueeze(0).unsqueeze(0)  # [1, 1, sz, sz]


def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """屏蔽 padding 位置"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]


# ============================================================
# 8. 训练 Demo —— 序列复刻任务
# ============================================================
def generate_copy_batch(batch_size: int, seq_len: int, vocab_size: int):
    """
    生成一批数据：
      src:  [BOS, 3, 7, 2, EOS, PAD, PAD]
      tgt:  [BOS, 3, 7, 2, EOS]
    输入序列随机，输出序列相同（复刻任务）
    """
    # 随机序列（跳过 PAD=0, BOS=1, EOS=2）
    data = torch.randint(3, vocab_size, (batch_size, seq_len))
    # src: [BOS] + 序列 + [EOS] + padding
    bos = torch.full((batch_size, 1), BOS_IDX)
    eos = torch.full((batch_size, 1), EOS_IDX)
    src = torch.cat([bos, data, eos], dim=1)  # [B, seq_len+2]
    # tgt_input:  [BOS] + 序列（去掉最后一个位置，Decoder 输入）
    tgt_input = torch.cat([bos, data], dim=1)  # [B, seq_len+1]
    # tgt_output: 序列 + [EOS]（Decoder 预测目标）
    tgt_output = torch.cat([data, eos], dim=1)  # [B, seq_len+1]
    return src, tgt_input, tgt_output


def train():
    # 超参数
    VOCAB_SIZE = 20        # 词表大小（含 PAD/BOS/EOS）
    D_MODEL = 128          # 模型维度
    N_HEADS = 4            # 注意力头数
    N_LAYERS = 3           # Encoder/Decoder 层数
    D_FF = 512             # 前馈网络隐藏层维度
    BATCH_SIZE = 64
    SEQ_LEN = 8            # 序列长度
    EPOCHS = 200
    LR = 1e-3
    DATASET_SIZE = 2000    # 预生成固定训练样本数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 预生成固定训练集，每次 epoch 遍历相同数据，加速收敛
    print("Generating training dataset...")
    dataset = [generate_copy_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE) for _ in range(DATASET_SIZE // BATCH_SIZE)]

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for src, tgt_in, tgt_out in dataset:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            tgt_mask = generate_causal_mask(tgt_in.size(1)).to(device)

            logits = model(src, tgt_in, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss / len(dataset):.4f}")

    # ---- 推理测试 ----
    model.eval()
    with torch.no_grad():
        src, _, tgt_out = generate_copy_batch(1, SEQ_LEN, VOCAB_SIZE)
        src = src.to(device)

        encoder_out = model.encode(src)
        generated = [BOS_IDX]
        for _ in range(SEQ_LEN + 1):
            tgt = torch.tensor([generated]).to(device)
            tgt_mask = generate_causal_mask(len(generated)).to(device)
            logits = model.decode(tgt, encoder_out, tgt_mask=tgt_mask)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            if next_token == EOS_IDX:
                break

        print(f"\nInput:     {src[0].tolist()}")
        print(f"Expected:  {tgt_out[0].tolist()}")
        print(f"Predicted: {generated}")


if __name__ == "__main__":
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    train()

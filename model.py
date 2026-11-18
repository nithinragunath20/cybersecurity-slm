
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[:head_dim // 2] / head_dim
        )
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    _, _, seq_len, head_dim = x.shape
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.float()).to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = d_in // num_heads
        self.d_out = num_heads * self.head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.scaling = self.head_dim ** -0.5

    def forward(self, x, mask, cos, sin):
        b, seq_len, _ = x.shape
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        q = q * self.scaling

        scores = q @ k.transpose(2, 3)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, seq_len, self.d_out)
        return self.out_proj(context)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, d_ff):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_groups)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask, cos, sin):
        attn_out = self.attention(self.norm1(x), mask, cos, sin)
        x = x + attn_out
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_groups,
                 n_layers, d_ff, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_groups, d_ff)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

        head_dim = d_model // n_heads
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            context_length=max_seq_len,
            dtype=torch.float32
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence exceeds maximum context length")

        x = self.tok_emb(input_ids) * (self.d_model ** 0.5)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        cos = self.cos[:seq_len].to(x.device)
        sin = self.sin[:seq_len].to(x.device)

        for block in self.blocks:
            x = block(x, mask, cos, sin)

        x = self.final_norm(x)
        logits = self.out_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonalize(M):
    # six step Newton-Schulz by @YouJiacheng
    # coefficients from: https://twitter.com/YouJiacheng/status/1893704552689303901
    # found by optimization: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae
    # the idea of stability loss was from @leloykun

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / torch.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = torch.eye(A.shape[0], device=M.device, dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


# Atomic
class Linear(nn.Linear):
    def __init__(self, fanin, fanout, target_norm=1.0):
        super().__init__(fanin, fanout, bias=False)
        self.fanin = fanin
        self.fanout = fanout
        self.register_buffer('target_norm', torch.tensor(target_norm, dtype=torch.float))
        
        # Initialize with orthogonal weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            weight = torch.randn(self.fanout, self.fanin)
            weight = orthogonalize(weight) * torch.sqrt(torch.tensor(self.fanout / self.fanin))
            self.weight.copy_(weight)
    
    def project_weights(self):
        """Project weights to the constraint manifold"""
        with torch.no_grad():
            weight = orthogonalize(self.weight) * torch.sqrt(torch.tensor(self.fanout / self.fanin))
            self.weight.copy_(weight)
    
    def dualize_gradients(self):
        """Apply dualization to gradients"""
        if self.weight.grad is not None:
            with torch.no_grad():
                grad = self.weight.grad
                d_weight = orthogonalize(grad) * torch.sqrt(torch.tensor(self.fanout / self.fanin)) * self.target_norm
                self.weight.grad.copy_(d_weight)

    @staticmethod
    def from_modula(m, w=None):
        """Convert from modula.atom.Linear"""
        with torch.no_grad():
            linear = Linear(m.fanin, m.fanout, getattr(m, 'target_norm', 1.0))
            if w is not None:
                linear.weight.copy_(w)
        return linear

    def __repr__(self):
        return f"Linear(fanin={self.fanin}, fanout={self.fanout}, target_norm={self.target_norm})"

class Embed(nn.Embedding):
    def __init__(self, num_embed, d_embed, target_norm, padding_idx=None):
        super().__init__(num_embed, d_embed, padding_idx=padding_idx)
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.sensitivity = 1
        self.register_buffer('target_norm', torch.tensor(target_norm, dtype=torch.float))
        
        # Initialize with normalized weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            weight = torch.randn(self.num_embed, self.d_embed)
            weight = F.normalize(weight, p=2, dim=1) * torch.sqrt(torch.tensor(self.d_embed, dtype=torch.float))
            self.weight.copy_(weight)
    
    def project_weights(self):
        """Project weights to the constraint manifold"""
        with torch.no_grad():
            weight = F.normalize(self.weight, p=2, dim=1) * torch.sqrt(torch.tensor(self.d_embed, dtype=torch.float))
            self.weight.copy_(weight)
    
    def dualize_gradients(self):
        """Apply dualization to gradients"""
        if self.weight.grad is not None:
            with torch.no_grad():
                grad = self.weight.grad
                # Normalize each embedding vector's gradient
                grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
                # Handle zero gradients to avoid NaN
                grad_norm = torch.where(grad_norm == 0, torch.ones_like(grad_norm), grad_norm)
                d_weight = grad / grad_norm * torch.sqrt(torch.tensor(self.d_embed, dtype=torch.float)) * self.target_norm
                # Handle any remaining NaN values
                d_weight = torch.nan_to_num(d_weight)
                self.weight.grad.copy_(d_weight)

    @staticmethod
    def from_modula(m, w=None):
        """Convert from modula.atom.Embed"""
        with torch.no_grad():
            embed = Embed(m.num_embed, m.d_embed, getattr(m, 'target_norm', 1.0))
            if w is not None:
                embed.weight.copy_(w)
        return embed

    def __repr__(self):
        return f"Embed(num_embed={self.num_embed}, d_embed={self.d_embed}, target_norm={self.target_norm})"

# Bond

class ReLU(nn.ReLU):
    pass # No changes needed, inherits from nn.ReLU

class GeLU(nn.GELU):
    def __init__(self):
        super().__init__(approximate='tanh')

    def forward(self, x):
        return super().forward(x) / 1.1289  # 1.1289 is the max derivative of gelu(x)


class SplitIntoHeads(nn.Module):
    """Reshapes an input to have heads.

    Input shape: (batch_size, sequence_length, embed_dim) 
    Output shape: (batch_size, num_heads, sequence_length, head_size)
    
    Adapted from Karpathy's nanoGPT.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
    
    def forward(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)

    @staticmethod
    def from_modula(m):
        """Convert from modula.bond.SplitIntoHeads"""
        return SplitIntoHeads(m.num_heads)


class MergeHeads(nn.Module):
    """Inverse of SplitIntoHeads."""
    def forward(self, x):
        B, num_heads, T, head_dim = x.shape
        return x.transpose(1, 2).reshape(B, T, num_heads * head_dim)


class AttentionQK(nn.Module):
    """Computes the query and key matrix multiplication in attention."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        q, k = x  # both shape [batch, n_heads, seq_len, d_query]
        scale = 1 / q.shape[-1]
        scores = q @ k.transpose(-2, -1) * scale
        return scores  # shape [batch, n_heads, seq_len, seq_len]


class CausalMask(nn.Module):
    """Masks the upper triangular part of the attention scores."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        scores = x
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device))
        return torch.where(mask, scores, torch.tensor(-float('inf'), device=scores.device))


class Softmax(nn.Module):
    """Softmax with a sharpness parameter."""
    def __init__(self, scale):
        super().__init__()
        self.sensitivity = scale
    
    def forward(self, x):
        return F.softmax(self.sensitivity * x, dim=-1)

    @staticmethod
    def from_modula(m):
        """Convert from modula.bond.Softmax"""
        return Softmax(m.sensitivity)


class ApplyAttentionScores(nn.Module):
    """Computes attention values from the scores."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        v, scores = x
        return scores @ v

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.sensitivity = scale

    def forward(self, x):
        v, (q, k) = x
        scale = self.sensitivity / q.shape[-1]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        return out

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., d:]     # Second half first
    x2 = x[..., :d]     # First half second
    y1 = cos * x1 + sin * x2
    y2 = -sin * x1 + cos * x2
    return torch.cat([y1, y2], 3)

class Rope(nn.Module):
    def __init__(self, d_head, base=10000):
        super().__init__()
        self.rope_dim = d_head // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        q, k = x
        seq_len = q.shape[2]  # Assuming shape [batch, n_heads, seq_len, d_head]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(q.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        
        # Shape: [1, 1, seq_len, rope_dim] to match [batch, n_heads, seq_len, d_head]
        cos = self.cos_cached[None, None, :, :]
        sin = self.sin_cached[None, None, :, :]
        
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    @staticmethod
    def from_modula(m):
        """Convert from modula.bond.Rope"""
        return Rope(2 * m.rope_dim, base=m.base)

# Abstract

class Identity(nn.Module):
    def forward(self, x):
        return x

class Add(nn.Module):
    def forward(self, x):
        a, b = x
        return a + b

class Mul(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.sensitivity = scalar

    def forward(self, x):
        return x * self.sensitivity

    @staticmethod
    def from_modula(m):
        """Convert from modula.abstract.Mul"""
        return Mul(m.sensitivity)

# Composite
class Parallel(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(*modules)

    def __getitem__(self, idx):
        return self.module_list[idx]

    def forward(self, inputs):
        return [module(inputs) for module in self.module_list]

def FlashAttention(num_heads, d_embed, d_query, d_value, attention_scale):
    return nn.Sequential(
        Parallel([
            nn.Sequential( # V
                Linear(num_heads * d_value, d_embed),
                SplitIntoHeads(num_heads=num_heads)
            ),
            nn.Sequential(
                Parallel([
                    nn.Sequential( # Q
                        Linear(num_heads * d_query, d_embed),
                        SplitIntoHeads(num_heads=num_heads)
                    ),
                    nn.Sequential( # K
                        Linear(num_heads * d_query, d_embed),
                        SplitIntoHeads(num_heads=num_heads)
                    )
                ]),
                Rope(d_query),
            )
        ]),
        ScaledDotProductAttention(attention_scale),
        Mul(1/3), 
        MergeHeads(),
        Linear(d_embed, num_heads * d_value)
    )

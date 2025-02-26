import jax
import jax.numpy as jnp

from modula.abstract import Bond

class ReLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.maximum(0, x)


class GeLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jax.nn.gelu(x) / 1.1289  # 1.1289 is the max derivative of gelu(x)

class SplitIntoHeads(Bond):
    """Reshapes an input to have heads.

    Input shape: (batch_size, sequence_length, embed_dim) 
    Output shape: (batch_size, num_heads, sequence_length, head_size)
    
    Adapted from Karpathy's nanoGPT.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
        self.num_heads = num_heads
    
    def forward(self, x, w):
        B, T, D = x.shape
        return jnp.reshape(x, (B, T, self.num_heads, D // self.num_heads)).transpose(0, 2, 1, 3)

class MergeHeads(Bond):
    """Inverse of SplitIntoHeads."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        B, num_heads, T, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, num_heads * head_dim)

class AttentionQK(Bond):
    """Computes the query and key matrix multiplication in attention."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # what is this sensitivity?
    
    def forward(self, x, w):
        q, k = x  # both shape [batch, n_heads, seq_len, d_query]
        scale = 1 / q.shape[-1]
        scores = q @ k.transpose(0, 1, 3, 2) * scale
        return scores  # shape [batch, n_heads, seq_len, seq_len]

class CausalMask(Bond):
    """Masks the upper triangular part of the attention scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # what is this sensitivity?
    
    def forward(self, x, w):
        scores = x
        mask = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
        return jnp.where(mask, scores, -jnp.inf)

class Softmax(Bond):
    """Softmax with a sharpness parameter."""
    def __init__(self, scale):
        super().__init__()
        self.smooth = True
        self.sensitivity = scale
    
    def forward(self, x, w):
        return jax.nn.softmax(self.sensitivity * x, axis=-1)

class ApplyAttentionScores(Bond):
    """Computes attention values from the scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        v, scores = x
        return scores @ v

class Rope(Bond):
    """Rotates queries and keys by relative context window distance."""
    def __init__(self, d_head, base=10000):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # rope is an orthogonal transformation

        self.rope_dim = d_head // 2
        self.inverse_frequencies = 1/base**(jnp.arange(self.rope_dim) / self.rope_dim)
        self.seq_len_cached = None
        self.sin_cached = None
        self.cos_cached = None
    
    def get_cached(self, seq_len):
        if self.seq_len_cached != seq_len:
            self.seq_len_cached = seq_len
            distance = jnp.arange(seq_len)
            freqs = jnp.outer(distance, self.inverse_frequencies)  # shape [seq_len, rope_dim]
            self.cos_cached = jnp.expand_dims(jnp.cos(freqs), (0, 1))  # shape [seq_len, rope_dim]
            self.sin_cached = jnp.expand_dims(jnp.sin(freqs), (0, 1))  # shape [seq_len, rope_dim]
        return self.sin_cached, self.cos_cached
    
    def rotate(self, x):
        batch, n_heads, seq_len, d_head = x.shape
        assert self.rope_dim == d_head // 2

        x1 = x[..., self.rope_dim:]  # shape [batch, n_heads, seq_len, rope_dim]
        x2 = x[..., :self.rope_dim]  # shape [batch, n_heads, seq_len, rope_dim]

        cos, sin = self.get_cached(seq_len)
        y1 =  cos * x1 + sin * x2
        y2 = -sin * x1 + cos * x2

        return jnp.concat([y1, y2], axis=-1)
    
    def forward(self, x, w):
        q, k = x
        return self.rotate(q), self.rotate(k)

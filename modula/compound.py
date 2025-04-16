import jax.numpy as jnp

from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width) @ ReLU()
    for _ in range(depth-2):
        m = m @ Linear(width, width) @ ReLU()
    return m @ Linear(width, input_dim)

def Attention(num_heads, d_embed, d_query, d_value, softmax_scale, causal, posemb="rope", bias=False):
    """Multi-head attention"""
    Q, K, V = Linear(num_heads * d_query, d_embed), Linear(num_heads * d_query, d_embed), Linear(num_heads * d_value, d_embed)
    Q = SplitIntoHeads(num_heads) @ (Q + Bias(num_heads * d_query) if bias else Q)
    K = SplitIntoHeads(num_heads) @ (K + Bias(num_heads * d_query) if bias else K)
    V = SplitIntoHeads(num_heads) @ (V + Bias(num_heads * d_value) if bias else V)
    W = Linear(d_embed, num_heads * d_value)
    W = (W + Bias(d_embed) if bias else W) @ MergeHeads()
    QK = (Q, K)
    if posemb == "rope":
        QK = Rope(d_query) @ QK
    attn = AttentionQK() @ QK
    if causal:
        attn = CausalMask() @ attn
    AttentionScores = Softmax(softmax_scale) @ attn
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, causal=True)
    mlp = Linear(d_embed, 4*d_embed) @ GeLU() @ Linear(4*d_embed, d_embed)
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed)

    return out @ blocks @ embed

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]

def ViT(num_classes, image_size=(28, 28), patch_size=(7, 7), num_heads=4, d_embed=32, d_query=8, d_value=8, num_blocks=4, blocks_mass=5, attention_scale=1.0, final_scale=1.0, LN=True, bias=True, scale=True):
    i1, i2 = image_size
    p1, p2 = patch_size
    h, w = i1 // p1, i2 // p2
    patchify = Linear(d_embed, p1 * p2) @ Patchify(patch_size)
    if bias:
        patchify = patchify + Bias(d_embed)
    posemb = Constant(posemb_sincos_2d(h, w, d_embed))

    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, causal=False, posemb="none", bias=bias)
    mlp = (Linear(d_embed, 4*d_embed) + Bias(d_embed) if bias else Linear(d_embed, 4*d_embed)) @ GeLU() @ (Linear(4*d_embed, d_embed) + Bias(4*d_embed) if bias else Linear(4*d_embed, d_embed))
    if LN:
        ln = LayerNorm()
        if bias and scale:
            ln = (Scale(d_embed) + Bias(d_embed)) @ ln
        elif bias:
            ln = ln + Bias(d_embed)
        elif scale:
            ln = Scale(d_embed) @ ln
        att = att @ ln
        mlp = mlp @ ln
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    gap = Mean(axis=1, size=h * w)
    out = final_scale * (Linear(num_classes, d_embed) + Bias(num_classes) if bias else Linear(num_classes, d_embed))

    ret = blocks @ (patchify + posemb)
    if LN:  # Final LN
        ret = ln @ ret
    return out @ gap @ ret

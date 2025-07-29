import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp

import modula.atom
import modula.bond

from torch_modules import (orthogonalize, Linear, Embed, ReLU, GeLU, SplitIntoHeads, MergeHeads,
                           AttentionQK, CausalMask, Softmax, ApplyAttentionScores,
                           Rope)

def test_orthogonalize():
    """Test PyTorch implementation against JAX implementation"""
    print("Testing PyTorch orthogonalize against JAX modula.atom.orthogonalize...")
    
    # Test cases with different shapes
    test_cases = [
        (4, 4),    # Square
        (6, 3),    # Tall
        (3, 6),    # Wide
        (5, 5),    # Another square
        (8, 2),    # Very tall
        (2, 8),    # Very wide
    ]
    
    for i, (rows, cols) in enumerate(test_cases):
        print(f"\nTest {i+1}: Matrix shape ({rows}, {cols})")
        
        # Create random matrix
        np_matrix = np.random.randn(rows, cols).astype(np.float32)
        
        # Convert to respective frameworks
        torch_matrix = torch.from_numpy(np_matrix)
        jax_matrix = jnp.array(np_matrix)
        
        # Apply orthogonalization
        torch_result = orthogonalize(torch_matrix)
        jax_result = modula.atom.orthogonalize(jax_matrix)
        
        # Convert back to numpy for comparison
        torch_result_np = torch_result.detach().numpy()
        jax_result_np = np.array(jax_result)
        
        # Check if results are close
        max_diff = np.max(np.abs(torch_result_np - jax_result_np))
        relative_error = max_diff / (np.max(np.abs(jax_result_np)) + 1e-8)
        
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Relative error: {relative_error:.2e}")
        print(f"  Results match (abs diff < 1e-6): {max_diff < 1e-6}")
        
        # Also check orthogonality of PyTorch result
        if torch_result.shape[1] <= torch_result.shape[0]:
            # Tall or square: check Q^T @ Q = I
            product = torch_result.T @ torch_result
            identity = torch.eye(torch_result.shape[1])
            ortho_error = torch.max(torch.abs(product - identity)).item()
        else:
            # Wide: check Q @ Q^T = I
            product = torch_result @ torch_result.T
            identity = torch.eye(torch_result.shape[0])
            ortho_error = torch.max(torch.abs(product - identity)).item()
        
        print(f"  PyTorch result orthogonality error: {ortho_error:.2e}")
        print(f"  Is orthogonal (error < 1e-5): {ortho_error < 1e-5}")
    
    # Test with different dtypes
    print("\nTest dtype consistency:")
    np_matrix = np.random.randn(4, 3).astype(np.float64)
    
    # torch_matrix_f32 = torch.from_numpy(np_matrix.astype(np.float32))
    torch_matrix_f64 = torch.from_numpy(np_matrix.astype(np.float64))
    jax_matrix = jnp.array(np_matrix)
    
    # torch_result_f32 = orthogonalize(torch_matrix_f32)
    torch_result_f64 = orthogonalize(torch_matrix_f64)
    jax_result = modula.atom.orthogonalize(jax_matrix)
    
    # Compare f64 results (should be most accurate)
    diff_f64 = np.max(np.abs(torch_result_f64.numpy() - np.array(jax_result)))
    print(f"  Float64 max difference: {diff_f64:.2e}")
    print(f"  Float64 results match: {diff_f64 < 1e-10}")
    
    # Test gradient compatibility (PyTorch-specific feature)
    print("\nTest gradient compatibility:")
    torch_matrix = torch.randn(4, 3, requires_grad=True)
    torch_result = orthogonalize(torch_matrix)
    loss = torch.sum(torch_result ** 2)
    loss.backward()
    
    has_grad = torch_matrix.grad is not None
    grad_finite = torch.all(torch.isfinite(torch_matrix.grad)) if has_grad else False
    print(f"  Gradients computed: {has_grad}")
    print(f"  Gradients finite: {grad_finite}")
    
    print("\nAll tests completed!")

def test_linear_modules():
    """Test that JAX and PyTorch Linear modules produce identical results"""
    # Set parameters
    fanin, fanout = 8, 4
    batch_size = 2
    
    # Create JAX module and initialize weights
    jax_linear = modula.atom.Linear(fanout, fanin)  # JAX module
    key = jax.random.PRNGKey(42)
    jax_weights = jax_linear.initialize(key)
    
    # Create PyTorch module and load the same weights
    w = torch.tensor(jax_weights[0])  # Convert JAX weights to PyTorch tensor
    torch_linear = Linear.from_modula(jax_linear, w)
    
    # Create identical input
    np.random.seed(42)
    input_np = np.random.randn(batch_size, fanin).astype(np.float32)
    
    jax_input = jnp.array(input_np)
    torch_input = torch.from_numpy(input_np)
    
    # Forward pass
    jax_output = jax_linear.forward(jax_input, jax_weights)
    torch_output = torch_linear(torch_input)
    
    # Compare outputs
    torch_output_np = torch_output.detach().numpy()
    jax_output_np = np.array(jax_output)
    
    print(f"JAX output shape: {jax_output_np.shape}")
    print(f"PyTorch output shape: {torch_output_np.shape}")
    print(f"Max absolute difference: {np.max(np.abs(jax_output_np - torch_output_np))}")
    
    np.testing.assert_allclose(jax_output_np, torch_output_np, rtol=1e-6, atol=1e-6)
    print("✓ Linear modules produce identical results")

def test_embed_modules():
    """Test that JAX and PyTorch Embed modules produce identical results"""
    # Set parameters
    num_embed, d_embed = 10, 6
    batch_size = 3
    seq_len = 4
    
    # Create JAX module and initialize weights
    jax_embed = modula.atom.Embed(d_embed, num_embed)  # JAX module
    key = jax.random.PRNGKey(123)
    jax_weights = jax_embed.initialize(key)
    
    # Create PyTorch module and load the same weights
    w = torch.tensor(jax_weights[0])  # Convert JAX weights to PyTorch tensor
    torch_embed = Embed.from_modula(jax_embed, w)
    
    # Create identical input (indices)
    np.random.seed(123)
    input_indices = np.random.randint(0, num_embed, size=(batch_size, seq_len))
    
    jax_input = jnp.array(input_indices)
    torch_input = torch.from_numpy(input_indices)
    
    # Forward pass
    jax_output = jax_embed.forward(jax_input, jax_weights)
    torch_output = torch_embed(torch_input)
    
    # Compare outputs
    torch_output_np = torch_output.detach().numpy()
    jax_output_np = np.array(jax_output)
    
    print(f"JAX output shape: {jax_output_np.shape}")
    print(f"PyTorch output shape: {torch_output_np.shape}")
    print(f"Max absolute difference: {np.max(np.abs(jax_output_np - torch_output_np))}")
    
    np.testing.assert_allclose(jax_output_np, torch_output_np, rtol=1e-6, atol=1e-6)
    print("✓ Embed modules produce identical results")

def test_activations():
    # Test data
    x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    # PyTorch
    torch_x = torch.tensor(x)
    torch_relu = ReLU()(torch_x).numpy()
    torch_gelu = GeLU()(torch_x).numpy()
    
    # JAX (assuming modula.bond.ReLU and modula.bond.GeLU are imported)
    jax_x = jnp.array(x)
    jax_relu = np.array(modula.bond.ReLU().forward(jax_x, None))
    jax_gelu = np.array(modula.bond.GeLU().forward(jax_x, None))
    
    # Compare
    np.testing.assert_allclose(torch_relu, jax_relu, rtol=1e-6)
    np.testing.assert_allclose(torch_gelu, jax_gelu, rtol=1e-4)
    
    print("✓ ReLU matches PyTorch")
    print("✓ GELU matches PyTorch")

def test_split_and_merge_heads():
    # Test parameters
    batch_size = 2
    sequence_length = 8
    embed_dim = 64
    num_heads = 8
    
    # Create test input (same for both versions)
    np.random.seed(42)
    input_data = np.random.randn(batch_size, sequence_length, embed_dim).astype(np.float32)
    
    print("Testing SplitIntoHeads...")
    
    # JAX version - Split
    jax_split = modula.bond.SplitIntoHeads(num_heads)
    jax_input = jnp.array(input_data)
    jax_split_output = jax_split.forward(jax_input, None)
    
    # PyTorch version - Split
    torch_split = SplitIntoHeads(num_heads)
    torch_input = torch.from_numpy(input_data)
    torch_split_output = torch_split.forward(torch_input)
    
    # Compare split outputs
    jax_split_np = np.array(jax_split_output)
    torch_split_np = torch_split_output.detach().numpy()
    
    print(f"Split - JAX output shape: {jax_split_np.shape}")
    print(f"Split - PyTorch output shape: {torch_split_np.shape}")
    
    split_close = np.allclose(jax_split_np, torch_split_np, atol=1e-6)
    print(f"Split outputs are numerically close: {split_close}")
    
    expected_split_shape = (batch_size, num_heads, sequence_length, embed_dim // num_heads)
    assert jax_split_np.shape == expected_split_shape, "JAX split shape mismatch"
    assert torch_split_np.shape == expected_split_shape, "PyTorch split shape mismatch"
    assert split_close, "Split outputs are not numerically equivalent"
    
    print("✅ SplitIntoHeads tests passed!")
    
    print("\nTesting MergeHeads...")
    
    # JAX version - Merge
    jax_merge = modula.bond.MergeHeads()
    jax_merge_output = jax_merge.forward(jax_split_output, None)
    
    # PyTorch version - Merge
    torch_merge = MergeHeads()
    torch_merge_output = torch_merge.forward(torch_split_output)
    
    # Compare merge outputs
    jax_merge_np = np.array(jax_merge_output)
    torch_merge_np = torch_merge_output.detach().numpy()
    
    print(f"Merge - JAX output shape: {jax_merge_np.shape}")
    print(f"Merge - PyTorch output shape: {torch_merge_np.shape}")
    
    merge_close = np.allclose(jax_merge_np, torch_merge_np, atol=1e-6)
    print(f"Merge outputs are numerically close: {merge_close}")
    
    expected_merge_shape = (batch_size, sequence_length, embed_dim)
    assert jax_merge_np.shape == expected_merge_shape, f"JAX merge shape mismatch"
    assert torch_merge_np.shape == expected_merge_shape, f"PyTorch merge shape mismatch"
    assert merge_close, "Merge outputs are not numerically equivalent"
    
    # Test round-trip: original -> split -> merge should equal original
    original_recovered_jax = np.allclose(np.array(jax_input), jax_merge_np, atol=1e-6)
    original_recovered_torch = np.allclose(input_data, torch_merge_np, atol=1e-6)
    
    print(f"JAX round-trip recovery: {original_recovered_jax}")
    print(f"PyTorch round-trip recovery: {original_recovered_torch}")
    
    assert original_recovered_jax, "JAX round-trip failed"
    assert original_recovered_torch, "PyTorch round-trip failed"
    
    print("✅ MergeHeads tests passed!")
    print("✅ All tests passed!")

def test_attention_components():
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 6
    d_query = 8
    scale = 2.0
    
    # Create test inputs
    np.random.seed(42)
    q_data = np.random.randn(batch_size, num_heads, seq_len, d_query).astype(np.float32)
    k_data = np.random.randn(batch_size, num_heads, seq_len, d_query).astype(np.float32)
    v_data = np.random.randn(batch_size, num_heads, seq_len, d_query).astype(np.float32)
    
    print("Testing AttentionQK...")
    
    # Test AttentionQK
    jax_qk = modula.bond.AttentionQK()
    jax_q = jnp.array(q_data)
    jax_k = jnp.array(k_data)
    jax_scores = jax_qk.forward((jax_q, jax_k), None)
    
    torch_qk = AttentionQK()
    torch_q = torch.from_numpy(q_data)
    torch_k = torch.from_numpy(k_data)
    torch_scores = torch_qk.forward((torch_q, torch_k))
    
    jax_scores_np = np.array(jax_scores)
    torch_scores_np = torch_scores.detach().numpy()
    
    print(f"QK - JAX shape: {jax_scores_np.shape}, PyTorch shape: {torch_scores_np.shape}")
    qk_close = np.allclose(jax_scores_np, torch_scores_np, atol=1e-6)
    print(f"QK outputs close: {qk_close}")
    assert qk_close, "AttentionQK outputs don't match"
    
    print("✅ AttentionQK tests passed!")
    
    print("\nTesting CausalMask...")
    
    # Test CausalMask
    jax_mask = modula.bond.CausalMask()
    jax_masked = jax_mask.forward(jax_scores, None)
    
    torch_mask = CausalMask()
    torch_masked = torch_mask.forward(torch_scores)
    
    jax_masked_np = np.array(jax_masked)
    torch_masked_np = torch_masked.detach().numpy()
    
    print(f"Mask - JAX shape: {jax_masked_np.shape}, PyTorch shape: {torch_masked_np.shape}")
    
    # For masked values, check that -inf values are in the same positions
    jax_is_neginf = jax_masked_np == -np.inf
    torch_is_neginf = torch_masked_np == -np.inf
    mask_positions_match = np.array_equal(jax_is_neginf, torch_is_neginf)
    
    # For non-masked values, check they're numerically close
    non_masked_close = np.allclose(jax_masked_np[~jax_is_neginf], torch_masked_np[~torch_is_neginf], atol=1e-6)
    
    print(f"Mask positions match: {mask_positions_match}")
    print(f"Non-masked values close: {non_masked_close}")
    assert mask_positions_match and non_masked_close, "CausalMask outputs don't match"
    
    print("✅ CausalMask tests passed!")
    
    print("\nTesting Softmax...")
    
    # Test Softmax
    jax_softmax = modula.bond.Softmax(scale)
    jax_soft_out = jax_softmax.forward(jax_masked, None)
    
    torch_softmax = Softmax(scale)
    torch_soft_out = torch_softmax.forward(torch_masked)
    
    jax_soft_np = np.array(jax_soft_out)
    torch_soft_np = torch_soft_out.detach().numpy()
    
    print(f"Softmax - JAX shape: {jax_soft_np.shape}, PyTorch shape: {torch_soft_np.shape}")
    softmax_close = np.allclose(jax_soft_np, torch_soft_np, atol=1e-6)
    print(f"Softmax outputs close: {softmax_close}")
    assert softmax_close, "Softmax outputs don't match"
    
    print("✅ Softmax tests passed!")
    
    print("\nTesting ApplyAttentionScores...")
    
    # Test ApplyAttentionScores
    jax_apply = modula.bond.ApplyAttentionScores()
    jax_v = jnp.array(v_data)
    jax_final = jax_apply.forward((jax_v, jax_soft_out), None)
    
    torch_apply = ApplyAttentionScores()
    torch_v = torch.from_numpy(v_data)
    torch_final = torch_apply.forward((torch_v, torch_soft_out))
    
    jax_final_np = np.array(jax_final)
    torch_final_np = torch_final.detach().numpy()
    
    print(f"Apply - JAX shape: {jax_final_np.shape}, PyTorch shape: {torch_final_np.shape}")
    apply_close = np.allclose(jax_final_np, torch_final_np, atol=1e-6)
    print(f"Apply outputs close: {apply_close}")
    assert apply_close, "ApplyAttentionScores outputs don't match"
    
    print("✅ ApplyAttentionScores tests passed!")
    print("✅ All attention component tests passed!")

def test_rope_equivalence():
    # Test parameters
    batch_size = 2
    n_heads = 8
    seq_len = 16
    d_head = 64
    base = 10000
    
    # Create test input
    np.random.seed(42)
    q_np = np.random.randn(batch_size, n_heads, seq_len, d_head).astype(np.float32)
    k_np = np.random.randn(batch_size, n_heads, seq_len, d_head).astype(np.float32)
    
    # Convert to respective frameworks
    q_torch = torch.from_numpy(q_np)
    k_torch = torch.from_numpy(k_np)
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    
    # Initialize models
    rope_jax = modula.bond.Rope(d_head, base=base)
    rope_torch = Rope(d_head, base=base)
    
    # Forward pass
    with torch.no_grad():
        q_out_torch, k_out_torch = rope_torch((q_torch, k_torch))
    
    q_out_jax, k_out_jax = rope_jax.forward((q_jax, k_jax), None)  # w=None since it's not used
    
    sin_jax, cos_jax = rope_jax.get_cached(seq_len)
    
    # Extract cached values
    sin_torch = rope_torch.sin_cached.numpy()
    cos_torch = rope_torch.cos_cached.numpy()

    # Check if base values match
    sin_match = np.allclose(sin_torch, sin_jax, rtol=1e-5, atol=1e-6)
    cos_match = np.allclose(cos_torch, cos_jax, rtol=1e-5, atol=1e-6)
    
    print(f"Sin values match: {sin_match}")
    print(f"Cos values match: {cos_match}")

    # Convert to numpy for comparison
    q_out_torch_np = q_out_torch.numpy()
    k_out_torch_np = k_out_torch.numpy()
    q_out_jax_np = np.array(q_out_jax)
    k_out_jax_np = np.array(k_out_jax)
    
    # Check equivalence
    q_close = np.allclose(q_out_torch_np, q_out_jax_np, rtol=1e-5, atol=1e-6)
    k_close = np.allclose(k_out_torch_np, k_out_jax_np, rtol=1e-5, atol=1e-6)
    
    print(f"Q outputs match: {q_close}")
    print(f"K outputs match: {k_close}")
    print(f"Max Q difference: {np.max(np.abs(q_out_torch_np - q_out_jax_np))}")
    print(f"Max K difference: {np.max(np.abs(k_out_torch_np - k_out_jax_np))}")
    
    assert q_close and k_close, "Rope implementations do not match!"
    print("✅ All tests passed! Both implementations produce identical outputs.")



test_orthogonalize()
test_linear_modules()
test_embed_modules()
test_activations()
test_split_and_merge_heads()
test_attention_components()
test_rope_equivalence()

import jax
import torch
import torch.nn as nn

import modula.abstract
import modula.atom
import modula.bond

import modula.torch_modules as torch_modules

substitute = (
    modula.atom.Linear, torch_modules.Linear,
    modula.atom.Embed, torch_modules.Embed,
    modula.bond.ReLU, nn.ReLU,
    modula.bond.GeLU, torch_modules.GeLU,
    modula.bond.SplitIntoHeads, torch_modules.SplitIntoHeads,
    modula.bond.MergeHeads, torch_modules.MergeHeads,
    modula.bond.AttentionQK, torch_modules.AttentionQK,
    modula.bond.CausalMask, torch_modules.CausalMask,
    modula.bond.Softmax, torch_modules.Softmax,
    modula.bond.ApplyAttentionScores, torch_modules.ApplyAttentionScores,
    modula.bond.Rope, torch_modules.Rope,
    modula.abstract.Mul, torch_modules.Mul,
    modula.abstract.Add, torch_modules.Add,
    modula.abstract.Identity, torch_modules.Identity,
)

def get_structure_signature(module):
    """
    Get a hashable signature representing the module's structure.
    
    Returns:
        tuple: A nested tuple representing the structure
    """
    if isinstance(module, modula.abstract.Atom):
        return ('Atom', type(module).__name__)
    
    if isinstance(module, modula.abstract.Bond):
        if isinstance(module, modula.abstract.Mul):
            # Include scalar for Mul since it affects structure
            return ('Bond', type(module).__name__, module.sensitivity)
        return ('Bond', type(module).__name__)
    
    if isinstance(module, modula.abstract.CompositeModule):
        m0, m1 = module.children
        return ('Composite', get_structure_signature(m0), get_structure_signature(m1))
    
    if isinstance(module, modula.abstract.TupleModule):
        child_sigs = tuple(get_structure_signature(child) for child in module.children)
        return ('Tuple', child_sigs)
    
    return ('Unknown', type(module).__name__)

# Usage example:
def modules_match_by_signature(module1, module2):
    """Fast structural comparison using signatures."""
    return get_structure_signature(module1) == get_structure_signature(module2)

def convert_modula_to_torch(modula_module, substitute_map=substitute):
    """
    Convert a modula module to its PyTorch equivalent using the substitution map.
    
    Args:
        modula_module: The modula module to convert
        substitute_map: Tuple of (modula_class, torch_class) pairs
        
    Returns:
        Corresponding PyTorch module
        
    Raises:
        ValueError: If no corresponding PyTorch module is found
    """
    if isinstance(modula_module, nn.Module):
        # If it's already a PyTorch module, return it as is
        return modula_module
    # Create a dictionary from the tuple for easier lookup
    substitution_dict = {}
    for i in range(0, len(substitute_map), 2):
        modula_class = substitute_map[i]
        torch_class = substitute_map[i + 1]
        substitution_dict[modula_class] = torch_class
    
    # Find the corresponding torch class
    modula_type = type(modula_module)
    if modula_type not in substitution_dict:
        raise ValueError(f"No PyTorch equivalent found for {modula_type}")
    
    torch_class = substitution_dict[modula_type]
    
    # Convert using the from_modula static method
    if hasattr(torch_class, 'from_modula'):
        # try to access a `weight` attribute if it exists
        if hasattr(modula_module, 'weight'):
            weight = torch.tensor(modula_module.weight, dtype=torch.float32)
            return torch_class.from_modula(modula_module, weight)
        else:
            return torch_class.from_modula(modula_module)
    else: # If no from_modula method, just instantiate the class
        return torch_class() # it won't have arguments if it doesn't have from_modula

# Reference Attetnion module for detection
def Attention(num_heads, d_embed, d_query, d_value, attention_scale):
    """Multi-head attention"""

    # For keys, queries, and values we add a heads dimension. For the out projection, we remove heads.
    # Remember modules compose right-to-left, and the order is modula.atom.Linear(d_out, d_in)! And @ means compose.
    Q = modula.bond.SplitIntoHeads(num_heads) @ modula.atom.Linear(num_heads * d_query, d_embed)
    K = modula.bond.SplitIntoHeads(num_heads) @ modula.atom.Linear(num_heads * d_query, d_embed)
    V = modula.bond.SplitIntoHeads(num_heads) @ modula.atom.Linear(num_heads * d_value, d_embed)
    W = modula.atom.Linear(d_embed, num_heads * d_value) @ modula.bond.MergeHeads()

    # Read right-to-left: rotate (Q, K) with RoPE, apply Q @ K.T, mask, softmax (with a scale we can choose).
    AttentionScores = modula.bond.Softmax(attention_scale) @ modula.bond.CausalMask() @ modula.bond.AttentionQK() @ modula.bond.Rope(d_query) @ (Q, K)

    # Read right-to-left: apply attention scores, multiply by 1/3 to fix the sensitivity to 1, project back to d_embed.
    return W @ (1/3 * modula.bond.ApplyAttentionScores()) @ (V, AttentionScores)

def extract_attention_parameters(module):
    """
    Extract attention parameters from an Attention module structure.
    
    Returns:
        dict: Dictionary with keys 'num_heads', 'd_embed', 'd_query', 'd_value', 'attention_scale'
              or None if the structure doesn't match expected Attention pattern
    """
    # First verify this looks like an attention module by checking high-level structure
    if not isinstance(module, modula.abstract.CompositeModule):
        return None
    
    # Walk through and collect all relevant modules
    split_heads_modules = []
    linear_modules = []
    softmax_modules = []
    
    def collect_modules(mod):
        # Import the actual module classes - adjust these imports based on your actual module structure
        if hasattr(mod, '__class__'):
            class_name = mod.__class__.__name__
            if class_name == 'SplitIntoHeads':
                split_heads_modules.append(mod)
            elif class_name == 'Linear':
                linear_modules.append(mod)
            elif class_name == 'Softmax':
                softmax_modules.append(mod)
    
    modula.abstract.traverse_forward_order(module, collect_modules)
    
    # Verify we have the expected number of modules
    if len(split_heads_modules) < 1 or len(linear_modules) < 4 or len(softmax_modules) < 1:
        return None
    
    try:
        # Extract num_heads from first SplitIntoHeads
        num_heads = split_heads_modules[0].num_heads
        
        # Extract attention_scale from Softmax
        attention_scale = softmax_modules[0].sensitivity  # Assuming it's stored as self.scale
        
        # For linear modules, based on the structure:
        # - First Linear encountered should be V (due to tuple ordering): fanout = num_heads * d_value, fanin = d_embed
        # - We need to find Q, K, V, and W linear modules
        
        # The V linear module (first one encountered)
        v_linear = linear_modules[0]
        d_embed = v_linear.fanin
        num_heads_times_d_value = v_linear.fanout
        d_value = num_heads_times_d_value // num_heads
        
        # Find Q or K linear module to get d_query
        # Q and K should have the same fanout = num_heads * d_query
        # We can identify them as having fanin = d_embed and fanout != num_heads * d_value
        qk_candidates = [lin for lin in linear_modules[1:] 
                        if lin.fanin == d_embed]
        
        if len(qk_candidates) < 2:  # Should have both Q and K
            print("Not enough Q/K candidates found.")
            return None
            
        num_heads_times_d_query = qk_candidates[0].fanout
        d_query = num_heads_times_d_query // num_heads
        
        # Verify consistency
        if (num_heads_times_d_query != qk_candidates[1].fanout or
            d_value * num_heads != num_heads_times_d_value or
            d_query * num_heads != num_heads_times_d_query):
            print("Inconsistent dimensions found in attention module.")
            return None
        
        return {
            'num_heads': num_heads,
            'd_embed': d_embed,
            'd_query': d_query,
            'd_value': d_value,
            'attention_scale': attention_scale
        }
        
    except (AttributeError, ZeroDivisionError, IndexError):
        raise
        return None

def sequentialise(module, substitute_flash=False, verbose=False):
    modules = []

    if substitute_flash:
        attention_signature = get_structure_signature(Attention(num_heads=1, d_embed=1, d_query=1, d_value=1, attention_scale=1.0))

    def _traverse_tuple(mod):
        return torch_modules.Parallel([sequentialise(child, substitute_flash=substitute_flash) for child in mod.children])

    def _traverse(mod):
        if substitute_flash:
            if verbose:
                print(f"Traversing module: {hash(get_structure_signature(mod))} - {get_structure_signature(mod)}")
            if get_structure_signature(mod) == attention_signature:
                # If the module matches the attention signature, convert it to a torch attention module
                mod = torch_modules.FlashAttention(**extract_attention_parameters(mod))
                for m in mod:
                    modules.append(m)
                mod = None
        if isinstance(mod, modula.abstract.CompositeModule):
            # For composite modules, traverse m0 first, then m1 (execution order)
            m0, m1 = mod.children
            _traverse(m0)
            _traverse(m1)
        elif isinstance(mod, modula.abstract.TupleModule):
            # For tuple modules, traverse all children (they execute in parallel)
            mod = _traverse_tuple(mod)
        
        # Apply function to current module after traversing children
        if not any((isinstance(mod, cls) for cls in [modula.abstract.CompositeModule, modula.abstract.TupleModule])):
            if mod is not None:
                modules.append(mod)
    
    _traverse(module)

    # convert modula.atom and modula.bond modules to torch modules
    modules = [convert_modula_to_torch(m) for m in modules]
    seq_module = nn.Sequential(*modules)
    return seq_module

def flash_sequentialise(module):
    state_dict = sequentialise(module, substitute_flash=False).state_dict()
    flash_module = sequentialise(module, substitute_flash=True, verbose=False)
    flash_module.load_state_dict(state_dict)
    return flash_module

if __name__ == "__main__":
    import numpy as np
    key = jax.random.PRNGKey(0)
    torch.manual_seed(0)  # For reproducibility in PyTorch

    # Example usage

    module = modula.atom.Linear(fanout=4, fanin=3)
    module @= (modula.atom.Linear(fanout=2, fanin=4), modula.atom.Linear(fanout=2, fanin=4))
    module @= modula.atom.Linear(fanout=2, fanin=4)

    print(sequentialise(module))

    attention = Attention(num_heads=2, d_embed=8, d_query=4, d_value=4, attention_scale=1.0)
    weights = attention.initialize(key)

    # equip attention modules with target norms and weights
    _ = modula.abstract.get_leaf_target_norms(attention, target_norm=1.0)
    modula.abstract.set_atomic_weights(attention, weights)
    torch_attention = sequentialise(attention)
    print(torch_attention)

    # Check they are equivalent
    x = torch.randn(2, 10, 8)  # batch size 2, sequence length 10, embedding size 8
    modula_out = attention(x.numpy(), weights)
    torch_out = torch_attention(x)
    rtol = 1e-3  # relative tolerance for comparison
    np.testing.assert_allclose(modula_out, torch_out.detach().numpy(), rtol=rtol)
    print(f"Attention conversion successful and outputs match to within {rtol} relative tolerance.")


    print([n for n, v in torch_attention.state_dict().items()])  # Print the names of the parameters in the torch attention module


    # flash = torch_modules.FlashAttention(num_heads=2, d_embed=8, d_query=4, d_value=4, attention_scale=1.0)
    flash = sequentialise(attention, substitute_flash=True)
    print([n for n, v in flash.state_dict().items()])  # Print the names of the parameters in the flash attention module

    flash.load_state_dict(torch_attention.state_dict())  # Load weights from the previous attention module
    print(flash)

    # test the Q, K and V modules are OK
    v, _v = torch_attention[0][0][0], flash[0][0][0]
    q, _q = torch_attention[0][1][0][0][0], flash[0][1][0][0][0]
    k, _k = torch_attention[0][1][0][1][0], flash[0][1][0][1][0]
    proj, _proj = torch_attention[4], flash[4]
    rtol = 1e-3  # relative tolerance for comparison
    np.testing.assert_allclose(v.weight.detach().numpy(), _v.weight.detach().numpy(), rtol=rtol)
    np.testing.assert_allclose(q.weight.detach().numpy(), _q.weight.detach().numpy(), rtol=rtol)
    np.testing.assert_allclose(k.weight.detach().numpy(), _k.weight.detach().numpy(), rtol=rtol)
    np.testing.assert_allclose(proj.weight.detach().numpy(), flash[4].weight.detach().numpy(), rtol=rtol)
    
    # Check they are equivalent
    torch_out_flash = flash(x)
    np.testing.assert_allclose(torch_out.detach().numpy(), torch_out_flash.detach().numpy(), rtol=rtol)
    print(f"Flash attention conversion successful and outputs match to within {rtol} relative tolerance.")

    # Do the same for a complete model
    def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
        # Set embed to have mass 1. This controls the proportion of feature learning that it contributes to the whole network.
        embed = modula.atom.Embed(d_embed, vocab_size)
        embed.tare()

        # Let's create attention and MLP layers. 
        att = Attention(num_heads, d_embed, d_query, d_value, attention_scale)
        print(f"att hash = {hash(get_structure_signature(att))}")
        mlp = modula.atom.Linear(d_embed, 4*d_embed) @ modula.bond.GeLU() @ modula.atom.Linear(4*d_embed, d_embed)

        # For our residual connections, L = 2*num_blocks because each block has two residual connections.
        att_block = (1-1/(2*num_blocks)) * modula.abstract.Identity() + 1/(2*num_blocks) * att
        mlp_block = (1-1/(2*num_blocks)) * modula.abstract.Identity() + 1/(2*num_blocks) * mlp

        # We can use powers of a module to compose it with itself many times!
        blocks = (mlp_block @ att_block) ** num_blocks

        # Set all transformer blocks to have mass 5 (by default).
        # So 5/7 of the change in the network output is due to the blocks,
        # and 2/7 of the change in output is due to the embedding and out projection.
        blocks.tare(absolute=blocks_mass)

        out = final_scale * modula.atom.Linear(vocab_size, d_embed)

        return out @ blocks @ embed

    vocab_size = 65
    num_heads = 4
    d_embed = 128
    d_query = 32
    d_value = 32
    num_blocks = 4
    attention_scale = 1
    final_scale = 1   

    model = GPT(
        vocab_size=vocab_size,
        num_heads=num_heads,
        d_embed=d_embed,
        d_query=d_query,
        d_value=d_value,
        num_blocks=num_blocks,
        attention_scale=attention_scale,
        final_scale=final_scale,
    ) 

    weights = model.initialize(key)
    _ = modula.abstract.get_leaf_target_norms(model, target_norm=1.0)
    modula.abstract.set_atomic_weights(model, weights)

    torch_model = sequentialise(model)
    print(torch_model)

    # Check they are equivalent
    x = torch.randint(0, vocab_size, (2, 10))  # batch size 2, sequence length 10
    modula_out = model(x.numpy(), weights)
    torch_out = torch_model(x)

    rtol = 1e-3  # relative tolerance for comparison
    np.testing.assert_allclose(modula_out, torch_out.detach().numpy(), rtol=rtol)
    print(f"GPT conversion successful and outputs match to within {rtol} relative tolerance.")

    # Test we can convert the model to FlashAttention
    flash_model = flash_sequentialise(model)
    print(flash_model)
    
    # Check they are equivalent
    torch_out_flash = flash_model(x)
    np.testing.assert_allclose(torch_out.detach().numpy(), torch_out_flash.detach().numpy(), rtol=rtol)
    print(f"Flash GPT conversion successful and outputs match to within {rtol} relative tolerance.")


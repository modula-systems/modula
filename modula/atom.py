import jax
import jax.numpy as jnp

from modula.abstract import Atom, Bond, CompositeModule, TupleModule

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
    M = M / jnp.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weights, x)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = orthogonalize(grad) * jnp.sqrt(self.fanout / self.fanin) * target_norm
        return [d_weight]


class Embed(Atom):
    def __init__(self, d_embed, num_embed):
        super().__init__()
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [num_embed, d_embed]
        return weights[x]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.num_embed, self.d_embed))
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = grad / jnp.linalg.norm(grad, axis=1, keepdims=True) * jnp.sqrt(self.d_embed) * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]


if __name__ == "__main__":
    from modula.abstract import get_leaf_modules, get_leaf_target_norms 

    def test_dualize_consistency(module, grad_w, target_norm=1.0, rtol=1e-6):
        """
        Test that get_unnormalized_dual and get_leaf_target_norms produce results
        consistent with the actual dualize method.
        
        Args:
            module: A Module instance
            grad_w: Weight gradient list
            target_norm: Target norm to test with
            rtol: Relative tolerance for comparison
            
        Returns:
            bool: True if consistent, False otherwise
        """
        # Get results from actual dualize
        actual_dual = module.dualize(grad_w, target_norm=target_norm)
        
        # Get results from our functions
        unnormalized_dual = get_unnormalized_dual(module, grad_w)
        leaf_modules = get_leaf_modules(module)
        target_norms = get_leaf_target_norms(module, target_norm=target_norm)
        
        # Apply target norms to unnormalized dual
        predicted_dual = []
        weight_idx = 0
        
        for leaf_module, leaf_target_norm in zip(leaf_modules, target_norms):
            if isinstance(leaf_module, (Atom)):  # Only atoms have weights
                leaf_weights = unnormalized_dual[weight_idx:weight_idx + leaf_module.atoms]
                # Apply the target norm
                scaled_weights = [w * leaf_target_norm for w in leaf_weights]
                predicted_dual.extend(scaled_weights)
                weight_idx += leaf_module.atoms
            # Bonds have no weights, so nothing to add to predicted_dual
        
        # Compare actual vs predicted
        if len(actual_dual) != len(predicted_dual):
            print(f"Length mismatch: actual {len(actual_dual)}, predicted {len(predicted_dual)}")
            return False
        
        for i, (actual, predicted) in enumerate(zip(actual_dual, predicted_dual)):
            if not jnp.allclose(actual, predicted, rtol=rtol):
                print(f"Mismatch at weight {i}")
                print(f"Actual shape: {actual.shape}, Predicted shape: {predicted.shape}")
                print(f"Max difference: {jnp.max(jnp.abs(actual - predicted))}")
                return False
        
        print("✓ Dualize consistency test passed!")
        return True

    # Example usage:
    def test_example():
        """Example test with a simple module"""
        # Create a simple module
        linear = Linear(fanout=4, fanin=3)
        linear @= Linear(fanout=4, fanin=4)  # Add another linear layer
        linear @= Linear(fanout=2, fanin=4)  # Add another linear layer
        
        # Initialize weights and create some gradient
        key = jax.random.PRNGKey(42)
        weights = linear.initialize(key)
        grad_w = [jax.random.normal(key, shape=w.shape) for w in weights]
        
        # Test consistency
        return test_dualize_consistency(linear, grad_w, target_norm=2.5)

    key = jax.random.PRNGKey(0)

    # sample a random d0xd1 matrix
    d0, d1 = 50, 100
    M = jax.random.normal(key, shape=(d0, d1))
    O = orthogonalize(M)

    # compute SVD of M and O
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    s = jnp.linalg.svd(O, compute_uv=False)

    # print singular values
    print(f"min singular value of O: {jnp.min(s)}")
    print(f"max singular value of O: {jnp.max(s)}")

    print(f"min singular value of M: {jnp.min(S)}")
    print(f"max singular value of M: {jnp.max(S)}")

    # check that M is close to its SVD
    error_M = jnp.linalg.norm(M - U @ jnp.diag(S) @ Vh) / jnp.linalg.norm(M)
    error_O = jnp.linalg.norm(O - U @ Vh) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in M's SVD: {error_M}")
    print(f"relative error in O: {error_O}")

    test_example()

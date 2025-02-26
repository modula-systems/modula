import jax
import jax.numpy as jnp

from modula.abstract import Atom

def orthogonalize(M):
    a, b, c = 3.0, -3.2, 1.2
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for _ in range(10):
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
        return x @ weights.transpose()  # shape is [..., fanout]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
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
        weights = w[0]  # shape [d_embed, num_embed]
        embed = weights[:, x]  # shape [d_embed, x_shape]
        return jnp.moveaxis(embed, 0, -1) # shape [x_shape, d_embed]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.d_embed, self.num_embed))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight, axis=0, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]


if __name__ == "__main__":

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

    # check that M is close to its SVD
    error_M = jnp.linalg.norm(M - U @ jnp.diag(S) @ Vh) / jnp.linalg.norm(M)
    error_O = jnp.linalg.norm(O - U @ Vh) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in M's SVD: {error_M}")
    print(f"relative error in O: {error_O}")

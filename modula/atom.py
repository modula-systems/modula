import jax
import jax.numpy as jnp

from modula.abstract import Atom

def orthogonalize(m):
    m = m / jnp.linalg.norm(m)
    for _ in range(10):
        m = 3/2 * m - 1/2 * m @ m.T @ m
    return m

class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.scale = jnp.sqrt(self.fanout / self.fanin)
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]
        return self.scale * weights @ x, [x]

    def backward(self, w, acts, grad_output):
        weights = w[0]
        input = acts[0]
        grad_input = self.scale * weights.T @ grad_output                         # oops: self.scale appears here
        grad_weight = self.scale * grad_output @ input.T                          # oops: self.scale appears here
        return [grad_weight], grad_input

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad_weight = grad_w[0]
        d_weight = orthogonalize(grad_weight)
        return [d_weight * target_norm]

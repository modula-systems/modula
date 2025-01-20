import jax.numpy as jnp

from modula.abstract import Bond

class ReLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.maximum(0, x), [x]

    def backward(self, w, acts, grad_output):
        input = acts[0]
        grad_input = (input > 0) * grad_output
        return [], grad_input

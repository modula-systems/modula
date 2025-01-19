import numpy as np

from modula.abstract import Atom

class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.scale = np.sqrt(self.fanout / self.fanin)
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

    def initialize(self):
        weight = np.random.normal(size=(self.fanout, self.fanin))
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = weight / np.linalg.norm(weight)
        for _ in range(10):
            weight = 3/2 * weight - 1/2 * weight @ weight.T @ weight
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad_weight = grad_w[0]
        grad_weight = grad_weight / np.linalg.norm(grad_weight)
        for _ in range(10):
            grad_weight = 3/2 * grad_weight - 1/2 * grad_weight @ grad_weight.T @ grad_weight
        return [grad_weight * target_norm]

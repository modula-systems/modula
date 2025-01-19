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
        # semi-orthogonal init
        A = np.random.normal(size=(self.fanout, self.fanin))
        A = A if self.fanout > self.fanin else A.T
        Q = np.linalg.qr(A, mode='reduced')[0]
        Q = Q if self.fanout > self.fanin else Q.T
        return [Q]

    def normalize(self, grad_w, target_norm=1.0):
        grad_weight = grad_w[0]
        spectral_norm = np.linalg.norm(grad_weight, ord=2)
        return [grad_weight / spectral_norm * target_norm]

class ShampooLinear(Linear):
    def __init__(self, fanout, fanin):
        super().__init__(fanout, fanin)

    def normalize(self, grad_w, target_norm=1.0):
        grad_weight = grad_w[0]
        U, S, Vt = np.linalg.svd(grad_weight, full_matrices=False)
        return [U @ Vt * target_norm]

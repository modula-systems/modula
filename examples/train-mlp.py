import numpy as np
import matplotlib.pyplot as plt

from modula.atom import Linear, ShampooLinear
from modula.bond import ReLU
from modula.error import SquareError

def train(module, error, input, target, steps, init_lr, normalize, tqdm=False):
    train_loss_list = []
    w = module.initialize()

    if tqdm:
        from tqdm.notebook import tqdm
    else:
        tqdm = lambda x: x

    for step in tqdm(range(steps)):
        schedule = 1 - step / steps

        output, activations = module(input, w)
        loss = error(output, target)
        error_grad = error.grad(output, target)
        grad_w, _ = module.backward(w, activations, error_grad)
        if normalize:
            grad_w = module.normalize(grad_w)

        for weight, grad_weight in zip(w, grad_w):
            weight -= init_lr * schedule * grad_weight

        train_loss_list.append(loss)

    return train_loss_list

input_dim = 11
width = 100
output_dim = 6
batch_size = 32
steps = 1001
lr_list = 10.0 ** np.arange(-4, 1)

error = SquareError()

x = np.random.rand(input_dim, batch_size)
y = np.random.rand(output_dim, batch_size)

results = {}
config_list = [(False, False), (True, False), (True, True)]

for normalize, shampoo in config_list:

    print("running config:", "normalize", normalize, "shampoo", shampoo)
    for init_lr in lr_list:
        print("init_lr", init_lr)

        M = ShampooLinear if shampoo else Linear
        m = M(output_dim, width) @ ReLU() @ M(width, input_dim)

        results[(normalize, shampoo, init_lr)] = train(m, error, x, y, steps, init_lr, normalize)

config_titles = ["Vanilla gradient descent", "Modular normalization using $G/||G||_*$", "Modular normalization using $G^0$"]

fig, ax = plt.subplots(1, len(config_list), figsize=(10, 3), sharey=True)

for i, (normalize, shampoo) in enumerate(config_list):
    for init_lr in lr_list:
        ax[i].plot(results[(normalize, shampoo, init_lr)], label=init_lr)
    ax[i].set_title(config_titles[i])
    ax[i].set_xlabel("Training iteration")
    ax[i].set_yscale("log")

ax[0].set_ylabel("Training loss")
ax[0].legend(title="Learning rate:", frameon=False)
ax[0].set_ylim(None, 1)

plt.tight_layout()
plt.show()

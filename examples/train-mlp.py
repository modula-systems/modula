import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modula.compound import MLP
from modula.error import SquareError

def train(module, error, input, target, steps, init_lr, dualize):

    w = module.initialize()
    w = module.project(w)

    train_loss_list = []

    for step in tqdm(range(steps)):
        schedule = 1 - step / steps

        output, activations = module(input, w)
        loss = error(output, target)
        error_grad = error.grad(output, target)
        grad_w, _ = module.backward(w, activations, error_grad)

        if dualize:
            d_w = module.dualize(grad_w)
        else:
            d_w = grad_w

        for weight, d_weight in zip(w, d_w):
            weight -= init_lr * schedule * d_weight

        w = module.project(w)

        train_loss_list.append(loss)

    return train_loss_list

input_dim = 11
width = 100
depth = 3
output_dim = 6
batch_size = 32
steps = 1001
lr_list = 10.0 ** np.arange(-4, 3)

error = SquareError()

x = np.random.rand(input_dim, batch_size)
y = np.random.rand(output_dim, batch_size)

sensitivity = 10
results = {'dualized': {}, 'non_dualized': {}}

print(f"sensitivity: {sensitivity}")
for init_lr in lr_list:
    print(f"init_lr: {init_lr}")

    m = sensitivity * MLP(output_dim, input_dim, width, depth)

    # Train with dualization
    loss_history = train(m, error, x, y, steps, init_lr, dualize=True)
    results['dualized'][init_lr] = loss_history[-1]
    
    # Train without dualization 
    loss_history = train(m, error, x, y, steps, init_lr, dualize=False)
    results['non_dualized'][init_lr] = loss_history[-1]

plt.figure(figsize=(8, 6))

dualized_losses = [results['dualized'][lr] for lr in lr_list]
non_dualized_losses = [results['non_dualized'][lr] for lr in lr_list]

plt.plot(lr_list, dualized_losses, 'o-', label='With dualization')
plt.plot(lr_list, non_dualized_losses, 'o-', label='Without dualization')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Final loss')
plt.title(f'Final Loss vs Learning Rate (Sensitivity = {sensitivity})')
plt.legend(frameon=False)
plt.grid(True)
plt.show()

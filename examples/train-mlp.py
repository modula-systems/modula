import jax
import matplotlib.pyplot as plt
from tqdm import tqdm

from modula.compound import MLP
from modula.error import SquareError

def train(module, error, input, target, steps, init_lr, beta = 0.0):

    w = module.initialize(jax.random.PRNGKey(0))
    w = module.project(w)

    train_loss_list = []
    dual_norm_list = []

    momentum = [0] * module.atoms

    for step in tqdm(range(steps)):
        output, activations = module(input, w)
        loss = error(output, target)
        error_grad = error.grad(output, target)
        grad_w, _ = module.backward(w, activations, error_grad)

        momentum = [beta * mom_i + (1-beta) * grad_i for mom_i, grad_i in zip(momentum, grad_w)]

        d_w = module.dualize(momentum)
        dual_norm = sum((d_weight * mom_weight).sum() for d_weight, mom_weight in zip(d_w, momentum))
        dual_norm_list.append(dual_norm)
        lr = init_lr * dual_norm

        w = [weight - lr * d_weight for weight, d_weight in zip(w, d_w)]
        w = module.project(w)

        train_loss_list.append(loss)

    return train_loss_list, dual_norm_list

input_dim = 10
width = 100
depth = 5
output_dim = 6
batch_size = 32
steps = 201
lr_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

error = SquareError()

key, subkey = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.uniform(key, shape=(input_dim, batch_size))
y = jax.random.uniform(subkey, shape=(output_dim, batch_size))

sensitivity = 10
results = {'dualized': {}, 'non_dualized': {}}

m = sensitivity * MLP(output_dim, input_dim, width, depth)
m.jit()
print(m)

for init_lr in lr_list:
    print(f"init_lr: {init_lr}")

    # Train with momentum
    loss_history, _ = train(m, error, x, y, steps, init_lr, beta=0.9)
    results['dualized'][init_lr] = loss_history[-1]
    
    # Train without momentum
    loss_history, _ = train(m, error, x, y, steps, init_lr, beta=0.0) 
    results['non_dualized'][init_lr] = loss_history[-1]

# Find best learning rates
best_lr_momentum = min(results['dualized'].items(), key=lambda x: x[1])[0]
best_lr_no_momentum = min(results['non_dualized'].items(), key=lambda x: x[1])[0]

print(f"\nBest learning rates:")
print(f"With momentum (β=0.9): {best_lr_momentum}")
print(f"Without momentum (β=0.0): {best_lr_no_momentum}")

# Plot dual norms and losses for best learning rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

loss_history_momentum, dual_norms_momentum = train(m, error, x, y, steps, best_lr_momentum, beta=0.9)
loss_history_no_momentum, dual_norms_no_momentum = train(m, error, x, y, steps, best_lr_no_momentum, beta=0.0)

# Plot dual norms
ax1.plot(dual_norms_momentum, label=f'With momentum (β=0.9, lr={best_lr_momentum})')
ax1.plot(dual_norms_no_momentum, label=f'Without momentum (β=0.0, lr={best_lr_no_momentum})')
ax1.set_yscale('log')
ax1.set_xlabel('Training step')
ax1.set_ylabel('Gradient dual norm')
ax1.set_title('Gradient Dual Norms with Best Learning Rates')
ax1.legend(frameon=False)
ax1.grid(True)

# Plot losses
ax2.plot(loss_history_momentum, label=f'With momentum (β=0.9, lr={best_lr_momentum})')
ax2.plot(loss_history_no_momentum, label=f'Without momentum (β=0.0, lr={best_lr_no_momentum})')
ax2.set_yscale('log')
ax2.set_xlabel('Training step')
ax2.set_ylabel('Loss')
ax2.set_title('Training Curves with Best Learning Rates')
ax2.legend(frameon=False)
ax2.grid(True)

plt.tight_layout()
plt.show()

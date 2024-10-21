<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/modula.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/modula_light.svg">
  <img alt="modula logo" src="assets/modula.svg">
</picture>

Modula is a deep learning framework designed for graceful scaling. Neural networks written in Modula automatically transfer learning rate across scale.

We are slowly writing [the Modula docs](https://jeremybernste.in/modula/). Check them out for an accessible introduction to scaling theory and the Modula API. Also, here are some [slides](https://docs.google.com/presentation/d/1mCp6weIty9BzFFmx7LUGk2MPmNi-m-yKjigQ9wnycng/edit?usp=sharing) for a talk that Jeremy gave, that provide a more visual introduction to Modula. And here is [Modulax](https://github.com/GallagherCommaJack/modulax/) by Jack Gallagher. And here is a barebones implementation of [Modula in NumPy](https://colab.research.google.com/drive/1lKS15RJilGsstYP5JDQKSn3Z7TUUYIDQ?usp=sharing).

Modula is an experimental framework based on our research paper: [Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813).

## Quick start

Install modula via pip:

```bash
pip install modula
```

Next, let's download the Shakespeare data:
```bash
pip install numpy datasets
python examples/data/shakespeare.py
```

And finally, let's train a GPT:
```bash
python examples/train-gpt.py
```

This runs on CPU and should get train loss: 1.65 and test loss: 1.80 after 2000 iterations.

## Project roadmap

These are the current plans for the project. Ideas and pull requests are welcome!

- [ ] finish writing [the docs](https://jeremybernste.in/modula/)
- [ ] improve MFU (see https://github.com/jxbz/modula/pull/1)
- [ ] add more examples

## Learning rate transfer with Modula

The following figures are all for 10k steps of training GPT on OpenWebText.

The left two panels of this first figure show that the learning rate of modular-normalized-Adam transfers across width and depth. The right two panels show scaling performance when learning rate is tuned at the scale marked by the gray dotted line. Modular normalization fixes width and depth scaling for both SGD and Adam.

![some results](/assets/aggregate.svg)

Our GPT implementation differs slightly from standard nanoGPT since we designed it to scale better. To check our implementation is not losing performance compared to standard nanoGPT, in the next figure we compare three setups:
1. our direct reimplementation of nanoGPT with Adam (column 1);
2. our custom GPT implementation with Adam and without modular normalization (column 2);
3. our custom GPT implementation with Adam and with modular normalization (column 3).

![some more results](/assets/nanogpt-vs-modula.svg)

Notice that our GPT implementation transfers learning rate better than nanoGPT, even without modular normalization (column 2 versus column 1). We also noticed other interesting behaviours: for example, our GPT implementation with modular normalization transfers learning rate quite well across context length:

![and some more results](/assets/gpt-owt-context.svg)

## Training an MLP in Modula

Let's start by building an MLP and initializing its weights:

```python
from modula.atom import Linear
from modula.bond import ReLU

mlp = Linear(10,10000) @ ReLU() @ Linear(10000, 1000)
weights = mlp.initialize(device="cpu")
```

Now let's fit this MLP to some random data:
```python
from torch import randn, no_grad

data, target = randn(1000), randn(10)

for step in range(steps:=20):
    output = mlp(data, weights)

    loss = (target - output).square().mean()
    loss.backward()

    with no_grad():
        mlp.normalize(grad := weights.grad())     # normalize the gradient in the modular norm
        weights -= 0.1 * grad
        weights.zero_grad()
    
        mlp.regularize(weights, strength = 0.01)  # regularize the weight vector

    print(step, loss.item())
```
## Modula abstractions

Modula provides two useful abstractions: the `Vector` class and the `Module` class.

### The `Vector` class

The `Vector` class is used to store the weights of the neural net. For instance, in the previous example the line `weights = mlp.initialize(device="cpu")` creates a `Vector` called `weights`. And `grad = weights.grad()` stores the gradient of `weights` as a `Vector` called `grad`. The point of all this is that you can do operations on `Vector` objects like:
```python
weights -= 0.1 * grad
```
This allows you to write optimization algorithms without doing for loops over lists of tensors everywhere.

### The `Module` class

The meat of Modula is found in the `Module` class. A `Module` `m` must have six attributes. Two numbers:
```python
m.mass: float           # sets the proportion of feature learning m contributes to any supermodule
m.sensitivity: float    # estimates the sensitivity of m to input perturbations
```
and four methods:
```python
m.forward(x: Tensor, w: Vector) -> Tensor    # maps an input and a weight vector to an output
m.initialize() -> Vector                     # randomly samples a weight vector
m.normalize(w: Vector)                       # scales vector w to have unit modular norm
m.regularize(w: Vector, strength: float)     # regularizes vector w in-place
```

There are three kinds of modules in Modula:
- Atoms are modules that have weights and where the attributes are hand-declared, e.g. `modula.atom.Linear`;
- Bonds are modules without weights and where the attributes are hand-declared, e.g. `modula.bond.GELU`;
- Compounds are modules built by combining atoms and bonds---their attributes are inferred automatically, e.g. `modula.compound.GPT`.

We provide the following basic operations for building compounds:
```python
M_2 @ M_1     # composes module M_2 with module M_1
(M_1, M_2)    # acts as a tuple module in any further composition
M_1 + M_2     # returns the module sum
a * M         # multiplies module M by scalar a
M ** L        # returns the Lth iterate of module M, i.e. M @ M @ ... @ M
```
So, for example, the following `residualize` function takes a block module `block` and a depth `L` and returns a resnet with this block:
```python
from modula.bond import Identity
residualize = lambda block, L : ((1 - 1/L) * Identity() + 1/L * block) ** L
```

The point of all this is that you can build a complicated compound module `m`, and all module attributes will be automatically inferred. Then during training, you can call `m.normalize` on the Adam or SGD updates, and the learning rate will automatically transfer when scaling the architecture.

## Repository structure

```
.
├── assets                          # figures, logos and such
    └── ...
├── examples
│   ├── hello-world.py              # simple training loop
│   ├── gradient-accumulation.py    # gradient accumulation for large batch training
│   └── multi-gpu.py                # multi GPU training with torch.distributed
├── modula
│   ├── __init__.py
│   ├── abstract.py                 # basic definitions: composition & concatenation, etc.
│   ├── atom.py                     # modules with weights: linear, conv2d etc.
│   ├── bond.py                     # modules without weights: ReLU, FunctionalAttention, etc.
│   ├── compound.py                 # derived modules: GPT, ResNet, etc.
│   └── vector.py                   # class for storing weight vectors
├── paper                           # code associated with the arXiv paper
    └── ...
├── LICENSE                         # MIT license
├── README.md                       # this file
└── setup.py                        # pip package stuff
```

## BibTeX

If Modula is useful in your research, consider citing [our paper](https://arxiv.org/abs/2405.14813):

```bibtex
@article{modula,
  author  = {Tim Large and Yang Liu and Minyoung Huh and Hyojin Bahng and Phillip Isola and Jeremy Bernstein},
  title   = {Scalable Optimization in the Modular Norm},
  journal = {arXiv:2405.14813},
  year    = 2024
}
```

## Acknowledgements
The design of Modula was influenced by [μP](https://github.com/microsoft/mup), [autobound](https://github.com/google/autobound), [AGD](https://github.com/jxbz/agd) and [PyTorch](https://github.com/pytorch/pytorch) itself.

## License
Modula is released under an [MIT license](/LICENSE).

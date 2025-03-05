<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/modula.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/modula_light.svg">
  <img alt="modula logo" src="assets/modula.svg">
</picture>

Modula is a deep learning library and a deep learning theory built hand-in-hand. Modula disentangles complex neural networks and turns them into structured mathematical objects called modules. This makes training faster and easier to scale, while also providing tools for understanding the properties of the trained network. Modula is built on top of [JAX](https://github.com/google/jax). More information is available in the [Modula docs](https://docs.modula.systems).

# Installation

Modula can be installed using pip:

```bash
pip install git+https://github.com/modula-systems/modula.git
```

Or you can clone the repository and install locally:

```bash
git clone https://github.com/modula-systems/modula.git
cd modula
pip install -e .
```

# Functionality

Modula provides a set of architecture-specific helper functions that are automatically constructed along with the network architecture itself. As an example, let’s build a multi-layer perceptron:

```python
from modula.atom import Linear
from modula.bond import ReLU

mlp = Linear(10, 256)
mlp @= ReLU()
mlp @= Linear(256, 256)
mlp @= ReLU()
mlp @= Linear(256, 784)

mlp.jit() # makes everything run faster
```

Behind the scenes, Modula builds a function to randomly initialize the weights of the network:

```python
import jax

key = jax.random.PRNGKey(0)
weights = mlp.initialize(key)
```

Supposing we have used JAX to compute the gradient of our loss and stored this as `grad`, then we can use Modula to dualize the gradient, thereby accelerating our gradient descent training:

```python
dualized_grad = mlp.dualize(grad)
weights = [w - 0.1 * dg for w, dg in zip(weights, dualized_grad)]
```

And after the weight update, we can project the weights back to their natural constraint set:

```python
weights = mlp.project(weights)
```

In short, Modula lets us think about the weight space of our neural network as a somewhat classical optimization space, complete with duality and projection operations.

# References

Modula is based on two papers. The first is on the [modular norm](https://arxiv.org/abs/2405.14813):

```bibtex
@inproceedings{modular-norm,
  title={Scalable Optimization in the Modular Norm},
  author={Tim Large and Yang Liu and Minyoung Huh and Hyojin Bahng and Phillip Isola and Jeremy Bernstein},
  booktitle={Neural Information Processing Systems},
  year={2024}
}
```

And the second is on [modular duality](https://arxiv.org/abs/2410.21265):

```bibtex
@article{modular-dualization,
  title   = {Modular Duality in Deep Learning},
  author  = {Jeremy Bernstein and Laker Newhouse},
  journal = {arXiv:2410.21265},
  year    = 2024
}
```

## Acknowledgements
We originally wrote Modula on top of PyTorch, but I ported the project over to JAX inspired by Jack Gallagher's [modulax](https://github.com/GallagherCommaJack/modulax).

## License
Modula is released under an [MIT license](/LICENSE).

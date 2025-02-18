Quickstart
===========

Modula is a neural networks library built on top of `JAX <https://github.com/google/jax>`_.

Installation
-------------

Modula can be installed using pip:

.. code-block:: bash

   pip install git+https://github.com/modula-systems/modula.git

Functionality
--------------

Modula provides a set of architecture-specific helper functions that are automatically constructed along with the network architecture itself. As an example, let's build a multi-layer perceptron:

.. code-block:: python

    from modula.atom import Linear
    from modula.bond import ReLU

    mlp = Linear(10, 256)
    mlp @= ReLU()
    mlp @= Linear(256, 256)
    mlp @= ReLU()
    mlp @= Linear(256, 784)

    mlp.jit() # makes everything run faster

Behind the scenes, Modula builds a function to randomly initialize the weights of the network:

.. code-block:: python

    import jax

    key = jax.random.PRNGKey(0)
    weights = mlp.initialize(key)

Supposing we have used JAX to compute the gradient of our loss and stored this as :code:`grad`, then we can use Modula to dualize the gradient, thereby accelerating our gradient descent training:

.. code-block:: python

    dualized_grad = mlp.dualize(grad)
    weights = [w - 0.1 * dg for w, dg in zip(weights, dualized_grad)]

And after the weight update, we can project the weights back to their natural constraint set:

.. code-block:: python

    weights = mlp.project(weights)

In short, Modula lets us think about the weight space of our neural network as a somewhat classical optimization space, complete with duality and projection operations.

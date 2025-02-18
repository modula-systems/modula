Metrized deep learning
=======================

.. warning::
   This section is under construction.

There are two main academic papers for understanding Modula. The first is called *"Scalable optimization in the modular norm"*. In this paper, we construct a recursive procedure for assigning a norm to the weight space of general neural architectures. Neural networks are automatically Lipschitz and (when possible) Lipschitz smooth in this norm with respect to their weights. The construction also provides means to track input-output Lipschitz properties. The paper is available here:

   | 📘 `Scalable optimization in the modular norm <https://arxiv.org/abs/2405.14813>`_
   |     Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola & Jeremy Bernstein
   |     NeurIPS 2024

The second paper builds on the first and is called *"Modular duality in deep learning"*. In this paper, we take the modular norm and use it to derive optimizers via a procedure called "modular dualization". Modular dualization chooses a weight update :math:`\Delta w` to minimize the linearization of the loss :math:`\mathcal{L}(w)` subject to a constraint on the modular norm :math:`\|\Delta w\|_{M}` of the weight update. In symbols, we solve:

.. math::

   \Delta w = \operatorname{arg min}_{\Delta w : \|\Delta w\|_{M} \leq \eta} \;\langle \Delta w, \nabla \mathcal{L}(w) \rangle,

where :math:`\eta` sets the learning rate. Due to the structure of the modular norm, this duality procedure can be solved recursively leveraging the modular structure of the neural architecture. This procedure leads to modular optimization algorithms, where different layer types can have different optimization rules depending on which norm is assigned to that layer. The paper is available here:

   | 📗 `Modular duality in deep learning <https://arxiv.org/abs/2410.21265>`_
   |     Jeremy Bernstein & Laker Newhouse
   |     arXiv 2024

.. We write neural networks by combining together basic building blocks, usually called layers. Each layer takes an input and a weight vector and produces an output.

..  For example, a linear layer takes an input :math:`x` and produces an output :math:`Wx`, where :math:`W` is a weight matrix. A ReLU nonlinearity takes an input :math:`x` and produces an output :math:`\max(0, x)`.

.. The core idea behind metrized deep learning is to assign norms to the different spaces inside a neural network in a principled way. These norms will enable us to predict the sensitivity of the network's internals under perturbations to both the inputs and the weights. 

.. When we talk about the "different spaces" inside the network, we are referring to:

.. :math:`\mathsf{M}(x; w)`

.. - the input space of each layer
.. - the output space of each layer
.. - the weight space of each layer
.. - the full weight space formed by concatenating the weights of all the layers

.. Here's a simple example of how modular dualization works in practice. Let's create a linear layer and dualize its gradient:

.. .. code-block:: python

..    import modula
..    import torch

..    # Create a linear layer module
..    linear = modula.Linear(784, 10)
   
..    # Create a random gradient
..    grad = torch.randn_like(linear.weight)
   
..    # Dualize the gradient with learning rate 0.1
..    update = linear.dualize(grad, eta=0.1)
   
..    # Apply the update
..    linear.weight.data.add_(update)

.. The ``dualize`` method automatically computes the optimal update direction and step size based on the modular norm of the layer. For a linear layer, this ends up being similar to natural gradient descent, but with automatic step size selection.



.. Suppose you have a neural network :math:`f(x; w)` that takes an input :math:`x` and a vector of parameters :math:`w`. You can think of this network as a function :math:`f : \mathcal{X} \times \mathcal{W} \to \mathcal{Y}` where :math:`\mathcal{X}` is the space of inputs, :math:`\mathcal{W}` is the space of parameters, and :math:`\mathcal{Y}` is the space of outputs.

.. Let's Taylor expand the network in its weights around some point :math:`(x, w)`:

.. .. math::

..    f(x; w + \Delta w) = f(x; w) + \underbrace{\nabla_w f(x; w)^\top \Delta w}_{\text{first-order change}} + \frac{1}{2} \underbrace{\Delta w^\top \nabla^2_w f(x; w) \Delta w}_{\text{second-order change}} + \cdots


.. What if we could bound the linear part and the quadratic part in some norm?

.. .. math::

..    \|\nabla_w f(x; w)^\top \Delta w\|_{\mathcal{Y}} & \leq \mu \cdot \|\Delta w\|_{\mathcal{W}} \\
..    \|\Delta w^\top \nabla^2_w f(x; w) \Delta w\|_{\mathcal{Y}} & \leq \alpha \cdot \|\Delta w\|_{\mathcal{W}}^2

.. Modules
.. --------

.. The modular norm
.. -----------------

.. There are two main academic papers for understanding Modula. The first is called *"Scalable optimization in the modular norm"*. In this paper, we construct a recursive procedure for assigning a norm to the weight space of general neural architectures. Neural networks are automatically Lipschitz and (when possible) Lipschitz smooth in this norm with respect to their weights. The construction also provides means to track input-output Lipschitz properties. The paper is available here:

..    | 📘 `Scalable optimization in the modular norm <https://arxiv.org/abs/2405.14813>`_
..    |     Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola & Jeremy Bernstein
..    |     NeurIPS 2024

.. Modular duality
.. ----------------

.. The second paper builds on the first and is called *"Modular duality in deep learning"*. In this paper, we take the modular norm and use it to derive optimizers via a procedure called "modular dualization". Modular dualization chooses a weight update :math:`\Delta w` to minimize the linearization of the loss :math:`\mathcal{L}(w)` subject to a constraint on the modular norm :math:`\|\Delta w\|_{M}` of the weight update. In symbols, we solve:

.. .. math::

..    \Delta w = \operatorname{arg min}_{\Delta w : \|\Delta w\|_{M} \leq \eta} \;\langle \Delta w, \nabla \mathcal{L}(w) \rangle,

.. where :math:`\eta` sets the learning rate. Due to the structure of the modular norm, this duality procedure can be solved recursively leveraging the modular structure of the neural architecture. This procedure leads to modular optimization algorithms, where different layer types can have different optimization rules depending on which norm is assigned to that layer. The paper is available here:

..    | 📗 `Modular duality in deep learning <https://arxiv.org/abs/2410.21265>`_
..    |     Jeremy Bernstein & Laker Newhouse
..    |     arXiv 2024


.. There are many other papers by myself and other authors that I feel contain important ideas on this topic. Here are some of them:
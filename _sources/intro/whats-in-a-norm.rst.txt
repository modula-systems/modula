What's in a norm?
==================

At the coarsest level, a neural network is just a function that maps an input and a weight vector to an output. Something that we would really like to understand is how the network behaves under perturbation. We would really like to be able to predict things like:

- If I change the input to my network, how much will the output change?
- If I change the weights of my network, how much will the output change?

In fact, we would really like to understand how a neural network behaves if we perturb both the inputs and the weights at the same time! To see why this is important, consider splitting a neural network :math:`f` into two pieces :math:`f = f_\mathrm{head} \circ f_\mathrm{tail}`. During training, if we perturb the weights of both :math:`f_\mathrm{head}` and :math:`f_\mathrm{tail}` simultaneously, then from the perspective of :math:`f_\mathrm{head}` both the inputs and the weights are changing!

Let's start to be a bit more formal. We will think of a neural network as a function :math:`f : \mathcal{X} \times \mathcal{W} \to \mathcal{Y}` that takes an input :math:`x \in \mathcal{X}` and a weight vector :math:`w \in \mathcal{W}` and produces an output :math:`y \in \mathcal{Y}`. If we Taylor expand the network in both the weights and inputs simultaneously, we get:

.. math::

   f(x + \Delta x; w + \Delta w) = f(x; w) + \nabla_w f(x; w)^\top \Delta w + \nabla_x f(x; w)^\top \Delta x +  \cdots.

So the first-order change in the output of the network is described by the two terms :math:`\nabla_w f(x; w)^\top \Delta w` and :math:`\nabla_x f(x; w)^\top \Delta x`. We would like to be able to predict the size of these terms, ideally for any weight perturbation :math:`\Delta w` and any input perturbation :math:`\Delta x`. If we could, we would like to predict the size of the second order terms too. To make progress, we now introduce "metrized deep learning".

Metrized deep learning
-----------------------

Given a neural network :math:`f : \mathcal{X} \times \mathcal{W} \to \mathcal{Y}`, what if we could supply three helpful tools:

- a norm :math:`\|\cdot\|_{\mathcal{X}}` on the input space :math:`\mathcal{X}`,
- a norm :math:`\|\cdot\|_{\mathcal{W}}` on the weight space :math:`\mathcal{W}`,
- a norm :math:`\|\cdot\|_{\mathcal{Y}}` on the output space :math:`\mathcal{Y}`.

These norms would allow us to talk meaningfully about the size of the inputs, the size of the weights and the size of the outputs of the network. Could we find norms that help us achieve our goal, of predicting---or at least bounding---the size of the first order change in the output of the network? Like:

.. math::

   \|\nabla_w f(x; w)^\top \Delta w\|_{\mathcal{Y}} & \leq \mu \cdot \|\Delta w\|_{\mathcal{W}}; \\
   \|\nabla_x f(x; w)^\top \Delta x\|_{\mathcal{Y}} & \leq \nu \cdot \|\Delta x\|_{\mathcal{X}}.

If these bounds hold, then in applied math we would say that the network is *Lipschitz-continuous* with respect to the given norms. If these Lipschitz bounds are to be really useful in helping us design training algorithms and to scale training, we would really like two extra properties to hold:

1. the bounds hold quite tightly for the kinds of perturbations :math:`\Delta w` and :math:`\Delta x` that arise during training;
2. the coefficients :math:`\mu` and :math:`\nu` are *non-dimensional*, meaning they do not depend on width or depth.

If these extra properties hold, then we can really start to think of the weight space norm :math:`\|\cdot\|_{\mathcal{W}}` as a kind of "measuring stick" for designing training algorithms that work well regardless of scale. But it might seem challenging to find norms that satisfy these properties. Afterall, neural networks have a complicated internal structure. And there are a plethora of different architectures to consider. Well, what if we construct a norm as a function of the architecture? This brings us to the *modular norm*!

The modular norm
-----------------

We proposed a procedure for assigning a useful norm to the weight space of general neural architectures. We call this norm the *modular norm*, and neural networks are automatically Lipschitz and (when possible) Lipschitz smooth in the modular norm with respect to their weights. The construction also provides means to track input-output Lipschitz properties. The paper is here:

   | 📘 `Scalable optimization in the modular norm <https://arxiv.org/abs/2405.14813>`_
   |     Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola & Jeremy Bernstein
   |     NeurIPS 2024


The idea of the modular norm is to break up the construction of the neural network into a sequence of "compositions" and "concatenations" of sub-networks that we call "modules", working all the way down to the "atomic modules" which are the individual network layers. If we can specify Lipschitz statements for atomic modules, and show how these statements pass through compositions and concatenations, then we can use the modular norm to produce Lipschitz statements for any network.

Modular dualization
--------------------

Perhaps the most exciting application of the modular norm is the idea of "modular dualization", which is a procedure for automatically constructing architecture-specific optimization algorithms. We describe this procedure in our paper:

   | 📗 `Modular duality in deep learning <https://arxiv.org/abs/2410.21265>`_
   |     Jeremy Bernstein & Laker Newhouse
   |     arXiv 2024


Modular dualization chooses a weight update :math:`\Delta w \in \mathcal{W}` to minimize the linearization of the loss function :math:`\mathcal{L} : \mathcal{W} \to \mathbb{R}` subject to a constraint on the modular norm :math:`\|\Delta w\|_{\mathcal{W}}` of the weight update. Constraining the modular norm of the weight update ensures none of the internal activations of the network change in an unstable way because of the update. In symbols, we choose an update by:

.. math::

   \Delta w = \eta \times \operatorname{arg min}_{t \in \mathcal{W} : \|t\|_{\mathcal{W}} \leq 1} \;\langle t, \nabla \mathcal{L}(w) \rangle,

where :math:`\eta` is the learning rate. Due to the structure of the modular norm, this duality procedure can be solved recursively leveraging the modular structure of the neural architecture. This procedure leads to modular optimization algorithms, where different layer types can have different optimization rules depending on which norm is assigned to that layer. The Modula package implements this procedure.
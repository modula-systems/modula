Reading list
=============

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


There are many other papers by myself and other authors that I feel contain important ideas on this topic. Here are some of them:

Optimization
-------------

- `Preconditioned spectral descent for deep learning <https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html>`_
- `The duality structure gradient descent algorithm: analysis and applications to neural networks <https://arxiv.org/abs/1708.00523>`_
- `On the distance between two neural networks and the stability of learning <https://arxiv.org/abs/2002.03432>`_
- `Automatic gradient descent: Deep learning without hyperparameters <https://arxiv.org/abs/2304.05187>`_
- `A spectral condition for feature learning <https://arxiv.org/abs/2310.17813>`_ 
- `Universal majorization-minimization algorithms <https://arxiv.org/abs/2308.00190>`_
- `Old optimizer, new norm: An anthology <https://arxiv.org/abs/2409.20325>`_
- `Muon: An optimizer for hidden layers in neural networks <https://kellerjordan.github.io/posts/muon/>`_

Generalization
---------------

- `Spectrally-normalized margin bounds for neural networks <https://arxiv.org/abs/1706.08498>`_
- `A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks <https://arxiv.org/abs/1707.09564>`_
- `Investigating generalization by controlling normalized margin <https://arxiv.org/abs/2205.03940>`_

New developments
-----------------

- `Preconditioning and normalization in optimizing deep neural networks <https://github.com/ZQZCalin/trainit/blob/master/optimizers/muon/mango_report.pdf>`_
- `Improving SOAP using iterative whitening and Muon <https://nikhilvyas.github.io/SOAP_Muon.pdf>`_
- `On the concurrence of layer-wise preconditioning methods and provable feature learning <https://arxiv.org/abs/2502.01763>`_
- `A note on the convergence of Muon and further improvements <https://arxiv.org/abs/2502.02900>`_
- `Training deep learning models with norm-constrained LMOs <https://arxiv.org/abs/2502.07529>`_
Stiefel manifold
=================

📚 *This page contains original research. To cite the Modula docs, here's some BibTeX:*

.. code::

   @misc{modula-docs,
      author  = {Jeremy Bernstein},
      title   = {The Modula Docs},
      url     = {https://docs.modula.systems/},
      year    = 2025
   }

On this page we shall consider a problem that I affectionately refer to as *manifold Muon*—or, more formally, the problem of *steepest descent under the spectral norm on the Stiefel manifold*. This problem arises when one is interested in taking the best possible optimization step in a spectral norm geometry (useful for accelerating training) while keeping the size of the weight matrices tightly regulated (potentially helpful for training stability and removing learning rate confounders). This page will generalize the analysis from :doc:`the square case <orthogonal>` to the full Stiefel manifold.

I posed manifold Muon as an open problem on the `Modula docs <https://docs.modula.systems/algorithms/manifold/orthogonal/#open-problem-extending-to-the-stiefel-manifold>`_ earlier this year, and two researchers Franz Louis Cesista (a.k.a. Leloy) and Jianlin Su recently proposed solutions. Leloy proposed a `heuristic solution <https://leloykun.github.io/ponder/steepest-descent-stiefel/>`_ via alternating projections, and Jianlin `solved the problem <https://kexue.fm/archives/11221>`_ by setting up a fixed point iteration. I heard about Leloy's work and an early version of Jianlin's approach (which did not yet work) and managed to solve the problem myself with a slightly different approach based on Lagrangian duality, which I will present in the next section. I also want to acknowledge that `Cédric Simal <https://scholar.google.com/citations?user=Vo3M-WIAAAAJ&hl>`_ independently proposed studying the dual problem to me and Leloy, after I had worked out the following analysis.

Formulating the problem
------------------------

Let's set up the problem mathematically. Say we have a matrix-valued optimization variable :math:`W \in \mathbb{R}^{m \times n}` where, without loss of generality, we take :math:`m\geq n` so that the matrix has more rows than columns. And we have a cost function :math:`\mathcal{C}:\mathbb{R}^{m \times n}\to\mathbb{R}` that we would like to minimize. We would also like to constrain the matrix :math:`W` to the following set:

.. math::

   \mathsf{Stiefel}(m,n) := \left\{ W \in \mathbb{R}^{m \times n} \mid W^T W = I_n \right\}.

This set is known as the *Stiefel manifold*. A matrix :math:`W\in\mathsf{Stiefel}(m,n)` for :math:`m>n` is known as a *semi-orthogonal* matrix—since it has too few columns to form a complete orthonormal basis. There are various alternative ways to characterize the Stiefel manifold. For example, it is equivalently defined as the set of :math:`m \times n` matrices with unit :math:`\ell_2 \to \ell_2` condition number. Suffice to say, the Stiefel manifold is a very well-behaved class of matrices.

We would like to be able to take optimization steps that lie tangent to this manifold. Just as in :doc:`the square case <orthogonal>`, we can show that the tangent space to the Stiefel manifold at semi-orthogonal matrix :math:`W\in\mathsf{Stiefel}(m,n)` is given by the following linear subspace of the ambient matrix space :math:`\mathbb{R}^{m \times n}`:

.. math::

   \mathsf{T}_W \mathsf{Stiefel}(m,n) = \left\{ A \in \mathbb{R}^{m \times n} \mid A^\top W + W^\top A = 0 \right\}.

In the context of Riemannian optimization, there are `established means <https://press.princeton.edu/absil>`_ of projecting the gradient to this linear subspace in order to take steps tangent to the Stiefel manifold. But to make life more interesting, we shall be interested in cost functions with a different sort of structure. In particular, suppose our cost :math:`\mathcal{C}` is Lipschitz-smooth in the *spectral norm*:

.. math::

   \mathcal{C}(W + \Delta W) \leq \mathcal{C}(W) + \langle \nabla \mathcal{C}(W), \Delta W\rangle + \tfrac{1}{2} \cdot \| \Delta W \|_{\mathrm{spectral}}^2,

where :math:`\langle \nabla \mathcal{C}(W), \Delta W\rangle \equiv \operatorname{trace} \nabla \mathcal{C}^\top \Delta W` is the Frobenius inner product between the derivative of the cost and the weight update, measuring the first-order change in cost. To motivate this smoothness structure, observe that matrices in a neural network act as *operators* on vectors, and the spectral norm respects this fact—see our `anthology <https://arxiv.org/abs/2409.20325>`_ for more on this. Spectral norm smoothness suggests taking optimization steps of controlled spectral norm. And since the spectral norm does not emerge from an inner product, spectral norm smoothness takes us outside the realm of Riemannian geometry.

All told, we would like to design a gradient descent algorithm whose updates exploit the spectral norm geometry of the cost function while lying tangent to the Stiefel manifold. Here we focus on the problem of choosing the *direction* of the step given these constraints, and offload the problem of choosing the *magnitude* to the learning rate. We formulate the optimal update direction as the matrix :math:`A` that solves the following minimization problem:

.. _eq-primal:

.. math::
   \min_{A \in \mathbb{R}^{m \times n}} \underbrace{\mathstrut \operatorname{trace}(G^\top A)}_{\text{linearization of cost}} \quad \text{subject to} \quad \underbrace{\|A\|_{\mathrm{spectral}} \leq 1}_{\text{spectral constraint}} \quad \text{and} \quad \underbrace{\mathstrut A^\top W + W^\top A = 0}_{\text{tangent space constraint}}. \qquad (1)

In this expression, :math:`W` is the current point on the manifold, :math:`G := \nabla \mathcal{C}(W)` is shorthand for the derivative of the cost, and :math:`A` is the update direction that we seek. In words, we want to find an update direction that squeezes out the most linear improvement in cost while lying inside the ball of unit spectral norm and also lying tangent to the Stiefel manifold.

Solving manifold Muon via Lagrangian duality
---------------------------------------------

Similar to Jianlin's approach, we introduce a matrix :math:`\Lambda\in\mathbb{R}^{n\times n}` of Lagrange multipliers, and define a Lagrangian function :math:`\mathcal{L}(A, \Lambda)` that incorporates the tangent space constraint:

.. math::

   \begin{align*}
   \mathcal{L}(A, \Lambda) &:= \operatorname{trace} G^\top A + \operatorname{trace}\Lambda^\top (A^\top W + W^\top A) \\
   &= \operatorname{trace}A^\top (G + 2W(\Lambda+\Lambda^\top)),
   \end{align*}

where the second equality follows by applying the cyclic property of the trace and transposing one term. One can check that our original problem :ref:`(1) <eq-primal>` is equivalent to the saddle point problem :math:`\min_{\|A\|_\mathrm{spectral} \leq 1} \max_{\Lambda} \mathcal{L}(A,\Lambda)` since for any :math:`A` that violates the tangent space constraint, the inner maximization with respect to :math:`\Lambda` would send the Lagrangian to infinity. By Sion's minimax theorem, we can swap the order of the :math:`\min` and :math:`\max` to obtain:

.. math::

   \min_{\|A\|_\mathrm{spectral} \leq 1} \max_{\Lambda} \mathcal{L}(A,\Lambda) = \max_{\Lambda} \min_{\|A\|_\mathrm{spectral} \leq 1}  \mathcal{L}(A,\Lambda).

Following `an argument <https://jeremybernste.in/writing/deriving-muon>`_ which is now standard in Muon lore, we recognize the optimal value :math:`A_\mathrm{opt}(\Lambda)` of the primal variable :math:`A` for a given dual variable :math:`\Lambda` as:

.. math::

   A_{\mathrm{opt}}(\Lambda) := \mathop{\mathrm{arg\,min}}_{\|A\|_\mathrm{spectral} \leq 1}  \mathcal{L}(A,\Lambda) = - \operatorname{msign} (G+2W(\Lambda+\Lambda^\top)),

where :math:`\operatorname{msign}` is the *matrix sign function*, defined as the elementwise sign function applied to the singular values of a matrix, or in PyTorch code:

.. code-block:: python

   import torch

   def msign(X):
       U, S, V = torch.svd(X)
       return U @ S.sign().diag() @ V.T

Note that :math:`\operatorname{msign}` can be computed efficiently on GPUs without taking an SVD via `Newton-Schulz iteration <https://arxiv.org/abs/2409.20325>`_ as in the recent `Polar Express <https://arxiv.org/abs/2505.16932>`_ algorithm.

Substituting :math:`A_\mathrm{opt}(\Lambda)` back into the Lagrangian, we uncover the dual problem:

.. _eq-dual:

.. math::

   \max_{\Lambda}\mathcal{L}(A_\mathrm{opt}(\Lambda), \Lambda) = \max_{\Lambda} -\|G + W (\Lambda+\Lambda^\top)\|_\mathrm{nuclear}.

In contrast to the primal problem :ref:`(1) <eq-primal>`, the dual problem is completely unconstrained. We may solve the dual problem by running gradient ascent on the Lagrangian dual function :math:`\mathcal{L}(A_\mathrm{opt}(\Lambda), \Lambda)`—a technique formally known as *dual ascent*. After some work, the gradient of the dual function—or, more precisely, a *subgradient*—is given by the following formula:

.. math::

   \begin{align*}
   H(\Lambda) &:= - \nabla_\Lambda \|G + W (\Lambda+\Lambda^\top)\|_\mathrm{nuclear} \\
   &= - [W^\top\mathrm{msign}(G + 2W (\Lambda+\Lambda^\top)) + \operatorname{msign}(G + 2W (\Lambda+\Lambda^\top))^\top W].
   \end{align*}

To obtain this expression, we have applied the chain rule and the fact that :math:`\operatorname{msign}(X)` is in the subdifferential of :math:`\|X\|_\mathrm{nuclear}`.

This expression for :math:`H(\Lambda)` also has an intuitive interpretation: it measures the deviation of the current setting of :math:`A_\mathrm{opt}(\Lambda)` from satisfying the tangent space condition. `Jianlin's solution <https://kexue.fm/archives/11221>`_ can be interpreted as running a fixed point iteration on the first-order optimality condition for the dual problem: :math:`H(\Lambda_\mathrm{opt}) = 0`. Instead of running this fixed point iteration, we propose a different approach known as *dual ascent*.

The dual ascent algorithm
------------------------

In this section, we write down a gradient ascent algorithm to solve the Lagrangian dual problem. Given a tolerance :math:`\mathtt{tol}>0` and a step size :math:`\alpha>0` for updating the dual variable :math:`\Lambda`, the algorithm is given by:

1. Initialize the dual variable: :math:`\Lambda = -\tfrac{1}{4} \times (W^\top G + G^\top W)`.
2. Compute the candidate update direction: :math:`A = - \operatorname{msign}(G + 2W \Lambda)`.
3. Measure the deviation of :math:`A` from the tangent space: :math:`H = W^\top A + A^\top W`.
4. Check the stopping criterion:

   a. If the deviation is small enough, i.e. :math:`\|H\|_\mathrm{F} / \sqrt{mn} < \mathtt{tol}`, then return :math:`A`.
   b. Otherwise, update the dual variable: :math:`\Lambda \gets \Lambda + \alpha \times H` and go back to step 2.

Observe that the dual variable :math:`\Lambda` remains symmetric throughout this procedure, so we can use :math:`2 \Lambda` in place of :math:`\Lambda + \Lambda^\top` at step 2. The motivation for the special initialization of :math:`\Lambda` is that it leads to the algorithm terminating on the first step if :math:`W` is square. This is because step 2 already recovers the optimal value of :math:`A` :doc:`for the square case <orthogonal>` and so :math:`H=0` at step 3. In actual neural network training, where :math:`G` may not change much between steps because of momentum, it might make more sense to warm start :math:`\Lambda` from the previous iteration.

Once this algorithm terminates, we take the returned value of the primal variable :math:`A` and make the tangent space update :math:`W \gets W + \eta \times A`. The final step is to retract the updated weights back to the manifold. We will work out a retraction map in the next section.

Working out the retraction map
-----------------------------

An update in the tangent space will diverge slightly from the manifold for finite step sizes :math:`\eta`. As such we need to find a retraction map to project the updated weights back to the manifold. It turns out that the retraction map can be implemented in a simple way, by introducing an extra matrix :math:`C` to the update:

.. math::

   W \gets (W + \eta \times A)\cdot C.

We just need to solve for the proper value of :math:`C`. Checking the semi-orthogonality condition and using the fact that :math:`W^\top A + A^\top W = 0` because the update direction :math:`A` belongs to the tangent space, we find that:

.. math::

   \begin{align*}
   C^\top(W - \eta A)^\top (W - \eta A)C &=C^\top [W^\top W - \eta \times [W^\top A + A^\top W] + \eta^2 A^\top A]C \\
   &= C^\top[I_n - A^\top A + (1+\eta^2) \cdot A^\top A]C.
   \end{align*}

Even though :math:`A` is an output of :math:`\operatorname{msign}`, it may not hold that :math:`A^\top A = I_n` because :math:`A` may be low rank. We need to find a matrix :math:`C` satisfying :math:`C^\top[I_n - A^\top A + (1+\eta^2) \cdot A^\top A]C = I_n`. This task is made substantially easier by observing that :math:`A^\top A` and :math:`I_n - A^\top A` are orthogonal projectors. We can then read off a suitable value for :math:`C` as:

.. math::

   C = C^\top = I_n - A^\top A + \frac{A^\top A}{\sqrt{1+\eta^2}}.

While it is nice to have an analytical expression for the retraction map, in practice it might be numerically advantageous just to use :math:`\operatorname{msign}` to project the updated weights back to the manifold.

PyTorch implementation
----------------------

Here we give a basic PyTorch implementation for solving manifold Muon via dual ascent. The code re-uses the ``msign`` function defined earlier in the post.

.. code-block:: python

   import math

   def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
       # Ensure that W and G are both tall matrices
       should_tranpose = W.shape[0] < W.shape[1]
       if should_tranpose:
           W = W.T
           G = G.T
       # Initialize the dual variable
       Lambda = -0.25 * (W.T @ G + G.T @ W)
       # Ascend on the dual problem to find the update direction A
       for step in range(steps):
           # Update the candidate direction A
           A = msign(G + 2 * W @ Lambda)
           # Measure deviation of A from the tangent space:
           H = W.T @ A + A.T @ W
           # Check the stopping criterion
           if torch.norm(H) / math.sqrt(H.numel()) < tol:
               break
           # Update the dual variable
           Lambda -= alpha * (1 - step / steps) * H
       # Descend on the primal problem
       new_W = W - eta * A
       # Retract to the manifold
       new_W += new_W @ A.T @ A * (1/math.sqrt(1 + eta**2) - 1)
       # Restore the shape of the solution and return
       return new_W.T if should_tranpose else new_W

Acknowledgments
----------------

I am grateful to `Leloy <https://leloykun.github.io/ponder/steepest-descent-stiefel/>`_ and `Jianlin Su <https://kexue.fm/archives/11221/>`_ for sharing their excellent work on this topic. I also want to acknowledge `Cédric Simal <https://scholar.google.com/citations?user=Vo3M-WIAAAAJ&hl>`_ who independently proposed studying the dual problem to me, after I had worked out this dual ascent approach. I am incredibly grateful to the team at `Thinking Machines <https://thinkingmachines.ai/>`_ for supporting me to explore this problem. Any mistakes in this writeup are my own responsibility.
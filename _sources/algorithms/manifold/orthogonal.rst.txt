Orthogonal manifold
====================

📚 *This page contains original research. To cite the Modula docs, here's some BibTeX:*

.. code::
   
   @misc{modula-docs,
      author  = {Jeremy Bernstein},
      title   = {The Modula Docs},
      url     = {https://docs.modula.systems/},
      year    = 2025
   }

On this page, we will work out an algorithm for performing gradient descent on the manifold of orthogonal matrices while taking steps that are steepest under the spectral norm. The algorithm will solve for the matrix of unit spectral norm that maximizes the linearized improvement in loss while lying tangent to the manifold. The "retraction map"---which sends the update from the tangent space back to the manifold---involves a few extra matrix multiplications.

Steepest descent on the orthogonal manifold
--------------------------------------------

Consider a square weight matrix :math:`W\in\mathbb{R}^{n \times n}` that is orthogonal, meaning that :math:`W^\top W = I_n`. Suppose that the "gradient matrix" :math:`G\in\mathbb{R}^{n\times n}` is the derivative of some loss function evaluated at :math:`W`. Given step size :math:`\eta > 0`, we claim that the following weight update is steepest under the spectral norm while staying on the orthogonal manifold. First, we take the matrix sign of the skew part of :math:`W^\top G`:

.. math::
   X = \operatorname{msign}[W^\top G - G^\top W],

where the matrix sign :math:`\mathrm{msign}` of a matrix :math:`M` returns the matrix with the same singular vectors as :math:`M` but all positive singular values are set to one. And then we make the update:

.. math::
   W \mapsto W \cdot (I_n - \eta X) \cdot \left(I_n - X^TX + \frac{X^TX}{\sqrt{1+\eta^2}}\right).

The final bracket constitutes the "retraction map", which snaps the updated weights back to the manifold. Curiously, the update can be written in a purely multiplicative form.


Non-Riemannian manifold methods
--------------------------------

One reason this algorithm is interesting is that it is an example of a manifold optimization algorithm that is *non-Riemannian*. A Riemmanian manifold is a manifold equipped with a structure called a *Riemannian metric*, which is an inner product defined at each point on the manifold. The inner product provides a way to measure distance and construct geometry-aware optimization algorithms. There has been a lot of research into Riemannian optimization methods. Some examples in a machine learning context are:

- `Fast and accurate optimization on the orthogonal manifold without retraction <https://arxiv.org/abs/2102.07432>`_;
- `Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform <https://arxiv.org/abs/2002.01113>`_.

However, there has seemingly been much less research into optimization algorithms on manifolds equipped with non-Riemannian structures. For instance, a matrix manifold equipped with the spectral norm at every point is non-Riemannian since the spectral norm does not emerge from an inner product. But we believe these kinds of non-Riemannian geometries are very important in deep learning.

The structure of the tangent space
-----------------------------------

We would like to make a weight update so that the updated weights stay on the orthogonal manifold. First we need to figure out the structure of the "tangent space" at a point on the manifold. Roughly speaking, the tangent space is the set of possible velocities a particle could have as it passes through that particular point. So we need to consider all curves passing through the point on the manifold.

If we consider a curve :math:`W(t)` on the manifold parameterized by time :math:`t \in \mathbb{R}`, then this curve must satisfy :math:`W(t)^\top W(t) = I_n`. Differentiating with respect to :math:`t`, we find that the velocity must satisfy:

.. math::

   \frac{\partial W(t)}{\partial t}^\top W(t) + W(t)^\top \frac{\partial W(t)}{\partial t} = 0.

So to be in the tangent space of a point :math:`W` on the manifold, a matrix :math:`A` must satisfy :math:`A^\top W + W^\top A = 0`. Conversely, if a matrix :math:`A` satisfies :math:`A^\top W + W^\top A=0`, then it is the velocity of a curve on the manifold that passes through :math:`W`, as evidenced by the curve :math:`W(t) = W \exp(tW^\top A)`. Therefore, the tangent space at :math:`W` is completely characterized by the set:

.. math::

   \{A\in \mathbb{R}^{n\times n}:A^\top W + W^\top A = 0\}.

Finally, if we use the orthogonal matrix :math:`W` to make the change of variables :math:`A = W X`, then we see that :math:`A` belongs to the tangent space at :math:`W` if and only if :math:`X` is skew-symmetric: :math:`X^\top + X = 0`. So the tangent space to the orthogonal manifold can be parameterized by skew-symmetric matrices.


Steepest direction in the tangent space
----------------------------------------

We will solve for the matrix :math:`A` that belongs to the tangent space to the orthogonal manifold at matrix :math:`W` and maximizes the linearized improvement in loss :math:`\operatorname{trace}(G^\top A)` under the constraint that :math:`A` has unit spectral norm. Formally, we wish to solve:

.. math::

   \operatorname{arg max}_{A\in \mathbb{R}^{n\times n}: \|A\|_*\leq 1 \text{ and } A^\top W + W^\top A = 0}\; \operatorname{trace}(G^\top A).

To simplify, we make the change of variables :math:`A = W X` so that we now only need to maximize over skew-symmetric matrices :math:`X` of unit spectral norm:

.. math::

   \operatorname{arg max}_{X\in \mathbb{R}^{n\times n}:\|X\|_*\leq 1 \text{ and } X^\top + X= 0}\; \operatorname{trace}([W^\top G]^\top X).

Next, we decompose :math:`W^\top G = \frac{1}{2}[W^\top G + G^\top W] + \frac{1}{2}[W^\top G - G^\top W]` into its symmetric and skew-symmetric components and realize that, because :math:`X` is skew-symmetric, the contribution to the trace from the symmetric part of :math:`W^\top G` vanishes. So the problem becomes:

.. math::
   \operatorname{arg max}_{X\in \mathbb{R}^{n\times n}:\|X\|_*\leq 1 \text{ and } X^\top + X= 0}\; \operatorname{trace}\left(\left[\frac{W^\top G - G^\top W}{2}\right]^\top X\right).

If we simply ignore the skew-symmetric constraint, the solution for :math:`X` is given by :math:`X = \operatorname{msign}[W^\top G - G^\top W]`. But this solution for :math:`X` actually satisfies the skew-symmetric constraint! This is because the matrix sign function preserves skew-symmetry. An easy way to see this is that :math:`\operatorname{msign}[W^\top G - G^\top W]` can be computed by running an odd polynomial iteration (see :doc:`Newton-Schulz <../newton-schulz>`) on :math:`W^\top G - G^\top W`, and odd polynomials preserve skew-symmetry. [#youla]_

Undoing the change of variables, our tangent vector is given by :math:`A = W \cdot \operatorname{msign}[W^\top G - G^\top W]`.


Finding the retraction map
---------------------------

The previous section suggests making the weight update :math:`W \mapsto W - \eta W X = W (I_n - \eta X)`. This update takes a step in the tangent space, which diverges slightly from the orthogonal manifold for finite step sizes. A relatively expensive way to fix this issue is to just apply the matrix sign function, i.e. :math:`W \mapsto \operatorname{msign}[W (I_n - \eta X)]`, to project the weights back to the manifold. But we will show in this section that there is actually a shortcut. 

As a warmup, let's first consider the case that :math:`W^\top G - G^\top W` is full rank. Then :math:`X` is an orthogonal matrix and :math:`[W (I_n - \eta X)]^\top [W (I_n - \eta X)] = (1 + \eta^2) I_n`. Therefore, in this case, we can project back to the manifold simply by dividing the updated weights through by the scalar :math:`\sqrt{1+\eta^2}`. 

In the general case where :math:`W^\top G - G^\top W` and therefore :math:`X = \operatorname{msign}[W^\top G - G^\top W]` may not be full rank, let us search for a matrix :math:`C` such that :math:`W \cdot (I_n - \eta X) \cdot C` is orthogonal. Checking the orthogonality condition yields:

.. math::

   (W \cdot (I_n - \eta X) \cdot C)^\top (W \cdot (I_n - \eta X) \cdot C) = C^\top (I_n + \eta^2 X^\top X) C.

The trick is to recognize :math:`X^\top X` as the orthogonal projector on to the row space of :math:`X`. The matrix :math:`X` conserves vectors in the null space of :math:`X` but scales up vectors in the row space of :math:`X` by a factor of :math:`1+\eta^2`. It therefore suffices to choose a matrix :math:`C` that inverts this transformation in two steps. Noting that :math:`I_n - X^\top X` projects on to the null space of :math:`X`, the following choice of :math:`C` is what we need:

.. math::

   C = C^\top = I_n - X^\top X + \frac{X^TX}{\sqrt{1+\eta^2}}.


Python code
---------------------------

Here is a basic JAX implementation for the algorithm:

.. code-block:: python

   import jax.numpy as jnp
   import math

   def orthogonalize(M, steps = 10):
      a, b, c = 3, -16/5, 6/5
      transpose = M.shape[1] > M.shape[0]
      if transpose:
         M = M.T
      M = M / jnp.linalg.norm(M)
      for _ in range(steps):
         A = M.T @ M
         I = jnp.eye(A.shape[0])
         M = M @ (a * I + b * A + c * A @ A)
      if transpose:
         M = M.T
      return M

   def update(W, G, eta, NS_steps=10):
      I = jnp.eye(d)
      X = orthogonalize(W.T @ G - G.T @ W, NS_steps)
      retraction_factor = I - (1 - math.sqrt(1/(1+eta**2))) * X.T @ X
      return W @ (I - eta * X) @ retraction_factor


Open problem: Extending to the Stiefel Manifold
------------------------------------------------

I initially thought that this solution easily extended to the *Stiefel manifold*—i.e. the set of :math:`m \times n` semi-orthogonal matrices. But this turns out not to be the case: the algorithm we derived is generally not optimal if :math:`W` is rectangular. To see this, let's consider an :math:`m \times n` matrix :math:`W` with :math:`m > n`, and suppose that it belongs to the Stiefel manifold :math:`W^\top W = I_n`. The problem with our derivation is that the change of variables :math:`A = W X` no longer parameterizes the full set of :math:`m \times n` matrices. Instead, we need to make the change of variable :math:`A = WX + \overline{W}Y` where the columns of :math:`\overline{W}` are the "missing" columns of :math:`W`. In other words, the combined matrix :math:`[W | \overline{W}]` is a square orthogonal matrix. For this parameterization, the tangent space to the Stiefel manifold is obtained by requiring that :math:`X\in\mathbb{R}^{n\times n}` is skew-symmetric while :math:`Y\in\mathbb{R}^{(m-n)\times n}` is completely unconstrained. I do not know how to analytically solve the resulting maximization problem in this parameterization.

.. [#youla] In fact, any odd function applied entrywise to the singular values of a matrix will preserve skew symmetry. To see this, one needs to understand the spectral structure of skew-symmetric matrices. An :math:`n\times n` matrix :math:`X` is skew symmetric if and only if it can be written :math:`X = \sum_{i=1}^k \sigma_i (u_iv_i^\top - v_i u_i^\top)`, where the :math:`\sigma_i` are non-negative, the :math:`\{u_i\}\cup\{v_i\}` are all orthonormal and :math:`k \leq \lfloor n/2 \rfloor`. In other words, :math:`X` must admit an SVD where the singular values come in pairs with conjugate singular vectors. But applying an odd function :math:`f` to the singular values yields :math:`\sum_{i=1}^k f(\sigma_i) (u_iv_i^\top - v_i u_i^\top)`, which leaves the skew-symmetric structure intact. For more reading on the spectral structure of skew-symmetric matrices, see `(Haber, 2016) <https://scipp.ucsc.edu/~haber/ph218/pfaffian15.pdf>`_ or `(Youla, 1961) <https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/normal-form-for-a-matrix-under-the-unitary-congruence-group/964D0AA8DAC0CDB9079F04331B61859D>`_.
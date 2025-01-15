Hypersphere
============

On this page, we will work out an algorithm for performing steepest descent under the Euclidean norm on the hypersphere. While this algorithm may seem obvious, it is a good warmup and the technical scaffolding will help in more complicated examples.

Steepest descent on the hypersphere
------------------------------------

Consider a weight vector :math:`w\in\mathbb{R}^{n}` on the unit hypersphere, meaning that the squared Euclidean norm :math:`\|w\|_2^2 = \sum_{i=1}^n w_i^2 = 1`. Suppose that the "gradient vector" :math:`g\in\mathbb{R}^{n}` is the derivative of some loss function evaluated at :math:`w`. Given step size :math:`\eta > 0`, we claim that the following weight update is steepest under the Euclidean norm while staying on the unit hypersphere:

.. math::
   w \mapsto \frac{1}{\sqrt{1+\eta^2}} \times \left[w - \eta \times \frac{ (I_n - w w^\top)g}{\left\|(I_n - w w^\top)g\right\|_2}\right].

So we simply project the gradient on to the subspace orthogonal to the weight vector and normalize to obtain a unit vector. We offload the problem of setting the size of the update to choosing the step size parameter :math:`\eta`. Dividing through by :math:`\sqrt{1 + \eta^2}` projects the update back to the hypersphere.

The structure of the tangent space
-----------------------------------

The tangent space to the unit hypersphere at vector :math:`w` is simply the set of vectors orthogonal to :math:`w`:

.. math::
   \{ a \in \mathbb{R}^n : w^\top a = 0 \}.

While it's probably overkill, let's show this formally. The tangent space at :math:`w` is the set of possible velocities of curves passing through :math:`w`. For a real-valued parameter :math:`t`, consider a curve :math:`w(t)` on the unit hypersphere. If we differentiate the condition :math:`w(t)^\top w(t) = 1`, we find that :math:`\frac{\partial w(t)}{\partial t}^\top w(t) = 0`. This means that a tangent vector at :math:`w` must be orthogonal to :math:`w`. Conversely, if a vector :math:`a` satisfies :math:`a^\top w = 0`, then :math:`a` is a tangent vector to the manifold at :math:`w`, as can be seen by studying the curve :math:`w(t) = w\cdot cos(t) + a\cdot sin(t)` at :math:`t = 0`. So the tangent space really is :math:`\{ a \in \mathbb{R}^n : w^\top a = 0 \}`.

Steepest direction in the tangent space
----------------------------------------

To find the steepest direction in the tangent space under the Euclidean norm, we must solve:

.. math::
   \operatorname{arg max}_{a \in \mathbb{R}^n: \|a\|_2\leq 1 \text{ and } a^\top w = 0}\; g^\top a.

We can solve this problem using the method of Lagrange multipliers. We write down the Lagrangian:

.. math::
   \mathcal{L}(a, \lambda, \mu) = g^\top a - \frac{\lambda}{2} a^\top a - \mu\,a^\top w.

Taking the derivative with respect to :math:`a` and setting to zero, we find that :math:`a = (g - \mu w) / \lambda`. We can solve for :math:`\lambda` and :math:`\mu` by substituting in the constraints that :math:`a^\top a = 1` and :math:`a^\top w = 0`. Finally, we obtain:

.. math::
   a = \frac{ (I_n - w w^\top)g}{\left\|(I_n - w w^\top)g\right\|_2}.

Finding the retraction map
---------------------------

Making a weight update :math:`w \mapsto w - \eta\cdot a` along the steepest direction in the tangent space that we calculated in the previous section will leave the hypersphere. In fact, by Pythagoras' theorem, we have that :math:`\|w - \eta\cdot a\|_2 = \sqrt{1 + \eta^2}`. So to project the update back to the manifold, we can simply divide through by this scalar:

.. math::
   w \mapsto \frac{1}{\sqrt{1+\eta^2}} \times \left[w - \eta \times \frac{ (I_n - w w^\top)g}{\left\|(I_n - w w^\top)g\right\|_2}\right].


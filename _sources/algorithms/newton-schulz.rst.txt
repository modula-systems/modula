Newton-Schulz
==============

On this page, we will work out a family of iterative algorithms for "orthogonalizing" a matrix, by which we mean transforming either the rows or the columns of the matrix to form an orthonormal set of vectors. These so-called "Newton-Schulz" iterations are a useful family of algorithms to keep in your toolbox. We proposed using these iterations for neural net optimization in our paper:

   | 📗 `Modular duality in deep learning <https://arxiv.org/abs/2410.21265>`_
   |     Jeremy Bernstein & Laker Newhouse
   |     arXiv 2024

Before that, we included the iteration in an appendix of our `workshop paper <https://arxiv.org/abs/2409.20325>`_, and before that I actually worked out `the ideas <https://x.com/jxbz/status/1821610284223791156>`_
`directly <https://x.com/jxbz/status/1821685090709336319>`_
`on <https://x.com/jxbz/status/1824076109647925260>`_ `Twitter <https://x.com/tmjlarge/status/1824243567037972768>`_ with my collaborator Tim Large. We used a particular `cursed quintic iteration <#a-cursed-quintic-iteration>`_ in the Muon optimizer, which was used to set speed records for training NanoGPT:

   | 📕 `Muon: An optimizer for hidden layers in neural networks <https://kellerjordan.github.io/posts/muon/>`_
   |     Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse & Jeremy Bernstein
   |     blog post 2024

Since then, the iteration has been applied in new optimizers such as `Scion <https://arxiv.org/abs/2502.07529>`_, `improved-SOAP <https://nikhilvyas.github.io/SOAP_Muon.pdf>`_ and `Mango <https://github.com/ZQZCalin/trainit/blob/master/optimizers/muon/mango_report.pdf>`_. At the bottom of this page, we provide further `historical connections <#id1>`_ on the techniques.

Problem statement
-----------------

We wish to approximate the map that sends a matrix :math:`M\in\mathbb{R}^{m\times n}` with reduced SVD :math:`M = U \Sigma V^\top` to the matrix :math:`U V^\top`. This map can be thought of as "snapping the singular values of :math:`M` to one"---with the exception that the iterations we consider will actually fix zero singular values at zero. But ignoring this detail, the map is given by:

.. math::
   M = U \Sigma V^\top \mapsto U V^\top.

This operation is sometimes referred to as `"symmetric orthogonalization" <https://en.wikipedia.org/wiki/Orthogonalization>`_ because no row or column of the matrix :math:`M` is treated as special in the procedure. This is in contrast to `Gram-Schmidt orthogonalization <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`_, which involves first picking out a certain row or column vector as special and then orthogonalizing the remaining vectors against this vector.


.. Why care about symmetric orthogonalization?
.. --------------------------------------------

.. To train a neural network stably, it is desirable that the outputs of the layers evolve in a controlled fashion during training. We argued in our first paper on `distance measures between neural networks <https://arxiv.org/abs/2002.03432v1>`_ that a good way to achieve this is to control the change in singular values of the weight matrices, and in our paper on `a spectral condition for feature learning <https://arxiv.org/abs/2310.17813>`_ we proposed controlling the spectral norms of the weight upates. Following on from this, it makes sense to ask "what is the largest weight update we can make to a layer that has a given spectral norm?" This question is answered by taking the sharp operator of the gradient matrix. Formally, for a matrix :math:`G\in\mathbb{R}^{m\times n}` thought of as the gradient of a loss function, the sharp operator solves the following problem:

.. .. math::
..    G^\sharp = \operatorname{arg max}_{T \in \mathbb{R}^{m\times n} \,:\, \|T\|_* \leq 1} \langle G , T \rangle,

.. where :math:`\langle \cdot, \cdot \rangle` denotes the Frobenius inner product and :math:`\|\cdot\|_*` denotes the spectral norm. In words, the sharp operator tells us the direction :math:`T` in matrix space that squeezes out the most linearized change in loss :math:`\langle G, T \rangle` while keeping the spectral norm under control. Keeping the spectral norm of the weight update under control is important as it allows us to guarantee that the features of the model change by a controlled amount.

Odd polynomial iterations
-------------------------

We will consider iterations based on odd matrix polynomials of the form:

.. math::
   p(X) = a X + b X X^\top X + c (X X^\top)^2 X + ...

which acts on a matrix :math:`X \in \mathbb{R}^{m \times n}`. The important property of an odd matrix polynomial of this form is that it *commutes* with the singular value decomposition, in the sense that:

.. math::
   p(U \Sigma V^\top) = U p(\Sigma) V^\top.

So, to apply an odd polynomial :math:`p` to the singular values, it is enough to apply it to the overall matrix :math:`X`. Since the matrix of singular values :math:`\Sigma` is diagonal, this reduces to applying the scalar polynomial

.. math::
   f(x) = a x + bx^3 + cx^5 + ...

to the diagonal entries of :math:`\Sigma`. In what follows we will simply specify formulae for scalar polynomials :math:`f` with the understanding that they will be extended to matrix polynomials :math:`p` as specified above. Then our task is just to produce odd scalar polynomials :math:`f(x)` that when iterated like :math:`f \circ f \circ f \circ ... \circ f(x)` converge to the sign function :math:`\operatorname{sign}(x)`.

While the classical way to select these polynomials is by Taylor-approximating the sign function `(e.g. Björck & Bowie, 1971) <https://www.jstor.org/stable/2949484>`_, this only leads to a very restricted set of polynomials. I had the idea of treating the polynomial coefficients as free parameters to be tuned for specific performance characteristics.

A cubic iteration
------------------

We begin with the simplest Newton-Schulz iteration, based on the cubic polynomial:

.. math::
    f(x) = \frac{3}{2}x - \frac{1}{2}x^3.

We plot :math:`f(x)` on the left and on the right we plot :math:`f(x)` iterated five times to yield :math:`f(f(f(f(f(x)))))`.

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/qzvof94grn?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/2d0ekimums?embed" width="47%" height="300px" frameborder="0"></iframe>

As can be seen, by iterating :math:`f` several times, the graph starts to resemble that of the sign function :math:`\operatorname{sign}(x)`, at least on the interval close to the origin. In fact, you can check that if we iterate :math:`f` an infinite number of times, we will obtain precisely the sign function on the interval :math:`[-\sqrt{3},\sqrt{3}]`. As a consequence, if we iterate the corresponding matrix polynomial :math:`p(X) = \frac{3}{2}X - \frac{1}{2}XX^\top X`, we will approximate the sign function element-wise on the singular values of :math:`X`, thereby orthogonalising the matrix. The only caveat is that we need to ensure all singular values of the initial matrix lie in the interval :math:`[-\sqrt{3},\sqrt{3}]`. We can achieve this via a simple pre-processing step, mapping :math:`X \mapsto X / \|X\|_F`.
 
A quintic iteration
--------------------

Using a higher-order polynomial provides more degrees of freedom in our design space, which we can use to obtain faster convergence. In this section, we consider the quintic iteration given by:

.. math::
    f(x) = 3x - \frac{16}{5}x^3 + \frac{6}{5}x^5.

Again, we plot one and five iterations of this polyomial:

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/fjjjpsnl2g?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/1aqrfjge22?embed" width="47%" height="300px" frameborder="0"></iframe>

As can be seen, after 5 iterations the quintic iteration has achieved a substantially closer approximation to the sign function than the cubic iteration, at least on the interval :math:`[-3/2,3/2]`. However, the approximation exhibits some oscillatory behaviour close to the origin.

A cursed quintic iteration
---------------------------

We applied a Newton-Schulz iteration in the `Muon optimizer <https://kellerjordan.github.io/posts/muon/>`_ used in the `NanoGPT speedrun <https://github.com/KellerJordan/modded-nanogpt>`_. Keller experimented with tuning the coefficients in the iteration and found that the most important thing for fast convergence of the optimizer was to inflate the small singular values as fast as possible. To keep the wall-clock time low, we needed to do this in the smallest number of iterations possible. This is achieved by making the first coefficient in the polynomial as large as possible, thereby maximizing the slope of the polynomial at :math:`x=0`. I had the idea for using a non-convergent iteration, and Keller settled on the following:

.. math::
    f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5.

Plotting the polynomial after one and five iterations, we see some peculiar behaviour:

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/4xsjfwa5vh?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/9yjpijk1fv?embed" width="47%" height="300px" frameborder="0"></iframe>

This iteration *oscillates* and in fact *does not converge*! To see why, observe that a convergent iteration must at the very least satisfy :math:`f(1) = 1` so that :math:`x=1` is a fixed point. In turn, this implies that the sum of the coefficients should equal 1. But for Keller's polynomial, the coefficients sum to 

.. math::
   3.4445 - 4.7750 + 2.0315 = 0.701 \neq 1.

In short, the cursed quintic iteration sacrifices convergence for speed.

Jiacheng's six-step
--------------------

Another idea is that we do not need to use the same polynomial coefficients at each step of the iteration. Using different coefficients at each step provides even more degrees of freedom to achieve desirable behaviour. For example, `You Jiacheng <https://twitter.com/YouJiacheng>`_ found the following iteration `by computational search <https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae>`_:

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/a2jxvmompc?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/hta59syabv?embed" width="47%" height="300px" frameborder="0"></iframe>

The six quintic polynomials on the left are executed in sequence. The last quintic (the orange curve) has a very flat slope close to :math:`x=1`, which has the effect of flattening out the oscillations present after the first five steps. The result is the very good approximation to the sign function on the right.

I have integrated this iteration as a basic primitive in `Modula <https://github.com/modula-systems/modula/blob/aed70cddf2d3ab74fa218b1377840d1fd795cfcf/modula/atom.py#L6C1-L31C13>`_ using the following JAX implementation:

.. code-block:: python

   import jax.numpy as jnp

   def orthogonalize(M):
      # by @YouJiacheng (with stability loss idea from @leloykun)
      # https://twitter.com/YouJiacheng/status/1893704552689303901
      # https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae

      abc_list = [
         (3955/1024, -8306/1024, 5008/1024),
         (3735/1024, -6681/1024, 3463/1024),
         (3799/1024, -6499/1024, 3211/1024),
         (4019/1024, -6385/1024, 2906/1024),
         (2677/1024, -3029/1024, 1162/1024),
         (2172/1024, -1833/1024,  682/1024)
      ]

      transpose = M.shape[1] > M.shape[0]
      if transpose:
         M = M.T
      M = M / jnp.linalg.norm(M)
      for a, b, c in abc_list:
         A = M.T @ M
         I = jnp.eye(A.shape[0])
         M = M @ (a * I + b * A + c * A @ A)
      if transpose:
         M = M.T
      return M

I also recommend checking out the `writeup <https://leloykun.github.io/ponder/muon-opt-coeffs/>`_ of Franz Cesista (@leloykun) on this topic.

Designing your own iteration
-----------------------------

Designing these polynomial iterations can be a surprisingly fun exercise. If you'd like to explore designing your own iteration, you can start with a polynomial of the form:

.. math::
   f(x) = a x + b x^3 + c x^5 + d x^7 + e x^9 + ...

And then choose the coefficients :math:`a,b,c,d,e,...` to achieve your desired behaviour. Three important things to consider are:

- What order do you want to truncate your polynomial at? A higher-order iteration can converge in fewer steps, but each step is more expensive. There is a trade-off here.
- Do you want the iterations to converge? If so, you at least need to enforce that the coefficients sum to 1 so that :math:`f(1) = 1`. You could consider enforcing additional derivative conditions, such as that :math:`\partial f / \partial x = 0` at :math:`x=1`, to further stabilize the convergence.
- Do you want to use different polynomials on different steps? This adds complexity but provides a larger design space.

After making these decisions, you may have leftover degrees of freedom. A fun way to fix these degrees of freedom is to open up `Desmos <https://desmos.com>`_ and play around with the coefficients using sliders.

Historical connections
----------------------

The procedure of symmetric orthogonalization appears in a number of different contexts:

- it is used in solving the `orthogonal Procrustes problem <https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem>`_.
- it computes the "orthogonal polar factor" in the `polar decomposition <https://en.wikipedia.org/wiki/Polar_decomposition>`_ of a matrix.
- it was used by `Per-Olov Löwdin <https://en.wikipedia.org/wiki/Per-Olov_L%C3%B6wdin>`_ in the 1950s to perform atomic and molecular orbital calculations.
- it is used for doing `Frank-Wolfe optimization <https://proceedings.mlr.press/v28/jaggi13>`_ over the spectral norm ball.
- it was proposed for deep learning optimization in the paper `"preconditioned spectral descent for deep learning" <https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html>`_---albeit computed via matrix sketching rather than Newton-Schulz iterations.
- A Newton-Schulz iteration was used to orthogonalize the weight matrices (but not the updates!) in deep learning in the paper `"sorting out Lipschitz function approximation" <https://arxiv.org/abs/1811.05381>`_.

The earliest references on the Newton-Schulz iteration itself seem to be `"some iterative methods for improving orthonormality" <https://epubs.siam.org/doi/10.1137/0707031>`_ (Kovarik, 1970) and `"an iterative algorithm for computing the best estimate of an orthogonal matrix" <https://www.jstor.org/stable/2949484>`_ (Björck & Bowie, 1971). To justify using the name "Newton-Schulz" for these iterations, we note that Higham used it in `these slides <https://convexoptimization.com/TOOLS/procrust94.pdf>`_. The idea of graphically tuning the coefficients of the iteration to obtain certain performance characteristics is, to the best of my knowledge, our own original idea.
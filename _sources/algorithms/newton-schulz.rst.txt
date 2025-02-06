Newton-Schulz
==============

On this page, we will work out a family of iterative algorithms for "orthogonalizing a matrix", by which we mean transforming either the rows or the columns of the matrix to form an orthonormal set of vectors. 
In particular, we will consider the map that sends a matrix :math:`M\in\mathbb{R}^{m\times n}` with reduced SVD :math:`M = U \Sigma V^\top` to the matrix :math:`U V^\top`. This operation can be thought of as "snapping the singular values of :math:`M` to one"---although the iterations we consider will actually fix zero singular values at zero. We will refer to the orthogonalized matrix corresponding to :math:`M` as :math:`M^\sharp`---pronounced "M sharp"---so that:

.. math::
   M = U \Sigma V^\top \mapsto M^\sharp = U V^\top.

This "sharp operation" is sometimes referred to as `"symmetric orthogonalization" <https://en.wikipedia.org/wiki/Orthogonalization>`_ because no row or column of the matrix :math:`M` is treated as special in the procedure. This is in contrast to `Gram-Schmidt orthogonalization <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`_, which involves first picking out a certain row or column vector as special and then orthogonalizing the remaining vectors against this vector.


Steepest descent under the spectral norm
-----------------------------------------

The reason we care about orthogonalization and the sharp operator in the context of neural network optimization is that it is an essential primitive for solving the problem of "steepest descent under the spectral norm". For a matrix :math:`G\in\mathbb{R}^{m\times n}` thought of as the gradient of a loss function, the sharp operator solves the following problem:

.. math::
   G^\sharp = \operatorname{arg max}_{\Delta W \in \mathbb{R}^{m\times n} \,:\, \|\Delta W\|_* \leq 1} \langle G , \Delta W \rangle,

where :math:`\langle \cdot, \cdot \rangle` denotes the Frobenius inner product and :math:`\|\cdot\|_*` denotes the spectral norm. In words, the sharp operator tells us the direction :math:`\Delta W` in matrix space that squeezes out the most linearized change in loss :math:`\langle G, \Delta W \rangle` while keeping the spectral norm under control. Keeping the spectral norm of the weight update under control is important as it allows us to guarantee that the features of the model change by a controlled amount.

Historical connections
-----------------------

The procedure of symmetric orthogonalization appears in a number of different contexts:

- it is used to solve the `orthogonal Procrustes problem <https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem>`_.
- it is used to compute the "orthogonal polar factor" in the `polar decomposition <https://en.wikipedia.org/wiki/Polar_decomposition>`_ of a matrix.
- it was used by `Per-Olov Löwdin <https://en.wikipedia.org/wiki/Per-Olov_L%C3%B6wdin>`_ in the 1950s to perform atomic and molecular orbital calculations.
- it is used in `Frank-Wolfe optimization <https://proceedings.mlr.press/v28/jaggi13>`_ over the spectral norm ball.
- `Preconditioned Spectral Descent for Deep Learning <https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html>`_.

Newton-Schulz iteration

- kovarik, bjorck and bowie
- higham: newton-schulz
- anil and grosse: for weights not updates

Polynomial iterations
---------------------

A cubic iteration
------------------

.. math::
    f(x) = \frac{3}{2}x - \frac{1}{2}x^3

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/qzvof94grn?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/2d0ekimums?embed" width="47%" height="300px" frameborder="0"></iframe>

some more text

A quintic iteration
--------------------

.. math::
    f(x) = 3x - \frac{16}{5}x^3 + \frac{6}{5}x^5

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/fjjjpsnl2g?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/1aqrfjge22?embed" width="47%" height="300px" frameborder="0"></iframe>

A speedy iteration
-------------------

.. math::
    f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/4xsjfwa5vh?embed" width="47%" height="300px" frameborder="0" style="margin-right: 4%"></iframe>
   <iframe src="https://www.desmos.com/calculator/9yjpijk1fv?embed" width="47%" height="300px" frameborder="0"></iframe>

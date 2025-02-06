Newton-Schulz
==============

.. admonition:: Warning
   :class: warning

   This page is still under construction.

History of orthogonalization
----------------------------

- procrustes problem
- loewdin symmetrization
- sharp-operator: frank-wolfe? nesterov?
- neural nets: carlin

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

   <iframe src="https://www.desmos.com/calculator/qzvof94grn?embed" width="48%" height="300px" frameborder="0"></iframe>&nbsp;&nbsp;&nbsp;
   <iframe src="https://www.desmos.com/calculator/2d0ekimums?embed" width="48%" height="300px" frameborder="0"></iframe>

some more text

A quintic iteration
--------------------

.. math::
    f(x) = 3x - \frac{16}{5}x^3 + \frac{6}{5}x^5

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/fjjjpsnl2g?embed" width="48%" height="300px" frameborder="0"></iframe>&nbsp;&nbsp;&nbsp;
   <iframe src="https://www.desmos.com/calculator/1aqrfjge22?embed" width="48%" height="300px" frameborder="0"></iframe>

A speedy iteration
-------------------

.. math::
    f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5

.. raw:: html

   <iframe src="https://www.desmos.com/calculator/4xsjfwa5vh?embed" width="48%" height="300px" frameborder="0"></iframe>&nbsp;&nbsp;&nbsp;
   <iframe src="https://www.desmos.com/calculator/9yjpijk1fv?embed" width="48%" height="300px" frameborder="0"></iframe>

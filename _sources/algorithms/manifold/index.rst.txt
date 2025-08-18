Manifold duality maps
======================

.. toctree::
   :hidden:

   hypersphere
   orthogonal
   stiefel

In this section we will derive steepest descent optimization algorithms for manifolds equipped with certain norms. These algorithms will be useful if we want to construct steepest descent optimizers for modules where the tensors obey some natural constraints. In particular, we will consider:

For vectors:

- :doc:`steepest descent under the Euclidean norm on the hypersphere; <hypersphere>`

For matrices:

- :doc:`steepest descent under the spectral norm on the orthogonal manifold. <orthogonal>`
- :doc:`steepest descent under the spectral norm on the Stiefel manifold. <stiefel>`

In each case, we will adopt the following strategy:

1. characterize the "tangent space" to the manifold;
2. solve for the steepest direction in the tangent space under the given norm;
3. work out a "retraction map", which projects a step taken in the tangent space back to the manifold.

We can think of the tangent space to the manifold as a plane lying tangent to the manifold. So each point on the manifold has its own tangent space. Since the manifold is curved, taking a discrete step in the tangent space will leave the manifold. Therefore we need to project back to the manifold using the retraction map. It's always helpful to keep in mind the following picture:

.. plot:: figure/tangent.py


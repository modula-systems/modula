Welcome to the Modula docs!
============================

Modula is a deep learning framework and a deep learning theory built hand-in-hand. The central idea of Modula is to metrize the neural architecture and construct the corresponding duality theory. This is leading to training algorithms that are faster and more intrinsically scalable. But I anticipate other benefits too.

Purpose of the docs
^^^^^^^^^^^^^^^^^^^^

I'm currently in the process of overhauling these docs. But the idea is to create a central place to learn about the theory, algorithms and code behind Modula. I hope that this will help inspire further research into metrized deep learning.

If something is unclear, first check `the FAQ <faq>`_, but then consider starting a `GitHub issue <https://github.com/jxbz/modula/issues>`_, making a `pull request <https://github.com/jxbz/modula/pulls>`_ or reaching out by email. Then we can improve the docs for everyone.

Navigating the docs
^^^^^^^^^^^^^^^^^^^^

You can use the :kbd:`←` and :kbd:`→` arrow keys to jump around the docs. You can also use the side panel.

Citing the docs
^^^^^^^^^^^^^^^^

The docs currently contain some original research contributions not published anywhere else---in particular, the section on manifold duality maps. If you want to cite the docs, here's some BibTeX:

.. code::
    
    @misc{modula-docs,
      author  = {Jeremy Bernstein},
      title   = {The Modula Docs},
      url     = {https://docs.modula.systems/},
      year    = 2025
    }

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction:

   intro/reading-list

.. .. toctree::
..    :hidden:
..    :maxdepth: 2
..    :caption: Theory of Modules:

..    theory/vector
..    theory/module
..    theory/atom/index
..    theory/bond/index
..    theory/compound/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Algorithms:

   algorithms/manifold/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples:

   examples/hello-world
   examples/hello-mnist
   examples/weight-erasure

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More on Modula:

   Modula FAQ <faq>
   Modula codebase <https://github.com/modula-systems/modula>
   Modula homepage <https://modula.systems/>

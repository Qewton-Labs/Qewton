======
Qewton
======

Welcome to **Qewton**, a Python library of deep learning methods 
for solving differential equations, simulating engineering problems and creating 
digital twins.

In Qewton, the goal is to provide easy-to-use but powerful deep learning tools for physics and engineering to everyone, lowering the barrier to entry the field of AI for simulations in any imaginable application. We pre-implement methods such as
 - Operator Learning (Fourier Neural Operators, PCA-Nets, DeepONets (WIP))
 - Differentiable Physics (WIP)
 - PINNs (Physics-Informed Neural Networks)
 - and many more to come.

To simplify their usage, we provide
 - a graph-based API for defining and training models
 - connections to existing engineering and deep learning software
 - automatic model selection and hyperparameter tuning
 - visualization tools
 - automatic shape and data configuration checking
 - pre-implemented differential operators, equations and training pipelines.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   license

The documentation of all modules, functions, methods and variable in 
**Qewton** can be found here:

.. toctree::
   :maxdepth: 1
   :caption: API

   Algorithms <api/qewton.algorithms>
   Backends <api/qewton.backends>
   Configs <api/qewton.config>
   Constraints <api/qewton.constraints>
   Data <api/qewton.data>
   Geometries <api/qewton.geometries>
   Graphs <api/qewton.graphs>
   Optim <api/qewton.optim>
   Visualization <api/qewton.visualization>
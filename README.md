# Qewton: Deep Learning for Physics and Engineering

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


### Note:
This library is currently in beta, therefore things are subject to change and there is no Pypi release yet. Feel free to try, contribute and suggest features.

### Structure:
The core structure of Qewton is based on few key components:
![Workflow in Qewton](images/struktur_light.png#gh-light-mode-only)
![Workflow in Qewton](images/struktur_dark.png#gh-dark-mode-only)

### About us:
We are the creators of [TorchPhysics](https://github.com/qewton-labs/torchphysics), a deep learning library for PDEs. With Qewton, we aim to provide a more user-friendly and flexible framework for deep learning in physics and engineering.
<h1 align='center'>SITHshow: Scale-Invariant Temporal History Showcase</h1>

## Overview

SITHshow is a library showcasing applications of the neurally inspired [SITH](https://direct.mit.edu/neco/article/24/1/134/7733/A-Scale-Invariant-Internal-Representation-of-Time) representation of working memory for use in neural networks. Because SITH has a fuzzy memory of the past, it can often outperform RNNs and deep learning models with fixed buffers.  

... **overview here** ...

## Installation

The easiest way to get started is to run the [example notebooks](#examples) in Google Colaboratory.

To install this package with the JAX CPU backend (Windows users), run

```bash
pip install --upgrade "jax[cpu]"

git clone https://github.com/compmem/SITHshow
cd SITHshow
pip install .
```

To install this package with the JAX GPU backend (NVIDIA only), run one of the corresponding code chunks compatible with your GPU.

```bash
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

git clone https://github.com/compmem/SITHshow
cd SITHshow
pip install .
```

```bash
# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

git clone https://github.com/compmem/SITHshow
cd SITHshow
pip install .
```


To install this package with the JAX TPU backend, run

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

git clone https://github.com/compmem/SITHshow
cd SITHshow
pip install .
```

Requires Python 3.9+ and JAX 0.4.13+. See the [JAX Docs](https://jax.readthedocs.io/en/latest/installation.html) for more information about cross-platform compatibility.


## Examples

Work through the notebooks in Google Colab...

**Get started with SITH**

- [CME Demo](https://github.com/compmem/SITHshow/blob/main/examples/cme_demo.ipynb)

**Train on one scale, generalize to many with SITHCon**

- [Morse Decoder](https://github.com/compmem/SITHshow/blob/main/examples/morse_code.ipynb) (suitable to run on cpu)
- [AudioMNIST](https://github.com/compmem/SITHshow/blob/main/examples/audio_mnist.ipynb) (more computationally intensive)

**Efficiently learn latent number lines with CNL**

- [Odometer](https://github.com/compmem/SITHshow/blob/main/examples/odometer.ipynb) (suitable to run on cpu) 

## More SITH

TODO do we put papers here?

DeepSITH
RL SITH
etc.
<h1 align='center'>SITHshow: Scale-Invariant Temporal History Showcase</h1>

<h4 align="center">
  <a href="#overview">Overview</a> | 
  <a href="#quickstart">Quickstart</a> |
  <a href="#installation">Installation</a> |
  <a href="#examples">Examples</a> |
  <a href="#more-sith">More SITH</a> 
</h4>


## Overview

SITHshow is a library showcasing applications of the neurally inspired [SITH](https://direct.mit.edu/neco/article/24/1/134/7733/A-Scale-Invariant-Internal-Representation-of-Time) representation of working memory for use in neural networks. Because SITH has a fuzzy memory of the past, it can often outperform RNNs and deep learning models with fixed buffers.  

## Quickstart

Jump right in by trying the [example notebooks](#examples) in Google Colaboratory!

## Installation


To install this package with the JAX CPU backend (Windows users without WSL 2), run

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

Requires Python 3.9+ and JAX 0.4.13+. See the [JAX Docs](https://jax.readthedocs.io/en/latest/installation.html) for more information about cross-platform compatibility.


## Examples

**Getting started with SITH**

- [SITH Intro](https://github.com/compmem/SITHshow/blob/main/examples/sith_intro.ipynb)

**Train on one scale, generalize to many with SITHCon**

- [Morse Decoder](https://github.com/compmem/SITHshow/blob/main/examples/morse_code.ipynb) (suitable to run on cpu)
- [AudioMNIST](https://github.com/compmem/SITHshow/blob/main/examples/audio_mnist.ipynb) (more computationally intensive, GPU recommended)

**Efficiently learn latent number lines with CNL**

- [Odometer](https://github.com/compmem/SITHshow/blob/main/examples/odometer.ipynb) (suitable to run on cpu) 

## More SITH


- [Learning long-range temporal dependencies with DeepSITH](https://proceedings.neurips.cc/paper/2021/hash/e7dfca01f394755c11f853602cb2608a-Abstract.html)   

- [Applications of SITH in RL](https://open.bu.edu/handle/2144/45979)
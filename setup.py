from setuptools import setup

_jax_version_constraints = ">=0.4.14"
_jaxlib_version_constraints = ">=0.4.14"

setup(name='sithshow',
      version='0.1',
      description='Scale-Invariant Temporal History Showcase',
      url='TODO',
      author='Computational Memory Lab',
      author_email='TODO',
      packages=['sithshow'],
      install_requires=[
        f"jax{_jax_version_constraints}",
        f"jaxlib{_jaxlib_version_constraints}",
        "jaxtyping",
        "equinox",
        "optax",
        "numpy",
        "matplotlib",
        "tqdm",
        ],
      extras_require={
        "cpu": f"jax[cpu]{_jax_version_constraints}",
        # TPU and CUDA installations, currently require to add package repository URL, i.e.,
        # pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
        "tpu": f"jax[tpu]{_jax_version_constraints}",
        "cuda": f"jax[cuda]{_jax_version_constraints}",
    },
      include_package_data=True,
      zip_safe=False)
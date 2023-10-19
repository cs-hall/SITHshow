from setuptools import setup

setup(name='sithshow',
      version='0.1',
      description='Scale-Invariant Temporal History Showcase',
      url='https://github.com/compmem/SITHshow',
      author='Computational Memory Lab',
      author_email='pbs5u@virginia.edu',
      packages=['sithshow'],
      install_requires=[
        "jaxtyping",
        "equinox",
        "optax",
        "numpy",
        "matplotlib",
        "tqdm",
        "ipykernel",
        ],
      include_package_data=True,
      zip_safe=False)
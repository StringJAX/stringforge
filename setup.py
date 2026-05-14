from setuptools import setup

setup(
    name='stringjax',
    version='0.0.2',
    description='Differentiable tools for string compactifications with JAX.',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/AndreasSchachner/stringjax',
    # Top-level package + the vacuavault subpackage (server-side
    # validation/curation tooling for the HF dataset repo).
    packages=['stringjax', 'stringjax.vacuavault'],
    python_requires='>=3.12',
    # NOTE: jaxvacua/kahlerjax/jaxiverse intentionally do NOT live in
    # install_requires — those are *downstream* sibling packages that
    # depend on stringjax (not the other way round).  Listing them as
    # hard deps would create a circular pip dependency.  They're
    # surfaced lazily via try/except in stringjax/__init__.py and via
    # the ``[full]`` extra below for users who want one-shot install.
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'optax',
        'matplotlib',
        'seaborn',
        'h5py',
        'pandas',
        'tqdm',
        'sympy',
        # Vault layer (cy_io + vacuavault) needs these:
        'pyarrow',
        'huggingface_hub',
        'jaxpolylog@git+https://github.com/AndreasSchachner/jaxpolylog.git#egg=jaxpolylog',
        'jaxvacua@git+https://github.com/AndreasSchachner/jaxvacua.git#egg=jaxvacua',
        'kahlerjax@git+https://github.com/AndreasSchachner/kahlerjax.git#egg=kahlerjax',
        'jaxiverse@git+https://github.com/AndreasSchachner/jaxiverse.git#egg=jaxiverse',
    ],
    license='GPL-3.0',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

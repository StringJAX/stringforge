from setuptools import setup

setup(
    name='stringjax',
    version='0.0.1',
    description='Differentiable tools for string compactifications with JAX.',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/AndreasSchachner/stringjax',
    packages=['stringjax'],
    python_requires='>=3.12',
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
        'jaxpolylog@git+https://github.com/AndreasSchachner/jaxpolylog.git#egg=jaxpolylog',
        'jaxvacua@git+https://github.com/AndreasSchachner/jaxvacua.git#egg=jaxvacua',
    ],
    extras_require={
        # Private submodules — install manually from local clones or
        # private repos if you have access.
        'full': [
            'kahlerjax',
            'jaxiverse',
        ],
    },
    license='GPL-3.0',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

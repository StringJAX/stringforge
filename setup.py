from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent

setup(
    name='stringforge',
    version='0.1.0',
    description='Differentiable tools for string compactifications with JAX.',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/AndreasSchachner/stringforge',
    # Top-level package + the vacuavault subpackage (server-side
    # validation/curation tooling for the HF dataset repo).
    packages=['stringforge', 'stringforge.vacuavault'],
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'pandas',
        'pyarrow',
        'huggingface_hub',
        'jaxpolylog',
        'jaxvacua',
    ],
    license='GPL-3.0-only',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

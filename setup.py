from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/AndreasSchachner/stringforge',
    packages=find_packages(include=['stringforge', 'stringforge.*']),
    license='GPL-3.0-only',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

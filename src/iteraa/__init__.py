"""
The IterAA Package
=====================
Descriptions

Features
--------
Provides
  1. Plotting functionalities to visualise output.

Documentations
--------------
Documentation is available in two forms: docstrings provided with the code,
and a loose standing reference guide, available from
`the IterAA homepage <https://iteraa.readthedocs.io/en/latest/>`_.

Code snippets in docstrings are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> import iteraa
  >>> help(iteraa.ArchetypalAnalysis)
  ... # docstring: +SKIP

Utilities
---------
test (To be implemented)
    Run IterAA tests.
__version__
    Return IterAA version string.
"""

# read version from installed package
from importlib.metadata import version
__version__ = version('iteraa')

# Populate package namespace
__all__ = ['constants', 'datasets', 'utils', 'plot', 'iaa', 'piaa']
from iteraa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, FIGS_DIR_PATH
from iteraa.datasets import getExampleBlobData, getExampleSquareData, getCaseStudyData
from iteraa.utils import explainedVariance 
from iteraa.plot import plotRadarDatapoints, createSimplexAx, mapAlfaToSimplex, plotTSNE
from iteraa.iaa import ArchetypalAnalysis
from iteraa.piaa import subsetSplit, submitAAjobs, runAA, fitPIAA


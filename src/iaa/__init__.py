"""
The IAA Package
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
`the IAA homepage <https://iaa.readthedocs.io/en/latest/>`_.

Code snippets in docstrings are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> import iaa
  >>> help(iaa.ArchetypalAnalysis)
  ... # docstring: +SKIP

Utilities
---------
test (To be implemented)
    Run IAA tests.
__version__
    Return IAA version string.
"""

# read version from installed package
from importlib.metadata import version
__version__ = version('iaa')

# Populate package namespace
__all__ = ['constants', 'datasets', 'utils', 'plot', 'iaa', 'piaa']
from iaa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH
from iaa.datasets import getExampleDataPath, getStrongScalingDataPath, getWeakScalingDataPaths, getValidationDataPath, getCaseStudyDataPaths
from iaa.utils import explainedVariance 
from iaa.plot import ternaryPlot, compareProfile, datapointProfile, plotRadarDatapoint, createSimplexAx, mapAlfaToSimplex, plotTSNE
from iaa.iaa import ArchetypalAnalysis
from iaa.piaa import subsetSplit, runAA, fitPIAA


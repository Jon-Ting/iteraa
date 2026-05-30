"""
The Iterative Archetypal Analysis (IterAA) Package
=====================
Descriptions

Features
--------
Provides
  1. Fast iterative approach to archetypal analysis execution
  2. Plotting functionalities to visualise output.
  3. Parallel execution on HPC systems.

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

from importlib.metadata import version

from iteraa import datasets, plot, utils
from iteraa.iaa import ArchetypalAnalysis
from iteraa.piaa import fitPIAA, runAA, submitAAjobs, subsetSplit

try:
    __version__ = version("iteraa")
except Exception:
    __version__ = "1.0.0"

__all__ = [
    "ArchetypalAnalysis",
    "__version__",
    "datasets",
    "fitPIAA",
    "plot",
    "runAA",
    "submitAAjobs",
    "subsetSplit",
    "utils",
]

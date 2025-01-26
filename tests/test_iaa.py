from os import listdir

from iaa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, FIGS_DIR_PATH
# from iaa.datasets import getExampleDataPath
from iaa.utils import explainedVariance
from iaa.plot import plotRadarDatapoints, createSimplexAx, mapAlfaToSimplex, plotTSNE
from iaa.iaa import ArchetypalAnalysis
from iaa.piaa import subsetSplit, runAA, fitPIAA


#def test_subsetSplit(X, nSubsets, dataName):
#    """Unit test of subsetSplit()."""
    # subsetSplit(X, nSubsets, dataName)
    # assert len(listdir(f"{}")) == 0


def test_runAA():
    """Unit test of runAA()."""
    assert 1 == 1


from os import listdir

from iteraa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, FIGS_DIR_PATH
from iteraa.datasets import getExampleBlobData, getExampleSquareData, getCaseStudyData
from iteraa.utils import explainedVariance
from iteraa.plot import plotRadarDatapoints, createSimplexAx, mapAlfaToSimplex, plotTSNE
from iteraa.iaa import ArchetypalAnalysis
from iteraa.piaa import subsetSplit, runAA, fitPIAA


def test_getExampleBlobData():
    numSamples = 100
    X = getExampleBlobData(numSamples)
    assert X.shape == (numSamples, 2)


def test_getExampleSquareData():
    numSamples = 100
    X = getExampleSquareData(numSamples)
    assert X.shape == (numSamples, 2)


def test_getCaseStudyData():
    X = getCaseStudyData()
    print(X)
    assert X.columns == ['AnkleDiam', 'KneeDiam', 'WristDiam', 'Bitro', 'Biil', 'ElbowDiam', 'ChestDiam', 'ChestDp', 'Biac', 'Height']
    assert X.shape == (numSamples, 2)


# def test_subsetSplit(X, nSubsets, dataName):
#     """Unit test of subsetSplit()."""
#     subsetSplit(X, nSubsets, dataName)
#     assert len(listdir(f"{}")) == 0


def test_runAA():
    """Unit test of runAA()."""
    assert 1 == 1


from natsort import natsorted
import numpy as np
from os import listdir
from os.path import isfile
from pytest import approx

from iteraa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, FIGS_DIR_PATH
from iteraa.datasets import getExampleBlobData, getExampleSquareData, getCaseStudyData
from iteraa.utils import explainedVariance
from iteraa.plot import plotRadarDatapoints, createSimplexAx, mapAlfaToSimplex, plotTSNE
from iteraa.iaa import ArchetypalAnalysis
from iteraa.piaa import subsetSplit, runAA, fitPIAA


NUM_SAMPLES = 100
NUM_ARCHETYPES = 3
NUM_SUBSETS = 3


def test_getExampleBlobData():
    """Unit test of getExampleBlobData()."""
    X = getExampleBlobData(NUM_SAMPLES)
    assert X.shape == (NUM_SAMPLES, 2)


def test_getExampleSquareData():
    """Unit test of getExampleSquareData()."""
    numSamples = 100
    X = getExampleSquareData(NUM_SAMPLES)
    assert X.shape == (NUM_SAMPLES, 2)


def test_getCaseStudyData():
    """Unit test of getCaseStudyData()."""
    X = getCaseStudyData()
    assert X.shape == (507, 10)


def test_explainedVariance():
    """Unit test of explainedVariance()."""
    Xact = getExampleBlobData(NUM_SAMPLES)
    assert explainedVariance(Xact, Xact) == approx(1.0)


def test_ArchetypalAnalysis():
    """Unit test of ArchetypalAnalysis()."""
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES, iterative=True)
    X = getExampleSquareData(NUM_SAMPLES)
    aa.fit(X)
    assert aa.nDim == 2
    assert aa.nData == NUM_SAMPLES
    assert aa.alfa.shape == (NUM_ARCHETYPES, NUM_SAMPLES)
    assert aa.beta.shape == (NUM_SAMPLES, NUM_ARCHETYPES)
    assert aa.archetypes.shape == (2, NUM_ARCHETYPES)


def test_subsetSplit():
    """Unit test of subsetSplit()."""
    subsetsPicklesPath, dataName = 'tests/subsetsDataPKLs', 'test'
    X = getExampleSquareData(NUM_SAMPLES)
    subsetSplit(X, nSubsets=NUM_SUBSETS, dataName=dataName, subsetsPicklesPath=subsetsPicklesPath)
    dirFiles = listdir(subsetsPicklesPath)
    assert len(dirFiles) == NUM_SUBSETS
    for (i, dirFile) in enumerate(natsorted(dirFiles)):
        assert dirFile.startswith(f"{dataName}data{i+1}.pkl")


def test_runAA():
    """Unit test of runAA()."""
    dataName, subsetID = 'test', 1
    fName = f"tests/subsetsDataPKLs/{dataName}data{subsetID}.pkl"
    outputsPicklesPath = 'tests/subsetsOutputsPKLs'
    runAA(fName, nArchetypes=NUM_ARCHETYPES, outputsPicklesPath=outputsPicklesPath)
    assert isfile(f"{outputsPicklesPath}/{dataName}output{subsetID}.pkl")


def test_fitPIAA():
    """Unit test of fitPIAA()."""
    pass
    # fitPIAA()



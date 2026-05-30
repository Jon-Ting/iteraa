from os import makedirs
from os.path import exists

import numpy as np
import pytest

from iteraa.datasets import getCaseStudyData, getExampleBlobData, getExampleSquareData
from iteraa.iaa import ArchetypalAnalysis
from iteraa.piaa import fitPIAA, runAA, subsetSplit
from iteraa.plot import createSimplexAx, mapAlfaToSimplex, plotRadarDatapoints, plotTSNE
from iteraa.utils import calcSSE, ecdf, explainedVariance, furthestSum, solveConstrainedNNLS

NUM_SAMPLES = 100
NUM_ARCHETYPES = 3
NUM_SUBSETS = 3
TEST_DATA_DIR = "tests/subsetsDataPKLs"
TEST_OUT_DIR = "tests/subsetsOutputsPKLs"


@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure test directories exist and are clean."""
    if not exists(TEST_DATA_DIR):
        makedirs(TEST_DATA_DIR)
    if not exists(TEST_OUT_DIR):
        makedirs(TEST_OUT_DIR)
    yield
    # Optional: cleanup after tests
    # rmtree(TEST_DATA_DIR)
    # rmtree(TEST_OUT_DIR)


def test_getExampleBlobData():
    X = getExampleBlobData(NUM_SAMPLES)
    assert X.shape == (NUM_SAMPLES, 2)


def test_getExampleSquareData():
    X = getExampleSquareData(NUM_SAMPLES)
    assert X.shape == (NUM_SAMPLES, 2)


def test_getCaseStudyData():
    # Note: requests might fail if no internet, but usually okay in this environment
    try:
        X = getCaseStudyData()
        assert X.shape == (507, 10)
    except Exception as e:
        pytest.skip(f"Could not fetch case study data: {e}")


def test_explainedVariance():
    X = getExampleBlobData(NUM_SAMPLES)
    assert explainedVariance(X.T, X.T) == pytest.approx(1.0)


def test_ArchetypalAnalysis_spams():
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES, iterative=False)
    X = getExampleSquareData(NUM_SAMPLES)
    aa.fit(X)
    assert aa.archetypes_.shape == (2, NUM_ARCHETYPES)
    assert aa.alfa_.shape == (NUM_ARCHETYPES, NUM_SAMPLES)
    assert aa.beta_.shape == (NUM_SAMPLES, NUM_ARCHETYPES)
    assert aa.explainedVariance_ > 0


def test_ArchetypalAnalysis_iterative():
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES, iterative=True, nSubsets=NUM_SUBSETS)
    X = getExampleSquareData(NUM_SAMPLES)
    aa.fit(X)
    assert aa.archetypes_.shape == (2, NUM_ARCHETYPES)
    assert aa.alfa_.shape == (NUM_ARCHETYPES, NUM_SAMPLES)


def test_ArchetypalAnalysis_classical():
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES, maxIter=10)
    X = getExampleSquareData(NUM_SAMPLES)
    aa.fitClassical(X)
    assert aa.archetypes_.shape == (2, NUM_ARCHETYPES)
    assert aa.alfa_.shape == (NUM_ARCHETYPES, NUM_SAMPLES)


def test_transform():
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES)
    X = getExampleSquareData(NUM_SAMPLES)
    aa.fit(X)
    X_new = getExampleSquareData(10)
    X_approx, alfa = aa.transform(X_new)
    assert X_approx.shape == (10, 2)
    assert alfa.shape == (NUM_ARCHETYPES, 10)


def test_fit_transform():
    aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES)
    X = getExampleSquareData(NUM_SAMPLES)
    X_approx = aa.fit_transform(X)
    assert X_approx.shape == (NUM_SAMPLES, 2)


def test_subsetSplit_and_runAA_and_fitPIAA():
    dataName = "pi_test"
    X = getExampleSquareData(NUM_SAMPLES)

    # 1. subsetSplit
    subsetSplit(X, nSubsets=NUM_SUBSETS, dataName=dataName, subsetsPicklesPath=TEST_DATA_DIR)

    # 2. runAA for each subset
    for i in range(1, NUM_SUBSETS + 1):
        fName = f"{TEST_DATA_DIR}/{dataName}data{i}.pkl"
        runAA(fName, nArchetypes=NUM_ARCHETYPES, outputsPicklesPath=TEST_OUT_DIR)

    # 3. fitPIAA
    aa = fitPIAA(
        X,
        nArchetypes=NUM_ARCHETYPES,
        numSubset=NUM_SUBSETS,
        dataName=dataName,
        outputsPicklesPath=TEST_OUT_DIR,
    )
    assert aa.archetypes_.shape == (2, NUM_ARCHETYPES)
    assert aa.alfa_.shape == (NUM_ARCHETYPES, NUM_SAMPLES)
    assert aa.explainedVariance_ > 0


def test_solveConstrainedNNLS():
    u = np.array([1.0, 0.0])
    t = np.array([[1.0, 0.0], [0.0, 1.0]])
    w, rnorm = solveConstrainedNNLS(u, t, C=1.0)
    assert w == pytest.approx([1.0, 0.0], abs=1e-5)


def test_furthestSum():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T  # (nDim, nData)
    init = [0]
    res = furthestSum(X, noc=3, i=init)
    assert len(res) == 3
    assert set(res).issubset({0, 1, 2, 3})


def test_ecdf():
    data = np.array([1, 2, 3, 4, 5])
    assert ecdf(data, 3) == 0.4  # [1, 2] are < 3
    assert ecdf(data, 6) == 1.0


def test_calcSSE():
    a = np.array([1, 2])
    b = np.array([1, 1])
    assert calcSSE(a, b) == 1.0


def test_plotting_smoke_tests(tmp_path):
    # Just check if they run without error
    import iteraa.constants as constants

    # Mock FIGS_DIR_PATH to use tmp_path
    original_path = constants.FIGS_DIR_PATH
    constants.FIGS_DIR_PATH = str(tmp_path)

    try:
        aa = ArchetypalAnalysis(nArchetypes=NUM_ARCHETYPES)
        X = getExampleSquareData(20)
        aa.fit(X)

        plotRadarDatapoints(aa, X, sampIDs=[0])
        createSimplexAx(aa)
        mapAlfaToSimplex(aa.alfa_, aa)
        plotTSNE(X, figNamePrefix="test")
    finally:
        constants.FIGS_DIR_PATH = original_path

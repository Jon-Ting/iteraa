
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from iteraa.constants import RANDOM_STATE


def getExampleBlobData(numSamples: int) -> np.ndarray:
    """Generates an example dataset with a blob distribution.

    Args:
        numSamples: Number of data points to generate.

    Returns:
        Xscaled: The standardized dataset. Shape: (numSamples, 2).
    """
    X, _ = make_blobs(n_samples=numSamples, centers=1, cluster_std=30, random_state=RANDOM_STATE)
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled


def getExampleSquareData(numSamples: int, randomState: int | None = RANDOM_STATE) -> np.ndarray:
    """Generates an example dataset with a uniform square distribution.

    Args:
        numSamples: Number of data points to generate.
        randomState: Random seed for reproducibility. Defaults to RANDOM_STATE.

    Returns:
        Xscaled: The standardized dataset. Shape: (numSamples, 2).
    """
    rng = np.random.default_rng(randomState)
    x = rng.uniform(-10, 10, numSamples)
    y = rng.uniform(-10, 10, numSamples)
    X = np.stack((x, y), axis=1)
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled


def getCaseStudyData() -> np.ndarray:
    """Retrieves the case study dataset (skeleton body measurements).

    Returns:
        Xscaled: The standardized dataset. Shape: (507, 10).
    """
    # Load the data
    r = requests.get("http://jse.amstat.org/datasets/body.dat.txt", timeout=10)
    data = np.array(list(map(lambda x: list(map(float, x.split())), r.text.splitlines())))

    # Order the columns in the similar order that appears in previous studies.
    columns = [
        "AnkleDiam",
        "KneeDiam",
        "WristDiam",
        "Bitro",
        "Biil",
        "ElbowDiam",
        "ChestDiam",
        "ChestDp",
        "Biac",
        "Height",
        "Gender",
    ]
    selectedCols = [8, 7, 6, 2, 1, 5, 4, 3, 0, 23, 24]
    df = pd.DataFrame(data[:, selectedCols], columns=columns)

    # Map the entries in the gender column into strings
    gender = {1.0: "male", 0.0: "female"}
    df["Gender"] = df["Gender"].map(gender)

    # Generate the feature set
    featNames = [
        "AnkleDiam",
        "KneeDiam",
        "WristDiam",
        "Bitro",
        "Biil",
        "ElbowDiam",
        "ChestDiam",
        "ChestDp",
        "Biac",
        "Height",
    ]
    X = df[featNames].values

    # Standardise all features
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled

from importlib.resources import files
from os import listdir
import requests

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def getExampleBlobData(numSamples):
    """Get example data set with circle distribution.

    Returns
    -------
    Xscaled : np.ndarray
        Scaled dataset.
    """
    X, _ = make_blobs(n_samples=numSamples, centers=1, cluster_std=30, random_state=0)
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled


def getExampleSquareData(numSamples):
    """Get example data set with uniform square distribution.

    Returns
    -------
    df : pd.DataFrame
        Dataset.
    """
    x = np.random.uniform(-10, 10, numSamples)
    y = np.random.uniform(-10, 10, numSamples)
    X = np.array([[xi, yi] for xi, yi in zip(x, y)])
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled


def getCaseStudyData():
    """Get case study data set (skeleton).

    Returns
    -------
    Xscaled : np.ndarray
        Scaled dataset.
    """
    # Load the data
    r = requests.get('http://jse.amstat.org/datasets/body.dat.txt')
    data = np.array(list(map(lambda x: list(map(float, x.split())), r.text.splitlines())))

    # Order the columns in the similar order that appears in previous studies.
    columns = ['AnkleDiam', 'KneeDiam', 'WristDiam', 'Bitro', 'Biil', 'ElbowDiam', 'ChestDiam', 'ChestDp', 'Biac', 'Height', 'Gender']
    selectedCols = [8, 7, 6, 2, 1, 5, 4, 3, 0, 23, 24]
    df = pd.DataFrame(data[:, selectedCols], columns=columns)

    # Map the entries in the gender column into strings
    gender = {1.0: 'male', 0.0: 'female'}
    df['Gender'] = df.apply(lambda row: gender[row['Gender']], axis = 1)

    # Generate the feature set
    featNames = ['AnkleDiam', 'KneeDiam', 'WristDiam', 'Bitro', 'Biil', 'ElbowDiam', 'ChestDiam', 'ChestDp', 'Biac', 'Height']
    X = df[featNames].values

    # Standardise all features
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    return Xscaled

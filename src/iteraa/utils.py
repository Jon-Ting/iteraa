"""Numerical and statistical utilities for archetypal analysis.

This module provides core mathematical functions including convergence metrics,
initialization algorithms (Furthest Sum), and constrained optimization wrappers.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from sklearn.metrics import explained_variance_score


def findFurthestPoint(xSearch: np.ndarray, xRef: np.ndarray) -> tuple[np.ndarray, int]:
    """Finds a data point in xSearch that is furthest from all points in xRef.

    In the context of archetypal analysis, xRef is typically the set of archetypes
    and xSearch is the dataset.

    Args:
        xSearch: The dataset to search for the furthest point. Shape: (nData, nDim).
        xRef: The reference points to calculate distances from. Shape: (nRef, nDim).

    Returns:
        A tuple containing the furthest point (ndarray) and its index in xSearch (int).
    """
    D = cdist(xRef, xSearch)
    Dsum = D.sum(axis=0)
    idxmax = np.argmax(Dsum)
    return xSearch[idxmax, :], int(idxmax)


def ecdf(X: np.ndarray, x: float) -> float:
    """Empirical Cumulative Distribution Function.

    Args:
        X: 1-D array of data points for a single feature, defining its distribution.
        x: The value to evaluate the ECDF at (e.g., an archetype component).

    Returns:
        The probability P(X <= x), representing how much of the data is covered by x.
    """
    return float(len(X[x > X]) / len(X))


def calcSSE(Xact: np.ndarray, Xapx: np.ndarray) -> float:
    """Calculates the Sum of Square Errors (SSE).

    Args:
        Xact: The actual data points.
        Xapx: The approximated data points.

    Returns:
        The sum of squared differences between Xact and Xapx.
    """
    return float(((Xact - Xapx) ** 2).sum())


def calcSST(Xact: np.ndarray) -> float:
    """Calculates the Total Sum of Squares (SST).

    Args:
        Xact: The actual data points.

    Returns:
        The sum of squared values of Xact.
    """
    return float((Xact**2).sum())


def explainedVariance(Xact: np.ndarray, Xapx: np.ndarray, method: str = "sklearn") -> float:
    """Calculates the explained variance.

    Args:
        Xact: The actual data points. Shape: (nDim, nData).
        Xapx: The approximated data points. Shape: (nDim, nData).
        method: The method to use ('sklearn' or 'classic'). Defaults to 'sklearn'.

    Returns:
        The explained variance score.
    """
    if method.lower() == "sklearn":
        # sklearn expects (n_samples, n_features), our data is (nDim, nData)
        return float(explained_variance_score(Xact.T, Xapx.T))
    SSE = calcSSE(Xact, Xapx)
    SST = calcSST(Xact)
    return float((SST - SSE) / SST)


def solveConstrainedNNLS(u: np.ndarray, t: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Solves ||u - t@w||^2 subject to w >= 0 and sum(w) = 1.

    The sum-to-one constraint is approximated by augmenting the system with a row
    of ones scaled by C.

    Args:
        u: The target vector. Shape: (mObservation,).
        t: The design matrix. Shape: (mObservation, nVariables).
        C: The scaling factor for the sum-to-one constraint.

    Returns:
        A tuple containing the solution vector w and the residual norm.
    """
    _, nVariables = t.shape
    # Scaling the observation and design matrix
    u_scaled = u * C
    t_scaled = t * C
    # Adding the sum-to-one constraint equation
    u_aug = np.append(u_scaled, [C], axis=0)
    t_aug = np.append(t_scaled, np.full((1, nVariables), C), axis=0)
    w, rnorm = nnls(t_aug, u_aug)
    return w, float(rnorm)


def furthestSum(
    K: np.ndarray, noc: int, i: list[int], exclude: np.ndarray | list[int] = []
) -> list[int]:
    """Furthest sum algorithm to efficiently generate initial seeds for archetypes.

    This function is based on the implementation from py_pcha by Ulf Aslak Jensen.
    Note: It handles transposed shapes (dimensions, examples) internally if needed.

    Args:
        K: Data matrix or kernel matrix. Shape: (I, J).
        noc: Number of candidate archetypes to extract.
        i: List of initial observation indices to start the search.
        exclude: Indices in K that cannot be used as candidates.

    Returns:
        A list of indices for the extracted candidate archetypes.
    """

    def maxIndVal(l):
        return max(zip(range(len(l)), l), key=lambda x: x[1])

    I, J = K.shape
    index = np.arange(J)
    if len(exclude) > 0:
        index[exclude] = 0

    # Mark initial indices as used
    for idx in i:
        index[idx] = -1

    indT = i[-1]
    sumDist = np.zeros((1, J))

    if noc * I < J:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, noc + 11):
            if k > noc - 1:
                Kq = np.dot(Kt[:, i[0]], Kt)
                sumDist -= np.sqrt(np.maximum(Kt2 - 2 * Kq + Kt2[i[0]], 0))
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = np.dot(Kt[:, indT].T, Kt)
            sumDist += np.sqrt(np.maximum(Kt2 - 2 * Kq + Kt2[indT], 0))
            ind, _ = maxIndVal(sumDist[:, t][0])
            indT = t[ind]
            i.append(indT)
            index[indT] = -1
    else:
        if I != J or not np.array_equal(K, K.T):  # Generate kernel if K not square/symmetric
            Kt = K
            K = np.dot(Kt.T, Kt)
            diagK = np.diag(K)
            K = np.sqrt(np.maximum(np.tile(diagK, (J, 1)) - 2 * K + np.tile(diagK, (J, 1)).T, 0))

        Kt2 = np.diag(K)
        for k in range(1, noc + 11):
            if k > noc - 1:
                sumDist -= np.sqrt(np.maximum(Kt2 - 2 * K[i[0], :] + Kt2[i[0]], 0))
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sumDist += np.sqrt(np.maximum(Kt2 - 2 * K[indT, :] + Kt2[indT], 0))
            ind, _ = maxIndVal(sumDist[:, t][0])
            indT = t[ind]
            i.append(indT)
            index[indT] = -1
    return i

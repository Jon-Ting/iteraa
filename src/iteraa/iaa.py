"""Archetypal Analysis implementation.

This module provides the ArchetypalAnalysis class for conducting archetypal analysis,
based on "Archetypal Analysis" by Adele Cutler and Leo Breiman (1994).
"""

from __future__ import annotations

from time import time

import numpy as np
from sklearn.model_selection import KFold
from spams import archetypalAnalysis

from iteraa.constants import RANDOM_STATE
from iteraa.utils import calcSSE, ecdf, explainedVariance, furthestSum, solveConstrainedNNLS


class ArchetypalAnalysis:
    """Conducts Archetypal Analysis on a dataset.

    Archetypal analysis finds a few representative "archetypes" in a dataset such that
    each data point can be represented as a convex combination of these archetypes,
    and each archetype is a convex combination of the data points.

    Attributes:
        nArchetypes: Number of archetypes to find.
        iterative: Whether to use an iterative subset-based approach for large datasets.
        nSubsets: Number of subsets for iterative fitting.
        shuffle: Whether to shuffle data before splitting into subsets.
        onlyZ: If True, only archetypes are computed; alfa and beta are skipped.
        subsetsSampleIdxs: Pre-defined indices for subsets.
        tolerance: Convergence tolerance.
        maxIter: Maximum number of iterations for classical fitting.
        randomState: Random seed for initialization.
        C: Constraint coefficient for sum-to-one constraint.
        initialize: Initialization method ('furthestsum', 'random', 'randomidx').
        redundancyTry: Number of attempts to resolve redundant archetypes.
        robust: Whether to use robust AA (SPAMS backend).
        computeXtX: Whether to compute XtX (SPAMS backend).
        stepsFISTA: Number of FISTA steps (SPAMS backend).
        stepsAS: Number of active subset steps (SPAMS backend).
        randominit: Whether to use random initialization (SPAMS backend).
        numThreads: Number of threads for SPAMS (-1 for all).
        verbose: Whether to print progress information.
        archetypes_: The fitted archetypes. Shape: (nDim, nArchetypes).
        alfa_: The weights to represent data as archetypes. Shape: (nArchetypes, nData).
        beta_: The weights to represent archetypes as data. Shape: (nData, nArchetypes).
        RSS_: Residual Sum of Squares of the fit.
        explainedVariance_: Explained variance of the fit.
        runTime_: Execution time of the fit in seconds.
    """

    def __init__(
        self,
        nArchetypes: int = 2,
        iterative: bool = False,
        nSubsets: int = 10,
        shuffle: bool = True,
        onlyZ: bool = False,
        subsetsSampleIdxs: list[np.ndarray] = [],
        tolerance: float = 0.001,
        maxIter: int = 200,
        randomState: int = RANDOM_STATE,
        C: float = 0.0001,
        initialize: str = "furthestSum",
        redundancyTry: int = 30,
        robust: bool = False,
        computeXtX: bool = False,
        stepsFISTA: int = 3,
        stepsAS: int = 50,
        randominit: bool = False,
        numThreads: int = -1,
        verbose: bool = False,
    ):
        self.nArchetypes = nArchetypes
        self.iterative = iterative
        self.nSubsets = nSubsets
        self.shuffle = shuffle
        self.onlyZ = onlyZ
        self.subsetsSampleIdxs = subsetsSampleIdxs
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.randomState = randomState
        self.C = C
        self.initialize = initialize.lower()
        self.redundancyTry = redundancyTry
        self.robust = robust
        self.computeXtX = computeXtX
        self.stepsFISTA = stepsFISTA
        self.stepsAS = stepsAS
        self.randominit = randominit
        self.numThreads = numThreads
        self.verbose = verbose

        # Fitted attributes
        self.archetypes_: np.ndarray | None = None
        self.alfa_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.RSS_: float | None = None
        self.explainedVariance_: float | None = None
        self.runTime_: float = 0.0
        self.nDim: int | None = None
        self.nData: int | None = None
        self.X: np.ndarray | None = None  # Transposed input data (nDim, nData)
        self.archetypeProfile: np.ndarray | None = None
        self.closeMatch: dict[int, tuple[int, np.ndarray]] = {}

    def fit(self, X: np.ndarray):
        """Fits the model to the data using the SPAMS backend.

        Args:
            X: The input data. Shape: (nData, nDim).
        """
        self.X = X.T
        self.nDim, self.nData = self.X.shape
        start_time = time()

        if self.iterative:
            self._fit_iterative(X)
        else:
            archetypes, A, B = archetypalAnalysis(
                np.asfortranarray(self.X),
                Z0=None,
                p=self.nArchetypes,
                returnAB=True,
                robust=self.robust,
                epsilon=self.tolerance,
                computeXtX=self.computeXtX,
                stepsFISTA=self.stepsFISTA,
                stepsAS=self.stepsAS,
                randominit=self.randominit,
                numThreads=self.numThreads,
            )
            self.archetypes_ = archetypes
            if not self.onlyZ:
                self.alfa_ = A.toarray()
                self.beta_ = B.toarray()

        self.runTime_ = time() - start_time
        if self.onlyZ or self.archetypes_ is None:
            return

        self._finalize_fit()

    def _fit_iterative(self, X: np.ndarray):
        """Internal helper for iterative fitting."""
        if not self.subsetsSampleIdxs:
            kFold = KFold(
                n_splits=self.nSubsets, shuffle=self.shuffle, random_state=self.randomState
            )
            self.subsetsSampleIdxs = [idxs for (_, idxs) in kFold.split(X)]

        subsetsZs = []
        subsetsAs = []
        subsetsBs = []

        for i, idxs in enumerate(self.subsetsSampleIdxs):
            if self.verbose:
                print(f"Subset {i + 1}")
            subsetX = X[idxs, :].T
            subsetZ, subsetA, subsetB = archetypalAnalysis(
                np.asfortranarray(subsetX),
                Z0=None,
                p=self.nArchetypes,
                returnAB=True,
                robust=self.robust,
                epsilon=self.tolerance,
                computeXtX=self.computeXtX,
                stepsFISTA=self.stepsFISTA,
                stepsAS=self.stepsAS,
                randominit=self.randominit,
                numThreads=self.numThreads,
            )
            subsetsZs.append(subsetZ)
            if not self.onlyZ:
                subsetsAs.append(subsetA.toarray())
                subsetsBs.append(subsetB.toarray())

        self._combine_subset_results(subsetsZs, subsetsAs, subsetsBs)

    def _combine_subset_results(
        self, subsetsZs: list[np.ndarray], subsetsAs: list[np.ndarray], subsetsBs: list[np.ndarray]
    ):
        """Combines archetypes from subsets into final archetypes.

        Args:
            subsetsZs: List of archetypes from each subset.
            subsetsAs: List of alfa matrices from each subset.
            subsetsBs: List of beta matrices from each subset.
        """
        allSubsetsZs = np.concatenate(subsetsZs, axis=1)
        self.archetypes_, Afinal, Bfinal = archetypalAnalysis(
            np.asfortranarray(allSubsetsZs),
            Z0=None,
            p=self.nArchetypes,
            returnAB=True,
            robust=self.robust,
            epsilon=self.tolerance,
            computeXtX=self.computeXtX,
            stepsFISTA=self.stepsFISTA,
            stepsAS=self.stepsAS,
            randominit=self.randominit,
            numThreads=self.numThreads,
        )

        if self.onlyZ:
            return

        # Reconstruct full alfa and beta
        allSampleIdxs = np.concatenate(self.subsetsSampleIdxs, axis=0)
        sortedXapproxIdxs = np.argsort(allSampleIdxs)

        A_final = Afinal.toarray()
        B_final = Bfinal.toarray()

        prevIdx = 0
        overallAs = []
        overallBs = []
        for i, subsetZ in enumerate(subsetsZs):
            nSubsetZs = subsetZ.shape[1]
            subsetOverallA = np.matmul(A_final[:, prevIdx : prevIdx + nSubsetZs], subsetsAs[i])
            subsetOverallB = np.matmul(subsetsBs[i], B_final[prevIdx : prevIdx + nSubsetZs, :])
            prevIdx += nSubsetZs
            overallAs.append(subsetOverallA)
            overallBs.append(subsetOverallB)

        self.alfa_ = np.concatenate(overallAs, axis=1)[:, sortedXapproxIdxs]
        self.beta_ = np.concatenate(overallBs, axis=0)[sortedXapproxIdxs]

    def fitClassical(self, X: np.ndarray):
        """Fits the model using the classical Cutler-Breiman algorithm (pure Python).

        Args:
            X: The input data. Shape: (nData, nDim).
        """
        self.X = X.T
        self.nDim, self.nData = self.X.shape
        start_time = time()

        if self.nArchetypes == 1:
            self.archetypes_ = self.X.mean(axis=1).reshape(self.nDim, 1)
            self.alfa_ = np.ones((1, self.nData))
            self.beta_ = np.full((self.nData, 1), 1.0 / self.nData)
        else:
            self._initializeArchetypes()
            RSS = 1.0
            RSSold = 100.0
            count = 0
            while count < self.maxIter:
                rel_change = abs(RSS - RSSold) / (RSSold + 1e-9)
                if rel_change <= self.tolerance:
                    break

                RSSold = RSS
                self._optimizeAlfa()
                hasError = self._optimizeBeta()
                if hasError:
                    # If redundancy resolution fails, we stop
                    break

                self.archetypes_ = np.matmul(self.X, self.beta_)
                RSS = np.sum(self.RSSi_)
                count += 1
            self.RSS_ = RSS

        self.runTime_ = time() - start_time
        self._finalize_fit()

    def _finalize_fit(self):
        """Common post-processing after fit."""
        self._rankArchetypes()
        Xapprox = np.matmul(self.archetypes_, self.alfa_)
        self.RSS_ = calcSSE(self.X, Xapprox)
        self.explainedVariance_ = explainedVariance(self.X, Xapprox, method="sklearn")
        self._extractArchetypeProfiles()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the model and returns the approximated data.

        Args:
            X: The input data. Shape: (nData, nDim).

        Returns:
            The approximated data points. Shape: (nData, nDim).
        """
        self.fit(X)
        Xapprox = np.matmul(self.archetypes_, self.alfa_)
        return Xapprox.T

    def transform(self, Xnew: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transforms new data into the archetype space.

        Args:
            Xnew: New data points. Shape: (nData_new, nDim).

        Returns:
            A tuple of (approximated data points, alfa matrix).
        """
        nData_new = Xnew.shape[0]
        alfa_new = np.zeros((self.nArchetypes, nData_new))
        for i in range(nData_new):
            alfa_new[:, i], _ = solveConstrainedNNLS(Xnew[i, :], self.archetypes_, self.C)

        XnewApprox = np.matmul(self.archetypes_, alfa_new)
        return XnewApprox.T, alfa_new

    def _initializeArchetypes(self):
        """Initializes archetypes based on the selected method."""
        if self.initialize == "random":
            self._randomInitialize()
        elif self.initialize == "randomidx":
            self._randomIdxInitialize()
        elif self.initialize == "furthestsum":
            try:
                self._furthestSumInitialize()
            except (IndexError, ValueError):
                if self.verbose:
                    print("FurthestSum initialization failed. Falling back to random.")
                self._randomInitialize()

    def _randomInitialize(self):
        from sklearn.preprocessing import MinMaxScaler

        rng = np.random.default_rng(self.randomState)
        sc = MinMaxScaler()
        sc.fit(self.X.T)
        archsNorm = rng.random((self.nArchetypes, self.nDim))
        self.archetypes_ = sc.inverse_transform(archsNorm).T

    def _furthestSumInitialize(self):
        rng = np.random.default_rng(self.randomState)
        init = [rng.integers(0, self.nData)]
        initArchIdx = furthestSum(self.X, self.nArchetypes, init)
        self.archetypes_ = self.X[:, initArchIdx]

    def _randomIdxInitialize(self):
        rng = np.random.default_rng(self.randomState)
        lstIdx = rng.choice(self.nData, self.nArchetypes, replace=False)
        self.archetypes_ = self.X[:, lstIdx]

    def _optimizeAlfa(self):
        """Optimizes the alfa matrix for fixed archetypes."""
        self.alfa_ = np.zeros((self.nArchetypes, self.nData))
        self.RSSi_ = np.zeros(self.nData)
        for i in range(self.nData):
            self.alfa_[:, i], self.RSSi_[i] = solveConstrainedNNLS(
                self.X[:, i], self.archetypes_, self.C
            )

    def _optimizeBeta(self) -> bool:
        """Optimizes the beta matrix for fixed alfas. Returns True if an error occurred."""
        self.beta_ = np.zeros((self.nData, self.nArchetypes))
        for l in range(self.nArchetypes):
            vBarL, hasError = self._compute_vbar(l)
            if hasError:
                return True
            self.beta_[:, l], _ = solveConstrainedNNLS(vBarL.flatten(), self.X, self.C)
        return False

    def _compute_vbar(self, l: int) -> tuple[np.ndarray, bool]:
        """Computes the v_bar vector for archetype l, handling redundancy."""
        eps = 1e-8
        alfa_l_sq_sum = np.sum(self.alfa_[l, :] ** 2)

        count = 0
        while alfa_l_sq_sum < eps and count < self.redundancyTry:
            # Replace redundant archetype
            self.archetypes_[:, l], _ = self._find_new_archetype_candidate(l)
            self._optimizeAlfa()
            alfa_l_sq_sum = np.sum(self.alfa_[l, :] ** 2)
            count += 1

        if alfa_l_sq_sum < eps:
            if self.verbose:
                print(f"Warning: Could not resolve redundancy for archetype {l + 1}.")
            return np.zeros((self.nDim, 1)), True

        sum_vec = np.zeros((self.nDim, 1))
        for i in range(self.nData):
            alfa_il = max(self.alfa_[l, i], 1e-6)
            # Reconstruct point i without archetype l
            recon_no_l = np.zeros((self.nDim, 1))
            for k in range(self.nArchetypes):
                if k != l:
                    recon_no_l[:, 0] += self.alfa_[k, i] * self.archetypes_[:, k]

            vi = (self.X[:, i].reshape(-1, 1) - recon_no_l) / alfa_il
            sum_vec += (self.alfa_[l, i] ** 2) * vi

        v_bar = sum_vec / alfa_l_sq_sum
        return v_bar, False

    def _find_new_archetype_candidate(self, k: int) -> tuple[np.ndarray, int]:
        """Finds a new candidate for a redundant archetype."""
        # Use current approximation error to find furthest point
        Xapprox = np.matmul(self.archetypes_, self.alfa_)
        RSSi = np.sum((self.X - Xapprox) ** 2, axis=0)

        top_k = min(10, self.nData)
        idMaxes = np.argsort(RSSi)[-top_k:]

        rng = np.random.default_rng(self.randomState)
        idMax = rng.choice(idMaxes)
        self.randomState += 1  # Update seed

        return self.X[:, idMax], int(idMax)

    def _rankArchetypes(self):
        """Ranks archetypes by their individual explained variance."""
        scores = []
        for i in range(self.nArchetypes):
            Xapprox = np.matmul(self.archetypes_[:, i : i + 1], self.alfa_[i : i + 1, :])
            scores.append(explainedVariance(self.X, Xapprox, method="sklearn"))

        rank = np.argsort(scores)[::-1]
        self.archetypes_ = self.archetypes_[:, rank]
        self.alfa_ = self.alfa_[rank, :]
        self.beta_ = self.beta_[:, rank]

    def _extractArchetypeProfiles(self):
        """Computes the feature profiles for each archetype."""
        self.archetypeProfile = np.zeros((self.nDim, self.nArchetypes))
        for i in range(self.nArchetypes):
            for j in range(self.nDim):
                self.archetypeProfile[j, i] = ecdf(self.X[j, :], self.archetypes_[j, i])

    # Compatibility properties for old code
    @property
    def archetypes(self):
        return self.archetypes_

    @property
    def alfa(self):
        return self.alfa_

    @property
    def beta(self):
        return self.beta_

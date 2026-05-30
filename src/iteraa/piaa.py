"""Parallel and iterative archetypal analysis utilities.

This module provides functions for splitting data into subsets, submitting jobs
to HPC schedulers, and combining results from distributed AA runs.
"""

import pickle
from os import listdir, mkdir
from os.path import exists, isfile
from subprocess import run
from time import time

import numpy as np
from natsort import natsorted
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from spams import archetypalAnalysis

from iteraa.constants import (
    AA_SCRIPT_PATH,
    JOBSCRIPTS_DIR_PATH,
    OUTPUTS_PICKLES_PATH,
    RANDOM_STATE,
    SUBSETS_PICKLES_PATH,
)
from iteraa.iaa import ArchetypalAnalysis


def subsetSplit(
    X: np.ndarray,
    nSubsets: int,
    dataName: str,
    subsetsSampleIdxs: list[np.ndarray] = [],
    subsetsPicklesPath: str = SUBSETS_PICKLES_PATH,
    postfixStr: str = "",
    shuffle: bool = True,
    randomState: int = RANDOM_STATE,
    verbose: bool = False,
) -> float:
    """Splits data into subsets and saves them as pickle files.

    Args:
        X: The dataset to split.
        nSubsets: Number of subsets.
        dataName: Name of the dataset for filename generation.
        subsetsSampleIdxs: Pre-defined sample indices for subsets.
        subsetsPicklesPath: Path to save subset pickles.
        postfixStr: Postfix for the saved filenames.
        shuffle: Whether to shuffle data before splitting.
        randomState: Random seed for shuffling.
        verbose: Whether to print progress information.

    Returns:
        The time spent on subset splitting in seconds.
    """
    startTime = time()
    if verbose:
        print(f"Splitting data into {nSubsets} subsets...")

    if not subsetsSampleIdxs:
        kFold = KFold(n_splits=nSubsets, shuffle=shuffle, random_state=randomState)
        subsetsSampleIdxs = [idxs for (_, idxs) in kFold.split(X)]

    if not exists(subsetsPicklesPath):
        mkdir(subsetsPicklesPath)

    for i, idxs in enumerate(subsetsSampleIdxs):
        if verbose:
            print(f"  Subset {i + 1}")
        fpath = f"{subsetsPicklesPath}/{dataName}data{i + 1}{postfixStr}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump((idxs, X[idxs, :].T), f)

    runTime = time() - startTime
    if verbose:
        print(f"Time spent on subset-splitting: {runTime:.3f} s")

    return runTime


def submitAAjobs(
    nArchetypes: int,
    dataName: str,
    splitKeyword: str = "data",
    postfixStr: str = "",
    jobscriptsDirPath: str = JOBSCRIPTS_DIR_PATH,
    subsetsPicklesPath: str = SUBSETS_PICKLES_PATH,
    outputsPicklesPath: str = OUTPUTS_PICKLES_PATH,
    AAscriptPath: str = AA_SCRIPT_PATH,
    project: str = "q27",
    queue: str = "normal",
    numCPUs: int = 48,
    wallTime: str = "00:05:00",
    mem: int = 5,
    jobFS: int = 1,
    email: str = "Jonathan.Ting@anu.edu.au",
    verbose: bool = False,
):
    """Submits AA jobs to a PBS-based job scheduler.

    Args:
        nArchetypes: Number of archetypes.
        dataName: Name of the dataset.
        splitKeyword: Keyword identifying data subsets.
        postfixStr: Postfix for subset filenames.
        jobscriptsDirPath: Directory to store generated jobscripts.
        subsetsPicklesPath: Path to subset pickles.
        outputsPicklesPath: Path to store output pickles.
        AAscriptPath: Path to the AA execution script.
        project: HPC project code.
        queue: HPC queue name.
        numCPUs: Number of CPUs per job.
        wallTime: Wall time limit for jobs.
        mem: Memory limit (GB).
        jobFS: JobFS limit (GB).
        email: Email for job notifications.
        verbose: Whether to print progress information.
    """
    subsetsPickles = [
        f
        for f in listdir(subsetsPicklesPath)
        if isfile(f"{subsetsPicklesPath}/{f}")
        and f"{dataName}{splitKeyword}" in f
        and f"{postfixStr}.pkl" in f
    ]

    if verbose:
        print("Generating archetypal analysis HPC jobs for all subsets...")
    if not exists(jobscriptsDirPath):
        mkdir(jobscriptsDirPath)

    for subsetPickle in subsetsPickles:
        jobscriptPath = f"{jobscriptsDirPath}/{subsetPickle.split('.')[0]}.sh"
        with open(jobscriptPath, "w") as f:
            f.write(f"#!/bin/bash\n#PBS -P {project}\n#PBS -q {queue}\n")
            f.write(f"#PBS -l ncpus={numCPUs},walltime={wallTime},mem={mem}GB,jobfs={jobFS}GB\n")
            f.write(f"#PBS -l storage=scratch/{project}\n#PBS -l wd\n")
            f.write(f"#PBS -M {email}\n#PBS -m a\n\n")
            f.write("module load python3/3.10.4\n\n")
            f.write("cd $PBS_O_WORKDIR\n")
            f.write(
                f"python3 {AAscriptPath} {subsetsPicklesPath}/{subsetPickle} "
                f"{nArchetypes} {outputsPicklesPath} {splitKeyword}"
            )

        run(["qsub", jobscriptPath], check=False)
        if verbose:
            print(f"  Submitted job for {subsetPickle}...")

    if verbose:
        print("All jobs submitted!")


def runAA(
    fName: str,
    nArchetypes: int,
    outputsPicklesPath: str = OUTPUTS_PICKLES_PATH,
    splitKeyword: str = "data",
    robust: bool = False,
    tolerance: float = 0.001,
    computeXtX: bool = False,
    stepsFISTA: int = 3,
    stepsAS: int = 50,
    randominit: bool = False,
    numThreads: int = -1,
    onlyZ: bool = False,
):
    """Executes archetypal analysis on a single data subset.

    Args:
        fName: Path to the subset pickle file.
        nArchetypes: Number of archetypes.
        outputsPicklesPath: Directory to save the output pickle.
        splitKeyword: Keyword identifying data subsets.
        robust: Whether to use robust AA.
        tolerance: Convergence tolerance.
        computeXtX: Whether to compute XtX.
        stepsFISTA: Number of FISTA steps.
        stepsAS: Number of active subset steps.
        randominit: Whether to use random initialization.
        numThreads: Number of threads.
        onlyZ: If True, only archetypes are saved.
    """
    startTime = time()
    with open(fName, "rb") as f:
        idxs, subsetX = pickle.load(f)

    subsetZ, subsetA, subsetB = archetypalAnalysis(
        np.asfortranarray(subsetX),
        Z0=None,
        p=nArchetypes,
        returnAB=True,
        robust=robust,
        epsilon=tolerance,
        computeXtX=computeXtX,
        stepsFISTA=stepsFISTA,
        stepsAS=stepsAS,
        randominit=randominit,
        numThreads=numThreads,
    )

    fname_only = fName.split("/")[-1]
    dataName = fname_only.split(splitKeyword)[0]
    subsetID = fname_only.split(splitKeyword)[-1].split(".pkl")[0]

    outputsDict = {"subsetZ": subsetZ, "runTime": time() - startTime}
    if not onlyZ:
        outputsDict["subsetA"] = subsetA.toarray()
        outputsDict["subsetB"] = subsetB.toarray()
        outputsDict["subsetsSampleIdxs"] = idxs

    if not exists(outputsPicklesPath):
        mkdir(outputsPicklesPath)

    out_path = f"{outputsPicklesPath}/{dataName}output{subsetID}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(outputsDict, f)


def fitPIAA(
    X: np.ndarray,
    nArchetypes: int,
    numSubset: int,
    dataName: str,
    outputsPicklesPath: str = OUTPUTS_PICKLES_PATH,
    postfixStr: str = "",
    shuffle: bool = True,
    robust: bool = False,
    onlyZ: bool = False,
    C: float = 0.0001,
    tolerance: float = 0.001,
    computeXtX: bool = False,
    stepsFISTA: int = 3,
    stepsAS: int = 50,
    randominit: bool = False,
    randomState: int = RANDOM_STATE,
    numThreads: int = -1,
    splitRunTime: float = 0.0,
    verbose: bool = True,
) -> ArchetypalAnalysis:
    """Combines results from distributed AA runs into a final model.

    Args:
        X: The full dataset.
        nArchetypes: Number of archetypes.
        numSubset: Number of subsets used.
        dataName: Name of the dataset.
        outputsPicklesPath: Path to subset output pickles.
        postfixStr: Postfix for filenames.
        shuffle: Whether shuffling was used.
        robust: Whether robust AA was used.
        onlyZ: Whether only archetypes were computed.
        C: Constraint coefficient.
        tolerance: Convergence tolerance.
        computeXtX: Whether to compute XtX.
        stepsFISTA: Number of FISTA steps.
        stepsAS: Number of active subset steps.
        randominit: Whether random initialization was used.
        randomState: Random seed.
        numThreads: Number of threads.
        splitRunTime: Time spent on subset splitting.
        verbose: Whether to print progress information.

    Returns:
        The fitted ArchetypalAnalysis object.
    """
    startTime = time()
    AA = ArchetypalAnalysis(
        nArchetypes=nArchetypes,
        iterative=True,
        robust=robust,
        onlyZ=onlyZ,
        nSubsets=numSubset,
        shuffle=shuffle,
        C=C,
        tolerance=tolerance,
        computeXtX=computeXtX,
        stepsFISTA=stepsFISTA,
        stepsAS=stepsAS,
        randominit=randominit,
        randomState=randomState,
        numThreads=numThreads,
        verbose=verbose,
    )
    AA.X = X.T
    AA.nDim, AA.nData = AA.X.shape

    subsetsZs = []
    subsetsAs = []
    subsetsBs = []
    subsetsSampleIdxs = []
    runTimes = []

    for f in natsorted(listdir(outputsPicklesPath)):
        if (
            not isfile(f"{outputsPicklesPath}/{f}")
            or f"{dataName}output" not in f
            or postfixStr not in f
        ):
            continue
        if verbose:
            print(f"  Loading subset: {f}")
        with open(f"{outputsPicklesPath}/{f}", "rb") as file:
            out = pickle.load(file)
        subsetsZs.append(out["subsetZ"])
        runTimes.append(out["runTime"])
        if not onlyZ:
            subsetsAs.append(out["subsetA"])
            subsetsBs.append(out["subsetB"])
            subsetsSampleIdxs.append(out["subsetsSampleIdxs"])

    AA.subsetsSampleIdxs = subsetsSampleIdxs
    AA._combine_subset_results(subsetsZs, subsetsAs, subsetsBs)

    AA.runTime_ = time() - startTime + splitRunTime + max(runTimes)

    if not onlyZ:
        # Standard finalize_fit without re-fitting
        AA._rankArchetypes()
        Xapprox = np.matmul(AA.archetypes_, AA.alfa_)
        AA.RSS_ = np.sum((AA.X - Xapprox) ** 2)
        AA.explainedVariance_ = explained_variance_score(AA.X.T, Xapprox.T)
        AA._extractArchetypeProfiles()
        if verbose:
            print(f"Explained variance: {AA.explainedVariance_:.3f}")

    return AA

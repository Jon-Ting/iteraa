from os import listdir, mkdir
from os.path import exists, isfile
import pickle
from subprocess import run
import sys
from time import time

from natsort import natsorted
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from spams import archetypalAnalysis

from iteraa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI, SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, JOBSCRIPTS_DIR_PATH, AA_SCRIPT_PATH
from iteraa.iaa import ArchetypalAnalysis


def subsetSplit(X, nSubsets, dataName, subsetsSampleIdxs=[], subsetsPicklesPath=SUBSETS_PICKLES_PATH, postfixStr='',
                shuffle=True, randomState=RANDOM_STATE, verbose=False):
    """Split data into subsets.

    Parameters
    ----------
    X : numpy.ndarray
        Whole data set.
    numSubset : int
        Number of subsets.
    subsetsSampleIdxs : list[int]
        Identifiers for subset samples.
    dataName : str
        Name of dataset.
    subsetsPicklesPath : str
        Path to directory containing data subset pickle files.
    postfixStr : str
        Postfix name.
    shuffle : bool
        Whether to shuffle data.
    randomState : int
        Random seed.
    verbose : bool
        Whether information is printed.

    Returns
    -------
    runTime : float
        Duration of execution.
    """
    startTime = time()
    if verbose:
        print(f"Splitting data into {nSubsets} subsets...")
    if len(subsetsSampleIdxs) == 0:
        kFold = KFold(n_splits=nSubsets, shuffle=shuffle, random_state=randomState)
        subsetsSampleIdxs = [idxs for (_, idxs) in kFold.split(X)]
        
    subsetsAs, subsetsBs = [], []
    for (i, idxs) in enumerate(subsetsSampleIdxs):
        if verbose:
            print(f"  Subset {i + 1}")
        with open(f"{subsetsPicklesPath}/{dataName}data{i + 1}{postfixStr}.pkl", 'wb') as f:
            pickle.dump((idxs, X[idxs, :].T), f)  # subsetX
            
    runTime = time() - startTime
    if verbose:
        print(f"Time spent on subset-splitting: {runTime:.3f} s")
        
    return runTime


def submitAAjobs(nArchetypes, dataName, splitKeyword='data', postfixStr='', 
                 jobscriptsDirPath=JOBSCRIPTS_DIR_PATH, 
                 subsetsPicklesPath=SUBSETS_PICKLES_PATH,
                 outputsPicklesPath=OUTPUTS_PICKLES_PATH, 
                 AAscriptPath=AA_SCRIPT_PATH,
                 project='q27', queue='normal', numCPUs=48, wallTime='00:05:00', mem=5, jobFS=1, email='Jonathan.Ting@anu.edu.au',
                 verbose=False):
    """Submit jobs of executing individual archetypal analysis to a job scheduler.

    Parameters
    ----------
    X : numpy.ndarray
        Whole data set.
    dataName : str
        Name of dataset.
    splitKeyword : str
        Keyword for data subsets.
    postfixStr : str
        Postfix name.
    jobscriptsPicklesPath : str
        Path to directory containing data jobscript files.
    subsetsPicklesPath : str
        Path to directory containing data subset pickle files.
    outputsPicklesPath : str
        Path to directory containing data output pickle files.
    AAscriptPath : str
        Path to script to run archetypal analysis.
    verbose : bool
        Whether information is printed.

    Returns
    -------
    None
    """
    subsetsPickles = [fName for fName in listdir(subsetsPicklesPath) 
                      if isfile(f"{subsetsPicklesPath}/{fName}") and f"{dataName}{splitKeyword}" in fName and f"{postfixStr}.pkl" in fName]
    if verbose:
        print('Generating archetypal analysis HPC jobs for all subsets...')
    if not exists(jobscriptsDirPath):
        mkdir(jobscriptsDirPath)
    for subsetsPickle in subsetsPickles:
        jobscriptPath = f"{jobscriptsDirPath}/{subsetsPickle.split('.')[0]}.sh"
        with open(jobscriptPath, 'w') as f:
            f.write(f"#!/bin/bash\n#PBS -P {project}\n#PBS -q {queue}\n")
            f.write(f"#PBS -l ncpus={numCPUs},walltime={wallTime},mem={mem}GB,jobfs={jobFS}GB\n")
            f.write(f"#PBS -l storage=scratch/{project}\n#PBS -l wd\n")
            f.write(f"#PBS -M {email}\n#PBS -m a\n\n")
            f.write("module load python3/3.10.4\n\n")
            f.write(f"cd $PBS_O_WORKDIR\npython3 {AAscriptPath} {subsetsPicklesPath}/{subsetsPickle} {nArchetypes} {outputsPicklesPath} {splitKeyword}")
        run(['qsub', jobscriptPath])
        if verbose:
            print(f"  Submitted job for {subsetsPickle}...")
    if verbose:
        print(f"All jobs submitted!")


def runAA(fName, nArchetypes, outputsPicklesPath=OUTPUTS_PICKLES_PATH, splitKeyword='data',
          robust=False, tolerance=0.001, computeXtX=False, stepsFISTA=3, stepsAS=50, 
          randominit=False, numThreads=-1, onlyZ=False):
    """Executes archetypal analysis.

    Parameters
    ----------
    fName : str
        Path to pickle file containing data.
    nArchetypes : int
        Number of archetypes.
    outputsPicklesPath : str
        Path to directory containing output pickle files.
    splitKeyword : str
        Keyword for data subsets.
    robust : bool
        Whether to use robust archetypal analysis.
    tolerance: float
        Tolerance.
    computeXtX : bool
        Whether to compute XtX.
    stepsFISTA : int
        Number of FISTA steps.
    stepsAS : int
        Number of active subset steps.
    randominit : bool
        Whether to initialise randomly.
    numThreads : int
        Number of threads for algorithm execution.
    onlyZ : bool
        Whether to stop early by returning only Z matrix.

    Returns
    -------
    None
    """
    startTime = time()
    with open(fName, 'rb') as f:
        idxs, subsetX = pickle.load(f)
    subsetZ, subsetA, subsetB = archetypalAnalysis(np.asfortranarray(subsetX), Z0=None, p=nArchetypes, 
                                                      returnAB=True, robust=robust, epsilon=tolerance, computeXtX=computeXtX, 
                                                      stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                                                      numThreads=numThreads)
    dataName = fName.split('/')[-1].split(splitKeyword)[0]
    subsetID = fName.split(splitKeyword)[-1].split('.pkl')[0]
    outputsDict = {'subsetZ': subsetZ, 'runTime': time() - startTime}
    if not onlyZ:    
        outputsDict['subsetA'] = subsetA.toarray()
        outputsDict['subsetB'] = subsetB.toarray()
        outputsDict['subsetsSampleIdxs'] = idxs
    with open(f"{outputsPicklesPath}/{dataName}output{subsetID}.pkl", 'wb') as f:
        pickle.dump(outputsDict, f)


def fitPIAA(X, nArchetypes, numSubset, dataName, outputsPicklesPath=OUTPUTS_PICKLES_PATH, postfixStr='', 
            shuffle=True, robust=False, onlyZ=False, C=0.0001, tolerance=0.001, computeXtX=False, 
            stepsFISTA=3, stepsAS=50, randominit=False, randomState=RANDOM_STATE, numThreads=-1, 
            splitRunTime=0.0, verbose=True):
    """Combining results from individual archetypal analysis runs to obtain final archetypes.

    Parameters
    ----------
    X : numpy.ndarray
        Whole data set.
    nArchetypes : int
        Number of archetypes.
    numSubset : int
        Number of subsets.
    dataName : str
        Name of dataset.
    outputsPicklesPath : str
        Path to directory containing output pickle files.
    postfixStr : str
        Postfix name.
    shuffle : bool
        Whether to shuffle data.
    robust : bool
        Whether to use robust archetypal analysis.
    onlyZ : bool
        Whether to stop early by returning only Z matrix.
    C : float
        C.
    tolerance: float
        Tolerance.
    computeXtX : bool
        Whether to compute XtX.
    stepsFISTA : int
        Number of FISTA steps.
    stepsAS : int
        Number of active subset steps.
    randominit : bool
        Whether to initialise randomly.
    randomState : int
        Random seed.
    numThreads : int
        Number of threads for algorithm execution.
    splitRuntime : float
        Execution duration for splitting of data into subsets (seconds)
    verbose : bool
        Whether information is printed.

    Returns
    -------
    AA : ArchetypalAnalysis object
        Object with fitted results.
    """
    startTime = time()
    # Initialise AA object to be filled in
    AA = ArchetypalAnalysis(nArchetypes=nArchetypes, iterative=True, robust=robust, onlyZ=onlyZ,  subsetsSampleIdxs=[], nSubsets=numSubset, shuffle=shuffle, 
                            C=C, tolerance=tolerance, computeXtX=computeXtX, stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                            randomState=randomState, numThreads=numThreads)
    AA.X = X.T
    AA.nDim, AA.nData = AA.X.shape
    
    AA.subsetsZs, subsetsAs, subsetsBs, AA.sampleIdxs, runTimes = [], [], [], [], []
    for fName in natsorted(listdir(outputsPicklesPath)):
        if not isfile(f"{outputsPicklesPath}/{fName}") or f"{dataName}output" not in fName or f"{postfixStr}.pkl" not in fName:
            continue
        if verbose:
            print(f"  Subset: {fName}")
        with open(f"{outputsPicklesPath}/{fName}", 'rb') as f:
            outputsDict = pickle.load(f)
        AA.subsetsZs.append(outputsDict['subsetZ'])
        subsetsAs.append(outputsDict['subsetA'])
        subsetsBs.append(outputsDict['subsetB'])
        AA.subsetsSampleIdxs.append(outputsDict['subsetsSampleIdxs'])
        runTimes.append(outputsDict['runTime'])
        if verbose:
            print(f": {outputsDict['runTime']} s")
    allSubsetsZs = np.concatenate(AA.subsetsZs, axis=1)  # (m*(k*p))
    AA.archetypes, Afinal, Bfinal = archetypalAnalysis(np.asfortranarray(allSubsetsZs), Z0=None, p=nArchetypes, 
                                                         returnAB=True, robust=robust, epsilon=tolerance, computeXtX=computeXtX, 
                                                         stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                                                         numThreads=numThreads)
    AA.runTime = time() - startTime + splitRunTime + max(runTimes)
    if AA.onlyZ:
        return AA
    # Rearrange the sample indices for subsequent comparison with the original data
    allSampleIdxs = np.concatenate(AA.subsetsSampleIdxs, axis=0)
    sortedXapproxIdxs = np.array(sorted(zip(range(len(allSampleIdxs)), allSampleIdxs), key=lambda tup: tup[1]))[:, 0]
    # Reconstruct data (n*m)
    A, B = Afinal.toarray(), Bfinal.toarray()
    allSubsetsZsApprox = np.matmul(AA.archetypes, A)  # (m*(k*p))
    prevIdx, subsetsZsApproxs, subsetsOverallAs, subsetsOverallBs = 0, [], [], []
    for (i, subsetZ) in enumerate(AA.subsetsZs):
        nSubsetZs = subsetZ.shape[1]
        subsetZapprox = allSubsetsZsApprox[:, prevIdx:prevIdx + nSubsetZs]  # Reconstructed subset archetypes from final archetypes
        subsetsOverallA = np.matmul(A[:, prevIdx:prevIdx + nSubsetZs], subsetsAs[i])
        subsetsOverallB = np.matmul(subsetsBs[i], B[prevIdx:prevIdx + nSubsetZs, :])
        prevIdx += nSubsetZs
        
        subsetsOverallAs.append(subsetsOverallA)
        subsetsOverallBs.append(subsetsOverallB)
        subsetsZsApproxs.append(subsetZapprox)
    AA.alfa = np.concatenate(subsetsOverallAs, axis=1)[:, sortedXapproxIdxs]
    AA.beta = np.concatenate(subsetsOverallBs, axis=0)[sortedXapproxIdxs]
    AA._rankArchetypes()
    AA.Xapx = np.matmul(AA.archetypes, AA.alfa)  # Note: self.archetypes = np.matmul(self.X, self.beta)
    AA.RSS_2 = ((AA.X - AA.Xapx) ** 2).sum()
    AA.explainedVariance_ = explained_variance_score(AA.X.T, AA.Xapx.T)
    AA._extractArchetypeProfiles()
    if verbose:
        print(f"Explained variance: {AA.explainedVariance_:.3f}")
    return AA


if __name__ == '__main__':
    runAA(fName=sys.argv[1], nArchetypes=int(sys.argv[2]), outputsPicklesPath=sys.argv[3], splitKeyword=sys.argv[4])

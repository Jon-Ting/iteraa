# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:39:27 2019
Modified on Mon July 1 14:17:29 2024 by Jonathan Yik Chang Ting

@original author: Benyamin Motevalli

This class is developed based on "Archetypa Analysis" by Adele Cutler and Leo
Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347
"""

from math import *
from os import listdir
from os.path import isfile
import pickle
import sys
from time import time

from natsort import natsorted
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from spams import archetypalAnalysis

from archetypes import ArchetypalAnalysis



RANDOM_STATE = 42
NUM_JOBS = 8
PALETTE = 'viridis'
DPI = None
SUBSETS_PICKLES_PATH = 'data/subsetsDataPKLs'
OUTPUTS_PICKLES_PATH = 'data/subsetsOutputsPKLs'



def subsetSplit(X, n_subsets, dataName, subsetsPicklesPath=SUBSETS_PICKLES_PATH,
                shuffle=True, random_state=RANDOM_STATE, verbose=False):
    startTime = time()
    if verbose:
        print(f"Splitting data into {n_subsets} subsets...")
    kFold = KFold(n_splits=n_subsets, shuffle=shuffle, random_state=random_state)
    subsets_As, subsets_Bs = [], []
    for (i, (_, idxs)) in enumerate(kFold.split(X)):
        if verbose:
            print(f"  Subset {i + 1}")
        with open(f"{subsetsPicklesPath}/{dataName}data{i + 1}.pkl", 'wb') as f:
            pickle.dump((idxs, X[idxs, :].T), f)  # subset_X
    return time() - startTime


def runAA(fName, n_archetypes, outputsPicklesPath=OUTPUTS_PICKLES_PATH, 
          robust=False, tolerance=0.001, computeXtX=False, stepsFISTA=3, stepsAS=50, 
          randominit=False, numThreads=-1, only_archetypes=False):
    startTime = time()
    with open(fName, 'rb') as f:
        idxs, subset_X = pickle.load(f)
    subset_Z, subset_A, subset_B = archetypalAnalysis(np.asfortranarray(subset_X), Z0=None, p=n_archetypes, 
                                                      returnAB=True, robust=robust, epsilon=tolerance, computeXtX=computeXtX, 
                                                      stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                                                      numThreads=numThreads)
    print(subset_Z)
    dataName = fName.split('/')[-1].split('data')[0]
    subsetID = fName.split('data')[-1].split('.pkl')[0]
    outputsDict = {'subset_Z': subset_Z, 'runTime': time() - startTime}
    if not only_archetypes:    
        outputsDict['subset_A'] = subset_A.toarray()
        outputsDict['subset_B'] = subset_B.toarray()
        outputsDict['subsets_sample_idxs'] = idxs
    with open(f"{outputsPicklesPath}/{dataName}output{subsetID}.pkl", 'wb') as f:
        pickle.dump(outputsDict, f)


def fitPIAA(X, n_archetypes, numSubset, dataName, outputsPicklesPath=OUTPUTS_PICKLES_PATH,
            shuffle=True, robust=False, only_archetypes=False, C=0.0001, tolerance=0.001, computeXtX=False, 
            stepsFISTA=3, stepsAS=50, randominit=False, random_state=RANDOM_STATE, numThreads=-1, 
            splitRunTime=0.0, verbose=True):
    startTime = time()
    # Initialise AA object to be filled in
    AA = ArchetypalAnalysis(n_archetypes=n_archetypes, iterative=True, robust=robust, only_archetypes=only_archetypes, n_subsets=numSubset, shuffle=shuffle, 
                            C=C, tolerance=tolerance, computeXtX=computeXtX, stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                            random_state=random_state, numThreads=numThreads)
    AA.X = X.T
    AA.n_dim, AA.n_data = AA.X.shape
    
    AA.subsets_Zs, subsets_As, subsets_Bs, AA.sampleIdxs, runTimes = [], [], [], [], []
    for fName in natsorted(listdir(outputsPicklesPath)):
        if not isfile(f"{outputsPicklesPath}/{fName}") or '.pkl' not in fName or f"{dataName}output" not in fName:
            continue
        if verbose:
            print(f"  Subset: {fName}")
        with open(f"{outputsPicklesPath}/{fName}", 'rb') as f:
            outputsDict = pickle.load(f)
        AA.subsets_Zs.append(outputsDict['subset_Z'])
        subsets_As.append(outputsDict['subset_A'])
        subsets_Bs.append(outputsDict['subset_B'])
        AA.subsets_sample_idxs.append(outputsDict['subsets_sample_idxs'])
        runTimes.append(outputsDict['runTime'])
    all_subsets_Zs = np.concatenate(AA.subsets_Zs, axis=1)  # (m*(k*p))
    AA.archetypes, A_final, B_final = archetypalAnalysis(np.asfortranarray(all_subsets_Zs), Z0=None, p=n_archetypes, 
                                                         returnAB=True, robust=robust, epsilon=tolerance, computeXtX=computeXtX, 
                                                         stepsFISTA=stepsFISTA, stepsAS=stepsAS, randominit=randominit, 
                                                         numThreads=numThreads)
    AA.run_time = time() - startTime + splitRunTime + max(runTimes)
    if AA.only_archetypes:
        return
    # Rearrange the sample indices for subsequent comparison with the original data
    allSampleIdxs = np.concatenate(AA.subsets_sample_idxs, axis=0)
    sortedXapproxIdxs = np.array(sorted(zip(range(len(allSampleIdxs)), allSampleIdxs), key=lambda tup: tup[1]))[:, 0]
    # Reconstruct data (n*m)
    A, B = A_final.toarray(), B_final.toarray()
    all_subsets_Zs_approx = np.matmul(AA.archetypes, A)  # (m*(k*p))
    prev_idx, subsets_Zs_approxs, subsets_overall_As, subsets_overall_Bs = 0, [], [], []
    for (i, subset_Z) in enumerate(AA.subsets_Zs):
        n_subset_archetypes = subset_Z.shape[1]
        subset_Z_approx = all_subsets_Zs_approx[:, prev_idx:prev_idx + n_subset_archetypes]  # Reconstructed subset archetypes from final archetypes
        subset_overall_A = np.matmul(A[:, prev_idx:prev_idx + n_subset_archetypes], subsets_As[i])
        subset_overall_B = np.matmul(subsets_Bs[i], B[prev_idx:prev_idx + n_subset_archetypes, :])
        prev_idx += n_subset_archetypes
        
        subsets_overall_As.append(subset_overall_A)
        subsets_overall_Bs.append(subset_overall_B)
        subsets_Zs_approxs.append(subset_Z_approx)
    AA.alfa = np.concatenate(subsets_overall_As, axis=1)[:, sortedXapproxIdxs]
    AA.beta = np.concatenate(subsets_overall_Bs, axis=0)[sortedXapproxIdxs]
    AA._rank_archetypes()
    AA.X_approx = np.matmul(AA.archetypes, AA.alfa)  # Note: self.archetypes = np.matmul(self.X, self.beta)
    AA.RSS_2 = ((AA.X - AA.X_approx) ** 2).sum()
    AA.explained_variance_ = explained_variance_score(AA.X.T, AA.X_approx.T)
    AA._extract_archetype_profiles()
    if verbose:
        print(f"Explained variance: {AA.explained_variance_:.3f}")
    return AA


if __name__ == '__main__':
    runAA(fName=sys.argv[1], n_archetypes=int(sys.argv[2]), outputsPicklesPath=OUTPUTS_PICKLES_PATH)
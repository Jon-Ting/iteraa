import pickle
import sys
from time import time

import numpy as np
from spams import archetypalAnalysis



def runAA(fName, nArchetypes, outputsPicklesPath='.',
          robust=False, tolerance=0.001, computeXtX=False, stepsFISTA=3, stepsAS=50, 
          randominit=False, numThreads=-1, onlyZ=False):
    """Executes archetypal analysis.

    Parameters
    ----------
    fName : str
        Path to pickle file containing data.

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
    print(subsetZ)
    dataName = fName.split('/')[-1].split('data')[0]
    subsetID = fName.split('data')[-1].split('.pkl')[0]
    outputsDict = {'subsetZ': subsetZ, 'runTime': time() - startTime}
    if not onlyZ:    
        outputsDict['subsetA'] = subsetA.toarray()
        outputsDict['subsetB'] = subsetB.toarray()
        outputsDict['subsetsSampleIdxs'] = idxs
    with open(f"{outputsPicklesPath}/{dataName}output{subsetID}.pkl", 'wb') as f:
        pickle.dump(outputsDict, f)


if __name__ == '__main__':
    runAA(fName=sys.argv[1], nArchetypes=int(sys.argv[2]), outputsPicklesPath=sys.argv[3])


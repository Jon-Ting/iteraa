# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:39:27 2019
Modified on Mon Jan 15 13:15:29 2024 by Jonathan Yik Chang Ting

@original author: Benyamin Motevalli

This class is developed based on "Archetypal Analysis" by Adele Cutler and Leo
Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347
"""

from copy import copy
from math import pi
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from spams import archetypalAnalysis

from iaa.constants import RANDOM_STATE, DPI, FIGS_DIR_PATH
from iaa.utils import ecdf, calcSSE, explainedVariance, solveConstrainedNNLS, furthestSum


class ArchetypalAnalysis():
    """
    Parameters:
    -----------
           
    nArchetypes:   Defines the number of archetypes.
    nDim:          Number of features (Dimensions).
    nData:         Number of data in dataset.
    
    X:              
        Dimension:  nDim x nData
        
                    Array of data points. It is the transpose of input data.
                    
    archetypes:
        
        Dimension:  nDim x nArchetypes
        
                    Array of archetypes. Each columns represents an archetype.
    
    alfa:
        Dimension:  nArchetypes x nData
        
                    Each column defines the weight coefficients for each 
                    archetype to approximate data point i.
                    Xi = sum([alfa]ik x Zk)
    
    beta:
        Dimension:  nData x nArchetypes
        
                    Each column defines the weight coefficients for each data 
                    point from which archetype k is constructed.
    
    tolerance:      Defines when to stop optimization.

    maxIter:       Defines the maximum number of iterations

    randomState:   Defines the random seed number for initialization. No effect if "furthestSum" is selected.       
                    
    C:              is a constraint coefficient to ensure that the summation of
                    alfa's and beta's equals to 1. C is conisdered to be inverse
                    of M^2 in the original paper.

    initialize:     Defines the initialization method to guess initial archetypes:

                        1. furthestSum (Default): the idea and code taken from https://github.com/ulfaslak/py_pcha and the original author is: Ulf Aslak Jensen.
                        2. random:  Randomly selects archetypes in the feature space. The points could be any point in space
                        3. randomIdx:  Randomly selects archetypes from points in the dataset.
    """
    def __init__(self, nArchetypes=2, iterative=False, nSubsets=10, shuffle=True, onlyZ=False, subsetsSampleIdxs=[],
                 tolerance=0.001, maxIter=200, randomState=RANDOM_STATE, C=0.0001, initialize='furthestSum', redundancyTry=30, 
                 robust=False, computeXtX=False, stepsFISTA=3, stepsAS=50, randominit=False, numThreads=-1, verbose=False):
        self.nArchetypes = nArchetypes
        self.nDim = None
        self.nData = None
        self.iterative = iterative
        self.nSubsets = nSubsets
        self.shuffle = shuffle
        self.onlyZ = onlyZ  # Stop running right after archetypes are computed, other functionalities won't be usable
        self.tolerance = tolerance
        self.C = C
        self.randomState = randomState
        self.archetypes = []
        self.alfa = []
        self.beta = []
        self.explainedVariance_ = []
        self.RSS_ = None
        self.RSS0_ = None
        self.RSSi_ = []
        self.initialize = initialize.lower()
        self.closeMatch = {}
        self.maxIter = maxIter
        self.redundancyTry = redundancyTry
        self.robust = robust
        self.computeXtX = computeXtX
        self.stepsFISTA = stepsFISTA
        self.stepsAS = stepsAS
        self.randominit = randominit
        self.numThreads = numThreads
        self.subsetsZs = []
        self.subsetsSampleIdxs = subsetsSampleIdxs
        self.runTime = 0.0
        self.verbose = verbose

    
    def fit(self, X):
        self.X = X.T
        self.nDim, self.nData = self.X.shape
        startTime = time()
        if self.iterative:
            # Split data into subsets and conduct archetypal analysis on each of them
            if len(subsetsSampleIdxs) == 0:
                kFold = KFold(n_splits=self.nSubsets, shuffle=self.shuffle, random_state=self.randomState)
                self.subsetsSampleIdxs = [idxs for (_, idxs) in kFold.split(X)]
            subsetsAs, subsetsBs = [], []
            for (i, idxs) in enumerate(self.subsetsSampleIdxs):
                if self.verbose:
                    print(f"Subset {i + 1}")
                subsetX = X[idxs, :].T
                subsetZ, subsetA, subsetB = archetypalAnalysis(np.asfortranarray(subsetX), Z0=None, p=self.nArchetypes, 
                                                                  returnAB=True, robust=self.robust, epsilon=self.tolerance, computeXtX=self.computeXtX, 
                                                                  stepsFISTA=self.stepsFISTA, stepsAS=self.stepsAS, randominit=self.randominit, 
                                                                  numThreads=self.numThreads)  # TODO: more flexible *nArchetypes*
                self.subsetsZs.append(subsetZ)
                if not self.onlyZ:    
                    subsetsAs.append(subsetA.toarray())
                    subsetsBs.append(subsetB.toarray())
            # Get final archetypes by conducting archetypal analysis on the archetypes obtained from each subset
            allSubsetsZs = np.concatenate(self.subsetsZs, axis=1)  # (m*(k*p))
            self.archetypes, Afinal, Bfinal = archetypalAnalysis(np.asfortranarray(allSubsetsZs), Z0=None, p=self.nArchetypes,
                                                                   returnAB=True, robust=self.robust, epsilon=self.tolerance, computeXtX=self.computeXtX, 
                                                                   stepsFISTA=self.stepsFISTA, stepsAS=self.stepsAS, randominit=self.randominit, 
                                                                   numThreads=self.numThreads)
            self.runTime = time() - startTime
            if self.onlyZ:
                return
            # Rearrange the sample indices for subsequent comparison with the original data
            allSampleIdxs = np.concatenate(self.subsetsSampleIdxs, axis=0)
            sortedXapproxIdxs = np.array(sorted(zip(range(len(allSampleIdxs)), allSampleIdxs), key=lambda tup: tup[1]))[:, 0]
            # Reconstruct data (n*m)
            A, B = Afinal.toarray(), Bfinal.toarray()
            allSubsetsZsApprox = np.matmul(self.archetypes, A)  # (m*(k*p))
            prevIdx, subsetsZsApproxs, subsetsOverallAs, subsetsOverallBs = 0, [], [], []
            for (i, subsetZ) in enumerate(self.subsetsZs):
                nSubsetZs = subsetZ.shape[1]
                subsetZapprox = allSubsetsZsApprox[:, prevIdx:prevIdx + nSubsetZs]  # Reconstructed subset archetypes from final archetypes
                subsetOverallA = np.matmul(A[:, prevIdx:prevIdx + nSubsetZs], subsetsAs[i])
                subsetOverallB = np.matmul(subsetsBs[i], B[prevIdx:prevIdx + nSubsetZs, :])
                prevIdx += nSubsetZs
                
                subsetsOverallAs.append(subsetOverallA)
                subsetsOverallBs.append(subsetOverallB)
                subsetsZsApproxs.append(subsetZapprox)
            self.alfa = np.concatenate(subsetsOverallAs, axis=1)[:, sortedXapproxIdxs]
            self.beta = np.concatenate(subsetsOverallBs, axis=0)[sortedXapproxIdxs]
        else:
            self.archetypes, A, B = archetypalAnalysis(np.asfortranarray(self.X), Z0=None, p=self.nArchetypes, 
                                         returnAB=True, robust=self.robust, epsilon=self.tolerance, computeXtX=self.computeXtX, 
                                         stepsFISTA=self.stepsFISTA, stepsAS=self.stepsAS, randominit=self.randominit, 
                                         numThreads=self.numThreads)
            self.runTime = time() - startTime
            if self.onlyZ:
                return
            self.alfa = A.toarray()
            self.beta = B.toarray()
        self._rankArchetypes()
        self.Xapprox = np.matmul(self.archetypes, self.alfa)  # Note: self.archetypes = np.matmul(self.X, self.beta)
        self.RSS_2 = calcSSE(self.X, self.Xapprox)
        self.explainedVariance_ = explainedVariance(self.X, self.Xapprox, method='sklearn')
        self._extractArchetypeProfiles()

    
    def fitClassical(self, X):
        self.X = X.T
        self.nDim, self.nData = self.X.shape
        self.RSS0_ = (self.X.var(axis = 1)).sum()
        self.RSS0_2 = (self.X.var(axis = 1) * self.nData).sum()
        
        if (self.nArchetypes == 1):
            self.archetypes = self.X.mean(axis=1).reshape(self.nDim, self.nArchetypes)
            self.alfa = np.ones([self.nArchetypes, self.nData])
            self.RSS_ = self.RSS0_
            self.RSS_2 = self.RSS0_2
        else:
            self._initializeArchetypes()
            self.RSS_ = 1.0
            RSSold = 100.0
            self.countConverg_ = 1
            while ((abs(self.RSS_ - RSSold)) / RSSold > self.tolerance):
                # print('old = ', RSSold, ' new = ', self.RSS_, 'err = ', abs(self.RSS_ - RSSold))
                RSSold = self.RSS_
                self._optimizeAlfa()
                hasError = self._optimizeBeta()
                if(hasError):
                    return
                self.archetypes = np.matmul(self.X, self.beta)
                self.RSS_ = self.RSSi_.sum()   
                if (self.countConverg_ > self.maxIter):
                    break
                self.countConverg_ += 1  
            self._rankArchetypes()
        self.Xapprox = np.matmul(self.archetypes , self.alfa)   
        self.RSS_2 = calcSSE(self.X, self.Xapprox)
        self.explainedVariance_ = explainedVariance(self.X, self.Xapprox, method = "_")
        self._extractArchetypeProfiles()
        
    
    def fit_transform(self, X):
        self.fit(X)
        Xapprox = np.matmul(self.archetypes , self.alfa)
        return Xapprox.T

    
    def transform(self, Xnew):
        XnewTrans = Xnew.T
        nData, nDim = Xnew.shape
        alfaNew, RSSi_new = self.__optimizeAlfaForTransform(Xnew, nData)
        # self.alfa, self.RSSi_ = self.__optimizeAlfaForTransform(Xnew, nData)
        XnewApprox = np.matmul(self.archetypes, alfaNew)
        return XnewApprox.T, alfaNew
    
    
    def _initializeArchetypes(self):
        if self.initialize == 'random':
            self._randomInitialize()
        elif self.initialize == 'randomIdx':
            self._randomIdxInitialize()
        elif self.initialize == 'furthestSum':
            try:
                self._furthestSumInitialize()
            except IndexError:
                class InitializationException(Exception): pass
                raise InitializationException("Initialization with furthest sum does not converge. Too few examples in dataset.\n"
                                              + "A random initialization is selected to continue.")
                self._randomInitialize()
            
            
    def _randomInitialize(self):
        from sklearn.preprocessing import MinMaxScaler
        np.random.seed(self.randomState)
        sc = MinMaxScaler()
        sc.fit(self.X.T)
        archsNorm = np.random.rand(self.nArchetypes, self.nDim)
        self.archetypesInit_ = sc.inverse_transform(archsNorm).T
        self.archetypes = copy(self.archetypesInit_)
        
        
    def _furthestSumInitialize(self):   
        init = [int(np.ceil(len(range(self.nData)) * np.random.rand()))]
        nDataRange = range(self.nData)
        initArchIdx = furthestSum(self.X[:, nDataRange], self.nArchetypes, init)
        
        self.archetypesInit_ = self.X[:,initArchIdx]
        self.archetypes = copy(self.archetypesInit_)
        
    
    def _randomIdxInitialize(self):
        lstIdx = []
        np.random.seed(self.randomState)
        for i in range(self.nArchetypes):
            lstIdx.append(np.random.randint(self.nData))
        self.archetypesInit_ = self.X[:,lstIdx]
        self.archetypes = copy(self.archetypesInit_)

    
    def __optimizeAlfaForTransform(self, Xnew, nData):
        """
        This functions aims to obtain corresponding alfa values for a new data
        point after the fitting is done and archetypes are determined.
        
        Having alfas, we can approximate the new data-points in terms of 
        archetypes.
        
        NOTE: Xnew dimension is nData x nDim. Here, the original data is passed
        instead of transpose.
        """
        alfa = np.zeros([self.nArchetypes, nData])
        RSSi_ = np.zeros([nData])
        for i, xi in enumerate(Xnew):
            alfa[:, i], RSSi_[i] = solveConstrainedNNLS(xi, self.archetypes, self.C)
        return alfa, RSSi_
    
    
    def _optimizeAlfa(self):
        """
        self.archetypes: has a shape of nDim x nArchetypes
        self.alfai:      has a shape of nArchetypes x 1.
        xi:              has a shape of nDim x 1.
        
        The problem to minimize is:
            
            xi = self.archetypes x self.alfai
        """
        self.alfa = np.zeros([self.nArchetypes, self.nData])
        self.RSSi_ = np.zeros([self.nData])
        for i, xi in enumerate(self.X.T):            
            self.alfa[:, i], self.RSSi_[i] = solveConstrainedNNLS(xi, self.archetypes, self.C)


    def _optimizeBeta(self):
        self.beta = np.zeros([self.nData, self.nArchetypes])
        for l in range(self.nArchetypes):
            vBarL, hasError = self._returnVbarL(l)
            if hasError:
                return hasError
            vBarL = vBarL.flatten()
            self.beta[:,l], _ = solveConstrainedNNLS(vBarL, self.X, self.C)
        return hasError

    
    def _findNewArchetype(self, k):
        """
        In some circumstance, summation of alfa's for an archetype k becomes zero. 
        That means archetype k is redundant. This function aims to find a new candidate
        from data set to replace archetype k.        
        """
        archK = copy(self.archetypes[:, k])
        if k == 0:
            alfaK = self.alfa[k:, :]
            archetypes = self.archetypes[:, k:]
        else:
            alfaK = self.alfa[k-1:, :]
            archetypes = self.archetypes[:, k-1:]
        Xapprox = np.matmul(archetypes, alfaK)
        
        RSSi = ((self.X.T - Xapprox.T) ** 2).sum(axis=1)
        if self.nData < 10:
            idMaxes = RSSi.argsort()
        else:
            idMaxes = RSSi.argsort()[-10:]

        if np.linalg.norm(self.X.T[idMaxes[-1], :] - archK):
            idMax = idMaxes[-1]
        else:
            import random
            idMax = random.Random(self.randomState).choice(list(idMaxes[:-1]))
            self.randomState += 10
        return self.X.T[idMax, :], idMax
  

    def _returnVbarL(self, l):        
        def returnVi(i, l, alfaIl):
            """
            This function calculates vi for each data point with respect to 
            archetype l.
            
            i:          ith data point (xi)
            l:          index of archetype that should be excluded.
            """
            eps = 0.000001
            vi = np.zeros([self.nDim,1])
            # for k, alfaik, zk in enumerate(zip(self.alfa[:,i], self.archetypes)):
            for k in range(self.nArchetypes):                
                if k != l:
                    xx = self.alfa[k,i] * self.archetypes[:,k]
                    vi[:,0] = vi[:,0] + xx
            if (alfaIl < eps):
                alfaIl = eps
            vi[:,0] = (self.X[:,i] - vi[:,0]) / alfaIl
            return vi
        
        
        def checkArchRedundancy():
            hasError = False
            eps = 0.00000001
            
            # CHECK SUM SQ OF alfaIl
            alftaIlSumsq = np.sum(self.alfa[l,:] ** 2)
            count = 1
            while(alftaIlSumsq < eps):
                
                # FINDING THE FURTHEST POINT AND REPLACING REDUNDANT ARCHETYPE l
                # archOld = copy(self.archetypes[:,l])
                self.archetypes[:,l], idMax  = self._findNewArchetype(l)
                # print(np.linalg.norm(archOld - self.archetypes[:,l]))

                # RE-OPTIMIZING ALFA
                self._optimizeAlfa()
                # self.alfa[l,:], _ = self.__optimizeAlfaForTransform(self.archetypes[:,l].reshape(1,-1), nData = 1)
                
                # RE-CALCULATE SUM SQ OF alfaIl
                alftaIlSumsq = np.sum(self.alfa[l,:] ** 2)

                if (count > self.redundancyTry):
                    hasError = True
                    break
                
                count += 1
            if (count > 1):
                # print(f'Warning: Archetype {l+1} was recognised redundant. The redundancy issue was resolved after {count -1} try(s) and a new candidate is replaced.')
                print(f'Warning: A redundant archetype was recognised. The redundancy issue was resolved after {count -1} try(s) and a new candidate is replaced.')
            return hasError, alftaIlSumsq
        hasError, alftaIlSumsq = checkArchRedundancy()
        if (hasError):
            self.nArchetypes = self.nArchetypes - 1
            print(f'Warning: After {self.redundancyTry} tries, the redundancy issue was not resolved. Hence, the number of archetypes is reduced to: {self.nArchetypes}')                    
            self.fit(self.X.T)  
            return None, hasError
        else:      
            eps = 0.000001
            sumAlfaIlSqVi = np.zeros([self.nDim,1])
            for i in range(self.nData):
                alfaIl = self.alfa[l,i]                
                vi = returnVi(i,l,alfaIl)
                sumAlfaIlSqVi = sumAlfaIlSqVi + alfaIl ** 2 * vi
            
            if (alftaIlSumsq < eps):
                alftaIlSumsq = eps
            vBar = sumAlfaIlSqVi / alftaIlSumsq
        return vBar, hasError
    
    
    def _rankArchetypes(self):
        """
        This function aims to rank archetypes. To do this, each data point is 
        approximated just using one of the archetypes. Then, we check how good
        is the approximation by calculating the explained variance. Then, we 
        sort the archetypes based on the explained variance scores. Note that,
        unlike to PCA, the summation of each individual explained variance 
        scores will not be equal to the calculated explained variance when all
        archetypes are considered.
        """
        expVarPerArch_ = []
        for i in range(self.nArchetypes):
            Xapprox = np.matmul(self.archetypes[:, i].reshape(self.nDim, 1) , self.alfa[i, :].reshape(1, self.nData))
            expVarPerArch_.append((i, explainedVariance(self.X, Xapprox, method = "_")))
        # self.expVarPerArch_ = tuple(self.expVarPerArch_)
        expVarPerArch_ = sorted(expVarPerArch_, key = lambda x: x[1], reverse=True )
        rank = [item[0] for item in expVarPerArch_]
        self.scorePerArch = [item[1] for item in expVarPerArch_]
        
        self.archetypes = self.archetypes[:,rank]
        self.alfa = self.alfa[rank,:]
        self.beta = self.beta[:,rank]
   
    
    def plotSimplex(self, alfa, archIDs=[0, 1, 2], plotArgs={}, 
                    gridOn=True, showLabel=True, labelAll=False, 
                    figSize=(3, 3), dpi=DPI, 
                    gridLineWidth=0.1, color='#303F9F', alpha=0.8, markerSize=20, 
                    figNamePrefix=''):
        """
        # groupColor = None, color = None, marker = None, size = None
        groupColor:    
            
            Dimension:      nData x 1
            
            Description:    Contains the category of data point.
        """
        if len(archIDs) == 0: 
            raise Exception("Archetype IDs can't be empty!")
        labels = ['A' + str(i + 1) for i in archIDs] if showLabel else []
        rotateLabels = True
        labelOffset = 0.10
        data = alfa[archIDs].T
        scaling = False
        sides = len(archIDs)
        
        basis = np.array([[np.cos(2*_*pi/sides + 90*pi/180), np.sin(2*_*pi/sides + 90*pi/180)] for _ in range(sides)])

        # If data is Nxsides, newdata is Nx2.
        if scaling:  # Scales data for you.
            newdata = np.dot((data.T / data.sum(-1)).T, basis)
        else:  # Assumes data already sums to 1.
            newdata = np.dot(data, basis)
        fig = plt.figure(figsize=figSize, dpi=dpi)
        ax = fig.add_subplot(111)

        if not labelAll and len(labels) < 8: 
            print('Number of labels <8, nullifying `labelAll`, showing all labels...')
            labelAll = True
        if showLabel:
            labelIdxs = np.round(np.linspace(0, len(labels)-1, 9)[:-1]).astype(int) if not labelAll else range(len(labels))
            for (i, l) in enumerate(labels):
                if not labelAll:
                    if i not in labelIdxs:
                        continue
                if i >= sides:
                    break
                x = basis[i, 0]
                y = basis[i, 1]
                if rotateLabels:
                    angle = 180*np.arctan(y/x)/pi + 90
                    if angle > 90 and angle <= 270:
                        angle = (angle + 180) % 360 # mod(angle + 180,360)
                else:
                    angle = 0
                ax.text(x*(1.05 + labelOffset), y*(1.05 + labelOffset),
                        l, horizontalalignment='center', verticalalignment='center', rotation=angle)
            
        # Clear normal matplotlib axes graphics.
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_frame_on(False)
        
        # Plot grid
        if gridOn:
            lstAx0 = []
            lstAx1 = []
            ignore = False
            for i in range(sides):
                for j in range(i + 2, sides):
                    ignore = True if (i==0 & j==sides) else False                    
                    if not (ignore):              
                        lstAx0.append(basis[i, 0] + [0,])
                        lstAx1.append(basis[i, 1] + [0,])
                        lstAx0.append(basis[j, 0] + [0,])
                        lstAx1.append(basis[j, 1] + [0,])
            ax.plot(lstAx0, lstAx1, color='#212121', linewidth=gridLineWidth, alpha=0.5, zorder=1)
        
        # Plot border
        lstAx0 = []
        lstAx1 = []
        for _ in range(sides):
            lstAx0.append(basis[_, 0] + [0,])
            lstAx1.append(basis[_, 1] + [0,])
        lstAx0.append(basis[0, 0] + [0,])
        lstAx1.append(basis[0, 1] + [0,])

        ax.plot(lstAx0, lstAx1, linewidth=1, zorder=2, color='k')  #, **edgeArgs) 
        if len(plotArgs) == 0:
            ax.scatter(newdata[:, 0], newdata[:, 1], color=color, zorder=3, alpha=alpha, s=markerSize,
                       edgecolor='k', linewidth=0.8)
        else:
            if ('marker' in plotArgs):   
                markerVals = plotArgs['marker'].values
                markerUnq = np.unique(markerVals)                
                
                for marker in markerUnq:
                    rowIdx = np.where(markerVals==marker)
                    tmpArg = {}
                    for keys in plotArgs:
                        if (keys!= 'marker'):
                            tmpArg[keys] = plotArgs[keys].values[rowIdx]
                    ax.scatter(newdata[rowIdx,0], newdata[rowIdx,1], **tmpArg, marker=marker, edgecolor='k', alpha=alpha, zorder=3)
            else:
                ax.scatter(newdata[:,0], newdata[:,1], **plotArgs, marker='s', zorder=3, alpha=alpha)
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_simplex.png", bbox_inches='tight')
            
               
    def parallelPlot(self, lstFeat, dfColor, featIDs=[0, 1, 2], archIDs=[0, 1, 2], sampIDs=[0, 1, 2], 
                     linewidth='0.3', archColor='k', 
                     figSize=(15, 5), dpi=DPI, figNamePrefix=''):
        """
        Based on source: http://benalexkeen.com/parallel-coordinates-in-matplotlib/
        
        lstFeat:
                    list of features.

        dfColor:
                    A dataframe of collection of colors corresponding to each
                    data point.
        """
        from matplotlib import ticker
        
        x = [i for (i, _) in enumerate(lstFeat)]
        df = pd.DataFrame(self.X[featIDs, :][:, sampIDs].T, columns=lstFeat)
        for i in archIDs:
            df.loc[-1, :] = list(self.archetypes[featIDs, i])
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index            
            
            dfColor.loc[-1] = 'arch'
            dfColor.index = dfColor.index + 1  # shifting index
            dfColor = dfColor.sort_index()  # sorting by index
                
        # Create (X-1) sublots along x axis
        fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=figSize, dpi=dpi)
        # Get min, max and range for each column
        # Normalise the data for each column
        minMaxRange = {}       
        for col in lstFeat:
            minMaxRange[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
            df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

        for (i, ax) in enumerate(axes):
            for idx in df.index:
                if (dfColor.loc[idx, 'color'] == 'arch'):
                    # Plot each archetype
                    ax.plot(x, df.loc[idx, lstFeat], color=archColor, alpha=0.8, linewidth='2.0')
                else:
                    # Plot each data point
                    ax.plot(x, df.loc[idx, lstFeat], color=dfColor.loc[idx, 'color'], alpha=0.3, linewidth=linewidth)
            ax.set_xlim([x[i], x[i+1]])
            
        # Set the tick positions and labels on y axis for each plot
        # Tick positions based on normalised data
        # Tick labels are based on original data
        def set_ticks_for_axis(dim, ax, ticks):
            minVal, maxVal, valRange = minMaxRange[lstFeat[dim]]
            step = valRange / float(ticks-1)
            tickLabels = [round(minVal + step * i, 2) for i in range(ticks)]
            normMin = df[lstFeat[dim]].min()
            normRange = np.ptp(df[lstFeat[dim]])
            normStep = normRange / float(ticks-1)
            ticks = [round(normMin + normStep * i, 2) for i in range(ticks)]
            ax.yaxis.set_ticks(ticks)
            ax.set_yticklabels(tickLabels)
            
        for dim, ax in enumerate(axes):
            ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
            set_ticks_for_axis(dim, ax, ticks=6)
            ax.set_xticklabels([lstFeat[dim]], rotation='vertical')
        # Move the final axis' ticks to the right-hand side
        ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        set_ticks_for_axis(len(axes), ax, ticks=6)
        ax.set_xticklabels([lstFeat[-2], lstFeat[-1]], rotation='vertical')
        # Remove space between subplots
        plt.subplots_adjust(wspace=0)       
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_parallel.png", bbox_inches='tight')
        
    
    def _extractArchetypeProfiles(self):
        """
        This function extracts the profile of each archetype. Each value in
        each dimension of archetypeProfile shows the portion of data that
        is covered by that archetype in that specific direction.
        """
        self.archetypeProfile = np.zeros([self.nDim, self.nArchetypes])
        for i in range(self.nArchetypes):
            for j in range(self.nDim):
                xArch = self.archetypes[j, i]
                xData = self.X[j, :]
                self.archetypeProfile[j, i] = ecdf(xData, xArch)
                
    
    def plotProfile(self, allFeatNames=None, featIDs=None, archIDs=[0, 1],
                    figSize=(14, 5), dpi=DPI, figNamePrefix=''):
        """
        This function plots the profile of the archetypes.

        allFeatNames:
            Optional input. list of all feature names.
        featIDs:
            Optional input. list of names of features of interest.
        """
        if len(archIDs) == 0:
            raise Exception("Archetype IDs can't be empty!")
        sns.set_style('ticks')
        xVals = np.arange(1, len(featIDs) + 1)
        for i in archIDs:
            plt.figure(figsize=figSize, dpi=dpi)
            plt.bar(xVals, self.archetypeProfile[featIDs, i] * 100.0, 
                    color='#D32F2F', edgecolor='#212121', linewidth=0.8)
            if (allFeatNames != None):
                plt.xticks(xVals, [allFeatNames[idx] for idx in featIDs], rotation='vertical')
            plt.ylim([0, 100])
            plt.ylabel('A' + str(i + 1))
            plt.grid(linestyle='dotted')
            plt.rcParams.update({'font.size': 10})
            plt.tight_layout()
            plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_A{i+1}_featProf.png", bbox_inches='tight')
            
    
    def plotRadarProfile(self, allFeatNames=None, featIDs=[0], archIDs=[0, 1], fillAlpha=0.2, linewidth=1, ncol=1,
                         sepArchs=False, showLabel=True, labelAll=False, showName=False, closeFig=False, showLegend=True, 
                         figSize=(6, 6), dpi=DPI, title=None, figNamePrefix=''):
        if len(archIDs) == 0: 
            raise Exception("Archetype IDs can't be empty!")
        if len(featIDs) < 1: 
            raise Exception('No feature is selected!')  
        
        # Getting labels
        labels = [allFeatNames[i] for i in featIDs] if showName else [f"F{i}" for i in featIDs]
        angles = np.linspace(0, 2*np.pi, len(featIDs), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        if not sepArchs:
            fig = plt.figure(figsize=figSize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)
            if showLegend:
                legend = []
        for (i, archID) in enumerate(archIDs):
            if sepArchs:
                fig = plt.figure(figsize=figSize, dpi=dpi)
                ax = fig.add_subplot(111, polar=True)
            c = sns.color_palette('husl', len(archIDs))[i]
            xClose = self.archetypeProfile[featIDs, archID]
            xClose = np.concatenate((xClose, [xClose[0]]))
            ax.plot(angles, xClose, '.-', linewidth=linewidth, markersize=5, zorder=-1, color=c)
            ax.fill(angles, xClose, alpha=fillAlpha, zorder=-2, color=c)
            # ax.set_rlabel_position(0)
            ax.set_rticks([0.2, 0.4, 0.6, 0.8])
            ax.set_rlim(0.0, 1.0)
            if showLabel:
                if labelAll:
                    labelIdxs = [j for j in range(len(angles) - 1)]
                else:  # Only label 8 standard corners
                    labelIdxs = np.round(np.linspace(0, len(angles)-1, 9)[:-1]).astype(int)
                ax.set_thetagrids(angles[labelIdxs] * 180.0/np.pi, [labels[idx] for idx in labelIdxs])
            else:
                ax.set_thetagrids(angles * 180.0/np.pi, [''] + [''] * len(featIDs))
            ax.set_title(title)
            ax.grid(linestyle='dotted', color='k', alpha=0.3, linewidth=1)
            ax.set_facecolor('#EAECEE')
            if not sepArchs:
                if showLegend:
                    legend.extend([f"A{archID + 1}", '_'])
            else:
                plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_radarProf_A{archID + 1}.png", bbox_inches='tight')
                if closeFig:
                    plt.close()
        if not sepArchs:
            if showLegend:
                ax.legend(legend, loc='center', bbox_to_anchor=(0.5, -0.3), ncol=ncol)   # (1.2, 0.5)
            plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_radarProf.png", bbox_inches='tight')
        
    
    def _extractCloseMatch(self):
        self.closeMatch = {}
        for i in range(self.nArchetypes):
            r = cdist(self.X.T, self.archetypes[:, i].reshape(1, self.nDim))
            iMin = np.argmin(r)
            self.closeMatch[i + 1] = (iMin, self.alfa[:, iMin])
            
    
    def plotCloseMatch(self, archIDs=[0, 1], archSpaceIDs=[0, 1], 
                       sepSamps=False, showLabel=True, labelAll=False, showLegend=False,
                       figSize=(6, 6), dpi=DPI, title=None, figNamePrefix=''):
        if len(archIDs) == 0 or len(archSpaceIDs) == 0: 
            raise Exception("Archetype IDs can't be empty!")

        # Extract most archetypal samples if not done
        if (len(self.closeMatch) == 0):
            self._extractCloseMatch()
            
        # Getting labels
        labels = [f"A{i+1}" for i in archSpaceIDs]
        angles = np.linspace(0, 2*np.pi, len(archSpaceIDs), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        if not sepSamps:
            fig = plt.figure(figsize=figSize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)
            if showLegend:
                legend = []
        for (i, archID) in enumerate(archIDs):
            if sepSamps:
                fig = plt.figure(figsize=figSize, dpi=dpi)
                ax = fig.add_subplot(111, polar=True)
            c = sns.color_palette('husl', len(archIDs))[i]
            xClose = self.closeMatch[archID + 1][1][archSpaceIDs]
            xClose = np.concatenate((xClose, [xClose[0]]))
            ax.plot(angles, xClose, '.-', linewidth=1, markersize=5, zorder=-1, color=c)
            ax.fill(angles, xClose, alpha=0.2, zorder=-2, color=c)
            # ax.set_rlabel_position(0)
            ax.set_rticks([0.2, 0.4, 0.6, 0.8])
            ax.set_rlim(0.0, 1.0)
            if showLabel:
                if not labelAll and len(archSpaceIDs) < 8: 
                    print('Number of labels <8, nullifying `labelAll`, showing all labels...')
                    labelAll = True
                if labelAll:
                    labelIdxs = [j for j in range(len(angles) - 1)]
                else:  # Only label 8 standard corners
                    labelIdxs = np.round(np.linspace(0, len(angles)-1, 9)[:-1]).astype(int)
                ax.set_thetagrids(angles[labelIdxs] * 180.0/np.pi, [labels[idx] for idx in labelIdxs])
            else:
                ax.set_thetagrids(angles * 180.0/np.pi, [''] + [''] * len(archIDs))
            ax.set_title(title)
            ax.grid(linestyle='dotted', color='k', alpha=0.3, linewidth=1)
            ax.set_facecolor('#EAECEE')
            if not sepSamps:
                if showLegend:
                    legend.extend([f"A{archID + 1}", '_'])
            else:
                plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_closeMatch_A{archID + 1}.png", bbox_inches='tight')
        if not sepSamps:
            if showLegend:
                ax.legend(legend, loc='center', bbox_to_anchor=(1.2, 0.5), ncol=1)
            plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_closeMatch.png", bbox_inches='tight')

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import explained_variance_score



def findFurthestPoint(xSearch, xRef):
    """
    This function finds a data point in x_search which has the furthest 
    distance from all data points in xRef. In the case of archetypes,
    xRef is the archetypes and xSearch is the dataset.
    
    Note:
        In both xSearch and xRef, the columns of the arrays should be the
        dimensions and the rows should be the data points.
    """
    from scipy.spatial.distance import cdist
    D = cdist(xRef, xSearch)
    Dsum = D.sum(axis=0)
    idxmax = np.argmax(Dsum)
    return xSearch[idxmax,:], idxmax


def ecdf(X, x):
    """Emperical Cumulative Distribution Function
    
    X: 
        1-D array. Vector of data points per each feature (dimension), defining
        the distribution of data along that specific dimension.
        
    x:
        Value. It is the value of the corresponding dimension of an archetype.
        
    P(X <= x):
        The cumulative distribution of data points with respect to the archetype
        (the probablity or how much of data in a specific dimension is covered
        by the archetype).
    """
    return float(len(X[X < x]) / len(X))


def calcSSE(Xact, Xapx):
    """
    This function returns the Sum of Square Errors.
    """
    return ((Xact - Xapx) ** 2).sum()


def calcSST(Xact):
    """
    This function returns the Sum of Square Errors.
    """
    return (Xact ** 2).sum()


def explainedVariance(Xact, Xapx, method='sklearn'):
    if method.lower() == 'sklearn':  
        return explained_variance_score(Xact.T, Xapx.T)
    else:
        SSE = calcSSE(Xact, Xapx)
        SST = calcSST(Xact)
        return (SST - SSE) / SST
        
        
def solveConstrainedNNLS(u, t, C):
    """
    This function solves the typical equation of ||U - TW||^2 where U and T are
    defined and W should be determined such that the above expression is 
    minimised. Further, solution of W is subjected to the following constraints:
        
        Constraint 1:       W >= 0
        Constraint 2:       sum(W) = 1
    
    Note that the above equation is a typical equation in solving alfa's and
    beta's.
    
    
    Solving for ALFA's:
    -------------------
    when solving for alfa's the following equation should be minimised:
        
        ||Xi - sum([alfa]ik x Zk)|| ^ 2.
    
    This equation should be minimised for each data point (i.e. nData is the
    number of equations), which results in nData rows of alfa's. In each 
    equation U, T, and W have the following dimensions:
        
    Equation (i):
        
        U (Xi):         It is a 1D-array of nDim x 1 dimension.
        T (Z):          It is a 2D-array of nDim x k dimension.
        W (alfa):       It is a 1D-array of k x 1 dimension.
    
    
    Solving for BETA's:
    -------------------   
    """
    mObservation, nVariables = t.shape
    # EMPHASIZING THE CONSTRAINT
    u = u * C
    t = t * C   
    # ADDING THE CONSTRAINT EQUATION
    u = np.append(u, [1], axis = 0)
    t = np.append(t, np.ones([1,nVariables]), axis = 0)
    w, rnorm = nnls(t, u)
    return w, rnorm


def furthestSum(K, noc, i, exclude=[]):
    """
    Note by Benyamin Motevalli:
        
        This function was taken from the following address:
            https://github.com/ulfaslak/py_pcha
            
        and the original author is: Ulf Aslak Jensen.
    """
    
    """
    Original Note:
        
    Furthest sum algorithm, to efficiently generat initial seed/archetypes.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).
    
    Parameters
    ----------
    K : numpy 2d-array
        Either a data matrix or a kernel matrix.

    noc : int
        Number of candidate archetypes to extract.

    i : int
        inital observation used for to generate the FurthestSum.

    exclude : numpy.1darray
        Entries in K that can not be used as candidates.

    Output
    ------
    i : int
        The extracted candidate archetypes
    """
    def maxIndVal(l):
        return max(zip(range(len(l)), l), key=lambda x: x[1])

    I, J = K.shape
    index = np.array(range(J))
    index[exclude] = 0
    index[i] = -1
    indT = i
    sumDist = np.zeros((1, J), np.complex128)

    if J > noc * I:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, noc + 11):
            if k > noc - 1:
                Kq = np.dot(Kt[:, i[0]], Kt)
                sumDist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = np.dot(Kt[:, indT].T, Kt)
            sumDist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[indT])
            ind, val = maxIndVal(sumDist[:, t][0].real)
            indT = t[ind]
            i.append(indT)
            index[indT] = -1
    else:
        if I != J or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = np.dot(Kt.T, Kt)
            K = np.lib.scimath.sqrt(
                np.tile(np.diag(K), (J, 1)) - 2 * K + \
                np.tile(np.mat(np.diag(K)).T, (1, J))
            )

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, noc + 11):
            if k > noc - 1:
                sumDist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sumDist += np.lib.scimath.sqrt(Kt2 - 2 * K[indT, :] + Kt2[indT])
            ind, val = maxIndVal(sumDist[:, t][0].real)
            indT = t[ind]
            i.append(indT)
            index[indT] = -1
    return i


from math import pi

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from MulticoreTSNE import MulticoreTSNE
import seaborn as sns
from sklearn.manifold import TSNE

from iaa.constants import RANDOM_STATE, NUM_JOBS, PALETTE, DPI
 

def ternaryPlot(data, scaling=True, startAngle=90, rotateLabels=True,
                labels=('one','two','three'), sides=3, labelOffset=0.10,
                edgeArgs={'color':'black','linewidth':1},
                figArgs = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},
                gridOn = True):
    '''
    source: https://stackoverflow.com/questions/701429/library-tool-for-drawing-ternary-triangle-plots
    
    This will create a basic "ternary" plot (or quaternary, etc.)
    
    DATA:           The dataset to plot. To show data-points in terms of archetypes
                    the alfa matrix should be provided.
    
    SCALING:        Scales the data for ternary plot such that the components along
                    each axis dimension sums to 1. This conditions is already imposed 
                    on alfas for archetypal analysis.
    
    startAngle:    Direction of first vertex.
    
    rotateLabels:  Orient labels perpendicular to vertices.
    
    labels:         Labels for vertices.
    
    sides:          Can accomodate more than 3 dimensions if desired.
    
    labelOffset:   Offset for label from vertex (percent of distance from origin).
    
    edgeArgs:      Any matplotlib keyword args for plots.
    
    figArgs:       Any matplotlib keyword args for figures.
    '''
    basis = np.array(
                    [
                        [
                            np.cos(2*_*pi/sides + startAngle*pi/180),
                            np.sin(2*_*pi/sides + startAngle*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data,basis)

#    fig = plt.figure(**figArgs)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotateLabels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = angle = (angle + 180) % 360 # mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + labelOffset),
                y*(1 + labelOffset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)
    
    # Plot border
    lstAx0 = []
    lstAx1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False                        
#                
            if not (ignore):
#            if (j!=i & j!=i+1 & j != i-1):                        
                lstAx0.append(basis[i,0] + [0,])
                lstAx1.append(basis[i,1] + [0,])
                lstAx0.append(basis[j,0] + [0,])
                lstAx1.append(basis[j,1] + [0,])

#    lstAx0.append(basis[0,0] + [0,])
#    lstAx1.append(basis[0,1] + [0,])
    
    ax.plot(lstAx0,lstAx1, color='#FFFFFF',linewidth=1, alpha = 0.5)
    
    # Plot border
    lstAx0 = []
    lstAx1 = []
    for _ in range(sides):
        lstAx0.append(basis[_, 0] + [0,])
        lstAx1.append(basis[_, 1] + [0,])

    lstAx0.append(basis[0,0] + [0,])
    lstAx1.append(basis[0,1] + [0,])
#    ax.plot(
#        [basis[_, 0] for _ in range(sides) + [0,]],
#        [basis[_, 1] for _ in range(sides) + [0,]],
#        **edgeArgs
#    )
#    
    ax.plot(lstAx0, lstAx1, linewidth=1) #, **edgeArgs ) 
    return newdata, ax 


def compareProfile(AAprof1, AAprof2, featureCols):
    """
    This function plots the profile of the archetypes.
    
    featureCols:
        Optional input. list of feature names to use to label x-axis.
    """               
    plt.style.use('ggplot')
    nDim = len(featureCols)
    xVals = np.arange(1, nDim + 1)
    plt.figure(figsize=(14,5))
    plt.bar(xVals, AAprof1 * 100.0, color='#413F3F', label='Minimum Case')
    plt.bar(xVals, AAprof2 * 100.0, color='#8A2BE2', alpha=0.5, label='Maximum Case')
    plt.xticks(xVals, featureCols, rotation='vertical')
    plt.ylim([0, 100])
#    plt.ylabel('A' + str(i + 1))
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()
    plt.legend(loc='upper left')
    
 
def datapointProfile(xPoint, xData):
    pointProfile = []
    for i, p in enumerate(xPoint):
        d = xData[i, :]
        pointProfile.append(ecdf(d, p))
    return np.array(pointProfile)


def plotRadarDatapoint(AA, X, title='Radar plot of datapoint'):
    _, alfaX = AA.transform(X)
    
    labels = ['A' + str(i+1) for i in range(AA.nArchetypes)]
    angles = np.linspace(0, 2*np.pi, AA.nArchetypes, endpoint=False)
    angles = np.concatenate((angles, [angles[0]])) 
    
    fig = plt.figure(figsize=(3, 3))
    alfaX = np.concatenate((alfaX, [alfaX[0]]))
    
    ax = fig.add_subplot(111, polar=True)
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, alfaX, 'o-', markersize=5, linewidth=1.5, color='#273746')
    ax.fill(angles, alfaX, alpha=0.25)
    ax.set_thetagrids(np.array(angles) * 180.0/np.pi, ['0'] + labels)
    ax.set_title(title)
    ax.grid(True)
    ax.set_rlim(0,1)
    ax.set_facecolor('#EAECEE')
    return ax


def createSimplexAx(AA, gridOn=True, gridcolor='#EAECEE',
                    bordercolor='#4A235A', fontcolor='#6C3483', figSize=(3, 3)):
        """
        # groupColor = None, color = None, marker = None, size = None
        groupColor:    
            
            Dimension:      nData x 1
            
            Description:    Contains the category of data point.
        """        
        # plt.style.use('seaborn-paper')
        nArchetypes = AA.nArchetypes
        # fig = plt.figure(figsize=[16,8])
        fig, ax = plt.subplots(figsize=figSize)
        
        labels = ('A'+str(i + 1) for i in range(nArchetypes))
        rotateLabels = True
        labelOffset = 0.10
        scaling = False
        sides = nArchetypes
        
        basis = np.array([[np.cos(2*_*pi/sides + 90*pi/180),
                           np.sin(2*_*pi/sides + 90*pi/180)] for _ in range(sides)])
    
        for i,l in enumerate(labels):
            if i >= sides:
                break
            x = basis[i,0]
            y = basis[i,1]
            if rotateLabels:
                angle = 180*np.arctan(y/x)/pi + 90
                if angle > 90 and angle <= 270:
                    angle = angle = (angle + 180) % 360 # mod(angle + 180, 360)
            else:
                angle = 0
            ax.text(x*(1 + labelOffset), y*(1 + labelOffset),
                    l, horizontalalignment='center', verticalalignment='center',
                    rotation=angle, fontsize=12, color=fontcolor)
    
        # Clear normal matplotlib axes graphics.
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_frame_on(False)

        # Plot Grids
        lstAx0 = []
        lstAx1 = []
        ignore = False
        for i in range(sides):
            for j in range(i + 2, sides):
                if (i == 0 & j == sides):
                    ignore = True
                else:
                    ignore = False                        
                if not (ignore):                    
                    lstAx0.append(basis[i,0] + [0,])
                    lstAx1.append(basis[i,1] + [0,])
                    lstAx0.append(basis[j,0] + [0,])
                    lstAx1.append(basis[j,1] + [0,])
        ax.plot(lstAx0, lstAx1, color=gridcolor, linewidth=0.5, alpha=0.5, zorder=1)
        
        # Plot border
        lstAx0 = []
        lstAx1 = []
        for _ in range(sides):
            lstAx0.append(basis[_, 0] + [0,])
            lstAx1.append(basis[_, 1] + [0,])
        lstAx0.append(basis[0, 0] + [0,])
        lstAx1.append(basis[0, 1] + [0,])
        ax.plot(lstAx0, lstAx1, linewidth=1.5, color=bordercolor, zorder=2)  #, **edgeArgs)
    
        return fig    


def mapAlfaToSimplex(alfa, AA):
    """
    alfa:    2D-array (nArchetypes x nData)
    """
    nArchetypes = AA.nArchetypes
    basis = np.array([[np.cos(2*_*pi/nArchetypes + 90*pi/180),
                       np.sin(2*_*pi/nArchetypes + 90*pi/180)] for _ in range(nArchetypes)])
    mappedAlfa = np.dot(alfa.T,basis)
    return mappedAlfa


def plotTSNE(X, figNamePrefix='', figSize=(3, 3), markerSize=1, numComponents=2, markIdxs=[],
              perplexity=30.0, earlyExaggeration=12.0, learningRate='auto', nIter=1000, angle=0.5, 
              metric='euclidean', init='pca', method='barnes_hut', minGradNorm=1e-7, 
              nIterWithoutProgress=300, nJobs=NUM_JOBS, randomState=RANDOM_STATE):
    # reducer = MulticoreTSNE(n_components=numComponents, init='random', method=method, angle=angle, random_state=RANDOM_STATE, 
    #                         perplexity=perplexity, early_exaggeration=earlyWxaggeration, learning_rate=200.0, n_iter=nIter,
    #                         n_iter_without_progress=nIterWithoutProgress, min_grad_norm=minGradNorm, metric=metric, 
    #                         verbose=0, n_jobs=nJobs)
    reducer = TSNE(n_components=numComponents, init=init, method=method, angle=angle, random_state=randomState, 
                   perplexity=perplexity, early_exaggeration=earlyExaggeration, learning_rate=learningRate, max_iter=nIter,
                   n_iter_without_progress=nIterWithoutProgress, min_grad_norm=minGradNorm, metric=metric, 
                   verbose=0, n_jobs=nJobs)
    embedding = reducer.fit_transform(X)
    sns.set_palette('icefire')
    plt.figure(figsize=figSize, dpi=DPI)
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1], s=markerSize, c=range(len(X)), cmap=PALETTE)
    # c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
    plt.scatter(x=embedding[markIdxs, 0], y=embedding[markIdxs, 1], marker='D', s=markerSize*30, facecolor='r', edgecolor='k', linewidth=0.8)
    plt.grid(linestyle='dotted')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(f"figs/{figNamePrefix}_tSNE.png", bbox_inches='tight')


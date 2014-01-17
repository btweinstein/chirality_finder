__author__ = 'bryan'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colormapChoice = plt.cm.jet

def getPositionData(coords, center):
    """Coordinates are in y,x form like that returned from getNonzeroCoordinates"""
    y, x = (coords[:, 0], coords[:, 1])

    deltaY = y - center[0]
    deltaX = x - center[1]

    # Get distance from center
    deltaR = np.column_stack((x, y, deltaX, deltaY))

    currentData = pd.DataFrame(deltaR, columns=['x', 'y', 'dx', 'dy'])
    currentData['r'] = np.sqrt(currentData['dx']**2 + currentData['dy']**2)
    currentData['theta'] = np.arctan2(currentData['dy'], currentData['dx'])

    return currentData

def getChiralityData(labels, center):

    chiralityData = pd.DataFrame()

    uniqueLabels = np.unique(labels)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    for currentLabel in uniqueLabels:
        # Unfortunately, getting from row/column to x/y
        # is somewhat irritating.
        coords = getNonzeroCoordinates(labels == currentLabel)

        data = getPositionData(coords, center)

        # Thin the data, only one point at each radius!
        # Be careful, however, as the mean of r or theta is not what you want. You want to
        # find the median (x, y) / (dx, dy) and calculate everything again based on that! As
        # the mean theta is not the same thing as the arctan2 of the mean dx/dy
        minR_int = np.floor(data['r'].min())
        maxR_int = np.ceil(data['r'].max())
        bins = np.arange(minR_int, maxR_int, 1)
        groups = data.groupby(pd.cut(data['r'], bins))
        meanData = groups.mean()
        # Recalculate r and theta as the average does not depend linearly on x/y
        meanData['r'] = np.sqrt(meanData['dx']**2 + meanData['dy']**2)
        meanData['theta'] = np.arctan2(meanData['dy'], meanData['dx'])
        # Now finish using the chirality data
        data = meanData
        #data = data.dropna()

        # Now rotate the coordinate system so that it is in the correct spot
        minRadiusIndex = data.r.idxmin()
        minRadiusRow = data.ix[minRadiusIndex]
        minRadiusTheta = minRadiusRow['theta']

        data['rotated'] = data['theta'] - minRadiusTheta
        data['rotated'][data['rotated'] < -np.pi] = 2*np.pi+ data['rotated']
        data['rotated'][data['rotated'] > np.pi] = -2*np.pi + data['rotated']

        data['label'] = currentLabel

        chiralityData = chiralityData.append(data)

    #chiralityData = chiralityData.reset_index(drop=True)

    return chiralityData

def getNonzeroCoordinates(binary):
    """Given a binary image return the x,y coordinates with nonzero values.
    We use IMAGE COORDINATES here, i.e. y increases downwardly.

    Returns a stack of [y, x]. Can be fed directly into an image to get the correct coordinates,
    as indexing for an image is row, column."""

    coords = np.transpose(np.nonzero(binary))
    return coords

def  makeChiralityPlot(chiralityData):
    """Makes a plot comparing the chirality of each labeled region."""

    f = plt.figure()
    numColors = len(chiralityData.label.unique())
    currentColor = 0
    for currentLabel in chiralityData.label.unique():
        labelData = chiralityData[chiralityData['label'] == currentLabel]
        labelData = labelData.sort(columns=['r'])
        # Somehow I still don't think we have a unique theta at every r. Let us fix that.
        # We can do that later. Let us apply this to the other replicate and see what happens.
        labelData.plot(x='r', y='rotated', style='+-', label=currentLabel, color=colormapChoice(1.*currentColor/numColors))
        currentColor += 1
    plt.xlabel('r')
    plt.ylabel('d$\\theta$')
    plt.title('Chirality')
    #box = plt.gca().get_position()
    #plt.gca().set_position([box.x0, box.y0, box.width*0.95, box.height])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return f

def visualizeSectors(chiralityData, overImage=False):
    """Plots the different sectors. If being plotted before an image to be overlayed,
    set overImage=True."""
    fig = plt.figure()
    groups = chiralityData.groupby('label')
    numColors = len(groups)
    currentColor = 0
    for name, group in groups:
        if not overImage:
            group.plot(x='x',y='y', style='+-', label=name, color=colormapChoice(1.*currentColor/numColors))
        else: # Make skinnier by getting rid of +, no grid lines
            group.plot(x='x',y='y', style='-', label=name, color=colormapChoice(1.*currentColor/numColors))
        currentColor += 1
    if not overImage:
        fig.gca().invert_yaxis()
    else:
        plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sector Boundaries')
    #box = plt.gca().get_position()
    #plt.gca().set_position([box.x0, box.y0, box.width*0.95, box.height])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    return fig
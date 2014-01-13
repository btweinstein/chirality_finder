__author__ = 'bryan'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        minR_int = np.floor(data['r'].min())
        maxR_int = np.ceil(data['r'].max())
        bins = np.arange(minR_int, maxR_int, 1)
        groups = data.groupby(pd.cut(data['r'], bins))
        meanData = groups.mean()
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
        labelData.plot(x='r', y='rotated', style='+-', label=currentLabel, color=plt.cm.jet(1.*currentColor/numColors))
        currentColor += 1
    plt.legend()
    return f

def visualizeChiralityData(chiralityData):
    """Plots the different sectors"""
    fig = plt.figure()
    groups = chiralityData.groupby('label')
    numColors = len(groups)
    print numColors
    currentColor = 0
    for name, group in groups:
        group.plot(x='x',y='y', label=name, color=plt.cm.jet(1.*currentColor/numColors))
        currentColor += 1
    fig.gca().invert_yaxis()
    plt.legend()
    return fig
"""
Created on Dec 14, 2013

@author: bryan
"""

import exifread
import pylab as pl
import skimage.io
import numpy as np
import pandas as pd

def niceDisplay(image, display, scaleFactor=0.25):
    newImage = image.applyLayers()
    newImage.resize( int(newImage.width*scaleFactor), int(newImage.height*scaleFactor) ).save(display)


def getScale(imagePath):

    f = open(imagePath, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    f.close()
    
    scaleFraction = tags['Image XResolution'].values[0]
    pixelPerMM= float(scaleFraction.num)/float(scaleFraction.den)
    
    return 1/pixelPerMM


def showImage(image):
    pl.figure()
    skimage.io.imshow(image, interpolation='None')

def loopTheta(theta):
    if theta < -np.pi:
        theta += 2*np.pi
    elif theta > np.pi:
        theta += -2*np.pi
    return theta

def getPositionData(coords, center):
    (x, y) = (coords[:, 0], coords[:, 1])
    deltaX = x - center[0]
    deltaY = y - center[1]

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
        (x,y) = np.nonzero(labels == currentLabel)
        coords = np.column_stack((x, y))
        data = getPositionData(coords, center)

        # Now rotate the coordinate system so that it is in the correct spot
        minRadiusIndex = data.r.idxmin()
        minRadiusRow = data.iloc[minRadiusIndex]
        minRadiusTheta = minRadiusRow['theta']

        data['rotated'] = data['theta'] - minRadiusTheta
        data['rotated'][data['rotated'] < -np.pi] = 2*np.pi+ data['rotated']
        data['rotated'][data['rotated'] > np.pi] = -2*np.pi + data['rotated']

        data['label'] = currentLabel

        chiralityData = chiralityData.append(data)

    chiralityData = chiralityData.reset_index(drop=True)
    return chiralityData
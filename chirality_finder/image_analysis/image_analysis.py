'''
Created on Dec 14, 2013

@author: bryan
'''

# Package imports
from image_analysis.utility import *
# External library imports
import skimage as ski
from ski import measure
from ski import color

import pandas as pd
import scipy as sp
import numpy as np
# Other imports 
import SimpleCV as cv

disp = cv.Display(displaytype='notebook')

def findBrightfieldCircle(brightfield, showPictures=True):
    '''Finds the circle (boundary) in a brightfield numpy image. 
    Returns the center and radius of the circle.'''
    
    brightfieldCV = cv.Image(brightfield)
    filtered = brightfieldCV.bandPassFilter(0.05,0.8)
    circs = filtered.findCircle(distance=2000, canny=50)
    # There will only be one circle as we only allow one due to "distance"
    circs = circs[0]    
    center = circs.coordinates()
    radius = circs.radius()
    
    if showPictures:
        circs.image = brightfieldCV
        circs.draw(width=12)
        brightfieldCV = brightfieldCV.applyLayers()
        showImage(brightfieldCV.getNumpy())
    
    return (center, radius)

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

def reduceLabelsToPixel(labels, center, showPictures=True):    
    '''Reduces the labeled sectors down to individual pixels by
    essentially taking the median theta at every radius in each blob.
    Currently, this cannot handle multi-valued sectors (i.e. sectors that
    annihilate) as the median will be in the middle of the sector.
    
    Returns the new labels.''' 
    
    # Make the median disk length greater than 1.4 so that you can
    # get 8 connected neighbors
    medianDiskLength = 1.5
    radiusStep = .5

    thinnedLabels = 0 * labels

    if showPictures: showImage(color.label2rgb(labels - 1))
    
    props = measure.regionprops(labels)
    for p in props:
        currentLabel = p.label
        coords = p.coords
        currentData = getPositionData(coords, center)
        
        minDistance = currentData['r'].min()
        maxDistance = currentData['r'].max()
        
        radiusList = np.arange(minDistance, maxDistance, radiusStep)
        
        for radius in radiusList:
            # Get all points that are essentially at this radius
            dataAtRadius = currentData[np.abs(radius - currentData['r']) <= medianDiskLength]
            
            medianTheta = dataAtRadius.theta.median()
            medianThetaRow = dataAtRadius[dataAtRadius.theta == medianTheta]
            medianX = medianThetaRow['x']
            medianY = medianThetaRow['y']

            # Get median line theta
            thinnedLabels[medianX, medianY] = currentLabel
            
    if showPictures: showImage(color.label2rgb(thinnedLabels - 1))
    showImage(color.label2rgb(thinnedLabels - 1))
    return thinnedLabels

def splitLabels(labels, center, showPictures=True):    
    '''Splits labels into pieces based on connected components
    as a funciton of R.''' 
    
    # Move in the opposite direction of growth and cut
    medianDiskLength = 1.5
    radiusStep = .5

    newLabels = 0 * labels
    labelCounter = 1

    if showPictures: showImage(ski.color.label2rgb(labels - 1))
    imageDimensions = np.shape(labels)
    props = ski.measure.regionprops(labels)
    for p in props:
        currentLabel = p.label
        coords = p.coords
        currentData = getPositionData(coords, center)
        
        minDistance = currentData['r'].min()
        maxDistance = currentData['r'].max()
        
        radiusList = np.arange(minDistance, maxDistance, radiusStep)
        
        currentBlobPic = np.zeros(imageDimensions)
        for radius in radiusList:
            # Get all points that are essentially at this radius
            dataAtRadius = currentData[np.abs(radius - currentData['r']) <= medianDiskLength]
            # Figure out if the regions are disconnected; if so, assign them
            # different names
            currentRadiusPic = np.zeros(imageDimensions)
            currentRadiusPic[dataAtRadius['x'], dataAtRadius['y']] = 1
            connectedComponents = ski.morphology.label(currentRadiusPic, neighbors=8, background=0) + 1
            
            representativeTheta = np.array([])
            radiusLabelMax = np.max(connectedComponents)
            for l in range(1, radiusLabelMax):
                (radX, radY) = np.nonzero(connectedComponents == l)
                radX = radX[0]
                radY = radY[0]
                currentThetaRow = dataAtRadius[(dataAtRadius['x']==radX) & (dataAtRadius['y']==radY)]
                
    return thinnedLabels

def filterSectors(labels, center, minLength=250, showPictures=True):
    ''' Returns a new labeled image with poor quality sectors
    filtered out. This basically means the sector is not long
    enough.'''
    
    if showPictures:
        labelImage = ski.color.label2rgb(labels - 1)
        showImage(labelImage)
    
    filteredLabels = labels * 0
    chiralityData = getChiralityData(labels, center)    
    
    #### Apply the filter ####
    # Region properties no longer works, as the regions are now
    # technically disconnected. So don't use that!

    uniqueLabels = np.unique(labels)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    
    for currentLabel in uniqueLabels:
        # Get current data
        currentData = chiralityData[chiralityData['label'] == currentLabel]
        # Get minimum and maximum r
        minR = currentData['r'].min()
        maxR = currentData['r'].max()
        if maxR - minR > minLength:
            filteredLabels[currentData['x'], currentData['y']] = currentLabel
    # Let's assume this does a good enough filtering job for now.
    # Now figure out the chirality!
    
    if showPictures: showImage(ski.color.label2rgb(labels - 1))
    
    return filteredLabels

def getChiralityData(labels, center):
    minLabel = 1
    maxLabel = np.max(labels)
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
"""
Created on Dec 14, 2013

@author: bryan
"""

# Package imports
from utility import *
# External library imports
import skimage as ski
import skimage.color
import pandas as pd
import numpy as np
import skimage.io
import skimage.filter
import skimage.morphology
import skimage.draw
import skimage.measure
import SimpleCV as cv
from prune_skeleton import prune_skeleton

def findBrightfieldCircle(brightfield, showPictures=False):
    """Finds the circle (boundary) in a brightfield numpy image.
    Returns the center and radius of the circle."""
    
    brightfieldCV = cv.Image(brightfield)
    filtered = brightfieldCV.bandPassFilter(0.05,0.8)
    circs = filtered.findCircle(distance=2000, canny=50)
    # There will only be one circle as we only allow one due to "distance"
    circs = circs[0]    
    center = circs.coordinates()
    radius = circs.radius()
    
    if showPictures:
        circs.image = brightfieldCV
        circs.draw(width=6)
        brightfieldCV = brightfieldCV.applyLayers()
        showImage(brightfieldCV.getNumpy())
    
    return center, radius

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
    """Reduces the labeled sectors down to individual pixels by
    essentially taking the median theta at every radius in each blob.
    Currently, this cannot handle multi-valued sectors (i.e. sectors that
    annihilate) as the median will be in the middle of the sector.
    
    Returns the new labels."""
    
    # Make the median disk length greater than 1.4 so that you can
    # get 8 connected neighbors
    medianDiskLength = 1.5
    radiusStep = .5

    thinnedLabels = 0 * labels

    if showPictures: showImage(ski.color.label2rgb(labels - 1))
    
    props = ski.measure.regionprops(labels)
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
            
    if showPictures: showImage(ski.color.label2rgb(thinnedLabels - 1))
    showImage(ski.color.label2rgb(thinnedLabels - 1))
    return thinnedLabels

def filterSectors(labels, center, minLength=250, showPictures=True):
    """ Returns a new labeled image with poor quality sectors
    filtered out. This basically means the sector is not long
    enough."""
    
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

def findSectors(path, homelandCutFactor=0.33, edgeCutFactor=0.9, showPictures=False):
    """Finds the sectors in an image by looking at the first
    fluorescence image (the first channel).

    Returns a labeled image with regions filed down to
    approximately a single pixel."""

    img = ski.io.imread(path)
    if showPictures:
        showImage(img)
    fluor1 = img[:, :, 0]
    fluor2 = img[:, :, 1]
    brightfield = img[:, :, 2]

    ### Use SimpleCV to find circle for now ###
    (center, radius) = findBrightfieldCircle(brightfield, showPictures=showPictures)

    # Filter, clean things up
    filtered = ski.filter.rank.median(fluor1, ski.morphology.disk(5))
    if showPictures: showImage(filtered)

    # Find sectors
    edges = ski.filter.sobel(filtered)
    if showPictures: showImage(edges)

    # Cut out center
    (rr, cc) = ski.draw.circle(center[0], center[1], homelandCutFactor*radius)
    edges[rr, cc] = 0
    # Cut out edge as we get artifacts there. Note that we could fix this by
    # using some sort of bandpass filter, but python is being a pain so we won't
    # do that right now

    (rr, cc) = ski.draw.circle(center[0], center[1], radius*edgeCutFactor)
    mask = np.zeros(np.shape(edges))
    mask[rr, cc] = 1

    edges = np.multiply(mask, edges)

    # Binarize
    binaryValue = ski.filter.threshold_otsu(edges)
    binary = edges > binaryValue



    prunedSkeleton = prune_skeleton(skeleton, showPictures)

    binaryLabels = ski.morphology.label(prunedSkeleton, neighbors=8, background=0) + 1
    if showPictures: showImage(ski.color.label2rgb(binaryLabels - 1))

    # Filter out small labels
    necessaryLength = .8*radius - homelandCutFactor*radius
    filteredLabels = filterSectors(binaryLabels, center, minLength=int(necessaryLength), showPictures=showPictures)

    if showPictures: showImage(ski.color.label2rgb(filteredLabels - 1))

    # Get the data of the sectors
    chiralityData = getChiralityData(filteredLabels, center)
    return filteredLabels, chiralityData
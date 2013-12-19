"""
Created on Dec 14, 2013

@author: bryan
"""

# Package imports
from utility import *
import lineage_seperator
# External library imports
import skimage as ski
import skimage.color
import numpy as np
import skimage.io
import skimage.filter
import skimage.morphology
import skimage.draw
import SimpleCV as cv

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

def findSectors(path, homelandCutFactor=0.33, edgeCutFactor=0.9, showPictures=False):
    """Finds the sectors in an image by looking at the first
    fluorescence image (the first channel).

    Returns a labeled image with regions filed down to
    approximately a single pixel."""

    img = ski.io.imread(path)
    if showPictures:
        showImage(img)
    fluor1 = img[:, :, 0]
    #fluor2 = img[:, :, 1]
    brightfield = img[:, :, 2]

    # Resize images so that my lineage separating program works well on them.
    #fluor1 = skimage.transform.rescale(fluor1, 0.5)
    #brightfield = skimage.transform.rescale(brightfield, 0.5)

    ### Use SimpleCV to find circle for now ###
    print 'Finding the center...'
    (center, radius) = findBrightfieldCircle(brightfield, showPictures=showPictures)

    # Filter, clean things up
    print 'Cleaning up image with filters...'
    filtered = ski.filter.rank.median(fluor1, ski.morphology.disk(2))
    if showPictures: showImage(filtered)

    # Find sectors
    print 'Finding edges...'
    edges = ski.filter.sobel(filtered)
    if showPictures: showImage(edges)

    # Cut out center
    print 'Cutting out center...'
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

    # Now seperate into distinct lineages using my circular seperator program.
    print 'Separating lineages...'
    lineage_seperator.padding_angle= 0.01
    circ = lineage_seperator.Circle(binary, center)
    circ.run()
    binaryLabels = circ.getLabelImage()

    if showPictures: showImage(binaryLabels)

    # Filter out small labels
    necessaryLength = .8*radius - homelandCutFactor*radius
    filteredLabels = filterSectors(binaryLabels, center, minLength=int(necessaryLength), showPictures=showPictures)

    if showPictures: showImage(ski.color.label2rgb(filteredLabels - 1))

    # Get the data of the sectors
    chiralityData = getChiralityData(filteredLabels, center)
    return filteredLabels, chiralityData
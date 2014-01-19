"""
Created on Dec 14, 2013

@author: bryan
"""

# Package imports
from utility import *
import InteractiveSelector
# External library imports
import skimage as ski
import skimage.color
import numpy as np
import skimage.io
import skimage.filter
import skimage.morphology
import skimage.draw
import skimage.segmentation
import SimpleCV as cv
import matplotlib.pyplot as plt

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

def selectPoint(image):
    class Select_Point:
        def __init__(self, image_matrix):
            self.center = None
            self.image = image_matrix
            self.figure = plt.figure()
            self.pointAxes = None
            self.figure.canvas.mpl_connect('button_press_event', self.buttonPressed)
            self.figure.canvas.mpl_connect('button_release_event', self.buttonReleased)
            ski.io.imshow(image_matrix)
            plt.show()

        def buttonPressed(self, event):
            if self.pointAxes is not None:
                plt.cla()
            self.center = (event.ydata, event.xdata)
            print 'Center set:' , self.center

        def buttonReleased(self, event):
            self.pointAxes = plt.plot(self.center[1], self.center[0], 'o')
            ski.io.imshow(self.image)
            plt.draw()

    p = Select_Point(image)

    return p.center

def getBinaryData(fluor, brightfield, homelandCutFactor=0.33, edgeCutFactor=0.9, threshold_factor = 1.0, showPictures=False,
                  select_center_manually = False, originalImage = None):
    """Finds the edges in an image by looking at the first
    fluorescence image (the first channel).

    Returns a labeled image with regions filed down to
    approximately a single pixel."""

    ### Use SimpleCV to find circle for now ###
    if not select_center_manually:
        print 'Finding the center...'
        (center, radius) = findBrightfieldCircle(brightfield, showPictures=showPictures)
    else:
        print 'Please select the center.'
        center = np.array(selectPoint(originalImage))
        print 'Please select where you think the boundary of the expansion is.'
        edge = np.array(selectPoint(originalImage))
        radius = np.linalg.norm(edge - center)

    # Filter, clean things up
    print 'Cleaning up image with filters...'
    filtered = ski.filter.rank.median(fluor, ski.morphology.disk(2))
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
    binaryEdges = edges > binaryValue*float(threshold_factor)

    if showPictures: showImage(binaryEdges)

    return binaryEdges, center, radius

def findSectors(fluor, brightfield, homelandCutFactor=0.33, edgeCutFactor=0.9, threshold_factor = 1.0, showPictures=False,
                exportBinaryEdges=False, select_center_manually = False, originalImage = None):
    """Finds the sectors in an image by looking at the first
    fluorescence image (the first channel).

    Returns a labeled image with regions filed down to
    approximately a single pixel."""

    binaryEdges, center, radius = getBinaryData(fluor, brightfield, homelandCutFactor=homelandCutFactor,
                        edgeCutFactor=edgeCutFactor, threshold_factor=threshold_factor, showPictures=showPictures,
                        select_center_manually=select_center_manually, originalImage=originalImage)
    if exportBinaryEdges:
        print 'Exporting image to test folder...'
        ski.io.imsave('binaryEdges.tiff', binaryEdges)
    lt = InteractiveSelector.InteractiveSelector(binaryEdges)
    editedBinary = lt.image
    binaryLabels = ski.morphology.label(editedBinary, neighbors=4, background=False) + 1

    if showPictures: showImage(binaryLabels)

    # Filter out small labels
    print 'Filtering out small labels...'
    necessaryLength = 150
    filteredLabels = ski.morphology.remove_small_objects(\
        binaryLabels, min_size=necessaryLength, connectivity=4, in_place=True)

    if showPictures: showImage(ski.color.label2rgb(filteredLabels, bg_label=0))

    filteredLabels, forwardmap, inversemap = ski.segmentation.relabel_sequential(filteredLabels, offset=1)

    # Get the labels of the different sectors
    print 'Done!'
    return filteredLabels, center, radius
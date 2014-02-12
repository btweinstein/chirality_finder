"""
Created on Dec 14, 2013

@author: bryan
"""

import exifread
import pylab as pl
import skimage.io
import numpy as np


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
    fig = pl.figure()
    skimage.io.imshow(image, interpolation='None')
    return fig

def loopTheta(theta):
    if theta < -np.pi:
        theta += 2*np.pi
    elif theta > np.pi:
        theta += -2*np.pi
    return theta
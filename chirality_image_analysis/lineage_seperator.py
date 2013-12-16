__author__ = 'bryan'
"""Responsible for separating lineages in a binary image
by utilizing the geometry of the range expansion."""

import skimage.draw
import numpy as np

from image_analysis import *


class Circle:
    inputImage = None
    center = None

    _radius = None
    _maxRadius = None
    _minRadius = None
    _xpoints = None
    _ypoints = None


    def __init__(self, inputImage, center):
        self.inputImage = inputImage
        self.center = center

        # Now find the minimum and maximum radius
        allPoints = np.where(inputImage)
        data = getPositionData(allPoints, self.center)
        self._minRadius = np.ceil(data.r.min())
        self._maxRadius = np.floor(data.r.max())

    def setPointsAtRadius(self):
        self._xpoints, self._ypoints = ski.draw.circle_perimeter(self.center[0], self.center[1], self._radius)

    def doStep(self):
        self.setPointsAtRadius()
        # Get a list of all points that are at that radius
        self.inputImage[self._xpoints, self._ypoints]

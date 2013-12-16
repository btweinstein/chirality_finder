__author__ = 'bryan'
"""Responsible for separating lineages in a binary image
by utilizing the geometry of the range expansion."""

import scipy as sp
import scipy.spatial

import skimage.draw

from image_analysis import *


eight_con_dist = 1.5

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
        [xcoords, ycoords] = np.where(inputImage != 0)
        stack = np.column_stack((xcoords, ycoords))
        data = getPositionData(stack, self.center)
        self._maxRadius = int(np.floor(data.r.max()))
        self._minRadius = int(np.ceil(data.r.min()))

        self._radius = self._maxRadius

    def setPointsAtRadius(self):
        self._xpoints , self._ypoints = ski.draw.circle_perimeter(self.center[0], self.center[1], self._radius)

    def doStep(self):
        self.setPointsAtRadius()
        # Get a list of all points that are at that radius
        index = np.where(self.inputImage[self._xpoints, self._ypoints])

        xPOI = self._xpoints[index]
        yPOI = self._ypoints[index]

        poi_coords = np.column_stack((xPOI, yPOI))
        poi_data = getPositionData(poi_coords, self.center)

        # Assume each point is separate, then link!
        dataLength = len(poi_data)
        poi_data['label'] = np.arange(dataLength)

         # Find the separation between every point
        pointVec = np.column_stack((poi_data['x'], poi_data['y']))
        distMat = sp.spatial.distance.cdist(pointVec, pointVec)
        distMat = np.triu(distMat)

        # Label connected components
        i, j = np.where(distMat <= eight_con_dist)
        poi_data.iloc[i].label = poi_data.iloc[j].label

        print
        print poi_data
        print
        self._radius -= 1

sector_length = 3.0
class Circle_Sector:
    _r = None
    _theta = None
    _minTheta = None
    _maxTheta = None

    def __init__(self, r, theta):
        self._r = r
        self._theta = theta
        self.setCurrentSpread()

    def setCurrentSpread(self):
        # We know s = r*d\theta
        dtheta = sector_length/self._r
        self._minTheta = self._theta - dtheta
        self._maxTheta = self._theta + dtheta
        # Make sure these fall in the coordinate system we
        # have defined, i.e. 0 to 2pi

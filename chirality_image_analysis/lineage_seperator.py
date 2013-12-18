__author__ = 'bryan'
"""Responsible for separating lineages in a binary image
by utilizing the geometry of the range expansion."""

import scipy as sp
import scipy.spatial

import skimage.draw
from image_analysis import *

######## Main Class ########

eight_con_dist = 1.50

class Circle:

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

        self._xpoints = None
        self._ypoints = None

        self._sectorHistory = []

    def setPointsAtRadius(self):
        xpoints, ypoints = ski.draw.circle_perimeter(self.center[0], self.center[1], self._radius)
        # Remove the points that fall outside the image
        coords = np.column_stack((xpoints, ypoints))
        coordArray = pd.DataFrame(data=coords, columns=['x', 'y'])
        coordArray = coordArray[(coordArray['x'] >= 0) & (coordArray['y'] >= 0)]
        coordArray = coordArray[(coordArray['x'] < self.inputImage.shape[0]) &  \
                                (coordArray['y'] < self.inputImage.shape[1])]
        self._xpoints = coordArray['x'].values
        self._ypoints = coordArray['y'].values

    def getLabeledPointsAtRadius(self):
        """Gets all points at the radius and their position data. Returns connected component labels
        (8 neighbors) too."""
        index = np.where(self.inputImage[self._xpoints, self._ypoints])

        xPOI = self._xpoints[index]
        yPOI = self._ypoints[index]

        poi_coords = np.column_stack((xPOI, yPOI))
        poi_data = getPositionData(poi_coords, self.center)

        # Assume each point is separate, then link!
        dataLength = len(poi_data)
        poi_data['c_label'] = np.arange(1, 1 + dataLength)

         # Find the separation between every point
        pointVec = np.column_stack((poi_data['x'], poi_data['y']))
        distMat = sp.spatial.distance.cdist(pointVec, pointVec)
        distMat = np.triu(distMat)

        # Label connected components
        i, j = np.where((distMat <= eight_con_dist) & (distMat > 0))
        if len(i) > 0 and len(j) > 0:
            for r, c in zip(i, j):
                poi_data.c_label.iloc[r] = poi_data.c_label.iloc[c]

        return poi_data

    def run(self):
        while self._radius >= self._minRadius:
            print 'Radius:' , self._radius
            self.doStep()

    def doStep(self):
        self.setPointsAtRadius()

        # Get a list of all points that are at that radius
        poi_data = self.getLabeledPointsAtRadius()

        # Based on the labels, create sectors
        groups = poi_data.groupby('c_label')
        currentSectors = []
        for label, g in groups:
            # Create a sector
            newSector = Circle_Sector(g['x'], g['y'], self._radius, self.center)
            newSector._label = label
            currentSectors.append(newSector)
        if len(self._sectorHistory) != 0:
            # Set children and parents of each sector
            lastSectors = self._sectorHistory[-1]

            # Find overlapping regions
            for oldSector in lastSectors:
                for newSector in currentSectors:
                    if oldSector.checkOverlap(newSector):
                        oldSector._childSectors.append(newSector)
                        newSector._parentSectors.append(oldSector)

            # Link the labels to the old, being wary of branch points (multiple children!)
            for oldSector in lastSectors:
                childrenNumber = len(oldSector._childSectors)
                if childrenNumber == 1: # Not a branch point
                    oldSector._childSectors[0]._label = oldSector._label
                elif childrenNumber > 1: # Branch Point
                    print 'branch point!'
                    oldSector._childSectors[0]._label = oldSector._label
                    # Get the maximum label number currently in use
                    maxLabel = -1
                    for s in currentSectors:
                        if s._label > maxLabel: maxLabel = s._label
                    for i in range(1, childrenNumber):
                        oldSector._childSectors[i]._label = maxLabel
                        maxLabel += 1
        self._sectorHistory.append(currentSectors)
        self._radius -= 1

    def getLabelImage(self, debug=False):
        labelImage = np.zeros(self.inputImage.shape, dtype=np.int)
        for i in range(len(self._sectorHistory)):
            for sector in self._sectorHistory[i]:
                if not debug:
                    labelImage[sector._xvalues, sector._yvalues] = sector._label
                else:
                    labelImage[sector._xvalues, sector._yvalues] = i
        return labelImage

    def getSectorLabelImage(self):
        """Labels each sector differently so that you can see the sectors."""
        labelImage = np.zeros(self.inputImage.shape, dtype=np.int)
        count = 1
        for i in range(len(self._sectorHistory)):
            for sector in self._sectorHistory[i]:
                labelImage[sector._xvalues, sector._yvalues] = count
                count +=1
        return labelImage

######## Sectors #########

from utility import *

padding_length = 1.0

class Circle_Sector:
    """Sectors contain connected pixels."""

    def __init__(self, xvalues, yvalues, radius, center):
        self._xvalues = xvalues
        self._yvalues = yvalues
        self._radius = radius
        self._center = center

        allPoints = np.column_stack((self._xvalues, self._yvalues))
        self._positionData = getPositionData(allPoints, self._center)

        self._maxTheta = self._positionData.theta.max()
        self._minTheta = self._positionData.theta.min()

        self._label = None
        self._parentSectors = []
        self._childSectors = []

    def checkOverlap(self, otherSector):
        """ Returns true if the two overlaps intersect (within a given tolerance)
        specified by padding_length."""
        # ds = r*dtheta
        dtheta = padding_length/self._radius
        # It is easiest to just compare all theta
        thetaSelf = self._positionData['theta']
        newTheta = otherSector._positionData['theta']

        origMesh, newMesh = np.meshgrid(thetaSelf, newTheta)
        overlap = np.abs(origMesh - newMesh) <= dtheta

        return np.any(overlap)
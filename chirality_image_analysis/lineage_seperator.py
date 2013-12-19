__author__ = 'bryan'
"""Responsible for separating lineages in a binary image
by utilizing the geometry of the range expansion."""

import skimage as ski
import skimage.draw
import skimage.morphology
from utility import *

######## Main Class ########

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

        self._maxLabel = 0

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
        (8 neighbors) too. Returns labels greater than the current maximum label."""
        index = np.where(self.inputImage[self._xpoints, self._ypoints])

        xPOI = self._xpoints[index]
        yPOI = self._ypoints[index]

        # Create a new image that we use connected components on
        cImage = np.zeros(self.inputImage.shape, dtype=np.int)
        cImage[xPOI, yPOI] = 1
        labelImage = skimage.morphology.label(cImage, neighbors=8) + 1

        poi_coords = np.column_stack((xPOI, yPOI))
        poi_data = getPositionData(poi_coords, self.center)
        # Label connected components
        poi_data['_clabel'] = labelImage[poi_data['x'], poi_data['y']]
        poi_data['_clabel'] += self._maxLabel

        return poi_data

    def run(self):
        while self._radius >= self._minRadius and self._radius <= self._maxRadius:
            print 'Radius:' , self._radius
            self.doStep()

    def doStep(self):
        self.setPointsAtRadius()

        # Get a list of all points that are at that radius
        poi_data = self.getLabeledPointsAtRadius()

        # Based on the labels, create sectors
        groups = poi_data.groupby('_clabel')
        currentSectors = []
        for clabel, g in groups:
            # Create a sector
            newSector = Circle_Sector(g['x'], g['y'], self._radius, self.center)
            newSector._clabel = clabel
            currentSectors.append(newSector)
        if len(self._sectorHistory) != 0:
            # Set children and parents of each sector
            lastSectors = self._sectorHistory[-1]

            # Find overlapping regions and link them
            for oldSector in lastSectors:
                for newSector in currentSectors:
                    if oldSector.checkOverlap(newSector):
                        oldSector._childSectors.append(newSector)
                        newSector._parentSectors.append(oldSector)

            # Link if not a branch point
            for oldSector in lastSectors:
                childrenNumber = len(oldSector._childSectors)
                if childrenNumber == 1: # Not a branch point
                    oldSector._childSectors[0]._clabel = oldSector._clabel
            #If you have multiple parents, likely a result of things not being connected
            #when they should be. Choose the piece with more labels in the image currently.
            for newSector in currentSectors:
                numParents = len(newSector._parentSectors)
                if numParents >= 2:
                    print 'Multiple parents! Linking...'
                    labelImage = self.getLabelImage()
                    maxNumPoints = -1
                    maxLabel = None
                    for psector in newSector._parentSectors:
                        # Get number of points with each label
                        numMembers = np.sum(psector._clabel == labelImage)
                        if numMembers > maxNumPoints:
                            maxNumPoints = numMembers
                            maxLabel = psector._clabel
                    newSector._clabel = maxLabel
                    for psector in newSector._parentSectors:
                        psector._clabel = maxLabel

        # Check what the maximum label at the current radius is
        for s in currentSectors:
            if self._maxLabel < s._clabel:
                self._maxLabel = s._clabel
        self._sectorHistory.append(currentSectors)
        self._radius -= 1

    def getLabelImage(self, debug=False):
        labelImage = np.zeros(self.inputImage.shape, dtype=np.int)
        for i in range(len(self._sectorHistory)):
            for sector in self._sectorHistory[i]:
                if not debug:
                    labelImage[sector._xvalues, sector._yvalues] = sector._clabel
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

        self._clabel = None
        self._parentSectors = []
        self._childSectors = []

    def checkOverlap(self, otherSector):
        """ Returns true if the two overlaps intersect (within a given tolerance)
        specified by padding_length."""
        # ds = r*dtheta
        dtheta = padding_length/float(self._radius)
        # It is easiest to just compare all theta

        thetaSelf = self._positionData['theta']
        newTheta = otherSector._positionData['theta']

        origMesh, newMesh = np.meshgrid(thetaSelf, newTheta)
        overlap = np.abs(origMesh - newMesh) <= dtheta

        return np.any(overlap)

    def mergeSectors(self, otherSector):
        # Just create a new sector at the current radius with points
        # combined.

        newXValues = np.hstack((self._xvalues, otherSector._xvalues))
        newYValues = np.hstack((self._yvalues, otherSector._yvalues))
        newRadius = self._radius
        newCenter = self._center

        return Circle_Sector(newXValues, newYValues, newRadius, newCenter)
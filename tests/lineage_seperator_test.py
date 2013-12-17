__author__ = 'bryan'

import skimage as ski
import skimage.io

from chirality_image_analysis import lineage_seperator
from chirality_image_analysis.utility import  *


testImage = ski.io.imread('test_image.tif')
testImage = (testImage > 0)
center = np.array([testImage.shape[0], testImage.shape[1]])/2
showImage(testImage)
circ = lineage_seperator.Circle(testImage, center)
print 'Max radius: ' , circ._maxRadius
print 'Min radius: ' , circ._minRadius
circ.run()
circ.plotSectorHistory()

pl.show()
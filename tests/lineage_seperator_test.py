__author__ = 'bryan'

import skimage as ski
import skimage.io
import skimage.color

from chirality_image_analysis import lineage_seperator
from chirality_image_analysis.utility import  *


testImage = ski.io.imread('test_image.tif')
testImage = (testImage > 0)
center = np.array([202, 240])
showImage(testImage)
circ = lineage_seperator.Circle(testImage, center)
print 'Max radius: ' , circ._maxRadius
print 'Min radius: ' , circ._minRadius
circ.run()
labelImage = circ.getLabelImage(debug=False)

showImage(ski.color.label2rgb(labelImage - 1))
pl.show()
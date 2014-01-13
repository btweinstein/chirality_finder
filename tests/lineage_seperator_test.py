__author__ = 'bryan'

import skimage as ski
import skimage.io
import skimage.color

from chirality_image_analysis import lineage_seperator
from chirality_image_analysis.utility import  *


testImage = ski.io.imread('binaryEdges.tiff')
testImage = (testImage > 0)
center = np.array([512, 696])
showImage(testImage)
circ = lineage_seperator.Circle(testImage, center)
print 'Max radius: ' , circ._maxRadius
print 'Min radius: ' , circ._minRadius
circ.run()
labelImage = circ.getLabelImage()

# Let's make sure that the labels are correct.
#props = skimage.measure.regionprops(labelImage)
#for p in props:
#    showImage(labelImage == p.label)

showImage(ski.color.label2rgb(labelImage - 1))
pl.show()
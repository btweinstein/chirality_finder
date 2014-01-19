__author__ = 'bryan'

import matplotlib.pyplot as plt
import skimage as ski
import skimage.color
import skimage.io

from chirality_data_analysis.example_data import *
from chirality_image_analysis import image_analysis as chi
from chirality_image_analysis.utility import *
from chirality_data_analysis import data_analysis as chd


currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

img = ski.io.imread(latestPath)
fluor1 = img[:, :, 0]
fluor2 = img[:, :, 1]
brightfield = img[:, :, 2]

filteredLabels, center, finalRadius = chi.findSectors(fluor1, brightfield, select_center_manually=True, originalImage=img)
showImage(ski.color.label2rgb(filteredLabels, bg_label=0))

# Now that we have the labels, we need to get the position data of each label
# and filter them all down to one pixel
chiralityData = chd.getChiralityData(filteredLabels, center)

# Look at the problem data, label #5
#problemData = chiralityData[chiralityData['label'] == 5]
#print problemData.head(30)

chd.makeChiralityPlot(chiralityData)
chd.visualizeSectors(chiralityData)

# Overlay the sectors on the image now
chd.visualizeSectors(chiralityData, overImage=True)
ski.io.imshow(img, interpolation='None')

plt.show()
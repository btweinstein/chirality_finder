__author__ = 'bryan'

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from chirality_data_analysis.example_data import *
from chirality_image_analysis import image_analysis as chi
from chirality_image_analysis.utility import *
from chirality_data_analysis import data_analysis as chd
import skimage as ski
import skimage.color

currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

filteredLabels, center, finalRadius = chi.findSectors(latestPath, showPictures=False)
showImage(ski.color.label2rgb(filteredLabels, bg_label=0))

# Now that we have the labels, we need to get the position data of each label
# and filter them all down to one pixel
chiralityData = getChiralityData(filteredLabels, center)
print chiralityData

chd.makeChiralityPlot(chiralityData)
chd.visualizeChiralityData(chiralityData)

plt.show()
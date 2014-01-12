__author__ = 'bryan'

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from chirality_data_analysis.example_data import *
from chirality_image_analysis import image_analysis as chi
from chirality_image_analysis.utility import *
import skimage as ski
import skimage.color

currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

filteredLabels = chi.findSectors(latestPath, showPictures=False)
showImage(ski.color.label2rgb(filteredLabels, bg_label=0))

for l in np.unique(filteredLabels):
    showImage(filteredLabels == l)

plt.show()
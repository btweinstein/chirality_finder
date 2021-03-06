__author__ = 'bryan'

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from chirality_data_analysis.example_data import *
from chirality_image_analysis.utility import *
from chirality_image_analysis import image_analysis as chi
import skimage as ski
import skimage.color
import skimage.morphology

currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

binaryEdges, center, radius = chi.getBinaryData(latestPath, showPictures=False)
labeled = ski.color.label2rgb(ski.morphology.label(binaryEdges, background=False))
showImage(labeled)
plt.show()
#chiFinder = LassoTool.LassoTool(binaryEdges)

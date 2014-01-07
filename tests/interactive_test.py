__author__ = 'bryan'

from matplotlib import pyplot as pl

from chirality_data_analysis.example_data import *
from chirality_image_analysis import image_analysis as chi
from chirality_image_analysis import LassoTool


currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

binaryEdges = chi.getBinaryEdges(latestPath, showPictures=False)
chiFinder = LassoTool.LassoTool(binaryEdges)
pl.show()
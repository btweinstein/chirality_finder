__author__ = 'bryan'

import pylab as pl

from chirality_data_analysis.example_data import *
from chirality_image_analysis import image_analysis as chi


currentData = desiredColonies[desiredColonies.name == 'ReplicateB']
latestDate = currentData.irow(currentData['date'].argmax())
latestPath = latestDate['path']

(filteredLabels, chiralityData) = chi.findSectors(latestPath, showPictures=True)
# You must use pl.show() at the end of the code if you want to see what plots were generated.
# As interactive plot generation is not enabled, I believe.
pl.show()
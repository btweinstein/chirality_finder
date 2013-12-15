__author__ = 'bryan'

import skimage as ski
import skimage.io

from example_data import *
from chirality_image_analysis import image_analysis as chi
from chirality_image_analysis import utility as ut


desiredColonies['diameter'] = None
for path in desiredColonies['path']:
    scale = ut.getScale(path)

    # Open the image
    img = ski.io.imread(path)

    fluor1 = img[:, :, 0]
    fluor2 = img[:, :, 1]
    brightfield = img[:, :, 2]

    (center, radius) = chi.findBrightfieldCircle(brightfield, True)
    print center, radius
    if center is not None and radius is not None:
        actualDiameter = 2.0*scale*radius
        # Assumes that if the diameter is less than 1, something bad happened.
        if actualDiameter > 1:
            desiredColonies.diameter[desiredColonies.path == path] = actualDiameter
    else:
        print 'Could not find a circle :('

# Plot the result
import pylab as pl

pl.figure()

uniqueNames = desiredColonies['name'].unique()
for name in uniqueNames:
    currentData = desiredColonies[desiredColonies.name == name]
    currentData = currentData.sort(columns='date')
    currentData.date = currentData.date - currentData.date.irow(0)

    currentData.plot('date', 'diameter', style='-o')
pl.legend(uniqueNames, loc=4)

pl.show()
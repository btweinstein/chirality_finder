__author__ = 'bryan'

from organize_data import *

imageData = getExperimentMetadata()

desiredColonies = imageData[(imageData['T']==37)&(imageData['mL']==25)]
# Find all unique names in the desired colonies
uniqueNames = desiredColonies['name'].unique()
numUnique = len(uniqueNames)
# Split up the data into unique names that you can analyze
uniqueColonies = []
for i in range(numUnique):
    currentName = uniqueNames[i]
    sameType = desiredColonies[desiredColonies['name']==currentName]
    uniqueColonies.append(sameType)
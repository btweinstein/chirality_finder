__author__ = 'bryan'

import glob
import os
import datetime

import pandas as pd


baseDirectory = '/home/bryan/Documents/Research_Data/Nelson/Chirality_Initial_Imaging'
analysisDirectory = baseDirectory + '/6_ToAnalyze_Clean'

def getExperimentMetadata():

    imageFiles = glob.glob(analysisDirectory + '/*.tif')
    imageData = pd.DataFrame(imageFiles, columns=['path'])

    #### Get Desired Information about images ####

    numImages = len(imageData)

    imageInfoArray = []
    for i in range(numImages):
        # Get the images
        imageName = os.path.basename(imageData['path'][i])
        # Chop off the end with the .zvi to avoid shennanigans
        imageArray = (imageName.split('-'))
        imageInfoStr = imageArray[0]

        imageInfo  = os.path.basename(imageInfoStr).split('_')
        imageInfoArray.append(imageInfo)

    #### Convert the array to a proper type ####
    columnNames = ['T', 'mL', 'name', 'date']

    for i in range(numImages):
        imageInfo = imageInfoArray[i]
        # Convert temperature to int
        temp = imageInfo[0]
        imageInfo[0] = int(temp[0:2])
        # Convert augerVolume to int
        augerVolume = imageInfo[1]
        imageInfo[1] = int(augerVolume[0:2])
        # Combine replicate info and other qualifiers to act as an identifier
        # I assume if the string is a digit, it is a date
        # I also assume that there is at most one extra qualifier
        if not imageInfo[3].isdigit():
            imageInfo[2] += '_' + imageInfo[3]
            imageInfo.pop(3)
        date = datetime.datetime(int(imageInfo[3]), int(imageInfo[4]), int(imageInfo[5]))
        del imageInfo [-3:]
        imageInfo.append(date)
        # Update
        imageInfoArray[i] = imageInfo

    imageInfo = pd.DataFrame(imageInfoArray, columns=columnNames)
    imageData = imageData.join(imageInfo)
    return imageData
__author__ = 'bryan'

import seaborn as sns
sns.set('talk')
import pandas as pd
import numpy as np
import time
import glob

baseDirectory = '/home/bryan/Documents/Research_Data/Nelson/Protein_Synthesis_Inhibitor'
analysisDirectory = baseDirectory + '/5_clean_try3'
growthDirectory = baseDirectory + '/Analysis/Growth'
editedDirectory = growthDirectory + '/Edited_Output'
dataDirectory = baseDirectory + '/Analysis/Chirality'

def get_all_data(output_folder = '/home/bryan/Documents/Research_Data/Nelson/all_analysis_data'):
    '''Returns growth data, chirality data for the chloramphenicol experiment'''
    # Load growth data

    growthData = np.load(baseDirectory +'/ipython/combined_growth_data.pkl')['growthData']
    growthData['colony_radius_um'] = growthData['colony_radius_mm'] * 10.**3.

    # Load chirality data

    folders = glob.glob(dataDirectory + '/*/')
    csvPaths = [f + '/chiralityData.csv' for f in folders]
    allData = pd.DataFrame()
    for path in csvPaths:
        allData = allData.append(pd.read_csv(path))

    allData.reset_index()

    # Fix up the loaded chirality data

    allData = allData.rename(columns = {'Unnamed: 0' : 'radius_interval'})
    allData['rotated_righthanded'] = -1.*allData['rotated']

    path_label_groups = allData.groupby(['path', 'label'])
    minRadii = path_label_groups.agg({'r_mm' : min})
    minRadii = minRadii.add_suffix('_min')

    mr_reset = minRadii.reset_index()
    allData = pd.merge(allData, mr_reset, on=['path', 'label'])

    allData['r_um'] = allData['r_mm']*10.**3.
    allData['r_um_min'] = allData['r_mm_min']*10.**3

    allData['log_r_div_ri'] = np.log(allData['r_um']/allData['r_um_min'])
    allData['1divri_minus_1divr_1divum'] = (1/allData['r_um_min']) - (1/allData['r_um'])

    # Give plates ID's to make our life easier

    individualPlates = allData.groupby(['medium', 'name', 'chlor'])
    allDataNew = pd.DataFrame()
    plateID = 0
    for name, group in individualPlates:
        group['plateID'] = plateID
        plateID += 1
        allDataNew = allDataNew.append(group)
    allData = allDataNew

    # Give growth data the same plates too

    individualPlates = allData.groupby(['medium', 'name', 'chlor'])
    growthDataNew = growthData.copy()
    growthDataNew['plateID'] = None
    for group_name, group in individualPlates:
        medium = group_name[0]
        name = group_name[1]
        chlor = group_name[2]

        currentPlateID = group['plateID'].iloc[0]

        growthDataNew['plateID'][(growthDataNew['medium'] == medium) & \
                      (growthData['name'] == name) & \
                      (growthData['chlor'] == chlor)] = currentPlateID
    growthData= growthDataNew

    # Getting the number of sectors

    radiusGroups = allData.groupby(['radius_interval', 'plateID'])

    def getNumSectors(x):
        numSectors = len(np.unique(x['label']))
        x['numSectors'] = numSectors
        return x

    allDataTest = radiusGroups.apply(getNumSectors)

    allData = allDataTest

    # Convert time to delta T
    growthData['time_s'] = growthData['date'].apply(lambda x: time.mktime(x.timetuple()))

    plate_groups = growthData.groupby(['medium', 'chlor', 'name'])
    timeDelta = plate_groups['time_s'].transform(lambda x: x - x.min())
    timeDelta.name = 'timeDelta'

    growthData['timeDelta'] = timeDelta

    # Create a \Delta R column

    def make_delta_r(x):
        min_radius = x['colony_radius_um'].min()
        x['deltaR'] = x['colony_radius_um'] - min_radius
        return x

    growthData = growthData.groupby('plateID').apply(make_delta_r)

    # Save data for a general audience
    growthData.to_pickle(output_folder + '/chlor_growthData.pkl')
    allData.to_pickle(output_folder + '/chlor_allData.pkl')

    return growthData, allData
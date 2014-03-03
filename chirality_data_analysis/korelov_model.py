__author__ = 'bryan'

import pymc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

growthData = None
chiralityData = None

########### Setting up the Analysis ###############

def getNumUnique(x):
        return len(np.unique(x))

def getTotalSectors(x):
    miniGroup = x.groupby('plateID')
    means = miniGroup.agg(np.mean)
    return np.sum(means['numSectors'])

# A list of everything you apply to the data
aggList = [np.mean, np.std, np.var, len]

def setup_analysis(group_on_name, group_on_value, lenToFilterChir = 0, lenToFilterDiff=0, numChirBins=50, numDiffBins=50):
    '''The following inputs MUST be in your pandas matrices or else everything will crash.

    Chirality Data:
    1. 'rotated_righthanded': \Delta \theta in a right handed coordinate system
    2. 'log_r_div_ri': The dimensionless radius desired
    3. '1divri_minus_1divr_1divum' : A different scaled radius
    4. 'plateID': All images taken from the same plate should have the same integer ID
    5. 'numSectors' : The number of sectors of a single colony at a given physical radius
    6. Column that you will separate on, i.e. group_on_name

    Growth Data:
    1. 'colony_radius_um': The colony radius in um
    2. 'timeDelta': The elapsed time in seconds since innoculation
    3. 'plateID': All images taken from the same plate should have the same integer ID
    4. Column that you will separate on, i.e. group_on_name
    '''

    print 'Setting up the simulation for' , group_on_name , '=' , group_on_value , '...'

    # Get the desired data
    plate_groups = growthData.groupby([group_on_name])
    current_growth_group = plate_groups.get_group(group_on_value)

    currentChiralityData = chiralityData[(chiralityData[group_on_name] == group_on_value)]
    print

    av_currentChiralityData = binChiralityData(currentChiralityData, numChirBins, 'log_r_div_ri', lenToFilter = lenToFilterChir)
    plot_av_chirality(av_currentChiralityData)

    print
    av_currentDiffusionData = binChiralityData(currentChiralityData, numDiffBins, '1divri_minus_1divr_1divum', lenToFilter = lenToFilterDiff)
    plot_diffusion(av_currentDiffusionData)

    print
    print 'Done setting up!'

    plt.show()

    return currentChiralityData, current_growth_group, av_currentChiralityData, av_currentDiffusionData

def binChiralityData(currentChiralityData, numChirBins, bin_on, lenToFilter = 150, verbose=True):
    '''Returns the average data per sector in each plate binned over the desired radius.'''

    min_x = currentChiralityData[bin_on].min() - 10.**-6.
    max_x = currentChiralityData[bin_on].max() + 10.**-6.
    numBins = numChirBins

    if verbose:
        print 'numBins for ', bin_on, ':' , numBins
        print 'Min ', bin_on, ':', min_x
        print 'Max ', bin_on, ':', max_x

    bins, binsize = np.linspace(min_x, max_x, numBins, retstep=True)

    if verbose: print 'Dimensionless Binsize for ', bin_on, ':' , binsize

    # Group by sector! Specified by a label and a plate
    sector_groups = currentChiralityData.groupby([pd.cut(currentChiralityData[bin_on], bins), 'plateID', 'label'])
    sectorData = sector_groups.agg(np.mean)
    sectorData = sectorData.rename(columns={bin_on : bin_on + '_mean'})
    sectorData = sectorData.reset_index()
    sectorData = sectorData.rename(columns={bin_on : 'bins',
                                           bin_on + '_mean' : bin_on})

    totalChirSectors = getTotalSectors(sectorData)

    av_currentChiralityData = sectorData.groupby(['bins']).agg(aggList)

    av_currentChiralityData['total_numSectors', 'mean'] = totalChirSectors

    # The key here is to filter out the pieces that have too few elements
    av_currentChiralityData = av_currentChiralityData[av_currentChiralityData['rotated_righthanded', 'len'] > lenToFilter]

    return av_currentChiralityData

def plot_av_chirality(av_currentChiralityData):
    x = av_currentChiralityData['log_r_div_ri', 'mean'].values
    y = av_currentChiralityData['rotated_righthanded', 'mean'].values
    yerr = av_currentChiralityData['rotated_righthanded', 'std'].values

    plt.figure()
    plt.plot(x , y, '+-')
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, antialiased=True)
    plt.xlabel('Average $\ln{(r/r_i)}$')
    plt.ylabel('Average d$\\theta$')
    plt.title('Average d$\\theta$ vs. Normalized Radius')

    #######################
    ## Number of Points ###
    #######################

    x = av_currentChiralityData['log_r_div_ri', 'mean'].values
    y = av_currentChiralityData['rotated_righthanded', 'len'].values

    plt.figure()
    plt.plot(x , y, '+-')

    plt.xlabel('Average $\ln{(r/r_i)}$')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples vs. scaled x-axis (Chirality)')

def plot_diffusion(av_currentDiffusionData):

    x = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    y = av_currentDiffusionData['rotated_righthanded', 'var'].values

    plt.figure()
    plt.plot(x , y, '+-')

    plt.xlabel('Average $1/r_i - 1/r$')
    plt.ylabel('Var(d$\\theta)$')
    plt.title('Variance of d$\\theta$ vs. Normalized Radius')

    ########################
    ### Number of Points ###
    ########################

    x = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    y = av_currentDiffusionData['rotated_righthanded', 'len'].values

    plt.figure()
    plt.plot(x , y, '+-')

    plt.xlabel('Average $1/r_i - 1/r$')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples vs. scaled x-axis (Diffusion)')

########### Creating Models #############

def make_model_constantRo(current_group, av_currentChiralityData, av_currentDiffusionData):

    ######################
    ### Velocity Piece ###
    ######################

    ro = pymc.Uniform('ro', lower=0, upper=10.*1000., value=3*1000.)
    vpar = pymc.Uniform('vpar', lower=10.**-6, upper=1, value=1.*10**-3)
    t = current_group['timeDelta'].values

    @pymc.deterministic
    def modeled_R(ro=ro, vpar=vpar, t=t):
        return ro + vpar*t

    R = pymc.TruncatedNormal('R', mu = modeled_R, tau=1.0/(0.1*1000)**2, a=0, \
                             value=current_group['colony_radius_um'].values, observed=True)

    #######################
    ### Chirality Piece ###
    #######################

    log_r_ri = av_currentChiralityData['log_r_div_ri', 'mean'].values

    vperp = pymc.Normal('vperp', mu=0, tau=1./(0.1)**2, value=0)

    @pymc.deterministic
    def modeled_dtheta(vperp=vperp, vpar=vpar, log_r_ri = log_r_ri):
        return (vperp/vpar) * log_r_ri

    dthetaDataChir = av_currentChiralityData['rotated_righthanded', 'mean'].values
    dthetaStdChir = av_currentChiralityData['rotated_righthanded', 'std'].values
    dthetaTauChir = 1.0/dthetaStdChir**2

    dtheta = pymc.Normal('dtheta', mu = modeled_dtheta, tau=dthetaTauChir, value=dthetaDataChir, observed=True)

    #######################
    ### Diffusion Piece ###
    #######################

    dif_xaxis = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    dtheta_variance = av_currentDiffusionData['rotated_righthanded', 'var'].values

    # Estimating the error of the variance
    numSamples = av_currentDiffusionData['rotated_righthanded', 'len'].values
    #dtheta_variance_error = np.sqrt(2*np.sqrt(dtheta_variance)**4/(numSamples - 1))

    ds = pymc.Uniform('ds', lower=10.**-2, upper=10**2, value=1.)

    @pymc.deterministic
    def modeled_variance(vpar=vpar, ds=ds, dif_xaxis=dif_xaxis):
        return (2*ds/vpar)*dif_xaxis

    dtheta_variance_std = pymc.Uniform('dtheta_variance_error', lower=10.**-6, upper=1, value=10.**-3.)

    var_dtheta = pymc.TruncatedNormal('var_dtheta', mu=modeled_variance, tau=1./dtheta_variance_std**2, a=0, \
                             value=dtheta_variance, observed=True)

    #######################
    ### Returning Model ###
    #######################

    return locals()

class chirality_pymc_model:
    def __init__(self, group_on_name, group_on_value, **kwargs):
        '''Keyword arguments:
        lenToFilterChir: Below what number of samples a bin should be thrown out for chirality
        lenToFilterDiff: Below what number of samples a bin should be thrown out for diffusion
        numChirBins: Number of chirality bins
        numDiffBins: Number of diffusion bins
        '''
        self.group_on_name = group_on_name
        self.group_on_value = group_on_value
        self.currentChiralityData, self.currentGrowth, self.av_chir, self.av_diff = setup_analysis(group_on_name, group_on_value, **kwargs)
        self.model = make_model_constantRo(self.currentGrowth, self.av_chir, self.av_diff)
        self.M = pymc.MCMC(self.model, db='pickle', dbname=group_on_name + '_' + str(group_on_value)+'.pkl')
        self.N = pymc.MAP(self.model)
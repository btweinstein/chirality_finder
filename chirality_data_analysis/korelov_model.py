__author__ = 'bryan'

import pymc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy as pat
import statsmodels.api as sm

import data_analysis


growthData = None
chiralityData = None

########### Setting up the Analysis ###############

def getNumUnique(x):
        return len(np.unique(x))

# A list of everything you apply to the data
aggList = [np.mean, np.std, np.var, len]

def setup_analysis(group_on_name, group_on_value, lenToFilterChir = 0, lenToFilterDiff=0, numChirBins=50, numDiffBins=50, **kwargs):
    '''The following inputs MUST be in your pandas matrices or else everything will crash.

    Chirality Data:
    1. 'rotated_righthanded': \Delta \theta in a right handed coordinate system
    2. 'log_r_div_ri': The dimensionless radius desired
    3. '1divri_minus_1divr_1divum' : A different scaled radius
    4. 'plateID': All images taken from the same plate should have the same integer ID
    5. Column that you will separate on, i.e. group_on_name

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

def binChiralityData(currentChiralityData, numChirBins, bin_on, lenToFilter = 0, verbose=True):
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

    # The sector data must now be corrected; the mean dx and dy should be used to recalculate all other quantities
    sectorData = data_analysis.recalculate_by_mean_position(sectorData)

    av_currentChiralityData = sectorData.groupby(['bins']).agg(aggList)

    # The key here is to filter out the pieces that have too few elements
    av_currentChiralityData = av_currentChiralityData[av_currentChiralityData['rotated_righthanded', 'len'] > lenToFilter]

    av_currentChiralityData = av_currentChiralityData.sort([(bin_on, 'mean')])

    ## At this point we should actually recalculate all positions/theta based on the mean position, i.e. dx and dy

    return av_currentChiralityData

def plot_av_chirality(av_currentChiralityData):
    x = av_currentChiralityData['log_r_div_ri', 'mean'].values
    y = av_currentChiralityData['rotated_righthanded', 'mean'].values
    yerr = av_currentChiralityData['rotated_righthanded', 'std'].values

    plt.figure()
    plt.plot(x , y, 'o-')
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
    plt.plot(x , y, 'o-')

    plt.xlabel('Average $\ln{(r/r_i)}$')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples vs. scaled x-axis (Chirality)')

def plot_diffusion(av_currentDiffusionData):

    x = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    y = av_currentDiffusionData['rotated_righthanded', 'var'].values

    plt.figure()
    plt.plot(x , y, 'o-')

    plt.xlabel('Average $1/r_i - 1/r$')
    plt.ylabel('Var(d$\\theta)$')
    plt.title('Variance of d$\\theta$ vs. Normalized Radius')

    ########################
    ### Number of Points ###
    ########################

    x = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    y = av_currentDiffusionData['rotated_righthanded', 'len'].values

    plt.figure()
    plt.plot(x , y, 'o-')

    plt.xlabel('Average $1/r_i - 1/r$')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples vs. scaled x-axis (Diffusion)')

########### Creating Models #############

def create_scale_invariant(name, lower = 10.**-20, upper=1, value = 10.**-5):

    @pymc.stochastic(name=name)
    def scale_invariant(value=value, lower=lower, upper=upper):

        def logp(value=value, lower=lower, upper=upper):
            if (lower < value) and (value < upper):
                return -np.log(np.log(upper/lower))
            else:
                return -np.inf

        def random(lower=lower, upper=upper):
            u = np.random.rand()
            return lower * (upper/lower)**u

    return scale_invariant

def create_weighted_potential(name, length):

    draws = np.random.rand(length)
    draws.sort()
    gaps = draws[1:] - draws[:-1]
    gaps = np.insert(gaps, 0, draws[0])
    gaps = np.append(gaps, 1-draws[-1])

    @pymc.stochastic(name=name + '_weight_stoch')
    def random_weights(value=gaps):
        numToDraw = len(gaps)

        def logp(value=value):
            return 0

        def random():
            draws = np.random.rand(numToDraw)
            draws.sort()
            gaps = draws[1:] - draws[:-1]
            gaps = np.insert(gaps, 0, draws[0])
            gaps = np.append(gaps, 1-draws[-1])

            return gaps

    w = random_weights

    @pymc.potential(name=name + '_weight_potential')
    def bootstrap(w=w):
        # Weigh the probability distribution of R randomly
        return np.sum(np.log(w))

    return bootstrap

def make_model_constantRo(current_group, av_currentChiralityData, av_currentDiffusionData, bootstrap=False, **kwargs):
    # Here we vastly improve uor old Bayesian analysis based on the fits. This way we
    # can compare, in a fair way, the frequentist bootstrapping vs. the bayesian methodology.

    ######################
    ### Velocity Piece ###
    ######################

    vpar = create_scale_invariant('vpar', lower=10.**-5, upper=1, value=1.*10**-3)
    t = current_group['timeDelta'].values

    @pymc.deterministic
    def modeled_R(vpar=vpar, t=t):
        return vpar*t

    R = pymc.TruncatedNormal('R', mu = modeled_R, tau=1.0/(0.1*1000)**2, a=0, \
                             value=current_group['deltaR'].values, observed=True)

    if bootstrap: # Incorporate soft potential
        r_weights = create_weighted_potential('R', len(R.value))

    #######################
    ### Chirality Piece ###
    #######################

    log_r_ri = av_currentChiralityData['log_r_div_ri', 'mean'].values

    vperp = create_scale_invariant('vperp', lower=10.**-10, upper=1, value=1.*10.**-3)

    dthetaDataChir = av_currentChiralityData['rotated_righthanded', 'mean'].values
    dthetaStdChir = av_currentChiralityData['rotated_righthanded', 'std'].values
    dthetaTauChir = 1.0/dthetaStdChir**2

    # Drop the 0 dtheta piece with infinite accuracy
    points = np.isfinite(dthetaTauChir)

    dthetaDataChir = dthetaDataChir[points]
    log_r_ri = log_r_ri[points]
    dthetaTauChir = dthetaTauChir[points]

    @pymc.deterministic
    def modeled_dtheta(vperp=vperp, vpar=vpar, log_r_ri = log_r_ri):
        return (vperp/vpar) * log_r_ri

    dtheta = pymc.Normal('dtheta', mu = modeled_dtheta, tau=dthetaTauChir, value=dthetaDataChir, observed=True)

    if bootstrap: # Incorporate soft potential
        dtheta_weights = create_weighted_potential('dtheta', len(dtheta.value))

    #######################
    ### Diffusion Piece ###
    #######################

    dif_xaxis = av_currentDiffusionData['1divri_minus_1divr_1divum', 'mean'].values
    dtheta_variance = av_currentDiffusionData['rotated_righthanded', 'var'].values

    # Estimating the error of the variance
    numSamples = av_currentDiffusionData['rotated_righthanded', 'len'].values
    dtheta_variance_error = np.sqrt(2*np.sqrt(dtheta_variance)**4/(numSamples - 1))
    dtheta_variance_tau = 1./dtheta_variance_error**2

    # Drop the infinite variance piece
    points = np.isfinite(dtheta_variance_tau)

    dtheta_variance = dtheta_variance[points]
    dif_xaxis = dif_xaxis[points]
    dtheta_variance_tau = dtheta_variance_tau[points]

    ds = create_scale_invariant('ds', lower=10.**-2, upper=10**2, value=1.)

    @pymc.deterministic
    def modeled_variance(vpar=vpar, ds=ds, dif_xaxis=dif_xaxis):
        return (2*ds/vpar)*dif_xaxis

    var_dtheta = pymc.TruncatedNormal('var_dtheta', mu=modeled_variance, tau=dtheta_variance_tau, a=0, \
                             value=dtheta_variance, observed=True)

    if bootstrap: # Incorporate soft potential
        var_weights = create_weighted_potential('var_dtheta', len(var_dtheta.value))

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
        self.model = make_model_constantRo(self.currentGrowth, self.av_chir, self.av_diff, **kwargs)
        self.M = pymc.MCMC(self.model, db='pickle', dbname=group_on_name + '_' + str(group_on_value)+'.pkl')
        self.N = pymc.MAP(self.model)

        self.Ro_results_freq= None
        self.dtheta_results_freq = None
        self.variance_results_freq = None

    def frequentist_fit(self, currentGrowth = None, av_chir = None, av_diff = None):
        if currentGrowth is None:
            currentGrowth = self.currentGrowth
        if av_chir is None:
            av_chir = self.av_chir
        if av_diff is None:
            av_diff = self.av_diff

        ##########################
        ##### Fit vparallel ######
        ##########################

        y, X = pat.dmatrices('colony_radius_um ~ timeDelta', data=currentGrowth,
                             return_type='dataframe')

        radiusSigma = (200.*10.**3) * np.ones_like(y)
        Ro_model = sm.WLS(y, X, weights=1/radiusSigma**2)
        self.Ro_results_freq = Ro_model.fit()

        ###########################
        ###  Fit dtheta ##########
        ##########################
        log_r_ri = av_chir['log_r_div_ri', 'mean'].values

        dthetaDataChir = av_chir['rotated_righthanded', 'mean'].values
        dthetaStdChir = av_chir['rotated_righthanded', 'std'].values
        dthetaTauChir = 1.0/dthetaStdChir**2

        # The first point error is technically 0 right now...so how do we include it?
        # Should we include it? We should include it so the model recognizes that it is
        # a real point. Actually, let's just drop it, it is probably easier.

        # If any point with tau=infty is here, we must kill it.

        points = np.isfinite(dthetaTauChir)

        dthetaDataChir = dthetaDataChir[points]
        log_r_ri = log_r_ri[points]
        dthetaTauChir = dthetaTauChir[points]

        # No intercept!
        y, X = pat.dmatrices('dtheta ~ 0 + log_r_ri', data={'dtheta' : dthetaDataChir,
                                                              'log_r_ri' : log_r_ri},
                             return_type='dataframe')

        dtheta_model = sm.WLS(y, X, weights=dthetaTauChir)
        self.dtheta_results_freq = dtheta_model.fit()

        ###################
        ### Fit Variance ##
        ###################

        dif_xaxis = av_diff['1divri_minus_1divr_1divum', 'mean'].values
        dtheta_variance = av_diff['rotated_righthanded', 'var'].values

        # Estimating the error of the variance
        numSamples = av_diff['rotated_righthanded', 'len'].values
        dtheta_variance_error = np.sqrt((2*dtheta_variance**2)/(numSamples - 1))
        dtheta_variance_tau = 1/dtheta_variance_error**2

        # Drop the first point as it has zero variance with infinite accuracy. It is
        # essentially pathological.


        points = np.isfinite(dtheta_variance_tau)

        dtheta_variance = dtheta_variance[points]
        dif_xaxis = dif_xaxis[points]
        dtheta_variance_tau = dtheta_variance_tau[points]

        y, X = pat.dmatrices('dtheta_variance ~ 0 + dif_xaxis', data={'dtheta_variance' : dtheta_variance,
                                                                      'dif_xaxis' : dif_xaxis},
                             return_type='dataframe')

        variance_model = sm.WLS(y, X, weights=dtheta_variance_tau)
        self.variance_results_freq = variance_model.fit()

    def get_params_frequentist(self):
        # Useful in bootstrapping estimate of error
        vpar = self.Ro_results_freq.params['timeDelta']
        vperp_div_vpar = self.dtheta_results_freq.params['log_r_ri']
        two_Ds_vpar = self.variance_results_freq.params['dif_xaxis']

        vperp = vperp_div_vpar * vpar
        Ds = (1./2.) * vpar * two_Ds_vpar

        return {'vpar' : vpar, 'vperp' : vperp, 'Ds' : Ds}

    @staticmethod
    def resample_df(df):
        rows = np.random.choice(df.index, len(df.index))
        return df.ix[rows]

    def frequentist_bootstrap(self, numIterations, plot=False):
        # This is a little silly as we calculate the error AFTER we have selected the sectors
        # and averaged them. We probably want to bootstrap SELECTING the sectors. Right?
        # It is worth figuring out regardless, assuming it does not take too long.

        vpar = np.empty(numIterations)
        vperp = np.empty(numIterations)
        Ds = np.empty(numIterations)

        for i in range(numIterations):
            # Sample from the data appropriately
            currentGrowth = self.resample_df(self.currentGrowth)
            if plot: plt.plot(currentGrowth['timeDelta'], currentGrowth['colony_radius_um'], 'o')

            if plot: plt.figure()
            av_chir = self.resample_df(self.av_chir)
            if plot: plt.plot(av_chir['log_r_div_ri', 'mean'], av_chir['rotated_righthanded', 'mean'], 'o')

            if plot: plt.figure()
            av_diff = self.resample_df(self.av_diff)
            if plot: plt.plot(av_diff['1divri_minus_1divr_1divum', 'mean'], av_diff['rotated_righthanded', 'var'], 'o')

            self.frequentist_fit(currentGrowth=currentGrowth, av_chir= av_chir, av_diff=av_diff)
            params = self.get_params_frequentist()

            vpar[i] = params['vpar']
            vperp[i] = params['vperp']
            Ds[i] = params['Ds']

        return pd.DataFrame({'vpar' : vpar, 'vperp' : vperp, 'Ds' : Ds})
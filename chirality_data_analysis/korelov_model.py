__author__ = 'bryan'

import pymc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy as pat
import statsmodels.api as sm

import data_analysis


### Utility Functions

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

def resample_df(df):
    rows = np.random.choice(df.index, len(df.index))
    return df.ix[rows]


### Main Code

class chirality_model:
    def __init__(self, growthData, chiralityData, group_on_name, group_on_value, **kwargs):

        '''Keyword arguments:
        lenToFilterChir: Below what number of samples a bin should be thrown out for chirality
        lenToFilterDiff: Below what number of samples a bin should be thrown out for diffusion
        numChirBins: Number of chirality bins
        numDiffBins: Number of diffusion bins
        verbose: Whether or not to be verbose
        '''

        # Define attributes
        self.growthData = growthData
        self.chiralityData = chiralityData
        self.group_on_name = group_on_name
        self.group_on_value = group_on_value
        self.kwargs = kwargs
        # A list of everything you apply to the data
        self.aggList = [np.mean, np.std, np.var, len]

        # Binned Data
        self.cur_growth = None
        self.cur_chir = None
        self.av_chir = None
        self.av_diff = None

        # Pymc Variables
        self.pymc_model = None

        # Frequentist Variables
        self.Ro_results_freq= None
        self.dtheta_results_freq = None
        self.variance_results_freq = None


        # Set up the system
        self.rebin_all_data(**self.kwargs)

        # Set up the pymc model
        self.remake_pymc_model(**self.kwargs)
        self.M = pymc.MCMC(self.pymc_model, db='pickle', dbname=self.group_on_name + '_' + str(self.group_on_value)+'.pkl')
        self.N = pymc.MAP(self.pymc_model)

    def rebin_all_data(self, lenToFilterChir = 0, lenToFilterDiff=0, numChirBins=50, numDiffBins=50, verbose=True, **kwargs):
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

        if verbose: print 'Setting up the simulation for' , self.group_on_name , '=' , self.group_on_value , '...'

        # Get the desired data
        plate_groups = self.growthData.groupby([self.group_on_name])

        self.cur_growth = plate_groups.get_group(self.group_on_value)
        self.cur_chir = self.chiralityData[(self.chiralityData[self.group_on_name] == self.group_on_value)]

        self.av_chir = self.binChiralityData(numChirBins, 'log_r_div_ri', lenToFilter = lenToFilterChir,
                                             verbose=verbose, **kwargs)
        if verbose: self.plot_av_chirality(**self.kwargs)

        self.av_diff = self.binChiralityData(numDiffBins, '1divri_minus_1divr_1divum', lenToFilter = lenToFilterDiff,
                                             verbose=verbose, **kwargs)
        if verbose: self.plot_av_diffusion(**kwargs)

        if verbose: print 'Done setting up!'

        if verbose: plt.show(**kwargs)

    def binChiralityData(self, numChirBins, bin_on, lenToFilter = 0, verbose=True, **kwargs):
        '''Returns the average data per sector in each plate binned over the desired radius.'''

        min_x = self.cur_chir[bin_on].min() - 10.**-6.
        max_x = self.cur_chir[bin_on].max() + 10.**-6.
        numBins = numChirBins

        if verbose:
            print 'numBins for ', bin_on, ':' , numBins
            print 'Min ', bin_on, ':', min_x
            print 'Max ', bin_on, ':', max_x

        bins, binsize = np.linspace(min_x, max_x, numBins, retstep=True)

        if verbose: print 'Dimensionless Binsize for ', bin_on, ':' , binsize

        # Group by sector! Specified by a label and a plate
        sector_groups = self.cur_chir.groupby([pd.cut(self.cur_chir[bin_on], bins), 'plateID', 'label'])
        sectorData = sector_groups.agg(np.mean)
        sectorData = sectorData.rename(columns={bin_on : bin_on + '_mean'})
        sectorData = sectorData.reset_index()
        sectorData = sectorData.rename(columns={bin_on : 'bins',
                                               bin_on + '_mean' : bin_on})

        # The sector data must now be corrected; the mean dx and dy should be used to recalculate all other quantities
        sectorData = data_analysis.recalculate_by_mean_position(sectorData)

        av_currentChiralityData = sectorData.groupby(['bins']).agg(self.aggList)

        # The key here is to filter out the pieces that have too few elements
        av_currentChiralityData = av_currentChiralityData[av_currentChiralityData['rotated_righthanded', 'len'] > lenToFilter]

        av_currentChiralityData = av_currentChiralityData.sort([(bin_on, 'mean')])

        ## At this point we should actually recalculate all positions/theta based on the mean position, i.e. dx and dy

        return av_currentChiralityData

    def plot_av_chirality(self, **kwargs):
        x = self.av_chir['log_r_div_ri', 'mean'].values
        y = self.av_chir['rotated_righthanded', 'mean'].values
        yerr = self.av_chir['rotated_righthanded', 'std'].values

        plt.figure()
        plt.plot(x , y, 'o-')
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, antialiased=True)
        plt.xlabel('Average $\ln{(r/r_i)}$')
        plt.ylabel('Average d$\\theta$')
        plt.title('Average d$\\theta$ vs. Normalized Radius')

        #######################
        ## Number of Points ###
        #######################

        x = self.av_chir['log_r_div_ri', 'mean'].values
        y = self.av_chir['rotated_righthanded', 'len'].values

        plt.figure()
        plt.plot(x , y, 'o-')

        plt.xlabel('Average $\ln{(r/r_i)}$')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples vs. scaled x-axis (Chirality)')

    def plot_av_diffusion(self, **kwargs):

        x = self.av_diff['1divri_minus_1divr_1divum', 'mean'].values
        y = self.av_diff['rotated_righthanded', 'var'].values

        plt.figure()
        plt.plot(x , y, 'o-')

        plt.xlabel('Average $1/r_i - 1/r$')
        plt.ylabel('Var(d$\\theta)$')
        plt.title('Variance of d$\\theta$ vs. Normalized Radius')

        ########################
        ### Number of Points ###
        ########################

        x = self.av_diff['1divri_minus_1divr_1divum', 'mean'].values
        y = self.av_diff['rotated_righthanded', 'len'].values

        plt.figure()
        plt.plot(x , y, 'o-')

        plt.xlabel('Average $1/r_i - 1/r$')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples vs. scaled x-axis (Diffusion)')

    ########### Creating Models #############

    def remake_pymc_model(self, **kwargs):
        # Here we vastly improve uor old Bayesian analysis based on the fits. This way we
        # can compare, in a fair way, the frequentist bootstrapping vs. the bayesian methodology.

        ######################
        ### Velocity Piece ###
        ######################

        vpar = create_scale_invariant('vpar', lower=10.**-5, upper=1, value=1.*10**-3)
        t = self.cur_growth['timeDelta'].values

        @pymc.deterministic
        def modeled_R(vpar=vpar, t=t):
            return vpar*t

        R = pymc.TruncatedNormal('R', mu = modeled_R, tau=1.0/(0.1*1000)**2, a=0, \
                                 value=self.cur_growth['deltaR'].values, observed=True)

        #######################
        ### Chirality Piece ###
        #######################

        log_r_ri = self.av_chir['log_r_div_ri', 'mean'].values

        vperp = pymc.Normal('vperp', mu=0, tau=1./(1.**2), value=1.*10.**-3)

        dthetaDataChir = self.av_chir['rotated_righthanded', 'mean'].values
        dthetaStdChir = self.av_chir['rotated_righthanded', 'std'].values
        dthetaTauChir = 1.0/dthetaStdChir**2

        # Drop the 0 dtheta piece with infinite accuracy; already accounted for in the model
        points = np.isfinite(dthetaTauChir)

        dthetaDataChir = dthetaDataChir[points]
        log_r_ri = log_r_ri[points]
        dthetaTauChir = dthetaTauChir[points]

        @pymc.deterministic
        def modeled_dtheta(vperp=vperp, vpar=vpar, log_r_ri = log_r_ri):
            return (vperp/vpar) * log_r_ri

        dtheta = pymc.Normal('dtheta', mu = modeled_dtheta, tau=dthetaTauChir, value=dthetaDataChir, observed=True)

        #######################
        ### Diffusion Piece ###
        #######################

        dif_xaxis = self.av_diff['1divri_minus_1divr_1divum', 'mean'].values
        dtheta_variance = self.av_diff['rotated_righthanded', 'var'].values

        # Estimating the error of the variance
        numSamples = self.av_diff['rotated_righthanded', 'len'].values
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

        #######################
        ### Returning Model ###
        #######################

        self.pymc_model = locals()


    def frequentist_fit(self, currentGrowth = None, av_chir = None, av_diff = None):
        if currentGrowth is None:
            currentGrowth = self.cur_growth
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

    def frequentist_linearfit_bootstrap(self, numIterations, plot=False):
        # This is a little silly as we calculate the error AFTER we have selected the sectors
        # and averaged them. We probably want to bootstrap SELECTING the sectors. Right?
        # It is worth figuring out regardless, assuming it does not take too long.

        vpar = np.empty(numIterations)
        vperp = np.empty(numIterations)
        Ds = np.empty(numIterations)

        for i in range(numIterations):
            # Sample from the data appropriately
            currentGrowth = resample_df(self.cur_growth)
            if plot: plt.plot(currentGrowth['timeDelta'], currentGrowth['colony_radius_um'], 'o')

            if plot: plt.figure()
            av_chir = resample_df(self.av_chir)
            if plot: plt.plot(av_chir['log_r_div_ri', 'mean'], av_chir['rotated_righthanded', 'mean'], 'o')

            if plot: plt.figure()
            av_diff = resample_df(self.av_diff)
            if plot: plt.plot(av_diff['1divri_minus_1divr_1divum', 'mean'], av_diff['rotated_righthanded', 'var'], 'o')

            self.frequentist_fit(currentGrowth=currentGrowth, av_chir= av_chir, av_diff=av_diff)
            params = self.get_params_frequentist()

            vpar[i] = params['vpar']
            vperp[i] = params['vperp']
            Ds[i] = params['Ds']

        return pd.DataFrame({'vpar' : vpar, 'vperp' : vperp, 'Ds' : Ds})
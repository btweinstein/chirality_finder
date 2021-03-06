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

def weighted_info(x):
    weights = x['binning_weights']
    x = x.drop('bins', axis=1)
    average = x.apply(np.average, weights=weights)
    variance = (x-average)**2.
    variance = variance.apply(np.average, weights=weights)
    std = variance.apply(np.sqrt)
    length = x.count()

    return pd.concat({'mean' : average, 'var' : variance, 'std' : std, 'len' : length})



### Main Code

class chirality_model:
    def __init__(self, growthData, chiralityData, group_on_name, group_on_value, bootstrap=False, **kwargs):

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

        # Current Data
        self.cur_growth = None
        self.cur_chir = None
        # One point per sector per bin
        self.cur_chir_sectors = None
        self.cur_diff_sectors = None
        # Binned Data
        self.av_chir = None
        self.av_diff = None

        self.cur_chir_bins = None
        self.cur_diff_bins = None

        # Pymc Variables
        self.pymc_model = None

        # Frequentist Variables
        self.Ro_results_freq= None
        self.dtheta_results_freq = None
        self.variance_results_freq = None

        # Get subset of data
        plate_groups = self.growthData.groupby([self.group_on_name])
        self.cur_growth = plate_groups.get_group(self.group_on_value)
        self.cur_chir = self.chiralityData[(self.chiralityData[self.group_on_name] == self.group_on_value)]
        # Set equal weights for the binning; changes for the bootstrap
        edge_groups = self.cur_chir.groupby(['plateID', 'label'])
        self.cur_chir['binning_weights'] = 1./len(edge_groups)

        label_count = 0
        df = pd.DataFrame()
        for n, g in edge_groups:
            g['edge_id'] = label_count
            df = df.append(g)
            label_count += 1
        self.cur_chir = df

        # Set up the system
        self.rebin_all_data(**self.kwargs)

        # Set up the pymc model
        if bootstrap:
            self.remake_pymc_bootstrap(**self.kwargs)
            dbname = self.group_on_name + '_' + str(self.group_on_value)+'_bootstrap.pkl'
        else:
            self.remake_pymc_model(**self.kwargs)
            dbname = self.group_on_name + '_' + str(self.group_on_value)+'.pkl'
        self.M = pymc.MCMC(self.pymc_model, db='pickle', dbname=dbname)
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

        self.cur_chir_bins, self.cur_chir_sectors, self.av_chir = \
            self.binChiralityData(numChirBins, 'log_r_div_ri', lenToFilter = lenToFilterChir,
                                             verbose=verbose, **kwargs)
        if verbose: self.plot_av_chirality(**self.kwargs)

        self.cur_diff_bins, self.cur_diff_sectors, self.av_diff = \
            self.binChiralityData(numDiffBins, '1divri_minus_1divr_1divum', lenToFilter = lenToFilterDiff,
                                             verbose=verbose, **kwargs)
        if verbose: self.plot_av_diffusion(**kwargs)

        if verbose: print 'Done setting up!'

        if verbose: plt.show(**kwargs)

    def rebin_chir_sectors(self):
        self.av_chir = self.cur_chir_sectors.groupby('bins').apply(weighted_info)
        self.av_chir = self.av_chir.swaplevel(1, 0, axis=1)
        self.av_chir.sort([('log_r_div_ri', 'mean')], inplace=True)

    def rebin_diff_sectors(self):
        self.av_diff = self.cur_diff_sectors.groupby('bins').apply(weighted_info)
        self.av_diff = self.av_diff.swaplevel(1,0, axis=1)
        self.av_diff.sort([('1divri_minus_1divr_1divum', 'mean')], inplace=True)

    def rebin_sectors(self):
        '''Sectors are already filtered by length. You just have to do a weighted mean here.'''

        self.av_chir = self.cur_chir_sectors.groupby('bins').apply(weighted_info)
        self.av_chir = self.av_chir.swaplevel(1, 0, axis=1)
        self.av_chir.sort([('log_r_div_ri', 'mean')], inplace=True)

        self.av_diff = self.cur_diff_sectors.groupby('bins').apply(weighted_info)
        self.av_diff = self.av_diff.swaplevel(1,0, axis=1)
        self.av_diff.sort([('1divri_minus_1divr_1divum', 'mean')], inplace=True)

    def binChiralityData(self, numBins, bin_on, lenToFilter = 0, verbose=True, bins=None, **kwargs):
        '''Returns the average data per sector in each plate binned over the desired radius.'''

        if bins is None:
            min_x = self.cur_chir[bin_on].min()
            max_x = self.cur_chir[bin_on].max()
            bins, binsize = np.linspace(min_x, max_x, numBins, retstep=True)

            if verbose:
                print 'numBins for ', bin_on, ':' , numBins
                print 'Min ', bin_on, ':', min_x
                print 'Max ', bin_on, ':', max_x
                print 'Dimensionless Binsize for ', bin_on, ':' , binsize

        # TAKE MEAN OF ALL EDGES. DO NOT BOOTSTRAP HERE

        sector_groups = self.cur_chir.groupby([pd.cut(self.cur_chir[bin_on], bins), 'plateID', 'label'])
        sectorData = sector_groups.agg(np.mean)

        sectorData = sectorData.rename(columns={bin_on : bin_on + '_mean'})
        sectorData = sectorData.reset_index()
        sectorData = sectorData.rename(columns={bin_on : 'bins',
                                               bin_on + '_mean' : bin_on})

        ### BOOTSTRAPPING BASED ON AVERAGE SECTORS #####

        # The sector data must now be corrected; the mean dx and dy should be used to recalculate all other quantities
        sectorData = data_analysis.recalculate_by_mean_position(sectorData)

        # Filter out sectordata that does not have enough points
        num_bins_df = sectorData.groupby('bins').agg(len)
        num_bins_df = num_bins_df.reset_index()
        # Get good bins
        goodBins = num_bins_df['bins'][num_bins_df['rotated_righthanded'] > lenToFilter]
        # Only keep the good bins
        sectorData = sectorData[sectorData['bins'].isin(goodBins)]

        av_currentChiralityData = sectorData.groupby(['bins']).agg(self.aggList)
        av_currentChiralityData = av_currentChiralityData.sort([(bin_on, 'mean')])

        return bins, sectorData, av_currentChiralityData

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

    def remake_pymc_bootstrap(self, **kwargs):

        # TODO: This is currently all wrong as you need to assign each sector the SAME weight regardless of binning...

        model_dict = {}
        ######################
        ### Velocity Piece ###
        ######################

        vpar = create_scale_invariant('vpar', lower=10.**-5, upper=1, value=1.*10**-3)
        model_dict['vpar'] = vpar

        plate_group = self.cur_growth.groupby('plateID')
        num_plates = len(plate_group)
        # Create a dirilect distribution to add fluctuations
        vpar_dir = pymc.Dirichlet('vpar_dir', np.ones(num_plates), trace=False)
        model_dict['vpar_dir'] = vpar_dir
        vpar_weights = pymc.CompletedDirichlet('vpar_weights', vpar_dir, trace=True)
        model_dict['vpar_weights'] = vpar_weights

        count = 0
        for n, g in plate_group:
            # Get all data corresponding to the plate group

            t = g['timeDelta'].values

            @pymc.deterministic(trace=False)
            def modeled_R(vpar=vpar, t=t):
                return vpar*t

            rname = 'R_' + str(int(n))
            R = pymc.TruncatedNormal(rname, mu = modeled_R, tau=1.0/(200)**2, a=0, \
                                     value=g['deltaR'].values, observed=True)
            model_dict[rname] = R

            # Make a potential to account for the weighting
            potential_name = 'R_weight_' + str(int(n))
            @pymc.potential(name=potential_name)
            def r_weight(weight = vpar_weights[0, count], R=R):
                return np.log(weight)

            model_dict[potential_name] = r_weight

            count += 1

        #######################
        ### Weighting Sectors #
        #######################

        # Create a Dirilecht distribution to deal with this
        num_edges = len(self.cur_chir_sectors.groupby(['plateID','label']))

        chir_dir = pymc.Dirichlet('chir_dir', np.ones(num_edges), trace=False)
        model_dict['chir_dir'] = chir_dir
        chir_weights = pymc.CompletedDirichlet('chir_weights', chir_dir, trace=True)
        model_dict['chir_weights'] = chir_weights
        # When calculating averages, use these weights!

        #####################
        ### Chirality Data ##
        #####################

        # We must dynamically bin this data, unfortunately :(

        @pymc.deterministic(dtype=float)
        def rebin_cur_chir_sectors(sector_weights = chir_weights[0]):
            group = self.cur_chir_sectors.groupby(['plateID', 'label'])
            count = 0
            df = pd.DataFrame()
            for n, g in group:
                g['binning_weights'] = sector_weights[count]
                df = df.append(g)
                count += 1
            self.cur_chir_sectors = df
            self.rebin_chir_sectors()

            # Calculate everything you need to

            log_r_ri = self.av_chir['log_r_div_ri', 'mean'].values
            dthetaDataChir = self.av_chir['rotated_righthanded', 'mean'].values
            dthetaStdChir = self.av_chir['rotated_righthanded', 'std'].values
            dthetaTauChir = 1.0/dthetaStdChir**2

            # Drop the 0 dtheta piece with infinite accuracy; already accounted for in the model
            points = np.isfinite(dthetaTauChir)

            log_r_ri = log_r_ri[points]
            dthetaDataChir = dthetaDataChir[points]
            dthetaTauChir = dthetaTauChir[points]

            return np.array([log_r_ri, dthetaDataChir, dthetaTauChir])

        vperp = pymc.Normal('vperp', mu=0, tau=1./(1.**2), value=1.*10.**-3)
        model_dict['vperp'] = vperp

        @pymc.deterministic
        def modeled_dtheta(vperp=vperp, vpar=vpar, log_r_ri = rebin_cur_chir_sectors[0]):
            return (vperp/vpar) * log_r_ri

        # Everything is specified here...so we now just have a potential, essentially,
        # as a stochastic is just confusing
        @pymc.potential()
        def dtheta(mu=modeled_dtheta, tau=rebin_cur_chir_sectors[2],
                   x=rebin_cur_chir_sectors[1]):
            return pymc.normal_like(x, mu, tau)

        model_dict['dtheta'] = dtheta

        ############################
        # Weighting Diffusion Data #
        ############################
        # Create a Dirilecht distribution to deal with this
        num_edges = len(self.cur_diff_sectors.groupby(['plateID','label']))

        diff_dir = pymc.Dirichlet('diff_dir', np.ones(num_edges), trace=False)
        model_dict['diff_dir'] = diff_dir
        diff_weights = pymc.CompletedDirichlet('diff_weights', diff_dir, trace=True)
        model_dict['diff_weights'] = diff_weights

        #############################
        # Modeling Diffusion Data ###
        #############################

        @pymc.deterministic(dtype=np.float)
        def rebin_cur_diff_sectors(sector_weights = diff_weights[0]):
            group = self.cur_diff_sectors.groupby(['plateID', 'label'])
            count = 0
            df = pd.DataFrame()
            for n, g in group:
                g['binning_weights'] = sector_weights[count]
                df = df.append(g)
                count += 1
            self.cur_diff_sectors = df
            self.rebin_diff_sectors()

            # Now calculate everything that you need to

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

            return np.array([dif_xaxis, dtheta_variance, dtheta_variance_tau])

        ds = create_scale_invariant('ds', lower=10.**-2, upper=10**2, value=1.)

        model_dict['ds'] = ds

        @pymc.deterministic
        def modeled_variance(vpar=vpar, ds=ds, dif_xaxis=rebin_cur_diff_sectors[0]):
            return (2*ds/vpar)*dif_xaxis

        @pymc.potential
        def var_dtheta(mu = modeled_variance, x=rebin_cur_diff_sectors[1],
                       tau = rebin_cur_diff_sectors[2]):
            return pymc.truncated_normal_like(x=x, mu=mu, tau=tau, a=0)

        model_dict['var_dtheta'] = var_dtheta

        ###### Done! #######
        self.pymc_model = model_dict

    def remake_pymc_model(self, **kwargs):
        # Here we vastly improve uor old Bayesian analysis based on the fits. This way we
        # can compare, in a fair way, the frequentist bootstrapping vs. the bayesian methodology.

        ######################
        ### Velocity Piece ###
        ######################

        plates = self.cur_growth.groupby('plateID')

        vpar = pymc.Uniform('vpar', lower=10.**-4., upper=10.**0., value=1.*10**-2)
        t = self.cur_growth['timeDelta'].values

        @pymc.deterministic
        def modeled_R(vpar=vpar, t=t):
            return vpar*t

        R = pymc.TruncatedNormal('R', mu = modeled_R, tau=1.0/(100.)**2., a=0, \
                                 value=self.cur_growth['deltaR'].values, observed=True)

        #######################
        ### Chirality Piece ###
        #######################

        log_r_ri = self.av_chir['log_r_div_ri', 'mean'].values

        vperp = pymc.Uniform('vperp', lower=-1., upper=1., value=0)

        dthetaDataChir = self.av_chir['rotated_righthanded', 'mean'].values
        # We must use the standard error of the mean or this makes no sense.
        dthetaStdChir = self.av_chir['rotated_righthanded', 'std'].values \
                        / np.sqrt(self.av_chir['numSectors', 'len'].values)
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
        numSamples = self.av_diff['numSectors', 'len'].values
        dtheta_variance_error = np.sqrt(2*np.sqrt(dtheta_variance)**4/(numSamples - 1))
        dtheta_variance_tau = 1./dtheta_variance_error**2

        # Drop the infinite variance piece as it is already accounted for
        points = np.isfinite(dtheta_variance_tau)

        dtheta_variance = dtheta_variance[points]
        dif_xaxis = dif_xaxis[points]
        dtheta_variance_tau = dtheta_variance_tau[points]

        ds = pymc.Uniform('ds', lower=10.**-3., upper=10., value=1.)

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

        radiusSigma = 200 * np.ones_like(y)
        Ro_model = sm.WLS(y, X, weights=1/radiusSigma**2)
        self.Ro_results_freq = Ro_model.fit()

        ###########################
        ###  Fit dtheta ##########
        ##########################
        log_r_ri = av_chir['log_r_div_ri', 'mean'].values

        dthetaDataChir = av_chir['rotated_righthanded', 'mean'].values
        dthetaStdChir = av_chir['rotated_righthanded', 'std'].values \
                        / np.sqrt(av_chir['numSectors', 'len'].values)
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

    def get_bootstrap_sectors(self, numIterations):
        '''Uses a bootstrap sampling scheme to get the mean and variance for the
        specified bins.'''

        #TODO: Make zero bins not count toward average

        # Get each edges' ID
        unique_edges = self.cur_chir['edge_id'].unique()

        def resample_sectors(df):
            sectors = np.random.choice(unique_edges, len(unique_edges))
            return df.ix[sectors]

        original_chir = self.cur_chir
        original_av_chir = self.av_chir
        original_av_diff = self.av_diff

        # Get the bins that we will repeatedly use

        original_chir_bins = self.cur_chir_bins
        original_diff_bins = self.cur_diff_bins

        bootstrap_av_chir_df = pd.DataFrame()
        bootstrap_av_diff_df = pd.DataFrame()

        for i in range(numIterations):

            new_chir = original_chir.set_index('edge_id')
            self.cur_chir = resample_sectors(new_chir)

            self.cur_chir_bins, self.cur_chir_sectors, self.av_chir = \
                self.binChiralityData(None, 'log_r_div_ri', bins=original_chir_bins, **self.kwargs)

            self.cur_diff_bins, self.cur_diff_sectors, self.av_diff = \
                self.binChiralityData(None, '1divri_minus_1divr_1divum', bins=original_diff_bins, **self.kwargs)

            # Add to the list of data

            self.av_chir['bootstrap_iter'] = i
            self.av_diff['bootstrap_iter'] = i

            bootstrap_av_chir_df = bootstrap_av_chir_df.append(self.av_chir)
            bootstrap_av_diff_df = bootstrap_av_diff_df.append(self.av_diff)

            if np.mod(i, 50) == 0:
                print i

        self.cur_chir = original_chir
        self.av_chir = original_av_chir
        self.av_diff = original_av_diff

        return bootstrap_av_chir_df, bootstrap_av_diff_df

    def frequentist_linearfit_bootstrap(self, numIterations, plot=False):

        vpar = np.empty(numIterations)
        vperp = np.empty(numIterations)
        Ds = np.empty(numIterations)

        unique_vpar_plates = self.cur_growth['plateID'].unique()
        # Give each sector an id
        unique_edges = self.cur_chir['edge_id'].unique()

        def resample_current_growth(df):
            plates = np.random.choice(unique_vpar_plates, len(unique_vpar_plates))
            return df.ix[plates]

        def resample_sectors(df):
            sectors = np.random.choice(unique_edges, len(unique_edges))
            return df.ix[sectors]

        original_chir = self.cur_chir
        original_av_chir = self.av_chir
        original_av_diff = self.av_diff

        for i in range(numIterations):
            # Sample from the data appropriately
            currentGrowth = self.cur_growth.set_index('plateID')
            currentGrowth = resample_current_growth(currentGrowth)
            if plot: plt.plot(currentGrowth['timeDelta'], currentGrowth['colony_radius_um'], 'o')

            new_chir = original_chir.set_index('edge_id')
            self.cur_chir = resample_sectors(new_chir)

            self.rebin_all_data(**self.kwargs)
            if plot: self.plot_av_chirality()
            if plot: self.plot_av_diffusion()

            self.frequentist_fit(currentGrowth=currentGrowth)
            params = self.get_params_frequentist()
            self.cur_chir=original_chir

            vpar[i] = params['vpar']
            vperp[i] = params['vperp']
            Ds[i] = params['Ds']

            if np.mod(i, 50) == 0:
                print i

        self.av_chir = original_av_chir
        self.av_diff = original_av_diff

        return pd.DataFrame({'vpar' : vpar, 'vperp' : vperp, 'Ds' : Ds})
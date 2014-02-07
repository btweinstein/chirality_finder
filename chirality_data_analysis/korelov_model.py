__author__ = 'bryan'

import pymc
import numpy as np

def make_model(current_group, av_currentChiralityData, av_currentDiffusionData):

    ######################
    ### Velocity Piece ###
    ######################

    ro = pymc.HalfNormal('ro', tau=1./1.5**2, value=1.0)
    vpar = pymc.Uniform('vpar', lower=0, upper=1, value=10**-6)
    t = current_group['timeDelta'].values

    @pymc.deterministic
    def modeled_R(ro=ro, vpar=vpar, t=t):
        return ro + vpar*t

    R = pymc.Normal('R', mu = modeled_R, tau=1.0/0.05**2, value=current_group['colony_radius_mm'], observed=True)

    velocityVariables = [ro, vpar, t, R]

    #######################
    ### Chirality Piece ###
    #######################

    log_r_ri = av_currentChiralityData['log_r_div_ri', 'mean']

    vperp = pymc.Normal('vperp', mu=0, tau=1./(10.**-3.)**2, value=0)

    @pymc.deterministic
    def modeled_dtheta(vperp=vperp, vpar=vpar, log_r_ri = log_r_ri):
        return (vperp/vpar) * log_r_ri

    dthetaData = av_currentChiralityData['rotated_righthanded', 'mean']
    dthetaStd = av_currentChiralityData['rotated_righthanded', 'std']
    dthetaTau = 1.0/dthetaStd**2

    dtheta = pymc.Normal('dtheta', mu = modeled_dtheta, tau=dthetaTau, value=dthetaData, observed=True)

    chiralityVariables = [log_r_ri, vperp, dtheta]

    #######################
    ### Diffusion Piece ###
    #######################

    dif_xaxis = av_currentDiffusionData['1_div_ri_minus_1_r', 'mean'].values
    dtheta_variance = av_currentDiffusionData['rotated_righthanded', 'mean'].values

    # Estimating the error of the variance
    numSamples = av_currentDiffusionData['rotated_righthanded', 'len'].values
    dtheta_variance_error = np.sqrt(2*dthetaStd**4/(numSamples - 1)).values

    ds = pymc.Uniform('ds', lower=0, upper=10.**-4, value=10.**-6)

    @pymc.deterministic
    def modeled_variance(vpar=vpar, ds=ds, dif_xaxis=dif_xaxis):
        return (2*ds/vpar)*dif_xaxis

    var_dtheta = pymc.Normal('var_dtheta', mu=modeled_variance, tau=1./dtheta_variance_error**2, \
                             value=dtheta_variance, observed=True)
    diffusionVariables = [dif_xaxis, ds, var_dtheta]

    #######################
    ### Returning Model ###
    #######################

    return [velocityVariables, chiralityVariables, diffusionVariables]
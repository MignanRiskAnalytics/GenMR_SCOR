"""
GenMR Dynamics Modelling
========================

This module provides functions to quantify peril interactions and temporal dependencies - IN EARLY PHASE OF CONSTRUCTION.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.1.2
:Date: 2026-01-12
:License: AGPL-3
"""

import numpy as np
import pandas as pd


###################################
## PERIL ONE-TO-ONE INTERACTIONS ##
###################################
def calc_lbd_CS2Li(h_km_CS, lat):
    '''
    Calculate the cloud-to-ground (CG) lightning rate per convective storm based on 
    storm top height and latitude, following the parameterizations of Price & Rind (1992, 1993).

    This function computes:
    1. The total lightning flash rate (flashes per minute per storm) as a function of 
       convective storm top height.
    2. The fraction of total lightning that is cloud-to-ground (CG), using an empirical 
       latitude-dependent formula for the mixed-phase layer depth.

    Notes
    -----
    - The CG fraction is empirical and not physically tied to the actual cloud height or 
      freezing level. It is latitude-dependent to reproduce observed CG maxima in subtropical regions.
    - The total lightning rate uses the continental storm scaling from Price & Rind (1992).

    Parameters
    ----------
    h_km_CS : float or ndarray
        Convective storm cloud top height in kilometres. This determines the total 
        lightning flash rate according to Price & Rind (1992).

    lat : float or ndarray
        Latitude in degrees. Used to compute the empirical cloud-to-ground fraction (Price & Rind, 1993).

    Returns
    -------
    lbd_Li_strike : float or ndarray
        Estimated cloud-to-ground lightning flash rate in flashes per minute per storm.

    rate_CG : float or ndarray
        Fraction of total lightning that is cloud-to-ground (dimensionless, between 0 and 1).


    References
    ----------
    Price & Rind (1992), A Simple Lightning Parameterization for Calculating Global Lightning Distributions.
    J. Geophys. Res. 97(D9), 9919-9933
    Price & Rind (1993), What determines the cloud-to-ground lightning fraction in thunderstorms?
    Geophys. Res. Lett. 20(6), 463-466
    '''
    lbd_Li_tot = 3.44 * 1e-5 * h_km_CS**4.9                # (flashes/min/storm) - Price & Rind (1992:eq.6)
    
    # deprecated:
    #T0, _, _ = calc_T0_EBCM(lat, mon)
    #z_freeze = calc_z_freeze(T0, lapse_rate)
    #dH_CS = h_km_CS - z_freeze
    dH_CS = -6.64e-5 * lat**2 -4.73e-3 * lat + 7.34                              # Price & Rind (1993:eq.3)
    
    IC2CG = .021*dH_CS**4 - .648*dH_CS**3 + 7.493*dH_CS**2 - 36.54*dH_CS +63.09  # Price & Rind (1993:eq.1)
    rate_CG = 1 / (1 + IC2CG)                                                    # Price & Rind (1993:eq.2)
    lbd_Li_strike = lbd_Li_tot * rate_CG
    
    return lbd_Li_strike, rate_CG


def calc_S_RS2FF(S_RS, par):
    '''
    # flow Q [m3/s] = RS [m/s] * A catchment [m2]
    '''
    S_FF = S_RS * 1e-3 / 3600 * par['A_km2'] * 1e6
    return np.round(S_FF)


def calc_S_TC2SS(v_max, relationship = 'generic'):
    '''
    Empirical relationships according to the Saffir-Simpson scale (generic) or 
    from Lin et al. (2010) (New York harbor).
    vmax: max wind speed [m/s] during storm passage
    S_SS: storm surge size at the source (coastline) 
    '''
    if relationship == 'generic':
        S_SS = .0011 * v_max**2 
    if relationship == 'New York harbor':
        S_SS = .031641 * v_max - .00075537 * v_max**2 + 3.1941e-5 * v_max**3
    return np.round(S_SS, decimals = 3)


def calc_S_WS2SS(v_max):
    '''
    Empirical relationship from Lin et al. (2010) - New York
    vmax: max wind speed [m/s] during storm passage
    '''
    S_SS = .031641 * v_max - .00075537 * v_max**2 + 3.1941e-5 * v_max**3
    return np.round(S_SS, decimals = 3)
"""
GenMR Dynamics Modelling
========================

This module provides functions to quantify peril interactions and temporal dependencies - IN CONSTRUCTION.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.2.1
:Date: 2026-06-30
:License: AGPL-3
"""

import numpy as np
import pandas as pd


###########################
## TIME SERIES MODELLING ##
###########################
def gen_YLT_1block(ELT, Nsim, distr, phi = 0.):
    '''
    Generate a Year Loss Table (YLT) from an Event Loss Table (ELT) using a
    single frequency distribution for all perils combined.

    Parameters
    ----------
    ELT : pandas.DataFrame
        Event Loss Table with columns:

        - ``evID`` : unique event identifier
        - ``lbd``  : annual occurrence rate of the event
        - ``loss`` : mean loss associated with the event
    Nsim : int
        Number of simulated years.
    distr : str
        Count distribution. Either ``'Poisson'`` or ``'negative binomial'``.
    phi : float, optional
        Overdispersion parameter (Mailier et al., 2006), defined as
        ``phi = var/lbd - 1``, where ``var`` is the variance of the annual
        event count and ``lbd`` is its mean. Required when
        ``distr='negative binomial'``; ignored otherwise.
        ``phi=0`` recovers the Poisson process (default).

    Returns
    -------
    YLT : pandas.DataFrame
        Year Loss Table with columns:

        - ``simID`` : simulated year identifier (1 to Nsim)
        - ``evID``  : sampled event identifier
        - ``loss``  : loss of the sampled event

    Notes
    -----
    Event sampling uses inverse-CDF on the normalised cumulative occurrence
    rate, implemented via ``np.searchsorted`` for O(n log m) complexity.
    '''
    # 1. Overall rate
    lbd = np.sum(ELT['lbd'])

    # 2. Simulate number of events per year
    if distr == 'Poisson':
        k = np.random.poisson(lbd, Nsim)
    elif distr == 'negative binomial':
        var = lbd * (1 + phi)
        p = lbd / var
        r = lbd * p / (1 - p)
        k = np.random.negative_binomial(r, p, Nsim)

    # 3. Simulation IDs
    simIDs = np.repeat(np.arange(1, Nsim + 1), k)

    # 4. Sample events — sort by lbd so EF is monotonically increasing
    ELT_s = ELT.sort_values(by='lbd', ascending=True).reset_index(drop=True)
    ELT_s['EF_cum'] = ELT_s['lbd'].cumsum()
    EF_norm = ELT_s['EF_cum'].values / lbd

    n = int(np.sum(k))
    u = np.random.random(n)

    # searchsorted is O(n log m) vs your O(n*m) list comprehension
    idx = np.searchsorted(EF_norm, u, side='left')
    idx = np.clip(idx, 0, len(ELT_s) - 1)
    evIDs = ELT_s['evID'].values[idx]

    # 5. Build YLT and attach losses
    loss_map = ELT.set_index('evID')['loss']
    YLT = pd.DataFrame({'simID': simIDs, 'evID': evIDs})
    YLT['loss'] = YLT['evID'].map(loss_map)

    return YLT


def gen_YLT(ELT, Nsim, distr_dict, phi_dict = None):
    '''
    Generate a Year Loss Table (YLT) from a multi-peril Event Loss Table (ELT),
    with a per-peril count distribution.

    Parameters
    ----------
    ELT : pandas.DataFrame
        Event Loss Table with columns:

        - ``evID`` : unique event identifier
        - ``lbd``  : annual occurrence rate of the event
        - ``loss`` : mean loss associated with the event
        - ``ID``   : peril identifier
    Nsim : int
        Number of simulated years.
    distr_dict : dict
        Mapping of peril identifier to count distribution
        Accepted values are ``'Poisson'`` and ``'negative binomial'``.
    phi_dict : dict, optional
        Mapping of peril identifier to overdispersion parameter
        ``phi = var/lbd - 1`` (Mailier et al., 2006). Required for perils
        assigned ``'negative binomial'`` in ``distr_dict``; ignored for
        Poisson perils. Default is ``None``.

    Returns
    -------
    YLT : pandas.DataFrame
        Year Loss Table with columns:

        - ``simID`` : simulated year identifier (1 to Nsim)
        - ``evID``  : sampled event identifier
        - ``loss``  : loss of the sampled event

    Notes
    -----
    Each peril is simulated independently. Annual event counts for peril
    ``p`` follow either Poisson(lbd_p) or NegativeBinomial(lbd_p, phi_p),
    where ``lbd_p = sum(ELT.loc[ELT.ID==p, 'lbd'])``. The sum of
    independent Poisson variates is itself Poisson, so results are
    statistically equivalent to ``gen_YLT_1block`` when all perils use
    ``'Poisson'``.
    '''
    years = np.arange(1, Nsim + 1)
    all_simIDs = []
    all_evIDs = []

    for peril, sub_ELT in ELT.groupby('ID'):
        distr = distr_dict[peril]
        lbd = sub_ELT['lbd'].sum()

        if distr == 'Poisson':
            k = np.random.poisson(lbd, Nsim)
        elif distr == 'negative binomial':
            var = lbd * (1 + phi_dict[peril])
            p = lbd / var
            r = lbd * p / (1 - p)
            k = np.random.negative_binomial(r, p, Nsim)

        simIDs = np.repeat(years, k)

        sub_s = sub_ELT.sort_values('lbd').reset_index(drop=True)
        EF_norm = sub_s['lbd'].cumsum().values / lbd
        u = np.random.random(int(k.sum()))
        idx = np.clip(np.searchsorted(EF_norm, u, side='left'), 0, len(sub_s) - 1)
        evIDs = sub_s['evID'].values[idx]

        all_simIDs.append(simIDs)
        all_evIDs.append(evIDs)

    loss_map = ELT.set_index('evID')['loss']
    YLT = pd.DataFrame({
        'simID': np.concatenate(all_simIDs),
        'evID' : np.concatenate(all_evIDs),
    })
    YLT['loss'] = YLT['evID'].map(loss_map)

    return YLT









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
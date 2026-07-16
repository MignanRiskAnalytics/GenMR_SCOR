"""
GenMR Dynamics Modelling
========================

This module provides functions to quantify peril interactions and temporal dependencies - IN CONSTRUCTION.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.2.1
:Date: 2026-07-15
:License: AGPL-3
"""

import numpy as np
import pandas as pd
import copy

from skimage import measure

from GenMR import environment as GenMR_env
from GenMR import perils as GenMR_perils
from GenMR import utils as GenMR_utils


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
    S_map = ELT.set_index('evID')['S']
    YLT = pd.DataFrame({
        'simID': np.concatenate(all_simIDs),
        'evID' : np.concatenate(all_evIDs),
    })
    YLT['S'] = YLT['evID'].map(S_map)
    YLT['loss'] = YLT['evID'].map(loss_map)

    return YLT



#################################
## T-COMPOUNDING (SEASONALITY) ##
#################################
def sample_DT_stoch(par, Nsim, Ndays):
    '''
    Draw the shared inter-annual + monthly advective temperature anomaly
    used to drive all thermodynamically-coupled perils (HW, Dr, HR).

    Returns
    -------
    DT_peryr : ndarray (Nsim,)
    DT_adv_month : ndarray (Nsim, 12)
        Monthly advective anomaly (°C). Sign gives the synoptic regime:
        > 0 anticyclonic (subsidence, clear sky) -> HW / Dr potential
        < 0 cyclonic (ascent, cloudy)            -> HR potential
    DT_adv_daily : ndarray (Nsim, Ndays)
        DT_adv_month repeated to daily resolution, for gen_YET_HW.
    '''
    days_per_mon = GenMR_utils.days_per_mon
    DT_peryr = np.random.normal(0, par['sigmaT_yearly'], Nsim)
    DT_adv_month = np.empty((Nsim, 12))
    DT_adv_daily = np.empty((Nsim, Ndays))
    for sim in range(Nsim):
        DT_adv_month[sim] = GenMR_perils.HazardFootprintGenerator.sample_T_advectivemodel(
            np.zeros(12), par['lat_deg'])
        DT_adv_daily[sim] = np.repeat(DT_adv_month[sim], days_per_mon)
    return DT_peryr, DT_adv_month, DT_adv_daily


## HEATWAVE ##
def gen_YET_HW(T0, T, par, DT_peryr, DT_adv_daily, Nsim):
    '''
    Generate a stochastic Year-Event Table (YET) of heatwave events with associated spatial footprints.

    For each simulated year, this function applies a precomputed inter-annual and
    advective temperature offset, generates a daily AR(1) temperature trajectory,
    detects heatwave events at a coastal reference location, and reconstructs the
    spatial footprint of each detected event over the full grid.

    This implementation uses a simplified detection scheme: the heatwave time window
    ``(start, end)`` for each event is determined once, at the coastal reference location
    (``T0``, ``T_loc_daily_stoch``), via :func:`get_HW_atloc`. That same ``(start, end)``
    window is then reused to slice the entire spatial field when computing the footprint,
    rather than re-detecting the heatwave independently at every grid cell.

    Why this works here: by construction, ``T0`` is the coastal/sea-level reference
    temperature, and the spatial field ``T`` is built via a fixed lapse-rate correction,
    decreasing monotonically with elevation ``z``. Since daily temperature variation is a
    single scalar time series shared across the whole domain, every grid cell's daily
    temperature is just ``T0``'s trajectory plus a constant spatial offset
    (elevation-dependent, but time-invariant). Detecting the heatwave once at ``T0`` is
    therefore equivalent to detecting it separately at each ``(x, y)`` location.

    Limitation: this shortcut breaks down if daily fluctuations were made to vary
    independently across space. In that more general case, different grid cells could
    cross the heatwave threshold on different days.

    Parameters
    ----------
    T0 : ndarray of shape (Ndays,)
        Daily baseline temperature (°C) at the coastal/sea-level reference location,
        before inter-annual, advective, or AR(1) stochastic perturbations are applied.
    T : ndarray of shape (Ndays, nx, ny)
        Daily baseline temperature field (°C) over the full spatial grid, built from
        ``T0`` via a fixed lapse-rate correction as a function of elevation.
    par : dict
        Dictionary of heatwave model parameters, expected to contain:

        - ``'T_AR1'`` : tuple of (phi, sigma), AR(1) persistence and innovation std. dev. (°C) for daily fluctuations.
        - ``'T_th'`` : float, heatwave temperature threshold (°C).
        - ``'Dt_da'`` : int, minimum number of consecutive days above threshold to qualify as a heatwave.
    DT_peryr : ndarray of shape (Nsim,)
        Precomputed inter-annual temperature offset (°C) for each simulated year.
    DT_adv_daily : ndarray of shape (Nsim, Ndays)
        Precomputed advective temperature offset (°C), resolved at daily resolution,
        for each simulated year.
    Nsim : int
        Number of simulated years to generate.

    Returns
    -------
    YET_HW : DataFrame
        Year Event Table of simulated heatwave events, with one row per event and columns:

        - ``'simID'`` : int, simulated year index (1-based).
        - ``'evID'`` : str, unique event identifier (``'HW'`` + running integer counter across all sims).
        - ``'ID'`` : str, peril code (``'HW'``).
        - ``'t'`` : float, event start time as a decimal-year fraction within the simulated year.
        - ``'S'`` : int, event duration in days (inclusive day count).
    catalog_hazFp_HW : dict
        Mapping from ``evID`` to the corresponding spatial footprint (ndarray of shape
        ``(nx, ny)``), giving the maximum temperature (°C) reached during heatwave days
        at each grid cell.
    T_sav_daily_stoch : ndarray of shape (Nsim,Ndays)
        Saved temperature time series
    '''
    Ndays = T.shape[0]

    DT_stoch = DT_peryr[:, None] + DT_adv_daily 

    YET_list = []
    catalog_hazFp_HW = {}
    evID_counter = 0
    T_sav_daily_stoch = np.full((Nsim, Ndays), np.nan, dtype=np.float32)
    for sim in range(Nsim):
        simID = sim + 1
        if simID % 1000 == 0:
            print(f'{simID}/{Nsim}', end='\r', flush=True)

        dT_daily_stoch = GenMR_perils.HazardFootprintGenerator.sample_T_AR1process(DT_stoch[sim], \
                                                                Ndays, par['T_AR1'][0], par['T_AR1'][1])
        T_loc_daily_stoch = T0 + dT_daily_stoch
        T_map_daily_stoch = T + dT_daily_stoch[:, None, None]
        T_sav_daily_stoch[sim,:] = T_loc_daily_stoch    # saved for other climatic perils

        HW_ti_stoch, _ = GenMR_perils.HazardFootprintGenerator.get_HW_atloc(T_loc_daily_stoch, \
                                                        par['T_th'],  par['Dt_da'])
        for (start, end) in HW_ti_stoch:
            evID_counter += 1
            evID = f'HW{evID_counter}'

            t = start / Ndays           # decimal-year start time
            duration = end - start + 1  # per-event duration in days

            window = T_map_daily_stoch[start:end]
            fp_HW, _ = GenMR_perils.HazardFootprintGenerator.get_HW_footprint(
                window, Tth = par['T_th'], Dt = par['Dt_da'])

            catalog_hazFp_HW[evID] = fp_HW

            YET_list.append({'simID': simID, 'evID': evID, 'ID': 'HW', 't': t, 'S': duration})


    YET_HW = pd.DataFrame(YET_list)
    return YET_HW, catalog_hazFp_HW, T_sav_daily_stoch


## DROUGHT & RAINSTORM ##
def sample_Dr_stress(par, Nsim):
    '''
    Draw a stochastic antecedent soil-moisture stress fraction per simulated year.

    Parameters
    ----------
    par : dict
        Expected keys:
        - 'Dr_stress_mean' : float, mean stress fraction in [0,1]
        - 'Dr_stress_k'    : float, Beta concentration (higher = tighter around mean)
    Nsim : int

    Returns
    -------
    Dr_stress : ndarray of shape (Nsim,)
        Fraction of stable max soil-water content depleted before the year starts.
    '''
    m, k = par['Dr_stress_mean'], par['Dr_stress_k']
    a, b = m * k, (1 - m) * k
    return np.random.beta(a, b, Nsim)


def gen_YET_Dr_HR(T0_mo, par, atmo_par, soil_par, DT_peryr, DT_adv_month, Dr_stress, Nsim):
    w_subs, w_asc = atmo_par['vz_subs_asc']
    z_tropo = GenMR_env.EnvLayer_atmo.calc_z_tropopause(par['lat_deg'])
    par_rain = {'p0': atmo_par['p0_kPa'], 'lapse_rate': atmo_par['lapse_rate_degC/km'],
                'eta_rain': atmo_par['eta_rain'], 'zmax_km': z_tropo}
    Smax = soil_par['hw_max_m'] * 1e3
    Dr_th = soil_par['hw_fc_m'] * 1e3 * par['hw_th']

    YET_Dr_list, YET_HR_list = [], []
    evID_Dr, evID_HR = 0, 0
    I_rain_saved = np.full((Nsim, 12), np.nan, dtype=np.float32)
    S_t_saved = np.full((Nsim, 12), np.nan, dtype=np.float32)
    for sim in range(Nsim):
        simID = sim + 1
        if simID % 1000 == 0:
            print(f'{simID}/{Nsim}', end='\r', flush=True)

        T_mo_stoch = T0_mo + DT_peryr[sim] + DT_adv_month[sim]

        cyclonic = DT_adv_month[sim] < 0        # (12,) bool: True = ascent/cloudy/wet
        w_mo = np.where(cyclonic, w_asc, w_subs)

        ET0 = GenMR_perils.calc_PET(T_mo_stoch, par['lat_deg'], cloudy = cyclonic)
        I_rain = GenMR_perils.gen_precipitation(T_mo_stoch, w_mo, par_rain)    # mm/day
        I_rain_mon = I_rain * GenMR_utils.days_per_mon 
        hw0 = (1. - Dr_stress[sim]) * soil_par['hw_fc_m'] * 1000. 
                                        # fraction of stable maximum water content (mm)
        S_t = GenMR_perils.update_soil_moisture(I_rain_mon, ET0, hw0, Smax)
        I_rain_saved[sim,:] = I_rain_mon
        S_t_saved[sim,:] = S_t

        Dr_events, Dr_dur = GenMR_perils.get_Dr(S_t, Dr_th)
        for (start, end), dur in zip(Dr_events, Dr_dur):
            evID_Dr += 1
            YET_Dr_list.append({'simID': simID, 'evID': f'Dr{evID_Dr}', 'ID': 'Dr',
                                 't': start/12, 'S': dur})

        HR_events, HR_dur = GenMR_perils.get_HR(I_rain_mon, par['HR_th'])
        for (start, end), dur in zip(HR_events, HR_dur):
            evID_HR += 1
            YET_HR_list.append({'simID': simID, 'evID': f'HR{evID_HR}', 'ID': 'HR', 't': start/12, 'S': dur})

    return pd.DataFrame(YET_Dr_list), pd.DataFrame(YET_HR_list), S_t_saved, I_rain_saved


## WILDFIRE ##
def sample_WF_t(D_t_1sim, lbd0):
    '''
    D_t_1sim : daily dryness index for one simulated year (0-1), shape (Ndays,)
    lbd0 : calibrated scale so that lambda(t) = lbd0 * D_t is a daily ignition rate
    '''
    lbd_t = lbd0 * D_t_1sim
    lbd_max = lbd_t.max()
    if lbd_max <= 0:
        return np.array([], dtype=int)

    Ndays = len(D_t_1sim)
    # homogeneous proposal process at rate lbd_max
    n_candidates = np.random.poisson(lbd_max * Ndays)
    candidate_days = np.random.randint(0, Ndays, size=n_candidates)

    # thinning step
    accept_prob = lbd_t[candidate_days] / lbd_max
    accept = np.random.random(n_candidates) < accept_prob
    occurrence_days = np.sort(candidate_days[accept])
    return occurrence_days

def percolate_cluster(seed_idx, p_edge_grid):
    '''
    NOT YET FULLY TESTED - so far not used
    '''
    ny, nx = p_edge_grid.shape
    visited = np.zeros_like(p_edge_grid, dtype=bool)
    sy, sx = np.unravel_index(seed_idx, p_edge_grid.shape)
    stack = [(sy, sx)]
    visited[sy, sx] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1,0), (-1,0), (0,1), (0,-1)):
            ny_, nx_ = y+dy, x+dx
            if 0 <= ny_ < ny and 0 <= nx_ < nx and not visited[ny_, nx_] and S[ny_, nx_] == 1:
                if np.random.random() < p_edge_grid[ny_, nx_]:
                    visited[ny_, nx_] = True
                    stack.append((ny_, nx_))
    return visited

def gen_YET_WF(src, urbLandLayer, D_t, method = 'deterministic'):
    '''
    '''
    Nsim = D_t.shape[0]    # use same number as HW and Dr
    landuse_S = copy.copy(urbLandLayer.S)

    # Fuel includes forest (S=1), later updated with wood building
    FuelClass = [1]

    # only used if method = 'probabilistic' (to move to env. layer):
    FuelCoef_by_class = {
                -1: 0.,   # water
                0: 0.,    # grassland
                1: 1.,    # forest
                2: 0.,    # urban - residential
                3: 0.,    # urban - industrial
                4: 0.,    # urban - commercial
                5: .6,    # wheat
                6: .6,    # maize
    }
    FuelCoef = np.vectorize(FuelCoef_by_class.get)(landuse_S)
    

    YET_WF_list = []
    catalog_hazFp_WF = {}
    evID_counter = 0
    for sim in range(Nsim):
        # stochastic fuel to grass distribution
        indFuel = np.where(np.isin(landuse_S.flatten(), FuelClass))[0]
        indFuel2Grass = np.random.choice(indFuel, size = int(len(indFuel) * src.par['WF']['ratio_grass']),
                        replace = False)    
        landuse_S4WF_flat = landuse_S.flatten()
        landuse_S4WF_flat[indFuel2Grass] = 0                           # grassland
        indFuel = np.where(np.isin(landuse_S4WF_flat, FuelClass))[0]   # updated
        # add wood buildings to fuel state:
        indwoodBldg = np.where(urbLandLayer.bldg_type.flatten() == 'W')[0]
        landuse_S4WF_flat[indwoodBldg] = 1                             # forest-like
        landuse_S4WF0 = landuse_S4WF_flat.reshape(landuse_S.shape)
        # connectivity
        indconnect = np.where(np.isin(landuse_S4WF_flat, FuelClass))[0]
        grid_connect_flat = np.zeros_like(landuse_S4WF_flat)
        grid_connect_flat[indconnect] = 1
        grid_connect0 = grid_connect_flat.reshape(landuse_S.shape)
        
        # can be updated within a year
        grid_connect = grid_connect0.copy()
        landuse_S4WF = landuse_S4WF0.copy()
        
        simID = sim + 1
        if simID % 1000 == 0:
            print(f'{simID}/{Nsim}', end='\r', flush=True)

        WF_days = sample_WF_t(D_t[sim, :], src.par['WF']['lbd0'])

        for ev_i in range(len(WF_days)):
            ignition_xy = np.random.choice(indFuel)
            fp_WF = np.full(grid_connect.shape, np.nan)

            if grid_connect.flat[ignition_xy] == 1:
                if method == 'deterministic':
                    S_clumps = measure.label(grid_connect, connectivity = 1)
                    clump_WF = S_clumps.flatten()[ignition_xy]
                    indWF = S_clumps == clump_WF
                elif method == 'probabilistic':
                    pmax = .5   # critical regime
                    p_edge = pmax * D_t[sim, WF_days[ev_i]] * FuelCoef
                    indWF = percolate_cluster(ignition_xy[0], p_edge)
                
                fp_WF[indWF] = 1
                grid_connect[indWF] = 0
                landuse_S4WF[indWF] = 0

                burntArea_cells = np.sum(fp_WF == 1)
                area_ha = burntArea_cells * (urbLandLayer.grid.w ** 2) * 100

                #burntBldgBlocks_cells = np.sum(indWF.flatten()[indwoodBldg])   # to use later for loss calc...

                if area_ha >= src.par['WF']['Smin_ha']:
                    evID_counter += 1
                    evID = f'WF{evID_counter}'
                    
                    catalog_hazFp_WF[evID] = indWF   #fp_WF

                    t = WF_days[ev_i] / 365.
                    YET_WF_list.append({'simID': simID, 'evID': evID, 'ID': 'WF', 't': t, 'S': area_ha})


    return pd.DataFrame(YET_WF_list), catalog_hazFp_WF




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
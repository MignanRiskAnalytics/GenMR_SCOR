"""
GenMR Peril Implementation
==========================

This module provides functions to define and implement perils in the GenMR digital template.
This includes tools for full catastrophe risk modelling, such as event loss table (ELT) definition,
intensity and loss footprint computation... TO DEVELOP/REWRITE

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-11-04
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import copy

import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
ls = plt_col.LightSource(azdeg=45, altdeg=45)

from GenMR import environment as GenMR_env
from GenMR import dynamics as GenMR_dynamics
from GenMR import utils as GenMR_utils



#################
# PERIL SOURCES #
#################

def get_peril_evID(evID):
    '''
    Return the peril identifiers for an array of event identifiers
    '''
    return np.array([evID[k][0:2] for k in range(len(evID))])


def gen_stochsrc_points(N, grid, rdm_seed = None):
    '''
    Return random uniform coordinates for N points in the grid
    '''
    if rdm_seed is not None:
        np.random.seed(rdm_seed)

    x_rdm = grid.xmin + np.random.random(N) * (grid.xmax - grid.xmin)
    y_rdm = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    return x_rdm, y_rdm


def gen_stochsrc_tracks(N, grid, npt, max_deviation, rdm_seed = None):
    '''
    Return coordinates of N storm tracks, defined as straight lines
    subject to random deviation (below max_deviation) along y at npt points.
    '''
    if rdm_seed is not None:
        np.random.seed(rdm_seed)

    ID = np.repeat(np.arange(N)+1, npt)
    x = np.tile(np.linspace(grid.xmin, grid.xmax, npt), N)
    ystart = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    yend = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    y = np.linspace(ystart, yend, npt, axis = 1).flatten()
    deviation = np.random.uniform(-max_deviation, max_deviation, size = N*npt)
    y += deviation
    return x, y, ID


class Src:
    '''
    Define the characteristics of the peril sources.
    
    Attributes:
        par (dict): A dictionary with nested keys ['perils',
                        'EQ'['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km'],
                        'FF'['riv_A_km', 'riv_lbd', 'riv_ome', 'riv_y0', 'Q_m3/s', 'A_km2'],
                        'VE'['x', 'y']]
    
    
    '''
    def __init__(self, par, grid, topoLayer = None):
        '''
        Initialize Src
        
        Args:
            par (dict): A dictionary with nested keys ['perils',
                        'EQ'['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km'],
                        'FF'['riv_A_km', 'riv_lbd', 'riv_ome', 'riv_y0', 'Q_m3/s', 'A_km2'],
                        'VE'['x', 'y']]
            grid (class): A class instance of RasterGrid
        '''
        par['grid_A_km2'] = (grid.xmax - grid.xmin) * (grid.ymax - grid.ymin)    # virtual region area (km2)
        self.par = par
        self.grid = copy.copy(grid)

        self.SS_char = None


    ## EQ characteristics ##
    @property
    def EQ_char(self):
        '''
        Calls the function get_char_srcLine(self, par) if EQ source provided, otherwise 
        returns error message.
        
        Args:
            self
        
        Returns:
            list: list of arrays (if EQ source defined)
                [0] (ndarray(dtype=float, ndim=1)): 1D array of src_xi (for all points)
                [1] (ndarray(dtype=float, ndim=1)): 1D array of src_yi (for all points)
                [2] (ndarray(dtype=float, ndim=1)): 1D array of src_id (for all points)
                [3] (ndarray(dtype=float, ndim=1)): 1D array of src_L (for all faults)
                [4] (ndarray(dtype=float, ndim=1)): 1D array of seg_id (for all points)
                [5] (ndarray(dtype=float, ndim=1)): 1D array of seg_strike (for all fault segments)
                [6] (ndarray(dtype=float, ndim=1)): 1D array of seg_L (for all fault segments)
        '''
        if 'EQ' in self.par['perils']:
            return self.get_char_srcLine(self.par['EQ'])
        else:
            return print('WARNING: No earthquake source initiated in source parameter list')
    
    def get_char_srcLine(self, par):
        '''
        Calculate the coordinates of fault sources based on their extrema and the provided 
        resolution , as well as lenghts and strikes of faults and fault segments.
        
        Args:
            par (dict): A dictionary with keys ['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km']
        
        Returns:
            ndarray(dtype=float, ndim=1): 1D array of src_xi (for all points)
            ndarray(dtype=float, ndim=1): 1D array of src_yi (for all points)
            ndarray(dtype=float, ndim=1): 1D array of src_id (for all points)
            ndarray(dtype=float, ndim=1): 1D array of src_L (for all faults)
            ndarray(dtype=float, ndim=1): 1D array of seg_id (for all points)
            ndarray(dtype=float, ndim=1): 1D array of seg_strike (for all fault segments)
            ndarray(dtype=float, ndim=1): 1D array of seg_L (for all fault segments)
        '''
        src_xi = np.array([])
        src_yi = np.array([])
        src_id = np.array([])
        src_L = np.array([])
        seg_id = np.array([])
        seg_strike = np.array([])
        seg_L = np.array([])
        seg = 0
        for src_i in range(len(par['x'])):
            Lsum = 0
            for seg_i in range(len(par['x'][src_i]) - 1):
                dx = par['x'][src_i][seg_i+1] - par['x'][src_i][seg_i]
                dy = par['y'][src_i][seg_i+1] - par['y'][src_i][seg_i]
                sign1 = dx / np.abs(dx)
                sign2 = dy / np.abs(dy)
                L = np.sqrt(dx**2 + dy**2)
                strike = np.arctan(dx/dy) * 180 / np.pi
                sign3 = np.sin(strike * np.pi / 180) / np.abs(np.sin(strike * np.pi / 180))
                npt = int(np.round(L / par['bin_km']))
                seg_xi = np.zeros(npt)
                seg_yi = np.zeros(npt)
                seg_xi[0] = par['x'][src_i][seg_i]
                seg_yi[0] = par['y'][src_i][seg_i]
                for k in range(1, npt):
                    seg_xi[k] = seg_xi[k-1] + sign1 * sign3 * par['bin_km'] * np.sin(strike * np.pi / 180)
                    seg_yi[k] = seg_yi[k-1] + sign2 * par['bin_km'] * np.cos(strike * np.pi / 180)
                src_xi = np.append(src_xi, np.append(seg_xi, par['x'][src_i][seg_i+1]))
                src_yi = np.append(src_yi, np.append(seg_yi, par['y'][src_i][seg_i+1]))
                src_id = np.append(src_id, np.repeat(src_i, len(seg_xi)+1))
                seg_id = np.append(seg_id, np.repeat(seg, len(seg_xi)+1))
                seg_strike = np.append(seg_strike, strike)
                seg_L = np.append(seg_L, L)
                seg += 1
                Lsum += L
            src_L = np.append(src_L, Lsum)
        return {'x': src_xi, 'y': src_yi, 'srcID': src_id, 'srcL': src_L, 'segID': seg_id, 'strike': seg_strike, 'segL': seg_L}


    ## TC characteristics ##
    @property
    def TC_char(self):
        '''
        Calls the function get_char_srcTrack(self, par) if TC source provided, otherwise 
        returns error message.
        
        Args:
            self
        
        Returns:
            X: xxx
        '''
        if 'TC' in self.par['perils']:
            return self.get_track_highres(self.par['TC'])
        else:
            return print('WARNING: No tropical cyclone source initiated in source parameter list')

    def get_track_highres(self, par):
        x_hires = np.array([])
        y_hires = np.array([])
        id_hires = np.array([])
        evID = np.unique(par['ID'])
        for i in range(len(evID)):
            indev = np.where(par['ID'] == evID[i])[0]
            x_ev = par['x'][indev]
            y_ev = par['y'][indev]
            for seg in range(len(x_ev) - 1):
                dx = x_ev[seg + 1] - x_ev[seg]
                dy = y_ev[seg + 1] - y_ev[seg]
                sign1 = dx / np.abs(dx)
                sign2 = dy / np.abs(dy)
                L = np.sqrt(dx**2 + dy**2)
                strike = np.arctan(dx/dy) * 180 / np.pi
                npt = int(np.round(L / par['bin_km']))
                seg_xi = np.zeros(npt)
                seg_yi = np.zeros(npt)
                seg_xi[0] = x_ev[seg]
                seg_yi[0] = y_ev[seg]
                for k in range(1, npt):
                    seg_xi[k] = seg_xi[k-1] + sign1 * sign2 * par['bin_km'] * np.sin(strike * np.pi / 180)
                    seg_yi[k] = seg_yi[k-1] + sign1 * sign2 * par['bin_km'] * np.cos(strike * np.pi / 180)
                x_hires = np.append(x_hires, np.append(seg_xi, x_ev[seg + 1]))
                y_hires = np.append(y_hires, np.append(seg_yi, y_ev[seg + 1]))
                id_hires = np.append(id_hires, np.repeat(evID[i], len(seg_xi)+1))
        return x_hires, y_hires, id_hires

    def __repr__(self):
        return 'Src({})'.format(self.par)





## EQ CASE ##
def calc_EQ_length2magnitude(L):
    '''
    Given the earthquake rupture L (km), calculate the magnitude M
    '''
    c1, c2 = [5., 1.22]     # reverse case, Fig. 2.6b, Wells and Coppersmith (1994)
    M = c1 + c2 * np.log10(L)
    return np.round(M, decimals = 1)

def calc_EQ_magnitude2length(M):
    '''
    Given the earthquake magnitude M, calculate the rupture length L (km)
    (for floating rupture computations)
    '''
    c1, c2 = [5., 1.22]     # reverse case, Fig. 2.6b, Wells and Coppersmith (1994)
    L = 10**((M - c1)/c2)
    return L


def gen_EQ_floatingRupture(evIDi, Si, src, srcEQ_char):
    '''
    '''
    nRup = len(Si)
    li = calc_EQ_magnitude2length(Si)
    flt_x = srcEQ_char['x']
    flt_y = srcEQ_char['y']
    flt_L = srcEQ_char['srcL']
    flt_id = srcEQ_char['srcID']
    indflt = GenMR_utils.partitioning(np.arange(len(flt_L)), flt_L / np.sum(flt_L), nRup)  # longer faults visited more often
    Rup_loc = np.zeros(nRup, dtype = object)
    Rup_coord = pd.DataFrame(columns = ['evID', 'x', 'y', 'loc'])
    i = 0
    while i < nRup:
        flt_target = np.random.choice(indflt, 1)
        indID = flt_id == flt_target
        src_x = flt_x[indID]
        src_y = flt_y[indID]
        src_L = flt_L[flt_target]
        init = np.floor((src_L - li[i]) / src['EQ']['bin_km'])
        if src_L >= li[i]:
            u = np.ceil(np.random.random(1) * init).astype(int)[0]         # random rupture start loc
            Rup_x = src_x[u:(u + li[i] / src['EQ']['bin_km']).astype(int)]
            Rup_y = src_y[u:(u + li[i] / src['EQ']['bin_km']).astype(int)]
            Rup_loc[i] = src['EQ']['object'] + str(flt_target[0] + 1)
            Rup_coord = pd.concat([Rup_coord, pd.DataFrame(data = {'evID': np.repeat(evIDi[i], len(Rup_x)), \
                                                              'x': Rup_x, 'y': Rup_y, \
                                                              'loc': np.repeat(Rup_loc[i], len(Rup_x))})], ignore_index=True)
            i += 1
            
    return Rup_coord



def calc_S_track(stochset, src, Track_coord):    
    indperil = np.where(stochset['ID'] == 'TC')[0]
    evIDs = stochset['evID'][indperil].values
    vmax_start = stochset['S'][indperil].values
    S_alongtrack = {}
    for i in range(src['TC']['N']):
        indtrack = np.where(Track_coord['ID'] == i+1)[0]
        track_x = Track_coord['x'][indtrack].values
        track_y = Track_coord['y'][indtrack].values

        npt = len(indtrack)
        track_vmax = np.repeat(vmax_start[i], npt)  # track over ocean at vmax_start

        # find inland section & reduce vmax
        d = [np.min(np.sqrt((track_x[j] - src['SS']['x'])**2 + \
                            (track_y[j] - src['SS']['y'])**2)) for j in range(npt)]
        indcoast = np.where(d == np.min(d))[0]
        d2coast = track_x[indcoast[0]:] - track_x[indcoast[0]]
        # ad-hoc decay relationship:
        track_vmax[indcoast[0]:] = vmax_start[i] * np.exp(-.1 / src['TC']['vforward_m/s'] * d2coast)
            
        S_alongtrack[evIDs[i]] = track_vmax
    return S_alongtrack




########################
# STOCHASTIC EVENT SET #
########################

## event rates ##
def incrementing(xmin, xmax, xbin, scale):
    '''
    Return evenly spaced values within a given interval in linear or log scale
    '''
    if scale == 'linear':
        xi = np.arange(xmin, xmax + xbin, xbin)
    if scale == 'log':
        xi = 10**np.arange(np.log10(xmin), np.log10(xmax) + xbin, xbin)
    return xi

def calc_Lbd_powerlaw(S, a, b):
    '''
    Calculate the cumulative rate Lbd according to a power-law (Eq. 2.38) given event size S
    '''
    Lbd = 10**(a - b * np.log10(S))
    return Lbd

def calc_Lbd_exponential(S, a, b):
    '''
    Calculate the cumulative rate Lbd according to an exponential law (Eq. 2.39) given event size S
    '''
    Lbd = 10**(a - b * S)
    return Lbd

def calc_Lbd_GPD(S, mu, xi, sigma, Lbdmin):
    '''
    Calculate the cumulative rate Lbd according to the Generalised Pareto Distribution (Eq. 2.50) given event size S
    '''
    Lbd = Lbdmin * (1 + xi * (S - mu) / sigma)**(-1 / xi)
    return Lbd

def transform_cum2noncum(S, par):
    '''
    Transform the rate from cumulative (Lbd) to non-cumulative (lbd) (e.g., Eq. 2.65)
    '''
    if par['Sscale'] == 'linear':
        S_lo = S - par['Sbin']/2
        S_hi = S + par['Sbin']/2
    elif par['Sscale'] == 'log':
        S_lo = 10**(np.log10(S) - par['Sbin']/2)
        S_hi = 10**(np.log10(S) + par['Sbin']/2)
    if par['distr'] == 'powerlaw':
        Lbd_lo = calc_Lbd_powerlaw(S_lo, par['a'], par['b'])
        Lbd_hi = calc_Lbd_powerlaw(S_hi, par['a'], par['b'])
    if par['distr'] == 'exponential':
        Lbd_lo = calc_Lbd_exponential(S_lo, par['a'], par['b'])
        Lbd_hi = calc_Lbd_exponential(S_hi, par['a'], par['b'])
    if par['distr'] == 'GPD':
        Lbd_lo = calc_Lbd_GPD(S_lo, par['mu'], par['xi'], par['sigma'], par['Lbdmin'])
        Lbd_hi = calc_Lbd_GPD(S_hi, par['mu'], par['xi'], par['sigma'], par['Lbdmin'])
    lbd = Lbd_lo - Lbd_hi
    return lbd


## generate event set ##
def gen_eventset(src, sizeDistr):
    ev_stoch = pd.DataFrame(columns = ['ID', 'evID', 'S', 'lbd'])
    Mmax2 = calc_EQ_length2magnitude(src.EQ_char['srcL'][1])  # smaller of 2 hardcoded faults in src

    for ID in src.par['perils']:
        if ID in sizeDistr['primary']:
            # event ID definition #
            evID = [ID + str(i+1) for i in range(src.par[ID]['N'])]

            # size incrementation #
            Si = incrementing(sizeDistr[ID]['Smin'], sizeDistr[ID]['Smax'], sizeDistr[ID]['Sbin'], sizeDistr[ID]['Sscale'])

            # weighting how Si size distributed over N event sources
            Si_n = len(Si)
            Si_ind = np.arange(Si_n)
            if ID == 'EQ':
                # smaller events more often to test more spatial combinations on fault segments
                qi = np.linspace(1,11,Si_n)
                qi /= np.sum(qi)
                qi = np.sort(qi)[::-1]
            else:
                # equal weight
                qi = np.repeat(1./Si_n, Si_n)
            Si_ind_vec = GenMR_utils.partitioning(Si_ind, qi, src.par[ID]['N'])  # distribute Si sizes into N event sources
            Si_vec = Si[Si_ind_vec]
            wi = 1 / np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
            wi_vec = [wi[Si_ind == i][0] for i in Si_ind_vec]    # weight of rate(Si) at each of N locations

            # rate calculation #
            # calibrate event productivity
            if sizeDistr[ID]['distr'] == 'powerlaw':
                if 'a' not in sizeDistr[ID].keys():
                    rescaled = src.par['grid_A_km2'] / GenMR_utils.fetch_A0(sizeDistr[ID]['region'])
                    sizeDistr[ID]['a'] = sizeDistr[ID]['a0'] + np.log10(rescaled)
            if sizeDistr[ID]['distr'] == 'GPD':
                if 'Lbdmin' not in sizeDistr[ID].keys():
                    rescaled = src.par['grid_A_km2'] / GenMR_utils.fetch_A0(sizeDistr[ID]['region'])
                    sizeDistr[ID]['Lbdmin'] = sizeDistr[ID]['Lbdmin0'] * rescaled
            # calculate event rate (weighted)
            lbdi = transform_cum2noncum(Si_vec, sizeDistr[ID])
            lbdi = lbdi * wi_vec
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'ID': np.repeat(ID, src.par[ID]['N']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

        if ID in sizeDistr['secondary']:
            trigger = sizeDistr[ID]['trigger']
            evID = [ID + '_from' + trigger + str(i+1) for i in range(src.par[trigger]['N'])]
            Si_vec = np.repeat(np.nan, src.par[trigger]['N'])
            lbdi = np.repeat(np.nan, src.par[trigger]['N'])
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'ID': np.repeat(ID, src.par[trigger]['N']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

    return ev_stoch.reset_index(drop = True)




#def pop_fix_points(N, par):
#    '''
#    Return fixed coordinates for N events 
#    '''
#    nloc = len(par['x'])
#    srcID = np.array([par['object'] + str(i + 1) for i in range(nloc)])
#    src_rdm_ind = np.random.choice(np.arange(nloc), N)
#    evID_loc = srcID[src_rdm_ind]
#    evID_x = np.array(par['x'])[src_rdm_ind]
#    evID_y = np.array(par['y'])[src_rdm_ind]
#    return evID_x, evID_y, evID_loc


#def pop_grid_area(N, grid):
#    '''
#    '''
#    box_x = np.tile([grid.xmin, grid.xmax, grid.xmax, grid.xmin, grid.xmin], N)
#    box_y = np.tile([grid.ymin, grid.ymin, grid.ymax, grid.ymax, grid.ymin], N)
#    return box_x, box_y


def pop_EQ_floatingRupture(evIDi, Si, src):
    '''
    '''
    nRup = len(Si)
    li = calc_EQ_M2L(Si)
    flt_x, flt_y, flt_id, flt_L, seg_id, seg_strike, seg_L = src.EQ_char    
    indflt = GenMR_utils.partitioning(np.arange(len(flt_L)), flt_L / np.sum(flt_L), nRup)  # longer faults visited more often
    Rup_loc = np.zeros(nRup, dtype=object)
    Rup_coord = pd.DataFrame(columns = ['evID', 'x', 'y', 'z'])
    i = 0
    while i < nRup:
        flt_target = np.random.choice(indflt, 1)
        indID = flt_id == flt_target
        src_x = flt_x[indID]
        src_y = flt_y[indID]
        src_L = flt_L[flt_target]
        src_z = src.par['EQ']['z_km'][indflt[i]]
        init = np.floor((src_L - li[i]) / src.par['EQ']['bin_km'])
        if src_L >= li[i]:
            u = np.ceil(np.random.random(1) * init).astype(int)[0]                         # random rupture start loc
            Rup_x = src_x[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_y = src_y[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_loc[i] = src.par['EQ']['object'] + str(flt_target[0] + 1)
            Rup_coord = pd.concat([Rup_coord, pd.DataFrame({'evID': np.repeat(evIDi[i], len(Rup_x)), \
                            'x': Rup_x, 'y': Rup_y, 'z': np.repeat(src_z, len(Rup_x))})], ignore_index=True)
            i += 1
    return Rup_coord['evID'], Rup_coord['x'], Rup_coord['y'], Rup_coord['z'], Rup_loc

def calc_EQ_M2L(M):
    '''
    '''
    return 10**((M - 4.49) / 1.49)    # for floating rupture computations

def get_highres(x0, y0, id0, par):
    '''
    '''
    x_hires = np.array([])
    y_hires = np.array([])
    id_hires = np.array([])
    evID = np.unique(id0)
    for i in range(len(evID)):
        indev = np.where(id0 == evID[i])[0]
        x_ev = x0[indev]
        y_ev = y0[indev]
        for seg in range(len(x_ev) - 1):
            dx = x_ev[seg + 1] - x_ev[seg]
            dy = y_ev[seg + 1] - y_ev[seg]
            sign1 = dx / np.abs(dx)
            sign2 = dy / np.abs(dy)
            L = np.sqrt(dx**2 + dy**2)
            strike = np.arctan(dx/dy) * 180 / np.pi
            npt = int(np.round(L / par['bin_km']))
            seg_xi = np.zeros(npt)
            seg_yi = np.zeros(npt)
            seg_xi[0] = x_ev[seg]
            seg_yi[0] = y_ev[seg]
            for k in range(1, npt):
                seg_xi[k] = seg_xi[k-1] + sign1 * sign2 * par['bin_km'] * np.sin(strike * np.pi / 180)
                seg_yi[k] = seg_yi[k-1] + sign1 * sign2 * par['bin_km'] * np.cos(strike * np.pi / 180)
            x_hires = np.append(x_hires, np.append(seg_xi, x_ev[seg + 1]))
            y_hires = np.append(y_hires, np.append(seg_yi, y_ev[seg + 1]))
            id_hires = np.append(id_hires, np.repeat(evID[i], len(seg_xi)+1))
    return x_hires, y_hires, id_hires




#####################
# HAZARD FOOTPRINTS #
#####################

# TO ADD...






###################
# LOSS FOOTPRINTS #
###################







############
# PLOTTING #
############

def plot_src(src, hillshading_z = '', file_ext = '-'):
    '''
    Plot peril sources in the spatial grid.
    
    Args:
        grid (class): An instance of class RasterGrid
        par (dict): A dictionary with nested keys ['perils',
                        'EQ'['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km'],
                        'FF'['riv_A_km', 'riv_lbd', 'riv_ome', 'riv_y0', 'Q_m3/s', 'A_km2'],
                        'VE'['x', 'y']]
        file_ext (str, optional): String representing the figure format ('jpg', 'pdf', etc., '-' by default)
    
    Returns:
        A plot (saved in file if file_ext not '-')
    '''

    handles = []
    labels = []

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    if len(hillshading_z) != 0:
        ax[0].contourf(src.grid.xx, src.grid.yy, ls.hillshade(hillshading_z, vert_exag=.1), cmap='gray', alpha = .2)
    if 'EQ' in src.par['perils']:
        for src_i in range(len(src.par['EQ']['x'])):
            h_eq, = ax[0].plot(src.par['EQ']['x'][src_i], src.par['EQ']['y'][src_i], color = GenMR_utils.col_peril('EQ'))
        handles.append(h_eq)
        labels.append('Fault segment: Earthquake (EQ)')
    if 'FF' in src.par['perils']:
        river_xi, river_yi, _, river_id = GenMR_env.calc_coord_river_dampedsine(src.grid, src.par['FF'])
        for src_i in range(len(src.par['FF']['riv_y0'])):
            indriv = river_id == src_i
            h_riv, = ax[0].plot(river_xi[indriv], river_yi[indriv], color = GenMR_utils.col_peril('FF'), linestyle = 'dashed')
            h_ff = ax[0].scatter(np.max(river_xi), src.par['FF']['riv_y0'][0], s=75, marker = 's', clip_on = False, color = GenMR_utils.col_peril('FF'))
        handles.append(h_ff)
        labels.append('River upstream point: Fluvial Flood (FF)')
        handles.append(h_riv)
        labels.append('River bed')
    if 'VE' in src.par['perils']:
        h_ve = ax[0].scatter(src.par['VE']['x'], src.par['VE']['x'], color = GenMR_utils.col_peril('VE'), s=75, marker='^')
        handles.append(h_ve)
        labels.append('Volcano: Volcanic Eruption (VE)')
    if 'SS' in src.par['perils']:
        h_ss, = ax[0].plot(src.SS_char['x'], src.SS_char['y'], color = GenMR_utils.col_peril('SS'))
        handles.append(h_ss)
        labels.append('Coastline: Storm surge (SS)')
    if 'TC' in src.par['perils']:
        for src_i in range(src.par['TC']['N']):
            indsrc = np.where(src.par['TC']['ID'] == src_i)[0]
            h_tc, = ax[0].plot(src.par['TC']['x'][indsrc], src.par['TC']['y'][indsrc], color = GenMR_utils.col_peril('TC'))
        handles.append(h_tc)
        labels.append('Storm track: Tropical cyclone (TC)')
    if 'AI' in src.par['perils']:
        h_ai = ax[0].scatter(src.par['AI']['x'], src.par['AI']['y'], color = GenMR_utils.col_peril('AI'), s=30, marker = '+', clip_on = False)
        handles.append(h_ai)
        labels.append('Impact site: Asteroid impact (AI)')

    h_box, = ax[0].plot([src.grid.xmin + src.grid.xbuffer, src.grid.xmax - src.grid.xbuffer, src.grid.xmax - src.grid.xbuffer, \
                src.grid.xmin + src.grid.xbuffer, src.grid.xmin + src.grid.xbuffer],
               [src.grid.ymin + src.grid.ybuffer, src.grid.ymin + src.grid.ybuffer, src.grid.ymax - src.grid.ybuffer, \
                src.grid.ymax - src.grid.ybuffer, src.grid.ymin + src.grid.ybuffer], linestyle='dotted', color='black')
    handles.append(h_box)
    labels.append('Active domain')
     
    ax[0].set_xlim(src.grid.xmin, src.grid.xmax)
    ax[0].set_ylim(src.grid.ymin, src.grid.ymax)
    ax[0].set_xlabel('$x$ (km)')
    ax[0].set_ylabel('$y$ (km)')
    ax[0].set_title('Peril source coordinates', size = 14, pad = 20)
    ax[0].set_aspect(1)


    lgd_src = ax[1].legend(handles, labels,
        loc='upper left',
        frameon=False,
        handletextpad=.8,   # spacing between patch & text
        borderaxespad=.5,   # spacing between legend & axes
        borderpad=.5        # padding in legend box
    )
    ax[1].add_artist(lgd_src)
    ax[1].axis('off')

    if file_ext != '-':
        plt.savefig('figs/DigitalTemplate_src.' + file_ext)


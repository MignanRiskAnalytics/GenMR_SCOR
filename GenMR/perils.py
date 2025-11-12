"""
GenMR Peril Implementation
==========================

This module provides functions to define and implement perils in the GenMR digital template.
This includes tools for full catastrophe risk modelling, such as event loss table (ELT) definition,
intensity and loss footprint computation... TO DEVELOP/REWRITE

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-11-11
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import copy
import re

import matplotlib.pyplot as plt

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


class Src:
    '''
    Define the characteristics of the peril sources.
    
    Attributes:
        par (dict): A dictionary with nested keys ['perils',
                        'EQ'['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km'],
                        'FF'['riv_A_km', 'riv_lbd', 'riv_ome', 'riv_y0', 'Q_m3/s', 'A_km2'],
                        'VE'['x', 'y']]
    
    
    '''
    def __init__(self, par, grid):
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
        par['EQ']['N'] = len(par['EQ']['x'])
        par['FF']['N'] = len(par['FF']['riv_A_km'])
        par['VE']['N'] = len(par['VE']['x'])
        par['EQ']['ID'] = np.char.add(par['EQ']['object'], (np.arange(par['EQ']['N'])+1).astype(str))
        par['FF']['ID'] = np.char.add(par['FF']['object'], (np.arange(par['FF']['N'])+1).astype(str))
        par['VE']['ID'] = np.char.add(par['VE']['object'], (np.arange(par['VE']['N'])+1).astype(str))
        self.par = par
        self.grid = copy.copy(grid)
        self.SS_char = None


    ## AI characteristics ##
    @property
    def AI_char(self):
        '''
        '''
        if 'AI' in self.par['perils']:
            srcID = [self.par['AI']['object'] + str(i + 1) for i in range(self.par['AI']['N'])]
            src_xi, src_yi = self.gen_stochsrc_points(self.par['AI']['N'], self.grid, self.par['rdm_seed'])
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            return print('WARNING: No earthquake source initiated in source parameter list')

    def gen_stochsrc_points(self, N, grid, rdm_seed = None):
        '''
        Return random uniform coordinates for N points in the grid
        '''
        if rdm_seed is not None:
            np.random.seed(rdm_seed)
        x_rdm = grid.xmin + np.random.random(N) * (grid.xmax - grid.xmin)
        y_rdm = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
        return x_rdm, y_rdm


    ## EQ characteristics ##
    @property
    def EQ_char(self):
        '''
        Calls the function get_char_srcLine(self, par) if EQ source provided, otherwise 
        returns error message.
        
        Args:
            self
        
        Returns:
            x: xxx
        '''
        if 'EQ' in self.par['perils']:
            return self.get_char_srcEQline(self.par['EQ'])
        else:
            return print('WARNING: No earthquake source initiated in source parameter list')
    
    def get_char_srcEQline(self, par):
        '''
        Calculate the coordinates of fault sources based on their extrema and the provided 
        resolution , as well as lenghts and strikes of faults and fault segments.
        
        Args:
            par (dict): A dictionary with keys ['x', 'y', 'w_km', 'dip_deg', 'z_km', 'mec', 'bin_km']
        
        Returns:
            x: xxx
        '''
        src_xi = np.array([])
        src_yi = np.array([])
        src_id = np.array([], dtype=int)
        src_L = np.array([])
        seg_id = np.array([], dtype=int)
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
        src_Mmax = calc_EQ_length2magnitude(src_L)
        srcID = np.char.add(self.par['EQ']['object'], (src_id+1).astype(str))
        return {'srcID': srcID, 'x': src_xi, 'y': src_yi, 'fltID': src_id, 'srcL': src_L, 'srcMmax': src_Mmax, 'segID': seg_id, 'strike': seg_strike, 'segL': seg_L}


    ## FF characteristics ##
    @property
    def FF_char(self):
        '''
        '''
        if 'FF' in self.par['perils']:
            src_ind = np.arange(self.par['FF']['N']) + 1
            srcID = np.char.add(self.par['FF']['object'], src_ind.astype(str))
            FF_x0 = np.repeat(self.grid.xmax, self.par['FF']['N'])
            return {'srcID': srcID, 'x': FF_x0, 'y': self.par['FF']['riv_y0']}
        else:
            return print('WARNING: No fluvial flood source initiated in source parameter list')


    ## TC characteristics ##
    @property
    def TC_char(self):
        '''
        '''
        if 'TC' in self.par['perils']:
            src_ind, src_xi, src_yi = self.gen_stochsrc_TCtracks(self.par['TC']['N'], self.grid, self.par['TC']['npt'], self.par['TC']['max_dev'], self.par['rdm_seed'])
            srcID = np.char.add(self.par['TC']['object'], src_ind.astype(str))
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            return print('WARNING: No tropical cyclone source initiated in source parameter list')

    def gen_stochsrc_TCtracks(self, N, grid, npt, max_deviation, rdm_seed = None):
        '''
        Return coordinates of N storm tracks, defined as straight lines
        subject to random deviation (below max_deviation) along y at npt points.
        '''
        if rdm_seed is not None:
            np.random.seed(rdm_seed)
        ind = np.repeat(np.arange(N)+1, npt)
        x = np.tile(np.linspace(grid.xmin, grid.xmax, npt), N)
        ystart = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
        yend = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
        y = np.linspace(ystart, yend, npt, axis = 1).flatten()
        deviation = np.random.uniform(-max_deviation, max_deviation, size = N*npt)
        y += deviation
        return ind, x, y


    ## VE characteristics ##
    @property
    def VE_char(self):
        '''
        '''
        if 'VE' in self.par['perils']:
            src_ind = np.arange(self.par['VE']['N']) + 1
            srcID = np.char.add(self.par['VE']['object'], src_ind.astype(str))
            return {'srcID': srcID, 'x': self.par['VE']['x'], 'y': self.par['VE']['y']}
        else:
            return print('WARNING: No volcanic eruption source initiated in source parameter list')


    def __repr__(self):
        return 'Src({})'.format(self.par)



################
# SCALING LAWS #
################

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
    '''
    '''
    ev_stoch = pd.DataFrame({'ID': pd.Series(dtype='object'), 'srcID': pd.Series(dtype='object'), 'evID': pd.Series(dtype='object'),
        'S': pd.Series(dtype=float), 'lbd': pd.Series(dtype=float)})
    srcIDs = []
    ev_char = pd.DataFrame({'evID': pd.Series(dtype='object'), 'x': pd.Series(dtype=float), 'y': pd.Series(dtype=float)})
    for ID in src.par['perils']:
        ev_ID = ev_x = ev_y = None
        if ID in sizeDistr['primary']:
            # event ID definition #
            evID = [ID + str(i+1) for i in range(sizeDistr[ID]['Nstoch'])]

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
            Si_ind_vec = GenMR_utils.partitioning(Si_ind, qi, sizeDistr[ID]['Nstoch'])  # distribute Si sizes into N event sources
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
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'ID': np.repeat(ID, sizeDistr[ID]['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

        if ID in sizeDistr['secondary']:
            trigger = sizeDistr[ID]['trigger']
            evID = [ID + '_from' + trigger + str(i+1) for i in range(sizeDistr[trigger]['Nstoch'])]
            Si_vec = np.repeat(np.nan, sizeDistr[trigger]['Nstoch'])
            lbdi = np.repeat(np.nan, sizeDistr[trigger]['Nstoch'])
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'ID': np.repeat(ID, sizeDistr[trigger]['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

        ## get event spatial characteristics ##
        if ID == 'AI':
            ev_ID, ev_x, ev_y = evID, src.AI_char['x'], src.AI_char['y']
            srcIDs = np.append(srcIDs, src.AI_char['srcID'])
        if ID == 'EQ':
            Rup_coord, Rup_loc = gen_EQ_floatingRupture(evID, Si_vec, src)
            ev_ID, ev_x, ev_y = Rup_coord['evID'], Rup_coord['x'], Rup_coord['y']
            srcIDs = np.append(srcIDs, Rup_loc)
        if ID == 'RS':
            srcIDs = np.append(srcIDs, np.repeat(src.par['RS']['object'], sizeDistr['RS']['Nstoch']))
        if ID == 'TC':
            track_coord = get_TCtrack_highres(evID, src)
            ev_ID, ev_x, ev_y = track_coord['evID'], track_coord['x'], track_coord['y']
            srcIDs = np.append(srcIDs, np.unique(src.TC_char['srcID']))
        if ID == 'VE':
            # WARNING: assumes for now that only one volcano source possible for now
            ev_ID, ev_x, ev_y = evID, np.repeat(src.VE_char['x'], sizeDistr['VE']['Nstoch']), np.repeat(src.VE_char['y'], sizeDistr['VE']['Nstoch'])
            srcIDs = np.append(srcIDs, np.repeat(src.VE_char['srcID'], sizeDistr['VE']['Nstoch']))
        if ID == 'FF':
            # WARNING: assumes for now that only one river source possible for now
            srcIDs = np.append(srcIDs, np.repeat(src.FF_char['srcID'], sizeDistr[trigger]['Nstoch']))
        if ID == 'LS':
            srcIDs = np.append(srcIDs, np.repeat(src.par['LS']['object'], sizeDistr[trigger]['Nstoch']))
        if ID == 'SS':
            srcIDs = np.append(srcIDs, np.repeat(src.par['SS']['object'], sizeDistr[trigger]['Nstoch']))
        if ev_ID is not None:
            ev_char = pd.concat([ev_char, pd.DataFrame({'evID': ev_ID, 'x': ev_x, 'y': ev_y})])
    ev_stoch['srcID'] = srcIDs
    return ev_stoch.reset_index(drop = True), ev_char.reset_index(drop = True)


## stochastic event characteristics ##
def gen_EQ_floatingRupture(evIDi, Si, src):
    '''
    '''
    nRup = len(Si)
    li = calc_EQ_magnitude2length(Si)
    flt_x = src.EQ_char['x']
    flt_y = src.EQ_char['y']
    flt_L = src.EQ_char['srcL']
    flt_id = src.EQ_char['fltID']
    indflt = GenMR_utils.partitioning(np.arange(len(flt_L)), flt_L / np.sum(flt_L), nRup)  # longer faults visited more often
    Rup_loc = np.zeros(nRup, dtype = object)
    Rup_coord = pd.DataFrame({'loc': pd.Series(dtype='object'), 'evID': pd.Series(dtype='object'), 'x': pd.Series(dtype=float), 'y': pd.Series(dtype=float)})
    i = 0
    while i < nRup:
        flt_target = np.random.choice(indflt, 1)
        indID = flt_id == flt_target
        src_x = flt_x[indID]
        src_y = flt_y[indID]
        src_L = flt_L[flt_target]
        init = np.floor((src_L - li[i]) / src.par['EQ']['bin_km'])
        if src_L >= li[i]:
            u = np.ceil(np.random.random(1) * init).astype(int)[0]         # random rupture start loc
            Rup_x = src_x[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_y = src_y[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_loc[i] = src.par['EQ']['object'] + str(flt_target[0] + 1)
            Rup_coord = pd.concat([Rup_coord, pd.DataFrame({'evID': np.repeat(evIDi[i], len(Rup_x)), 'x': Rup_x, 'y': Rup_y})], ignore_index=True)
            i += 1
    return Rup_coord, Rup_loc

def get_TCtrack_highres(evIDi, src):
    '''
    '''
    x_hires = np.array([])
    y_hires = np.array([])
    id_hires = np.array([])
    srcID = np.unique(src.TC_char['srcID'])  # match 1:1 between srcID and evID
    for i in range(len(evIDi)):
        indev = np.where(src.TC_char['srcID'] == srcID[i])[0]
        x_ev = src.TC_char['x'][indev]
        y_ev = src.TC_char['y'][indev]
        for seg in range(len(x_ev) - 1):
            dx = x_ev[seg + 1] - x_ev[seg]
            dy = y_ev[seg + 1] - y_ev[seg]
            sign1 = dx / np.abs(dx)
            sign2 = dy / np.abs(dy)
            L = np.sqrt(dx**2 + dy**2)
            strike = np.arctan(dx/dy) * 180 / np.pi
            npt = int(np.round(L / src.par['TC']['bin_km']))
            seg_xi = np.zeros(npt)
            seg_yi = np.zeros(npt)
            seg_xi[0] = x_ev[seg]
            seg_yi[0] = y_ev[seg]
            for k in range(1, npt):
                seg_xi[k] = seg_xi[k-1] + sign1 * sign2 * src.par['TC']['bin_km'] * np.sin(strike * np.pi / 180)
                seg_yi[k] = seg_yi[k-1] + sign1 * sign2 * src.par['TC']['bin_km'] * np.cos(strike * np.pi / 180)
            x_hires = np.append(x_hires, np.append(seg_xi, x_ev[seg + 1]))
            y_hires = np.append(y_hires, np.append(seg_yi, y_ev[seg + 1]))
            id_hires = np.append(id_hires, np.repeat(evIDi[i], len(seg_xi)+1))
    Track_coord = pd.DataFrame({'evID': id_hires, 'x': x_hires, 'y': y_hires})
    return Track_coord



#####################
# HAZARD FOOTPRINTS #
#####################

# analytical
def calc_I_shaking_ms2(S, r):
    PGA_g = 10**(-1.34 + .23*S - np.log10(r))     # size = magnitude
    g_earth = 9.81                   # [m/s^2]
    PGA_ms2 = PGA_g * g_earth
    return PGA_ms2

def calc_I_blast_kPa(S, r):
    Z = r * 1e3 / (S * 1e6)**(1/3)                # size = energy in kton TNT
    p_kPa = (1772/Z**3 - 114/Z**2 + 108/Z)
    return p_kPa

def calc_I_ash_m(S, r):
    # assumes h0 proportional to V - e.g h0 = 1e-3 km for V=3 km3 (1980 Mt. St. Helens)
    h0 = 1e-3 /3 * S                                                # size = volume in km3 
    r_half = np.sqrt(S * np.log(2)**2 / (2* np.pi * h0) )
    h_m = ( h0 * np.exp (-np.log(2) * r / r_half) ) * 1e3   # m
    return h_m

def calc_I_v_ms(S, r, par):
    '''
    Eq. 2.24
    '''
    rho_atm = 1.15                                       # air density [kg/m3]
    Omega = 7.2921e-5                                    # [rad/s]
    f = 2 * Omega * np.sin(par['lat_deg'] * np.pi/180)   # Coriolis parameter
    
    pn = par['pn_mbar'] * 100                            # [Pa]
    B = par['B_Holland']
    
    R = 51.6 * np.exp(-.0223 * S + .0281 * par['lat_deg'])   # see caption of Fig. 2.19
    pc = pn - 1 / B * (rho_atm * np.exp(1) * S**2)
    
    v_ms = ( B * R**B * (pn - pc) * np.exp(-(R/r)**B) / (rho_atm * r**B) + r**2 * f**2 / 4 )**.5 - r*f/2
    return v_ms

def add_v_forward(vf, vtan, track_x, track_y, grid, t_i):
    # components of forward motion vector
    if t_i < len(track_x)-1:
        dx = track_x[t_i+1]-track_x[t_i]
        dy = track_y[t_i+1]-track_y[t_i]
        if dx == 0 and dy == 0:
            dx = track_x[t_i]-track_x[t_i-1]
            dy = track_y[t_i]-track_y[t_i-1]
    else:
        # assumes same future direction
        dx = track_x[t_i]-track_x[t_i-1]
        dy = track_y[t_i]-track_y[t_i-1]

    beta = np.arctan(dy/dx)
    if dx > 0:
        vf_x = vf * np.cos(beta)
        vf_y = vf * np.sin(beta)
    else:
        vf_x = -vf * np.cos(beta)
        vf_y = -vf * np.sin(beta)
        
    # components of gradient-based azimuthal wind vector
    dx = grid.xx - track_x[t_i]
    dy = grid.yy - track_y[t_i]
    alpha = np.arctan(dy/dx)
    # if x > x0
    vtan_x = -vtan * np.sin(alpha)
    vtan_y = vtan * np.cos(alpha)
    # if x < x0
    indneg = np.where(grid.xx < track_x[t_i])
    vtan_x[indneg] = vtan[indneg] * np.sin(alpha[indneg])
    vtan_y[indneg] = -vtan[indneg] * np.cos(alpha[indneg])

    vtot_x = vtan_x + vf_x
    vtot_y = vtan_y + vf_y
    vtot = np.sqrt(vtot_x**2 + vtot_y**2)
    return vtot, vtot_x, vtot_y, vtan_x, vtan_y

def calc_S_track(stochset, src, Track_coord):    
    indperil = np.where(stochset['ID'] == 'TC')[0]
    evIDs = stochset['evID'][indperil].values
    vmax_start = stochset['S'][indperil].values
    S_alongtrack = {}
    for i in range(len(evIDs)):
        indtrack = np.where(Track_coord['evID'] == evIDs[i])[0]
        track_x = Track_coord['x'][indtrack].values
        track_y = Track_coord['y'][indtrack].values

        npt = len(indtrack)
        track_vmax = np.repeat(vmax_start[i], npt)  # track over ocean at vmax_start

        # find inland section & reduce vmax
        d = [np.min(np.sqrt((track_x[j] - src.SS_char['x'])**2 + \
                            (track_y[j] - src.SS_char['y'])**2)) for j in range(npt)]
        indcoast = np.where(d == np.min(d))[0]
        d2coast = track_x[indcoast[0]:] - track_x[indcoast[0]]
        # ad-hoc decay relationship:
        track_vmax[indcoast[0]:] = vmax_start[i] * np.exp(-.1 / src.par['TC']['vforward_m/s'] * d2coast)
        
        S_alongtrack[evIDs[i]] = track_vmax
    return S_alongtrack


# threshold model
def model_SS_Bathtub(I_trigger, src, topo_z):
        vmax_coastline = np.zeros(src.grid.ny)
        for j in range(src.grid.ny):
            indx = np.where(src.grid.x > src.SS_char['x'][j]-1e-6)[0][0]
            vmax_coastline[j] = I_trigger[indx,j]
        S_SS = GenMR_dynamics.calc_S_TC2SS(vmax_coastline, src.par['SS']['bathy'])
        I_SS = np.zeros((src.grid.nx, src.grid.ny))
        for j in range(src.grid.ny):
            I_alongx = S_SS[j] - topo_z[:,j]
            I_alongx[I_alongx < 0] = 0
            I_alongx[src.grid.x < src.SS_char['x'][j]] = 0
            I_SS[:,j] = I_alongx
        return I_SS


def gen_hazFootprints(stochset, evchar, src, topo_z):
    catalog_hazFootprints = {}
    print('generating footprints for:')
    for ID in src.par['perils']:
        indperil = np.where(stochset['ID'] == ID)[0]
        Nev_peril = len(indperil)

        if ID == 'AI':
            print(ID)
            AIcoord = evchar[get_peril_evID(evchar['evID']) == 'AI'].reset_index(drop = True)
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                r = np.sqrt((src.grid.xx - AIcoord['x'][i])**2 + (src.grid.yy - AIcoord['y'][i])**2)   # point source
                catalog_hazFootprints[evID] = calc_I_ash_m(S, r)

        if ID == 'VE':
            print(ID)
            VEcoord = evchar[get_peril_evID(evchar['evID']) == 'VE'].reset_index(drop = True)
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                r = np.sqrt((src.grid.xx - VEcoord['x'][0])**2 + (src.grid.yy - VEcoord['y'][0])**2)   # point source
                catalog_hazFootprints[evID] = calc_I_blast_kPa(S, r)

        if ID == 'EQ':
            print(ID)
            EQcoord = evchar[get_peril_evID(evchar['evID']) == 'EQ'].reset_index(drop = True)
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                srcID = stochset['srcID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                evID_coords = EQcoord[EQcoord['evID'] == evID]
                npt = len(evID_coords)
                d2rupt = np.zeros((src.grid.nx, src.grid.ny, npt))
                for k in range(npt):
                    d2rupt[:,:,k] = np.sqrt((src.grid.xx - evID_coords['x'].values[k])**2 + (src.grid.yy - evID_coords['y'].values[k])**2)
                dmin = d2rupt.min(axis = 2)
                z = np.array(src.par['EQ']['z_km'])[src.par['EQ']['ID'] == srcID]
                r = np.sqrt(dmin**2 + z**2)                                                        # line source
                catalog_hazFootprints[evID] = calc_I_shaking_ms2(S, r)

        if ID == 'TC':
            print(ID)
            TCcoord = evchar[get_peril_evID(evchar['evID']) == 'TC'].reset_index(drop = True)
            S_alongtrack = calc_S_track(stochset, src, TCcoord) # ad-hoc inland decay of windspeed
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                indtrack = np.where(TCcoord['evID'] == evID)[0]
                track_x = TCcoord['x'][indtrack].values
                track_y = TCcoord['y'][indtrack].values
                track_S = S_alongtrack[evID]
                npt = len(indtrack)
                I_t = np.zeros((src.grid.nx, src.grid.ny, npt))
                for j in range(npt):
                    r = np.sqrt((src.grid.xx - track_x[j])**2 + (src.grid.yy - track_y[j])**2)             # point source at time t
                    I_sym_t = calc_I_v_ms(track_S[j], r, src.par['TC'])
                    I_t[:,:,j], _, _, _, _ = \
                        add_v_forward(src.par['TC']['vforward_m/s'], I_sym_t, track_x, track_y, src.grid, j)
                catalog_hazFootprints[evID] = np.nanmax(I_t, axis = 2)                                # track source

        if ID == 'SS':
            print(ID)
            pattern = re.compile(r'TC(\d+)')  # match "TC" followed by numbers
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                evID_trigger = re.search(pattern, evID).group()
                I_trigger = catalog_hazFootprints[evID_trigger]
                catalog_hazFootprints[evID] = model_SS_Bathtub(I_trigger, src, topo_z)

    print('... catalogue completed')
    return catalog_hazFootprints






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
        ax[0].contourf(src.grid.xx, src.grid.yy, GenMR_env.ls.hillshade(hillshading_z, vert_exag=.1), cmap='gray', alpha = .2)
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
            h_ff = ax[0].scatter(src.FF_char['x'], src.FF_char['y'], s=75, marker = 's', clip_on = False, color = GenMR_utils.col_peril('FF'))
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
        for src_id in np.unique(src.TC_char['srcID']):
            indsrc = np.where(src.TC_char['srcID'] == src_id)[0]
            h_tc, = ax[0].plot(src.TC_char['x'][indsrc], src.TC_char['y'][indsrc], color = GenMR_utils.col_peril('TC'))
        handles.append(h_tc)
        labels.append('Storm track: Tropical cyclone (TC)')
    if 'AI' in src.par['perils']:
        h_ai = ax[0].scatter(src.AI_char['x'], src.AI_char['y'], color = GenMR_utils.col_peril('AI'), s=30, marker = '+', clip_on = False)
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
        plt.savefig('figs/src.' + file_ext)



def plot_hazFootprints(catalog_hazFootprints, grid, topoLayer_z, plot_Imax, nstoch = 5, file_ext = '-'):
    evIDs = np.array(list(catalog_hazFootprints.keys()))
    ev_peril = get_peril_evID(evIDs)
    perils = np.unique(ev_peril)
    nperil = len(perils)

    plt.rcParams['font.size'] = '18'
    _, ax = plt.subplots(nperil, nstoch, figsize=(20, nperil*20/nstoch))

    for i in range(nperil):
        indperil = np.where(ev_peril == perils[i])[0]
        nev = len(indperil)
        nplot = np.min([nstoch, nev])
        evID_shuffled = evIDs[indperil]
        if nev > nplot:
            np.random.shuffle(evID_shuffled)
        Imax = plot_Imax[perils[i]]
        for j in range(nplot):
            I_plt = np.copy(catalog_hazFootprints[evID_shuffled[j]])
            I_plt[I_plt >= Imax] = Imax
            ax[i,j].contourf(grid.xx, grid.yy, I_plt, cmap = 'Reds', levels = np.linspace(0, Imax, 100))
            ax[i,j].contourf(grid.xx, grid.yy, GenMR_env.ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
            ax[i,j].set_xlim(grid.xmin, grid.xmax)
            ax[i,j].set_ylim(grid.ymin, grid.ymax)
            ax[i,j].set_xlabel('$x$ (km)')
            ax[i,j].set_ylabel('$y$ (km)')
            ax[i,j].set_title(evID_shuffled[j], pad = 10)
            ax[i,j].set_aspect(1)
        if nplot < nstoch:
            for j in np.arange(nplot, nstoch):
                ax[i,j].set_axis_off()
    plt.tight_layout()    

    if file_ext != '-':
        plt.savefig('figs/hazFootprints.' + file_ext)

    plt.pause(1)
    plt.show()
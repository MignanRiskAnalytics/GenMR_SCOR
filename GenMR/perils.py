"""
GenMR Peril Implementation
==========================

This module provides functions to define and implement perils in the GenMR digital template.
This includes tools for full catastrophe risk modelling, such as event loss table (ELT) definition,
intensity and loss footprint computation... TO DEVELOP/REWRITE

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-11-17
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import os
import copy
import re
import warnings

#import matplotlib
#matplotlib.use('Agg')   # avoid kernel crash

import matplotlib.pyplot as plt
import imageio

from GenMR import environment as GenMR_env
from GenMR import dynamics as GenMR_dynamics
from GenMR import utils as GenMR_utils



#################
# PERIL SOURCES #
#################

def get_peril_evID(evIDs):
    '''
    Return the peril identifiers for an array of event identifiers
    '''
    return np.array([evID[:2] for evID in evIDs])


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
            src_xi, src_yi = self._gen_stochsrc_points(self.par['AI']['N'], self.grid, self.par['rdm_seed'])
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            warnings.warn('No AI source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}
        
    def _gen_stochsrc_points(self, N, grid, rdm_seed = None):
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
        # Check if result already cached
        if hasattr(self, '_EQ_char_cache'):
            return self._EQ_char_cache
        # Compute only if EQ peril exists
        if 'EQ' in self.par['perils']:
            result = self._get_char_srcEQline(self.par['EQ'])
            self._EQ_char_cache = result
            return result
        else:
            warnings.warn('No EQ source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}
    
    def _get_char_srcEQline(self, par):
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
                sign1 = np.sign(dx)
                sign2 = np.sign(dy)
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
            warnings.warn('No FF source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}


    ## TC characteristics ##
    @property
    def TC_char(self):
        '''
        '''
        if 'TC' in self.par['perils']:
            src_ind, src_xi, src_yi = self._gen_stochsrc_TCtracks(self.par['TC']['N'], self.grid, self.par['TC']['npt'], self.par['TC']['max_dev'], self.par['rdm_seed'])
            srcID = np.char.add(self.par['TC']['object'], src_ind.astype(str))
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            warnings.warn('No TC source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}

    def _gen_stochsrc_TCtracks(self, N, grid, npt, max_deviation, rdm_seed = None):
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
            warnings.warn('No VE source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}


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


## event set ##
class EventSetGenerator:
    '''
    Generates stochastic event set based on source and size distribution parameters.
    '''

    def __init__(self, src, sizeDistr, utils):
        self.src = src
        self.sizeDistr = sizeDistr
        self.utils = utils  # e.g. GenMR_utils module or similar
        self.ev_stoch = pd.DataFrame({'ID': pd.Series(dtype='object'), 'srcID': pd.Series(dtype='object'), 'evID': pd.Series(dtype='object'),
                                      'S': pd.Series(dtype=float), 'lbd': pd.Series(dtype=float)})
        self.ev_char = pd.DataFrame({'evID': pd.Series(dtype='object'), 'x': pd.Series(dtype=float), 'y': pd.Series(dtype=float)})
        self.srcIDs = []

    def generate(self):
        for ID in self.src.par['perils']:
            ev_ID = ev_x = ev_y = None

            ## PRIMARY EVENTS ##
            if ID in self.sizeDistr['primary']:
                evID, Si_vec, lbdi = self._generate_primary(ID)
                self.ev_stoch = pd.concat([
                    self.ev_stoch,
                    pd.DataFrame({'ID': np.repeat(ID, self.sizeDistr[ID]['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})
                ], ignore_index=True)

            ## SECONDARY EVENTS ##
            if ID in self.sizeDistr['secondary']:
                ID_trigger, evID, Si_vec, lbdi = self._generate_secondary(ID)
                self.ev_stoch = pd.concat([
                    self.ev_stoch,
                    pd.DataFrame({'ID': ID_trigger, 'evID': evID, 'S': Si_vec, 'lbd': lbdi})
                ], ignore_index=True)

            ## Spatial characteristics ##
            ev_ID, ev_x, ev_y = self._add_spatial_characteristics(ID, evID, Si_vec)
            if ev_ID is not None:
                self.ev_char = pd.concat([
                    self.ev_char,
                    pd.DataFrame({'evID': ev_ID, 'x': ev_x, 'y': ev_y})
                ])

        self.ev_stoch['srcID'] = self.srcIDs
        return self.ev_stoch.reset_index(drop=True), self.ev_char.reset_index(drop=True)


    def _generate_primary(self, ID):
        N = self.sizeDistr[ID]['Nstoch']
        evID = [f"{ID}{i+1}" for i in range(N)]

        # Generate size distribution
        Si = incrementing(self.sizeDistr[ID]['Smin'], self.sizeDistr[ID]['Smax'], self.sizeDistr[ID]['Sbin'], self.sizeDistr[ID]['Sscale'])

        # Weighting logic
        Si_vec, wi_vec = self._distribute_sizes(ID, Si)

        # Calibration for rate calculation
        if self.sizeDistr[ID]['distr'] == 'powerlaw':
            if 'a' not in self.sizeDistr[ID]:
                rescaled = self.src.par['grid_A_km2'] / self.utils.fetch_A0(self.sizeDistr[ID]['region'])
                self.sizeDistr[ID]['a'] = self.sizeDistr[ID]['a0'] + np.log10(rescaled)
        if self.sizeDistr[ID]['distr'] == 'GPD':
            if 'Lbdmin' not in self.sizeDistr[ID]:
                rescaled = self.src.par['grid_A_km2'] / self.utils.fetch_A0(self.sizeDistr[ID]['region'])
                self.sizeDistr[ID]['Lbdmin'] = self.sizeDistr[ID]['Lbdmin0'] * rescaled

        # Calculate weighted event rates
        lbdi = self._transform_cum2noncum(Si_vec, self.sizeDistr[ID]) * wi_vec
        return evID, Si_vec, lbdi

    def _generate_secondary(self, ID):
        trigger = self.sizeDistr[ID]['trigger']
        if ID == 'Ex':
            ID_trigger = np.array([ID])
            evID = np.array(['Ex_fromCIf'])
            Si_vec = np.array([self.sizeDistr[ID]['S']])
            lbdi = np.array([np.nan])
        else:
            N = self.sizeDistr[trigger]['Nstoch']
            ID_trigger = np.repeat(ID, self.sizeDistr[trigger]['Nstoch'])
            evID = [f"{ID}_from{trigger}{i+1}" for i in range(N)]
            Si_vec = np.repeat(np.nan, N)
            lbdi = np.repeat(np.nan, N)
        return ID_trigger, evID, Si_vec, lbdi


    def _distribute_sizes(self, ID, Si):
        Si_n = len(Si)
        Si_ind = np.arange(Si_n)

        if ID == 'EQ':
            qi = np.linspace(1, 11, Si_n)
            qi /= np.sum(qi)
            qi = np.sort(qi)[::-1]
        else:
            qi = np.repeat(1. / Si_n, Si_n)

        Si_ind_vec = self.utils.partitioning(Si_ind, qi, self.sizeDistr[ID]['Nstoch'])
        Si_vec = Si[Si_ind_vec]
        wi = 1 / np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
        wi_vec = [wi[Si_ind == i][0] for i in Si_ind_vec]
        return Si_vec, wi_vec

    def _transform_cum2noncum(self, S, par):
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

    def _add_spatial_characteristics(self, ID, evID, Si_vec):
        ev_ID = ev_x = ev_y = None

        if ID == 'AI':
            ev_ID, ev_x, ev_y = evID, self.src.AI_char['x'], self.src.AI_char['y']
            self.srcIDs = np.append(self.srcIDs, self.src.AI_char['srcID'])

        elif ID == 'EQ':
            Rup_coord, Rup_loc = self._gen_EQ_floatingRupture(evID, Si_vec, self.src)
            ev_ID, ev_x, ev_y = Rup_coord['evID'], Rup_coord['x'], Rup_coord['y']
            self.srcIDs = np.append(self.srcIDs, Rup_loc)

        elif ID == 'RS':
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['RS']['object'], self.sizeDistr['RS']['Nstoch']))

        elif ID == 'TC':
            track_coord = self._get_TCtrack_highres(evID, self.src)
            ev_ID, ev_x, ev_y = track_coord['evID'], track_coord['x'], track_coord['y']
            self.srcIDs = np.append(self.srcIDs, np.unique(self.src.TC_char['srcID']))

        elif ID == 'VE':
            ev_ID, ev_x, ev_y = evID, np.repeat(self.src.VE_char['x'], self.sizeDistr['VE']['Nstoch']), np.repeat(self.src.VE_char['y'], self.sizeDistr['VE']['Nstoch'])
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.VE_char['srcID'], self.sizeDistr['VE']['Nstoch']))

        elif ID == 'FF':
            trigger = self.sizeDistr[ID]['trigger']
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.FF_char['srcID'], self.sizeDistr[trigger]['Nstoch']))
        elif ID == 'LS':
            trigger = self.sizeDistr[ID]['trigger']
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['LS']['object'], self.sizeDistr[trigger]['Nstoch']))
        elif ID == 'SS':
            trigger = self.sizeDistr[ID]['trigger']
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['SS']['object'], self.sizeDistr[trigger]['Nstoch']))
        elif ID == 'Ex':
            ev_ID, ev_x, ev_y = np.array(['Ex_fromCIf']), np.array([self.src.par['Ex']['x']]), np.array([self.src.par['Ex']['y']])
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['Ex']['object'], 1))

        return ev_ID, ev_x, ev_y

    ## stochastic event characteristics ##
    def _gen_EQ_floatingRupture(self, evIDi, Si, src):
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

    def _get_TCtrack_highres(self, evIDi, src):
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

class HazardFootprintGenerator:
    def __init__(self, stochset, evchar, src, topo_z):
        self.stochset = stochset
        self.evchar = evchar
        self.src = src
        self.topo_z = topo_z
        self.catalog_hazFootprints = {}

    ## ANALYTICAL EXPRESSIONS ##
    @staticmethod
    def calc_I_shaking_ms2(S, r):
        PGA_g = 10 ** (-1.34 + 0.23 * S - np.log10(r))
        g_earth = 9.81
        return PGA_g * g_earth

    @staticmethod
    def calc_I_blast_kPa(S, r):
        Z = r * 1e3 / (S * 1e6) ** (1 / 3)          # size = energy in kton TNT
        return 1772 / Z**3 - 114 / Z**2 + 108 / Z

    @staticmethod
    def calc_I_ash_m(S, r):
        # assumes h0 proportional to V - e.g h0 = 1e-3 km for V=3 km3 (1980 Mt. St. Helens)
        h0 = 1e-3 / 3 * S                           # size = volume in m3
        r_half = np.sqrt(S * np.log(2)**2 / (2 * np.pi * h0))
        h_m = (h0 * np.exp(-np.log(2) * r / r_half)) * 1e3
        return h_m

    @staticmethod
    def calc_I_v_ms(S, r, par):
        rho_atm = 1.15                                       # air density (kg/m3)
        Omega = 7.2921e-5                                    # (rad/s)
        f = 2 * Omega * np.sin(par['lat_deg'] * np.pi/180)   # Coriolis parameter
        pn = par['pn_mbar'] * 100                            # (Pa)
        B = par['B_Holland']
        R = 51.6 * np.exp(-.0223 * S + .0281 * par['lat_deg'])   # see caption of Mignan (2024:fig.2.19)
        pc = pn - 1 / B * (rho_atm * np.exp(1) * S**2)

        v_ms = (B * R**B * (pn - pc) * np.exp(-(R / r)**B) / (rho_atm * r**B) + r**2 * f**2 / 4) ** 0.5 - r * f / 2
        return v_ms

    @staticmethod
    def add_v_forward(vf, vtan, track_x, track_y, grid, t_i):
        if t_i < len(track_x) - 1:
            dx = track_x[t_i + 1] - track_x[t_i]
            dy = track_y[t_i + 1] - track_y[t_i]
            if dx == 0 and dy == 0:
                dx = track_x[t_i] - track_x[t_i - 1]
                dy = track_y[t_i] - track_y[t_i - 1]
        else:
            dx = track_x[t_i] - track_x[t_i - 1]
            dy = track_y[t_i] - track_y[t_i - 1]

        beta = np.arctan(dy / dx)
        if dx > 0:
            vf_x, vf_y = vf * np.cos(beta), vf * np.sin(beta)
        else:
            vf_x, vf_y = -vf * np.cos(beta), -vf * np.sin(beta)

        dx = grid.xx - track_x[t_i]
        dy = grid.yy - track_y[t_i]
        alpha = np.arctan(dy / dx)

        vtan_x = -vtan * np.sin(alpha)
        vtan_y = vtan * np.cos(alpha)

        indneg = np.where(grid.xx < track_x[t_i])
        vtan_x[indneg] = vtan[indneg] * np.sin(alpha[indneg])
        vtan_y[indneg] = -vtan[indneg] * np.cos(alpha[indneg])

        vtot_x = vtan_x + vf_x
        vtot_y = vtan_y + vf_y
        vtot = np.sqrt(vtot_x**2 + vtot_y**2)
        return vtot, vtot_x, vtot_y, vtan_x, vtan_y

    @staticmethod
    def calc_S_track(stochset, src, Track_coord):
        indperil = np.where(stochset['ID'] == 'TC')[0]
        evIDs = stochset['evID'][indperil].values
        vmax_start = stochset['S'][indperil].values
        S_alongtrack = {}

        for i, evID in enumerate(evIDs):
            indtrack = np.where(Track_coord['evID'] == evID)[0]
            track_x = Track_coord['x'][indtrack].values
            track_y = Track_coord['y'][indtrack].values
            npt = len(indtrack)
            track_vmax = np.repeat(vmax_start[i], npt)

            d = [np.min(np.sqrt((track_x[j] - src.SS_char['x'])**2 + (track_y[j] - src.SS_char['y'])**2)) for j in range(npt)]
            indcoast = np.where(d == np.min(d))[0]
            d2coast = track_x[indcoast[0]:] - track_x[indcoast[0]]
            track_vmax[indcoast[0]:] = vmax_start[i] * np.exp(-.1 / src.par['TC']['vforward_m/s'] * d2coast)

            S_alongtrack[evID] = track_vmax
        return S_alongtrack

    @staticmethod
    def model_SS_Bathtub(I_trigger, src, topo_z):
        vmax_coastline = np.zeros(src.grid.ny)
        for j in range(src.grid.ny):
            indx = np.where(src.grid.x > src.SS_char['x'][j] - 1e-6)[0][0]
            vmax_coastline[j] = I_trigger[indx, j]

        S_SS = GenMR_dynamics.calc_S_TC2SS(vmax_coastline, src.par['SS']['bathy'])
        I_SS = np.zeros((src.grid.nx, src.grid.ny))

        for j in range(src.grid.ny):
            I_alongx = S_SS[j] - topo_z[:, j]
            I_alongx[I_alongx < 0] = 0
            I_alongx[src.grid.x < src.SS_char['x'][j]] = 0
            I_SS[:, j] = I_alongx

        return I_SS

    ## INTENSITY FOOTPRINT GENERATOR ##
    def generate(self):
        print('generating footprints for:', end=' ')
        for ID in self.src.par['perils']:
            indperil = np.where(self.stochset['ID'] == ID)[0]
            Nev_peril = len(indperil)
            print(ID, end=', ')

            if ID == 'AI':
                AIcoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'AI'].reset_index(drop=True)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    r = np.sqrt((self.src.grid.xx - AIcoord['x'][i])**2 + (self.src.grid.yy - AIcoord['y'][i])**2)
                    self.catalog_hazFootprints[evID] = self.calc_I_blast_kPa(S, r)

            elif ID == 'VE':
                VEcoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'VE'].reset_index(drop=True)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    r = np.sqrt((self.src.grid.xx - VEcoord['x'][0])**2 + (self.src.grid.yy - VEcoord['y'][0])**2)
                    self.catalog_hazFootprints[evID] = self.calc_I_ash_m(S, r)

            elif ID == 'Ex':
                Excoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'Ex'].reset_index(drop=True)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    r = np.sqrt((self.src.grid.xx - Excoord['x'][i])**2 + (self.src.grid.yy - Excoord['y'][i])**2)
                    self.catalog_hazFootprints[evID] = self.calc_I_blast_kPa(S, r)

            elif ID == 'EQ':
                EQcoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'EQ'].reset_index(drop=True)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    srcID = self.stochset['srcID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    evID_coords = EQcoord[EQcoord['evID'] == evID]
                    npt = len(evID_coords)
                    d2rupt = np.zeros((self.src.grid.nx, self.src.grid.ny, npt))
                    for k in range(npt):
                        d2rupt[:, :, k] = np.sqrt((self.src.grid.xx - evID_coords['x'].values[k])**2 +
                                                  (self.src.grid.yy - evID_coords['y'].values[k])**2)
                    dmin = d2rupt.min(axis=2)
                    z = np.array(self.src.par['EQ']['z_km'])[self.src.par['EQ']['ID'] == srcID]
                    r = np.sqrt(dmin**2 + z**2)
                    self.catalog_hazFootprints[evID] = self.calc_I_shaking_ms2(S, r)

            elif ID == 'TC':
                TCcoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'TC'].reset_index(drop=True)
                S_alongtrack = self.calc_S_track(self.stochset, self.src, TCcoord)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    indtrack = np.where(TCcoord['evID'] == evID)[0]
                    track_x = TCcoord['x'][indtrack].values
                    track_y = TCcoord['y'][indtrack].values
                    track_S = S_alongtrack[evID]
                    npt = len(indtrack)
                    I_t = np.zeros((self.src.grid.nx, self.src.grid.ny, npt))
                    for j in range(npt):
                        r = np.sqrt((self.src.grid.xx - track_x[j])**2 + (self.src.grid.yy - track_y[j])**2)
                        I_sym_t = self.calc_I_v_ms(track_S[j], r, self.src.par['TC'])
                        I_t[:, :, j], *_ = self.add_v_forward(
                            self.src.par['TC']['vforward_m/s'], I_sym_t, track_x, track_y, self.src.grid, j)
                    self.catalog_hazFootprints[evID] = np.nanmax(I_t, axis=2)

            elif ID == 'SS':
                pattern = re.compile(r'TC(\d+)')
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    evID_trigger = re.search(pattern, evID).group()
                    I_trigger = self.catalog_hazFootprints[evID_trigger]
                    self.catalog_hazFootprints[evID] = self.model_SS_Bathtub(I_trigger, self.src, self.topo_z)
                


        print('... catalogue completed')
        return self.catalog_hazFootprints

    ## TC CASE ##
    def get_TC_timeshot(self, evID, t):
        '''
        Compute a single time-step (snapshot) of a tropical cyclone (TC) event.

        Args:
            evID (str): Event ID
            t (int): Time index along the cyclone track (0 ≤ t < npt)

        Returns:
            tuple: 
                I_sym_t (ndarray): Symmetric wind field (static storm).
                I_asym_t (ndarray): Asymmetric wind field (storm in motion).
                vtot_x (ndarray): Total velocity component along x.
                vtot_y (ndarray): Total velocity component along y.
                vtan_x (ndarray): Tangential velocity component along x.
                vtan_y (ndarray): Tangential velocity component along y.
        '''
        Track_coord = self.evchar[get_peril_evID(self.evchar['evID']) == 'TC'].reset_index(drop=True)
        S_alongtrack = self.calc_S_track(self.stochset, self.src, Track_coord)   # always calculate for all, rewrite for one track...

        indtrack = np.where(Track_coord['evID'] == evID)[0]
        track_x = Track_coord['x'][indtrack].values
        track_y = Track_coord['y'][indtrack].values
        track_S = S_alongtrack[evID]
        npt = len(indtrack)

        if t < 0 or t >= npt:
            raise ValueError(f"t={t} is out of range (0 ≤ t < {npt}).")

        r = np.sqrt((self.src.grid.xx - track_x[t])**2 + (self.src.grid.yy - track_y[t])**2)
        I_sym_t = self.calc_I_v_ms(track_S[t], r, self.src.par['TC'])
        I_asym_t, vtot_x, vtot_y, vtan_x, vtan_y = self.add_v_forward(
            self.src.par['TC']['vforward_m/s'], I_sym_t, track_x, track_y, self.src.grid, t)

        return I_sym_t, I_asym_t, vtot_x, vtot_y, vtan_x, vtan_y



class DynamicHazardFootprintGenerator:
    def __init__(self, stochset, src, soilLayer):
        self.stochset = stochset
        self.src = src
        self.soil = soilLayer
        self.catalog_hazFootprints = {}
        self.cache_dir = 'io/cache_dynHazFootprints'
        os.makedirs(self.cache_dir, exist_ok = True)

    def _cache_path(self, evID):
        return os.path.join(self.cache_dir, f'{evID}.npy')

    ## INTENSITY FOOTPRINT GENERATOR ##
    def generate(self, selected_perils = None, force_recompute = False):
        self.force_recompute = force_recompute
        if selected_perils is None:
            selected_perils = self.src.par['perils']

        print('generating footprints for:')
        for ID in selected_perils:
            indperil = np.where(self.stochset['ID'] == ID)[0]
            Nev_peril = len(indperil)

            if ID == 'FF':
                self._run_FF(indperil, Nev_peril)
            elif ID == 'LS':
                self._run_LS(indperil, Nev_peril)

        print('... catalogue completed')
        return self.catalog_hazFootprints


    def _run_FF(self, indperil, Nev_peril):
        pattern = re.compile(r'RS(\d+)')
        movie = {'create': False}
        for i in range(Nev_peril):
            evID = self.stochset['evID'][indperil].values[i]
            cache_file = self._cache_path(evID)

            if os.path.exists(cache_file) and not self.force_recompute:
                self.catalog_hazFootprints[evID] = np.load(cache_file)
                print(f'{evID} (loaded from cache)')
            else:
                print(f'{evID} (computing)')
                evID_trigger = re.search(pattern, evID).group()

                S_trigger = self.stochset['S'][self.stochset['evID'] == evID_trigger].values
                I_RS = S_trigger * 1e-3 / 3600    # (mm/hr) to (m/s)

                FF_CA = CellularAutomaton_FF(I_RS, self.src, self.soil.grid, self.soil.topo.z, movie)
                FF_CA.run()

                FF_footprint_hmax = FF_CA.result()                    
                self.catalog_hazFootprints[evID] = FF_footprint_hmax
                np.save(cache_file, FF_footprint_hmax)


    def _run_LS(self, indperil, Nev_peril):
        pattern = re.compile(r'RS(\d+)')
        movie = {'create': False}
        for i in range(Nev_peril):
            evID = self.stochset['evID'][indperil].values[i]
            cache_file = self._cache_path(evID)

            if os.path.exists(cache_file) and not self.force_recompute:
                self.catalog_hazFootprints[evID] = np.load(cache_file)
                print(f'{evID} (loaded from cache)')
            else:
                print(f'{evID} (computing)')
                evID_trigger = re.search(pattern, evID).group()
                        
                S_trigger = self.stochset['S'][self.stochset['evID'] == evID_trigger].values
                hw = S_trigger * 1e-3 * self.src.par['RS']['duration']    # water column (m)
                wetness = hw / self.soil.h
                wetness[wetness > 1] = 1                  # max possible saturation
                wetness[self.soil.h == 0] = 0             # no soil case

                LS_CA = CellularAutomaton_LS(self.soil, wetness, movie)
                LS_CA.run()

                LS_footprint_hmax = LS_CA.result()                    
                self.catalog_hazFootprints[evID] = LS_footprint_hmax
                np.save(cache_file, LS_footprint_hmax)


## FLUVIAL FLOOD CASE ##
class CellularAutomaton_FF:
    def __init__(self, I_RS, src, grid, topoLayer_z, movie):
        self.I_RS = I_RS
        self.src = src
        self.grid = grid
        self.z = topoLayer_z
        movie['path'] = 'figs/FF_CA_frames/'
        self.movie = movie

        A_catchment = src.par['FF']['A_km2'] * 1e6   # (m2)
        self.Qp = I_RS * A_catchment                 # (m3/s)
        self.tmax = int(src.par['RS']['duration'] * 3600)

        river_xi, river_yi, _, _ = GenMR_env.calc_coord_river_dampedsine(grid, src.par['FF'])
        self.src_indx = np.where(grid.x > river_xi[-1] - 1e-6)[0][0]
        self.src_indy = np.where(grid.y > river_yi[-1] - 1e-6)[0][0]

        # source discharge to 2 cells (hardcoded river channel width)
        self.l_src_max = self.Qp / (2 * (grid.w * 1e3)**2)  # m/s

        self.mask_offshore = np.zeros((grid.nx, grid.ny), dtype=bool)
        for j in range(grid.ny):
            self.mask_offshore[grid.x >= src.SS_char["x"][j], j] = True

        self.FFfootprint_t = np.zeros((grid.nx, grid.ny))
        self.FFfootprint_hmax = np.zeros((grid.nx, grid.ny))

        self.c = .5
        self.t = 0
        self.k_movie = 0

        if movie['create'] and not os.path.exists(movie['path']):
            os.makedirs(movie['path'])

    def __iter__(self):
        self.t = 0
        return self

    def __next__(self):
        if self.t >= self.tmax:
            raise StopIteration

        t = self.t

        if t % 3600 == 0: 
            print(t/3600, "hr /", self.tmax/3600, end = '\r', flush = True)

        FF = self.FFfootprint_t

        # source input (two cells)
        FF[self.src_indx, self.src_indy]   = self.l_src_max
        FF[self.src_indx, self.src_indy-1] = self.l_src_max

        l = self.z + FF

        # slopes (Mignan 2024 fig. 4.5)
        dl_a = np.pad((l[:,1:] - l[:,:-1]), [(0,0),(1,0)])  # left
        dl_b = np.pad((l[:-1,:] - l[1:,:]), [(0,1),(0,0)])  # bottom
        dl_c = np.pad((l[:,:-1] - l[:,1:]), [(0,0),(0,1)])  # right
        dl_d = np.pad((l[1:,:] - l[:-1,:]), [(1,0),(0,0)])  # top

        dl_all = np.stack((dl_a, dl_b, dl_c, dl_d))
        dl_all[dl_all < 0] = 0
        dl_sum = np.sum(dl_all, axis=0)
        dl_sum[dl_sum == 0] = np.inf

        weight_dl = dl_all / dl_sum

        lmax = np.minimum(FF, np.amax(self.c * dl_all, axis=0))
        lmov_all = weight_dl * lmax

        # incoming fluxes
        lIN_a = np.pad(lmov_all[0,:,1:], [(0,0),(0,1)])
        lIN_b = np.pad(lmov_all[1,:-1,:], [(1,0),(0,0)])
        lIN_c = np.pad(lmov_all[2,:,:-1], [(0,0),(1,0)])
        lIN_d = np.pad(lmov_all[3,1:,:], [(0,1),(0,0)])

        lOUT_0 = np.sum(lmov_all, axis=0)

        # update state
        FF = FF + (lIN_a + lIN_b + lIN_c + lIN_d) - lOUT_0
        FF = FF * self.mask_offshore

        self.FFfootprint_t = FF
        self.FFfootprint_hmax = np.maximum(self.FFfootprint_hmax, FF)

        if self.movie['create']:
            dt = 60.*10 # snapshot only every 10 minutes
            t_min = t % dt
            if t_min == 0 and t / dt >= self.movie['tmin']:
                self._save_frame()

        self.t += 1
        return self.FFfootprint_t

    def _save_frame(self):
        k = self.k_movie

        plt.rcParams['font.size'] = '20'
        fig, ax = plt.subplots(1, 1, figsize=(10,10), facecolor='white')

        h_plot = copy.copy(self.FFfootprint_t)
        h_plot[h_plot == 0] = np.nan
        ax.contourf(self.grid.xx, self.grid.yy, h_plot, cmap = 'Blues', alpha = .9, vmin = 0, vmax = 5)
        ax.contourf(self.grid.xx, self.grid.yy, GenMR_env.ls.hillshade(self.z, vert_exag = .1),
                    cmap = 'gray', alpha = .1)

        ax.set_xlabel('$x$ (km)')
        ax.set_ylabel('$y$ (km)')
        ax.set_aspect(1)
        ax.set_xlim(self.movie['xmin'], self.movie['xmax'])
        ax.set_ylim(self.movie['ymin'], self.movie['ymax'])
        ax.set_title(f'FF iteration $t=${self.t/60}min', pad=10)

        k_str = f"{k:04d}"
        fig.savefig(self.movie['path'] + f'iter{k_str}.png', dpi=300, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
        self.k_movie += 1

    def run(self):
        for _ in self:
            pass

    def result(self):
        return self.FFfootprint_hmax

    def write_gif(self):
        fd = self.movie['path']
        filenames = sorted([f for f in os.listdir(fd) if f.startswith('iter')])
        img = [imageio.imread(fd + f) for f in filenames]
        if not os.path.exists('movs'):
            os.makedirs('movs')
        imageio.mimsave(f"movs/FF_CA_xrg{self.movie['xmin']}_{self.movie['xmax']}_yrg{self.movie['ymin']}_{self.movie['ymax']}.gif", \
                        img, duration = 500, loop = 0)


## LANDSLIDE CASE ##
class CellularAutomaton_LS:
    '''
    '''
    def __init__(self, soilLayer, wetness, movie, kmax = 20):
        '''
        '''
        self.soil = copy.deepcopy(soilLayer)
        self.grid = self.soil.grid
        self.z = self.soil.topo.z.copy()
        self.h0 = self.soil.h.copy()
        self.h = self.soil.h.copy()
        self.wetness = wetness
        self.kmax = kmax
        movie['path'] = 'figs/LS_CA_frames/'
        self.movie = movie

        LS_footprint = np.zeros((self.grid.nx, self.grid.ny))
        FS = GenMR_env.calc_FS(self.soil.topo.slope, self.h, self.wetness, self.soil.par)
        FS_state = GenMR_env.get_FS_state(FS)
        LS_footprint[FS_state == 2] = 1     # initiates LS where slope is unstable
        nx, ny = int(self.grid.xbuffer/self.grid.w), int(self.grid.ybuffer/self.grid.w)
        LS_footprint = GenMR_utils.zero_boundary_2d(LS_footprint, nx, ny)    # no LS in buffer zone
        self.LS_footprint = LS_footprint

        self.LS_footprint_hmax = np.zeros((self.grid.nx, self.grid.ny))
        self.k = 1

        if movie['create'] and not os.path.exists(movie['path']):
            os.makedirs(movie['path'])

    def __iter__(self):
        return self

    def __next__(self):
        if self.k > self.kmax:
            raise StopIteration

        print(f'\riteration {self.k} / {self.kmax}', end = '', flush = True)

        LS_footprint = self.LS_footprint
        h  = self.h
        z  = self.z
        h0 = self.h0
        grid = self.grid
        w = self.wetness

        # select movable cells
        indmov = np.where(np.logical_and(LS_footprint == 1, h > 0))

        for kk in range(len(indmov[0])):
            i, j = indmov[0][kk], indmov[1][kk]

            # slope and aspect
            z_pad = np.pad(z, 1, 'edge')
            tan_slope, aspect = GenMR_env.calc_topo_attributes(z_pad[i:i+3, j:j+3], grid.w)
            slope = np.degrees(np.arctan(tan_slope[1,1]))
            steepestdir = int(np.round(aspect[1,1] * 7 / 360))

            slope_stable = self._calc_stableSlope(h[i,j], w[i,j], self.soil.par)

            if slope > slope_stable:
                # neighbor indices
                i_nbor, j_nbor = GenMR_utils.get_neighborhood_ind(i, j, (grid.nx, grid.ny), 1, method='Moore')
                steepestdir_rfmt = GenMR_utils.get_ind_aspect2moore(steepestdir)

                i1 = i_nbor[steepestdir_rfmt]
                j1 = j_nbor[steepestdir_rfmt]

                if steepestdir % 2 == 0:
                    dh_stable = grid.w*1e3 * np.tan(np.radians(slope_stable))
                    dz = (grid.w*1e3 * np.tan(np.radians(slope)) - dh_stable)/2
                else:
                    dh_stable = grid.w*1e3 * np.sqrt(2) * np.tan(np.radians(slope_stable))
                    dz = (grid.w*1e3 * np.sqrt(2) * np.tan(np.radians(slope)) - dh_stable)/2

                dz = min(dz, h[i,j])

                z[i,j]  -= dz
                z[i1,j1] += dz
                h[i,j]  -= dz
                h[i1,j1] += dz

                LS_footprint[i1,j1] = 1

        self.LS_footprint_hmax = np.maximum(self.LS_footprint_hmax, h - h0)

        if self.movie['create']:
            self._save_frame()

        self.k += 1
        return self  # return self so the user can inspect state if needed

    def _calc_stableSlope(self, h, w, par):
        slope_i = np.arange(1, 50, .1)
        FS_i = GenMR_env.calc_FS(slope_i, h, w, par)
        slope_stable = slope_i[FS_i > 1.5][-1]
        return slope_stable

    def _save_frame(self):
        k = self.k

        plt.rcParams['font.size'] = '20'
        _, ax = plt.subplots(1, 1, figsize=(10,10), facecolor='white')

        h_plot = GenMR_utils.col_state_h(self.h, self.h0)
        ax.contourf(self.grid.xx, self.grid.yy, h_plot, cmap = GenMR_utils.col_h, vmin = 0, vmax = 5)
        ax.contourf(self.grid.xx, self.grid.yy, GenMR_env.ls.hillshade(self.z, vert_exag = .1),
                    cmap = 'gray', alpha = .1)

        ax.set_xlabel('$x$ (km)')
        ax.set_ylabel('$y$ (km)')
        ax.set_aspect(1)
        ax.set_xlim(self.movie['xmin'], self.movie['xmax'])
        ax.set_ylim(self.movie['ymin'], self.movie['ymax'])
        ax.set_title(f'LS iteration {k}', pad=10)

        k_str = f"{k:02d}"
        plt.savefig(self.movie['path'] + f'iter{k_str}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        for _ in self:
            pass

    def result(self):
        return self.LS_footprint_hmax

    def write_gif(self):
        fd = self.movie['path']
        filenames = sorted([f for f in os.listdir(fd) if f.startswith('iter')])
        img = [imageio.imread(fd + f) for f in filenames]
        if not os.path.exists('movs'):
            os.makedirs('movs')
        imageio.mimsave(f"movs/LS_CA_xrg{self.movie['xmin']}_{self.movie['xmax']}_yrg{self.movie['ymin']}_{self.movie['ymax']}.gif", \
                        img, duration=500, loop=0)







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

    _, ax = plt.subplots(1, 2, figsize=(10,4))
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
        h_ai =ax[0].scatter(src.AI_char['x'], src.AI_char['y'], color = GenMR_utils.col_peril('AI'), s=30, marker = '+', clip_on = False)
        handles.append(h_ai)
        labels.append('Impact site: Asteroid impact (AI)')
    if 'Ex' in src.par['perils']:
        h_ex = ax[0].scatter(src.par['Ex']['x'], src.par['Ex']['y'], color = GenMR_utils.col_peril('Ex'), s=90, marker = '+', clip_on = False)
        handles.append(h_ex)
        labels.append('Harbor refinery: Explosion (Ex)')


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
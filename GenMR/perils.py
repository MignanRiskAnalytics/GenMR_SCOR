"""
GenMR Peril Implementation
==========================

This module offers functions for defining and implementing perils within the GenMR digital template.
It provides tools for comprehensive catastrophe risk modeling, including event loss table (ELT) creation, 
as well as intensity and loss footprint calculations.

Peril models (v1.1.1)
---------------------
* AI: Asteroid impact
* EQ: Earthquake
* FF: Fluvial flood
* LS: Landslide
* RS: Rainstorm
* SS: Storm surge
* TC: Tropical cyclone
* VE: Volcanic eruption
* WF: Wildfire
* Ex: Explosion (industrial)

Peril models (v1.1.2)
---------------------
* CS: Convective storm
* Dr: Drought - in construction
* HW: Heatwave
* Li: Lightning
* To: Tornado
* WS: Windstorm

Planned peril models (v1.1.2)
-----------------------------
* PI: Pest infestation
* BO: Blackout
* BI: Business interruption
* Sf: Public service failure
* SU: Social unrest


:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.1.2
:Date: 2026-01-28
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import os
import copy
import re
import warnings
from tqdm import tqdm, trange

from collections import defaultdict

#import matplotlib
#matplotlib.use('Agg')   # avoid kernel crash (during flood modelling), still needed? to check

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import LineString, Polygon

import imageio
from skimage import measure

import scipy
from scipy.stats import beta, norm

from GenMR import environment as GenMR_env
from GenMR import dynamics as GenMR_dynamics
from GenMR import utils as GenMR_utils



#################
# PERIL SOURCES #
#################

def get_peril_evID(evIDs):
    '''
    Extract the peril identifiers from an array of event identifiers.

    Each event identifier is assumed to be a string where the first two 
    characters correspond to the peril type.

    Parameters
    ----------
    evIDs : array-like of str
        Array or list of event identifiers.

    Returns
    -------
    np.ndarray
        Array of peril identifiers corresponding to each event ID.
    '''
    return np.array([evID[:2] for evID in evIDs])


class Src:
    '''
    Define the characteristics of the peril sources.

    This class stores the parameters and derived attributes of different peril sources

    Parameters
    ----------
    par : dict
        Dictionary containing the peril source parameters, with nested keys.
            
    grid : RasterGrid
        Instance of RasterGrid class defining the computational domain.
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
        par['grid_A_km2'] = (grid.xmax_nobuffer - grid.xmin_nobuffer) * (grid.ymax_nobuffer - grid.ymin_nobuffer)    # active domain
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
        Generate the characteristics of asteroid impact (AI) sources, if defined.

        Returns
        -------
        dict
            Dictionary containing:
            - 'srcID' (list of str): Unique identifiers for each asteroid impact source.
            - 'x' (ndarray): x-coordinates of asteroid impact sources.
            - 'y' (ndarray): y-coordinates of asteroid impact sources.
        '''
        if 'AI' in self.par['perils']:
            srcID = [self.par['AI']['object'] + str(i + 1) for i in range(self.par['AI']['N'])]
            src_xi, src_yi = self._gen_stochsrc_points(self.par['AI']['N'], self.grid, rdm_seed = self.par['rdm_seed'])
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            warnings.warn('No AI source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}

    ## EQ characteristics ##
    @property
    def EQ_char(self):
        '''
        Return the characteristics of earthquake (EQ) sources.

        This property calls the helper function `_get_char_srcEQline` to compute
        the coordinates and identifiers of EQ sources if they exist. Results
        are cached to avoid repeated computation.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'srcID' (list of str): Unique source identifiers for each earthquake.
            - 'x' (ndarray): x-coordinates of the source points.
            - 'y' (ndarray): y-coordinates of the source points.
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
    
    ## FF characteristics ##
    @property
    def FF_char(self):
        '''
        Generate characteristics of flood (FF) sources.

        This property returns the coordinates and identifiers of riverine flood sources
        defined in the model parameters.

        Returns
        -------
        dict
            Dictionary containing:
            - 'srcID' (ndarray of str): Unique identifiers for each flood source.
            - 'x' (ndarray of float): x-coordinates of the flood source points.
            - 'y' (ndarray of float): y-coordinates of the flood source points.
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
        Generate characteristics of tropical cyclone (TC) sources.

        This property returns the coordinates and identifiers of tropical cyclone tracks
        defined in the model parameters. The track points are generated stochastically 
        according to the number of cyclones (`N`), number of points per track (`npt`), 
        maximum deviation (`max_dev`), and random seed (`rdm_seed`). Each TC source is 
        assigned a unique identifier.

        Returns
        -------
        dict
            Dictionary containing:
            - 'srcID' (ndarray of str): Unique identifiers for each tropical cyclone source.
            - 'x' (ndarray of float): x-coordinates of the TC track points.
            - 'y' (ndarray of float): y-coordinates of the TC track points.
        '''
        if 'TC' in self.par['perils']:
            src_ind, src_xi, src_yi = self._gen_stochsrc_TCtracks(self.par['TC']['N'], self.grid, self.par['TC']['npt'], self.par['TC']['max_dev'], self.par['rdm_seed'])
            srcID = np.char.add(self.par['TC']['object'], src_ind.astype(str))
            return {'srcID': srcID, 'x': src_xi, 'y': src_yi}
        else:
            warnings.warn('No TC source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}

    ## To characteristics ##
    @property
    def To_char(self):
        '''
        Generate the characteristics of tornado (To) sources, if defined.

        Returns
        -------
        dict
            Dictionary containing:
            - 'srcID' (list of str): Unique identifiers for each asteroid impact source.
            - 'x0' (ndarray): x-coordinates of tornado source initiation points.
            - 'y0' (ndarray): y-coordinates of tornado source initiation points.
        '''
        if 'To' in self.par['perils']:
            srcID = [self.par['To']['object'] + str(i + 1) for i in range(self.par['To']['N'])]
            src_xi, src_yi = self._gen_stochsrc_points(self.par['To']['N'], self.grid, xmax_bypass=self.par['To']['seed_xmax'], rdm_seed=self.par['rdm_seed'])
            return {'srcID': srcID, 'x0': src_xi, 'y0': src_yi}
        else:
            warnings.warn('No To source initiated in source parameter list')
            return {'srcID': [], 'x0': [], 'y0': []}

    ## VE characteristics ##
    @property
    def VE_char(self):
        '''
        Retrieve the characteristics of volcanic eruption (VE) sources.

        Returns
        -------
        dict
            Dictionary containing:
            - 'srcID' (ndarray of str): Unique identifiers for each volcanic eruption source.
            - 'x' (ndarray of float): x-coordinates of the flood source points.
            - 'y' (ndarray of float): y-coordinates of the flood source points.
        '''
        if 'VE' in self.par['perils']:
            src_ind = np.arange(self.par['VE']['N']) + 1
            srcID = np.char.add(self.par['VE']['object'], src_ind.astype(str))
            return {'srcID': srcID, 'x': self.par['VE']['x'], 'y': self.par['VE']['y']}
        else:
            warnings.warn('No VE source initiated in source parameter list')
            return {'srcID': [], 'x': [], 'y': []}


    def _gen_stochsrc_points(self, N, grid, xmax_bypass = np.nan, rdm_seed = None):
        '''
        Generate N random points uniformly distributed over the active domain of the grid.

        Parameters
        ----------
        N : int
            Number of random points to generate.
        grid : object
            A grid object with attributes `xmin`, `xmax`, `ymin`, `ymax` defining the spatial domain.
        rdm_seed : int, optional
            Seed for the random number generator, for reproducibility. Default is None.

        Returns
        -------
        tuple of ndarray
            - x_rdm (ndarray): x-coordinates of the random points.
            - y_rdm (ndarray): y-coordinates of the random points.
        '''
        if rdm_seed is not None:
            np.random.seed(rdm_seed)

        if np.isnan(xmax_bypass):
            x_rdm = grid.xmin_nobuffer + np.random.random(N) * (grid.xmax_nobuffer - grid.xmin_nobuffer)
        else:
            x_rdm = grid.xmin_nobuffer + np.random.random(N) * (xmax_bypass - grid.xmin_nobuffer)
        y_rdm = grid.ymin_nobuffer + np.random.random(N) * (grid.ymax_nobuffer - grid.ymin_nobuffer)
        return x_rdm, y_rdm

    def _get_char_srcEQline(self, par):
        '''
        Calculate coordinates and properties of earthquake fault sources.

        This function interpolates points along fault segments using the
        specified bin size and computes segment strikes and lengths. It also
        estimates the maximum earthquake magnitude from the total fault length.

        Parameters
        ----------
        par : dict
            Dictionary describing fault sources with the following keys:
            - 'x' (list of list of float): x-coordinates of fault segment vertices.
            - 'y' (list of list of float): y-coordinates of fault segment vertices.
            - 'w_km' (list of float): Width of fault in km (optional for some calculations).
            - 'dip_deg' (list of float): Dip angles in degrees.
            - 'z_km' (list of float): Depth of fault in km.
            - 'mec' (list): Moment efficiency coefficient (optional).
            - 'bin_km' (float): Resolution distance along the fault for interpolation.

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - 'srcID' (ndarray of str): Unique identifiers for each source.
            - 'x' (ndarray of float): x-coordinates of interpolated points along faults.
            - 'y' (ndarray of float): y-coordinates of interpolated points along faults.
            - 'fltID' (ndarray of int): Fault source index for each point.
            - 'srcL' (ndarray of float): Total length of each fault source (km).
            - 'srcMmax' (ndarray of float): Estimated maximum magnitude for each source.
            - 'segID' (ndarray of int): Segment index for each interpolated point.
            - 'strike' (ndarray of float): Strike angle (degrees) of each segment.
            - 'segL' (ndarray of float): Length of each segment (km).
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

    def _gen_stochsrc_TCtracks(self, N, grid, npt, max_deviation, rdm_seed = None):
        '''
        Generate coordinates for N stochastic tropical cyclone tracks.

        Each track is approximated as a straight line with `npt` points, but 
        a random deviation along the y-axis (bounded by `max_deviation`) is added 
        to simulate track variability.

        Parameters
        ----------
        N : int
            Number of tropical cyclone tracks to generate.
        grid : RasterGrid
            Grid object defining the spatial domain with attributes `xmin`, `xmax`, `ymin`, `ymax`.
        npt : int
            Number of points along each track.
        max_deviation : float
            Maximum deviation along the y-axis from the straight line.
        rdm_seed : int, optional
            Random seed for reproducibility (default is None).

        Returns
        -------
        ind : ndarray of int
            Array of source indices repeated for each point along the track.
        x : ndarray of float
            x-coordinates of all track points.
        y : ndarray of float
            y-coordinates of all track points with random deviations applied.
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

    def __repr__(self):
        return 'Src({})'.format(self.par)



################
# SCALING LAWS #
################

def calc_EQ_length2magnitude(L):
    '''
    Convert earthquake rupture length to moment magnitude.

    Given the rupture length of an earthquake fault (km), this 
    function calculates the corresponding earthquake magnitude using the 
    empirical relation from Wells and Coppersmith (1994:fig. 2.6b).

    Parameters
    ----------
    L : float or ndarray
        Earthquake rupture length in kilometers.

    Returns
    -------
    M : float or ndarray
        Calculated moment magnitude, rounded to one decimal place.
    '''
    c1, c2 = [5., 1.22]     # reverse case, Fig. 2.6b, Wells and Coppersmith (1994)
    M = c1 + c2 * np.log10(L)
    return np.round(M, decimals = 1)

def calc_EQ_magnitude2length(M):
    '''
    Convert earthquake magnitude to rupture length.

    Given a moment magnitude, this function calculates the expected 
    rupture length (in kilometers) using the empirical relation from 
    Wells and Coppersmith (1994:fig. 2.6b). Useful for floating rupture 
    computations.

    Parameters
    ----------
    M : float or ndarray
        Earthquake moment magnitude.

    Returns
    -------
    L : float or ndarray
        Estimated rupture length in kilometers.
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
    Generate an array of evenly spaced values within a specified interval.

    The spacing can be either linear or logarithmic, depending on the `scale` parameter.

    Parameters
    ----------
    xmin : float
        The minimum value of the interval.
    xmax : float
        The maximum value of the interval.
    xbin : float
        The increment (step size) between consecutive values.
        For logarithmic scale, this represents the step in log10 space.
    scale : {'linear', 'log'}
        Type of spacing. 'linear' for evenly spaced linear values, 
        'log' for logarithmically spaced values.

    Returns
    -------
    xi : ndarray
        Array of values from `xmin` to `xmax` spaced according to `xbin` and `scale`.
    '''
    if scale == 'linear':
        xi = np.arange(xmin, xmax + xbin, xbin)
    if scale == 'log':
        xi = 10**np.arange(np.log10(xmin), np.log10(xmax) + xbin, xbin)
    return xi

def calc_Lbd_powerlaw(S, a, b):
    '''
    Calculate the cumulative rate of events following a power-law distribution.

    This corresponds to Mignan (2024:eq. 2.38), where the cumulative occurrence rate 
    `Lbd` is a function of event size `S`.

    Parameters
    ----------
    S : float or ndarray
        Event size (e.g., magnitude, intensity, or volume).
    a : float
        Intercept parameter in the power-law relation (log10 scale).
    b : float
        Exponent parameter controlling the slope of the power-law in log10 scale.

    Returns
    -------
    Lbd : float or ndarray
        Cumulative rate corresponding to event size `S`.
    '''
    Lbd = 10**(a - b * np.log10(S))
    return Lbd

def calc_Lbd_exponential(S, a, b):
    '''
    Calculate the cumulative rate of events following an exponential law.

    This corresponds to Mignan (2024:eq. 2.39), where the cumulative occurrence rate 
    `Lbd` decreases exponentially with event size `S`.

    Parameters
    ----------
    S : float or ndarray
        Event size (e.g., magnitude, intensity, or volume).
    a : float
        Intercept parameter in the exponential relation (log10 scale).
    b : float
        Exponential decay parameter controlling how fast the rate decreases with `S`.

    Returns
    -------
    Lbd : float or ndarray
        Cumulative rate corresponding to event size `S`.
    '''
    Lbd = 10**(a - b * S)
    return Lbd

def calc_Lbd_GPD(S, mu, xi, sigma, Lbdmin):
    '''
    Calculate the cumulative rate of events following a Generalised Pareto Distribution (GPD).

    This corresponds to Mignan (2024:eq. 2.50), where the cumulative occurrence rate `Lbd` is 
    derived from the GPD parameters.

    Parameters
    ----------
    S : float or ndarray
        Event size (e.g., magnitude, intensity, or volume).
    mu : float
        Threshold parameter (location) of the GPD.
    xi : float
        Shape parameter of the GPD.
    sigma : float
        Scale parameter of the GPD.
    Lbdmin : float
        Minimum cumulative rate (normalization factor).

    Returns
    -------
    Lbd : float or ndarray
        Cumulative rate corresponding to event size `S`.
    '''
    Lbd = Lbdmin * (1 + xi * (S - mu) / sigma)**(-1 / xi)
    return Lbd


## event set ##
class EventSetGenerator:
    '''
    Generates stochastic event set based on source and size distribution parameters.

    Attributes
    ----------
    src : object
        Source object (e.g., instance of `Src`) containing hazard source locations and characteristics.
    sizeDistr : object
        Size distribution object defining the probability distribution of event magnitudes, volumes, or intensities.
    utils : module
        Utility module (e.g., `GenMR_utils`) providing helper functions.
    ev_stoch : pandas.DataFrame
        DataFrame storing stochastic event realizations with columns:
        - 'ID': unique event identifier
        - 'srcID': source identifier
        - 'evID': event identifier
        - 'S': event size
        - 'lbd': rate or frequency
    ev_char : pandas.DataFrame
        DataFrame storing event characteristics with columns:
        - 'evID': event identifier
        - 'x': x-coordinate of event
        - 'y': y-coordinate of event
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

            ## SPECIAL EVENTS ##
            if ID in self.sizeDistr['special']:
                if ID == 'CS':
                    ev_stoch_To = self.ev_stoch[self.ev_stoch['ID'] == 'To'].reset_index(drop = True)
                    evID = 'CS_to' + ev_stoch_To['evID'].values
                    Si_vec = np.repeat(self.src.par['CS']['S'], self.sizeDistr['CS']['Nstoch'])
                    lbdi = np.repeat(np.nan, self.sizeDistr['CS']['Nstoch'])
                    self.ev_stoch = pd.concat([
                        self.ev_stoch,
                        pd.DataFrame({'ID': np.repeat('CS', self.sizeDistr['CS']['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})
                    ], ignore_index=True)
                if ID == 'Dr':
                    evID = [f"{ID}{i+1}" for i in range(self.sizeDistr['Dr']['Nstoch'])]
                    Si_vec = self.src.par['Dr']['Si_mo']
                    lbdi = np.repeat(np.nan, self.sizeDistr['Dr']['Nstoch'])
                    self.ev_stoch = pd.concat([
                        self.ev_stoch,
                        pd.DataFrame({'ID': np.repeat('Dr', self.sizeDistr['Dr']['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})
                    ], ignore_index=True)
                if ID == 'HW':
                    evID = [f"{ID}{i+1}" for i in range(self.sizeDistr['HW']['Nstoch'])]
                    Si_vec = self.src.par['HW']['Si_da']
                    lbdi = np.repeat(np.nan, self.sizeDistr['HW']['Nstoch'])
                    self.ev_stoch = pd.concat([
                        self.ev_stoch,
                        pd.DataFrame({'ID': np.repeat('HW', self.sizeDistr['HW']['Nstoch']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})
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
        if self.sizeDistr[ID]['distr'] == 'exponential':
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

        if ID == 'EQ' or ID == 'WF':
            qi = np.linspace(1, 11, Si_n)
            qi /= np.sum(qi)
            qi = np.sort(qi)[::-1]
        else:
            qi = np.repeat(1. / Si_n, Si_n)

        Si_ind_vec = self.utils.partitioning(Si_ind, qi, self.sizeDistr[ID]['Nstoch'])
        Si_vec = Si[Si_ind_vec]

        counts = np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
        counts_safe = np.where(counts == 0, np.nan, counts)  # or np.inf
        wi = 1 / counts_safe
#        wi = 1 / np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
        wi_vec = [wi[Si_ind == i][0] for i in Si_ind_vec]
        return Si_vec, wi_vec

    def _transform_cum2noncum(self, S, par):
        '''
        Transform the rate from cumulative (Lbd) to non-cumulative (lbd) (e.g., Eq. 2.65)
        '''
        if par['Sscale'] == 'linear':
            S_lo = S
            S_hi = S + par['Sbin']
        elif par['Sscale'] == 'log':
            S_lo = S
            S_hi = 10**(np.log10(S) + par['Sbin'])
        if par['distr'] == 'powerlaw':
            Lbd_lo = calc_Lbd_powerlaw(S_lo, par['a'], par['b'])
            Lbd_hi = calc_Lbd_powerlaw(S_hi, par['a'], par['b'])
        if par['distr'] == 'exponential':
            Lbd_lo = calc_Lbd_exponential(S_lo, par['a'], par['b'])
            Lbd_hi = calc_Lbd_exponential(S_hi, par['a'], par['b'])
        if par['distr'] == 'GPD':
            if par['xi'] < 0:
                Smax = par['mu'] - par['sigma'] / par['xi']  # xi < 0
                if np.max(S_hi) > Smax:
                    print(f"WARNING: GPD upper endpoint exceeded: S_hi={np.max(S_hi):.3g} > Smax={Smax:.3g}")
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
        elif ID == 'To':
            line_coord = self._get_Toline_highres(evID, Si_vec, self.src)
            ev_ID, ev_x, ev_y = line_coord['evID'], line_coord['x'], line_coord['y']
            self.srcIDs = np.append(self.srcIDs, np.unique(self.src.To_char['srcID']))
        elif ID == 'VE':
            ev_ID, ev_x, ev_y = evID, np.repeat(self.src.VE_char['x'], self.sizeDistr['VE']['Nstoch']), np.repeat(self.src.VE_char['y'], self.sizeDistr['VE']['Nstoch'])
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.VE_char['srcID'], self.sizeDistr['VE']['Nstoch']))
        elif ID == 'WF':
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['WF']['object'], self.sizeDistr['WF']['Nstoch']))
        elif ID == 'WS':
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['WS']['object'], self.sizeDistr['WS']['Nstoch']))

        elif ID == 'CS':
            # retrieve tornado path lines
            lines = defaultdict(list)
            ev_char_To = self.ev_char[self.ev_char['evID'].str[:2] == 'To'].reset_index(drop = True)
            for evID_To, x, y in zip(ev_char_To['evID'], ev_char_To['x'], ev_char_To['y']):
                lines[evID_To].append((x, y))
            ev_ID_list, ev_x_list, ev_y_list = [], [], []
            for i, (evID_To, coords) in enumerate(lines.items()):
                line = LineString(coords)
                x, y = line.buffer(self.src.par['CS']['R_km']).exterior.xy
                ev_ID_list.append(np.repeat(evID[i], len(x)))
                ev_x_list.append(np.asarray(x))
                ev_y_list.append(np.asarray(y))
            ev_ID = np.concatenate(ev_ID_list)
            ev_x = np.concatenate(ev_x_list)
            ev_y = np.concatenate(ev_y_list)
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['CS']['object'], self.sizeDistr['CS']['Nstoch']))
        elif ID == 'Dr':
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['Dr']['object'], self.sizeDistr['Dr']['Nstoch']))
        elif ID == 'HW':
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['HW']['object'], self.sizeDistr['HW']['Nstoch']))

        elif ID == 'FF':
            trigger = self.sizeDistr[ID]['trigger']
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.FF_char['srcID'], self.sizeDistr[trigger]['Nstoch']))
        elif ID == 'Li':
            trigger = self.sizeDistr[ID]['trigger']
            Si_CS = self.ev_stoch[self.ev_stoch['ID'] == 'CS']['S'].values
            pointSet_coord = self._get_LipointSet(evID, Si_CS)
            ev_ID, ev_x, ev_y = pointSet_coord['evID'], pointSet_coord['x'], pointSet_coord['y']
            self.srcIDs = np.append(self.srcIDs, np.repeat(self.src.par['Li']['object'], self.sizeDistr[trigger]['Nstoch']))
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

    def _get_Toline_highres(self, evIDi, Si, src):
        '''
        '''
        Nstoch = len(evIDi)
        theta_rdm = np.random.uniform(0., 2*np.pi, Nstoch)
        u = np.random.uniform(0, 1, size = Nstoch)
        dxy_step = src.par['To']['bin_km']

        x_hires = np.array([])
        y_hires = np.array([])
        id_hires = np.array([])
        for i in range(Nstoch):
            if int(Si[i]) == 3:
                # EF-3 length statistics
                Beta_alpha, Beta_beta, Beta_lmax = src.par['To']['L_alpha_beta_Lmax_EF3']
            if int(Si[i]) == 4:
                # EF-4 length statistics
                Beta_alpha, Beta_beta, Beta_lmax = src.par['To']['L_alpha_beta_Lmax_EF4']
            if int(Si[i]) == 5:
                # EF-5 length statistics
                Beta_alpha, Beta_beta, Beta_lmax = src.par['To']['L_alpha_beta_Lmax_EF5']
            Lstoch = beta.ppf(u[i], Beta_alpha, Beta_beta) * Beta_lmax
            Lstoch = max(Lstoch, 2 * dxy_step)
            n_steps = int(np.ceil(Lstoch / dxy_step))
            x0 = src.To_char['x0'][i]
            y0 = src.To_char['y0'][i]
            dx = dxy_step * np.sin(theta_rdm[i])
            dy = dxy_step * np.cos(theta_rdm[i])
            x_hires = np.append(x_hires, x0 + np.arange(n_steps) * dx)
            y_hires = np.append(y_hires, y0 + np.arange(n_steps) * dy)
            id_hires = np.append(id_hires, np.repeat(evIDi[i], n_steps))
        Line_coord = pd.DataFrame({'evID': id_hires, 'x': x_hires, 'y': y_hires})
        return Line_coord

    def _get_LipointSet(self, evIDi, Si_trigger):
        '''
        '''
        lbd_Li_strike, _ = GenMR_dynamics.calc_lbd_CS2Li(Si_trigger, self.src.par['CS']['lat_deg'])    # strikes/min/storm
        CS_char = self.ev_char[self.ev_char['evID'].str[:2] == 'CS']                                   # to get CS footprint
        evIDs_CS = CS_char['evID'].unique()
        To_char = self.ev_char[self.ev_char['evID'].str[:2] == 'To']                                   # to get CS length = To length
        To_L_details = (To_char.groupby('evID').agg(x0=('x', 'first'),y0=('y', 'first'), x1=('x', 'last'), y1=('y', 'last'))
                        .assign(L_km=lambda df: np.hypot(df.x1 - df.x0, df.y1 - df.y0)).reset_index())
        L_km_CS = To_L_details['L_km'].values
        Dt_min_CS = L_km_CS / (self.src.par['CS']['vforward_m/s'] * 1e-3 * 60.)                        # storm duration as path length / speed from start to end
        x_Li, y_Li, evID_Li = [], [], []
        for i, evID_CS in enumerate(evIDs_CS):
            CS_coord = CS_char[CS_char['evID'] == evID_CS][['x', 'y']].values
            footprint_CS = Polygon(CS_coord)
#            N_Li = np.random.poisson(lbd_Li_strike * Dt_min_CS.iloc[i])       # no stochasticity in Tutorial 2 
            N_Li = int(lbd_Li_strike[i] * Dt_min_CS[i])
            pts = GenMR_utils.sample_points_in_polygon(footprint_CS, N_Li)
            x_Li.extend(pts[:, 0])
            y_Li.extend(pts[:, 1])
            evID_Li.extend(np.repeat(evIDi[i], N_Li))
        PointSet_coord = pd.DataFrame({'evID': evID_Li, 'x': x_Li, 'y': y_Li})

        return PointSet_coord


## Dr CASE ## NB: consider moving some of the following functions to atmospheric layer class
def calc_e0(T):
    '''
    Estimate the saturation vapor pressure from the Clausius–Clapeyron equation 
    under typical atmospheric conditions neglecting temperature dependence of the latent heat
    according to the August–Roche–Magnus formula.
    
    Parameters
    ----------
    T : float or array_like
        Mean monthly air temperature (°C), length = 12

    Returns
    -------
    e0 : float or ndarray
        Saturation vapor pressure (kPa)
    '''
    e0_hPa = 6.1094 * np.exp((17.625 * T)/(T + 243.04))    # e.g., Lawrence (2005:eq.6)
    e0_kPa = e0_hPa / 10
    return e0_kPa

def calc_qsat(T_degC, p_kPa):
    '''
    Compute the saturation specific humidity as a function of air temperature
    and ambient pressure.

    The saturation specific humidity is derived from the saturation mixing
    ratio, which follows directly from the ideal gas law applied to moist air.

    Parameters
    ----------
    T_degC : float or array_like
        Air temperature (°C).
    p_kPa : float or array_like
        Ambient atmospheric pressure (kPa).

    Returns
    -------
    q_s : float or ndarray
        Saturation specific humidity (kg/kg), defined as the ratio of the
        mass of water vapor to the total mass of moist air.

    References
    ----------
    Wallace and Hobbs (2006), Atmospheric Science: An Introductory Survey (2nd ed.). Academic Press, 483 pp.
    '''
    e_s = calc_e0(T_degC)               # saturation vapor pressure (kPa), August–Roche–Magnus formula
    Rd_Rv = .622                        # spe. gas constant for dry air / ... for water vapor (eq.3.14)
    w_s = Rd_Rv * e_s / (p_kPa - e_s)   # saturation mixing ratio (eq.3.63)
    q_s = w_s / (1. + w_s)              # just after eq.3.57
    return q_s

def gen_precipitation(Ti, w, par):
    '''
    Compute column-integrated precipitation from large-scale condensation.

    Precipitation is diagnosed by vertically integrating the condensation
    rate induced by upward motion in a moist atmosphere. Condensation is
    computed from the vertical gradient of saturation specific humidity and
    a prescribed vertical velocity profile, and converted to surface
    precipitation using a constant precipitation efficiency.

    Parameters
    ----------
    Ti : array_like
        Near-surface air temperature (°C).

    w : array_like
        Vertical velocity profile (positive upward, m/s).

    par : dict
        Dictionary of physical parameters with required keys:

        lapse_rate : float
            Environmental lapse rate (K/km).
        p0 : float
            Surface pressure (kPa).
        eta_rain : float
            Precipitation efficiency (dimensionless).
        zmax_km : float
            Tropopause altitude (km).

    Returns
    -------
    rain : ndarray
        Daily precipitation rate for each temperature profile (mm/day).
    '''
    z_tropopause_m = np.round(par['zmax_km'], decimals = 1) * 1e3
    zi_m = np.arange(0,z_tropopause_m,100)    # atmosphere profile

    nT, nz = len(Ti), len(zi_m)
    pz = np.zeros((nT,nz))
    qsat = np.zeros((nT,nz))
    rain = np.zeros(nT)
    for i in range(nT):
        pz[i,:] = GenMR_env.EnvLayer_atmo.calc_p_hydrostatic(zi_m, par['lapse_rate'] * 1e-3, Ti[i], par['p0'])
        Tz = GenMR_env.EnvLayer_atmo.calc_T_z(zi_m * 1e-3, Ti[i], lapse_rate = par['lapse_rate'])
        qsat[i,:] = calc_qsat(Tz, pz[i,:])
        dqsdz = np.gradient(qsat[i,:], zi_m)
        C = np.maximum(0., -w * dqsdz)                                   # condensation rate kg/kg/s
        rho = (pz[i,:] * 1e3)/(287.05 * (Tz + 273.15))                   # air density from ideal gas law
        rain[i] = par['eta_rain'] * np.trapz(rho * C, zi_m) *86400       # precipitation mass flux (mm/day)
    return rain

def calc_PET(T_monthly, lat_deg, cloudy = False):  # WARNING: replace with FAO 56 Penman-Monteith equation?
    '''
    Estimate the monthly potential evapotranspiration (PET) according to Thornthwaite (1948),
    which assumes a clear sky. When cloudy = True, 

    Parameters
    ----------
    T_monthly : array_like
        Mean monthly air temperature (°C), length = 12
    lat_deg : float
        Latitude (°)

    Returns
    -------
    PET : ndarray
        Monthly potential evapotranspiration (mm/month)

    References
    ----------
    Thornthwaite (1948), An approach toward a rational classification of climate.
    Geographical Review, 38(1), 55–94.
    '''
    # Cloud impact
    if cloudy:
        corr_cloud = .5
    else:
        corr_cloud = 0.
    
    # Heat index
    I = np.sum((T_monthly / 5.) ** 1.514)                          # p.89, just before eq.9

    # Exponent
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + .49239    # eq.9

    # Day length factor
    lat = np.radians(lat_deg)
    ndays_inMonth = GenMR_utils.get_ndays_inMonth()
    J = np.cumsum(ndays_inMonth) - ndays_inMonth / 2               # Julian day
    decl = .409 * np.sin(2. * np.pi * J / 365. - 1.39)             # solar declination (Allen et al., 1998:24)
    ws = np.arccos(-np.tan(lat) * np.tan(decl))                    # solar hour angle (sunrise eq.)
    L = 24. / np.pi * ws
    
    T_d = T_monthly
    T_d[T_d < 0.] = 0.
    PET_nocloud = 16. * (L / 12.) * (ndays_inMonth / 30.) * (10. * T_d / I) ** a  # eq.10: 16.*(10. * T_d / I) ** a
    PET = (1 - corr_cloud) * PET_nocloud
    return PET

def update_soil_moisture(P, ET, S0, Smax):
    '''
    Monthly soil water balance.

    Parameters
    ----------
    P : ndarray
        Monthly precipitation (mm/month)
    ET : ndarray
        Monthly evapotranspiration (mm/month)
    S0 : float
        Initial soil moisture (mm)
    Smax : float
        Maximum soil water storage (mm)

    Returns
    -------
    S : ndarray
        Soil moisture storage per month (mm)
    '''
    n = len(P)
    S = np.zeros(n)
    S[0] = S0
    for t in range(1, n):
        S[t] = S[t-1] + P[t-1] - ET[t-1]
        if S[t] > Smax:
            S[t] = Smax
        elif S[t] < 0:
            S[t] = 0
    return S

def get_Dr(S_t, Dr_th):
    '''
    Identify drought events and extract their durations and indices.

    A drought is defined as a contiguous sequence of months with S_t < Dr_th.

    Parameters
    ----------
    S_t : ndarray
        Monthly soil moisture (mm), length = 12
    Dr_th : float
        Soil moisture threshold to define drought (mm)

    Returns
    -------
    events : list of tuple
        List of (start_index, end_index) pairs for each drought,
        where indices are inclusive and zero-based.
    durations : list of int
        Durations (in months) of all detected droughts.
    '''
    ind_Dr = S_t < Dr_th
    durations = []
    events = []
    start = None
    for i, dry in enumerate(ind_Dr):
        if dry and start is None:
            start = i
        elif not dry and start is not None:
            length = i - start
            durations.append(length)
            events.append((start, i - 1))
            start = None
    if start is not None:
        length = len(S_t) - start
        durations.append(length)
        events.append((start, len(S_t) - 1))
    return events, durations

def calc_lbd_Dr(par, atmo_par, soil_par, Nsim = int(1e6)):
    '''
    Estimate drought event rates via Monte Carlo simulation.

    This function performs simulations of monthly soil water balance driven by 
    stochastic temperature variability and atmospheric circulation regimes. 
    For each simulated year, it identifies at most one drought event and maps its 
    severity to an upper-tail discretization.

    Parameters
    ----------
    par : dict
        Drought source parameters.

    atmo_par : dict
        Atmospheric parameters from the atmospheric environmental layer.

    soil_par : dict
        Soil parameters from the soil environmental layer.

    Nsim : int, optional
        Number of Monte Carlo simulations (years). Default is 1e6.

    Returns
    -------
    lbdi : ndarray
        Estimated event rates
   '''
    # temperature time series
    moni = np.arange(12)+1
    T0_mo, _, _ = GenMR_env.EnvLayer_atmo.calc_T0_EBCM(par['lat_deg'], moni)   # mean monthly temperature
    DT0_stoch = np.random.normal(0, par['sigmaT_yearly'], Nsim)
    DTadv_stoch = HazardFootprintGenerator.sample_T_advectivemodel(np.zeros(Nsim), par['lat_deg'])
    T0_mo_stoch = T0_mo[:, np.newaxis] + DT0_stoch[np.newaxis, :] + DTadv_stoch[np.newaxis, :]

    # atmopsheric regime: anticyclone vs cyclone
    w = np.where(DTadv_stoch >= 0, atmo_par['vz_subs_asc'][0], atmo_par['vz_subs_asc'][1])
    cloudy = np.where(DTadv_stoch >= 0, False, True)
    z_tropopause = GenMR_env.EnvLayer_atmo.calc_z_tropopause(par['lat_deg'])
    par_rain = {'p0': atmo_par['p0_kPa'], 'lapse_rate': atmo_par['lapse_rate_degC/km'], \
                'eta_rain': atmo_par['eta_rain'], 'zmax_km': z_tropopause}

    Dr_S_list = np.zeros(Nsim)
    for sim in trange(Nsim, desc = 'Simulating droughts'):
        #evapotranspiration from soils
        ET0 = calc_PET(T0_mo_stoch[:,sim], par['lat_deg'], cloudy = cloudy[sim])
        # precipitation
        I_rain = gen_precipitation(T0_mo_stoch[:,sim], w[sim], par_rain)
        # standard bucket / vertical water balance model
        hw_mo = update_soil_moisture(I_rain, ET0, soil_par['hw0_m']*1e3, soil_par['hw_max_m']*1e3)   # (mm)
        # get drought (none if Dr_S empty)
        Dr_ti, Dr_S = get_Dr(hw_mo, soil_par['hw_fc_m'] * par['hw_th'])
        if len(Dr_S) != 0:
            Dr_S_list[sim] = Dr_S[0]  # only one possible per year by construction
    
    # retrieve rates
    Si = par['Si_mo']
    S_map = pd.DataFrame({'S_raw': Dr_S_list[Dr_S_list > 0.]})
    S_map['S'] = S_map['S_raw'].apply(lambda s: GenMR_utils.map2upperTail(s, Si))
    nS = len(Si)
    lbdi = np.zeros(nS)
    for i in range(nS):
        lbdi[i] = np.sum(S_map['S'] == Si[i]) / Nsim
    return lbdi



#####################
# HAZARD FOOTPRINTS #
#####################

class HazardFootprintGenerator:
    '''
    Generate hazard footprints for a given stochastic event set.

    This class takes a stochastic set of hazard events, their characteristics, the hazard source
    information, and the topography to generate hazard footprints (e.g., spatial distribution 
    of hazard intensity). The footprints are stored in a catalog for further analysis or visualization.

    Parameters
    ----------
    stochset : pandas.DataFrame
        DataFrame containing the stochastic event set with event IDs, source IDs, event sizes, 
        and occurrence rates.
    evchar : pandas.DataFrame
        DataFrame with event characteristics, including event coordinates and any other metadata.
    src : object
        Source object containing hazard source information (e.g., locations, type, geometry).
    topo_z : ndarray
        2D array representing the topographic elevation grid (used for footprint modeling).
    '''
    def __init__(self, stochset, evchar, src, topo_z, atmoLayer, force_recompute = False):
        self.stochset = stochset
        self.evchar = evchar
        self.src = src
        self.topo_z = topo_z
        self.atmoLayer = atmoLayer
        self.catalog_hazFootprints = {}
        self.force_recompute = force_recompute
        self.cache_dir = 'io/cache_thresholdHazFootprints'
        os.makedirs(self.cache_dir, exist_ok = True)
        self.rate_HW = np.full(len(src.par['HW']['Si_da']), np.nan)

    def _cache_path(self, evID):
        return os.path.join(self.cache_dir, f'{evID}.npy')

    ## SIMPLE ANALYTICAL EXPRESSIONS ##
    @staticmethod
    def calc_I_shaking_ms2(S, r):
        '''
        Calculate peak ground acceleration (PGA) in m/s² from earthquake magnitude and distance.

        Parameters
        ----------
        S : float
            Earthquake magnitude.
        r : float
            Distance from the source to the site (km).

        Returns
        -------
        float
            Peak ground acceleration (m/s²).
        '''
        PGA_g = 10 ** (-1.34 + 0.23 * S - np.log10(r))
        g_earth = 9.81
        return PGA_g * g_earth

    @staticmethod
    def calc_I_blast_kPa(S, r):
        '''
        Calculate blast overpressure in kPa from an explosive event.

        Parameters
        ----------
        S : float
            Explosive yield in kilotons TNT equivalent.
        r : float
            Distance from the explosion center (km).

        Returns
        -------
        float
            Blast overpressure (kPa).
        '''
        Z = r * 1e3 / (S * 1e6) ** (1 / 3)          # size = energy in kton TNT
        return 1772 / Z**3 - 114 / Z**2 + 108 / Z

    @staticmethod
    def calc_I_ash_m(S, r):
        '''
        Calculate volcanic ash deposit thickness at a distance from the eruption.

        The model assumes an exponential decay of deposit thickness with distance,
        calibrated from historical eruptions (e.g., Mt. St. Helens 1980).

        Parameters
        ----------
        S : float
            Erupted volume in km³.
        r : float
            Distance from the volcanic vent (km).

        Returns
        -------
        float
            Ash deposit thickness (m).
        '''
        # assumes h0 proportional to V - e.g h0 = 1e-3 km for V=3 km3 (1980 Mt. St. Helens)
        h0 = 1e-3 / 3 * S                           # size = volume in m3
        r_half = np.sqrt(S * np.log(2)**2 / (2 * np.pi * h0))
        h_m = (h0 * np.exp(-np.log(2) * r / r_half)) * 1e3
        return h_m


    ## TROPICAL CYCLONE FUNCTIONS ##
    @staticmethod
    def calc4TC_I_v_ms(S, r, par):
        '''
        Calculate the maximum wind speed (m/s) of a tropical cyclone at a given distance.

        This model accounts for atmospheric density, Coriolis effect, and storm parameters
        following the approach in Mignan (2024:fig. 2.19).

        Parameters
        ----------
        S : float
            Windspeed (m/s) at point along track.
        r : float
            Distance from the storm center (km).
        par : dict
            Dictionary of cyclone parameters, including:
            - 'lat_deg' : Latitude in degrees
            - 'pn_mbar' : Ambient pressure in mbar
            - 'B_Holland' : Holland B parameter for wind profile

        Returns
        -------
        float
            Maximum wind speed in meters per second (m/s) at distance r from the storm center.
        '''
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
        '''
        Combine forward and tangential wind components to compute total wind field.

        This function calculates the total wind speed and its x and y components at each 
        grid point, accounting for the forward motion of the storm along its track and 
        the tangential wind profile.

        Parameters
        ----------
        vf : float
            Forward speed of the storm (m/s).
        vtan : ndarray
            Tangential wind speed array (m/s) at each grid point relative to storm center.
        track_x : ndarray
            X-coordinates of the storm track (km).
        track_y : ndarray
            Y-coordinates of the storm track (km).
        grid : object
            Grid object containing `xx` and `yy` 2D arrays for spatial coordinates.
        t_i : int
            Current time step index along the storm track.

        Returns
        -------
        vtot : ndarray
            Total wind speed at each grid point (m/s).
        vtot_x : ndarray
            X-component of total wind speed (m/s).
        vtot_y : ndarray
            Y-component of total wind speed (m/s).
        vtan_x : ndarray
            X-component of tangential wind (m/s).
        vtan_y : ndarray
            Y-component of tangential wind (m/s).
        '''
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
        with np.errstate(invalid='ignore', divide='ignore'):
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
        '''
        Calculate along-track hazard intensity for tropical cyclones (TC) 
        accounting for decay over land after coastal landfall.

        The method computes the intensity of each TC event along its track,
        reducing the intensity after landfall based on distance from coast.

        Parameters
        ----------
        stochset : pandas.DataFrame
            Stochastic event set containing columns ['ID', 'evID', 'S', ...].
        src : object
            Source object containing storm characteristics, including `SS_char` 
            (storm source coordinates) and `par` parameters.
        Track_coord : pandas.DataFrame
            DataFrame containing storm track coordinates with columns ['evID', 'x', 'y'].

        Returns
        -------
        S_alongtrack : dict
            Dictionary mapping each TC event ID to an array of along-track 
            maximum intensity values (same length as track points).
        '''
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


    ## TORNADO FUNCTIONS ##
    @staticmethod
    def calc_lateral_distance_signed(xx, yy, x, y):
        '''
        Compute the signed perpendicular distance to a line segment.

        This function evaluates the signed lateral (cross-track) distance from
        each point of a two-dimensional mesh grid to a high-resolution line.
        The distance is defined as the perpendicular distance to the locally
        closest line segment, with the sign determined by the right-hand normal
        to the segment direction.

        Distances are only defined where an orthogonal projection of the grid
        point falls within the extent of at least one line segment. Points
        located beyond the tips of the line are assigned np.nan.

        Parameters
        ----------
        xx : ndarray
            Two-dimensional array of x-coordinates (mesh grid).
        yy : ndarray
            Two-dimensional array of y-coordinates (mesh grid).
        x : array_like
            One-dimensional array of x-coordinates defining the polyline vertices.
        y : array_like
            One-dimensional array of y-coordinates defining the polyline vertices.

        Returns
        -------
        r : ndarray
            Two-dimensional array of signed lateral distances to the line.
            Negative values indicate points located on the left side of the
            line direction, positive values indicate points on the right
            side, and np.nan indicates points for which no orthogonal
            projection onto the line exists.
        '''
        r_out = np.full(xx.shape, np.nan)
        r_abs = np.full(xx.shape, np.inf)
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            L = np.hypot(dx, dy)
            if L == 0:
                continue
            tx, ty = dx / L, dy / L
            nx, ny = ty, -tx
            Xx = xx - x[i]
            Xy = yy - y[i]
            s = Xx * tx + Xy * ty
            r = Xx * nx + Xy * ny
            valid = (s >= 0) & (s <= L)
            update = valid & (np.abs(r) < r_abs)
            r_out[update] = r[update]
            r_abs[update] = np.abs(r[update])
        return r_out

    @staticmethod
    def calc_v_from_EFscale(EF):
        '''
        Derive tornado wind velocity components from Enhanced Fujita (EF) scale.

        This function maps an EF intensity level (EF3–EF5) to a characteristic
        maximum wind speed derived from the official scale and decomposes this
        wind speed into tangential, radial, and forward velocity components.

        The decomposition follows fixed ratios inferred from Holland et al. (2006),
        assuming that the EF-based wind speed represents the tangential component and
        that proportionality holds for different EF levels.

        Parameters
        ----------
        EF : int or float
            Enhanced Fujita (EF) intensity level of the tornado (e.g., 3, 4, or 5).

        Returns
        -------
        Vt : float
            Tangential wind velocity (m/s), assumed equal to the EF-based maximum wind speed.
        Vr : float
            Radial wind velocity (m/s), defined as a fixed fraction of the tangential velocity.
        Vf : float
            Forward velocity (m/s), defined as a fixed fraction of the tangential velocity.

        References
        ----------
        Holland et al. (2006), A Simple Model for Simulating Tornado Damage in Forests. 
        J. Appl. Meteorologicaly Climatology, 45, 1597-1611.
        '''
        vf_H2006 = 15         # values from Holland (2006)
        vmax_tan_H2006 = 60
        vmax_rad_H2006 = 30
        cr = vmax_rad_H2006 / vmax_tan_H2006
        cf = vf_H2006 / vmax_tan_H2006
        vmax = GenMR_utils.map_EF2vmax[int(EF)]
        Vt = vmax
        Vr = cr * Vt
        Vf = cf * Vt
        return Vt, Vr, Vf

    @staticmethod
    def model_To_analytical(x, Rmax, Vr, Vt, Vf):
        '''
        Analytical lateral wind profile for an idealized tornado vortex.

        This function computes the maximum near-surface wind speed as a function
        of lateral (cross-track) distance from a tornado track using an analytical,
        axisymmetric vortex model. The formulation follows Burow et al. (2020).

        Parameters
        ----------
        x : ndarray
            Signed lateral distance (m) from the tornado track.
        Rmax : float
            Radius of maximum tangential wind (m).
        Vr : float
            Radial wind velocity component at Rmax (m/s).
        Vt : float
            Tangential wind velocity component at Rmax (m/s).
        Vf : float
            Forward velocity of the tornado (m/s).

        Returns
        -------
        vmax : ndarray
            Maximum near-surface wind speed (m/s).

        References
        ----------
        Burow et al. (2020), Damage analysis of three long-track tornadoes using high-resolution satellite imagery. 
        Atmosphere, 11, 613
        '''
        vmax = np.zeros(len(x))
        xa = np.abs(x)
        indin = xa <= Rmax
        indout = xa > Rmax
        vmax_in = np.sqrt((np.sin((np.pi*x)/(2*Rmax)) * Vt + np.cos((np.pi*x)/(2*Rmax)) * Vr + Vf)**2 +\
                        (np.cos((np.pi*x)/(2*Rmax)) * Vt - np.sin((np.pi*x)/(2*Rmax)) * Vr)**2)
        vmax_out = np.sqrt((Vt * Rmax / x + Vf)**2 + (Vr * Rmax / x)**2)
        vmax[indin] = vmax_in[indin]
        vmax[indout] = vmax_out[indout]
        return vmax

    @staticmethod
    def calc4To_I_v_ms(S, r_pm_km, par):
        '''
        Compute the maximum wind speed profile for a tornado based on its size (EF scale) and lateral distance from the track.

        Parameters
        ----------
        S : int or float
            Tornado size, represented as an EF scale value (e.g., 3, 4, 5).
        r_pm_km : float or ndarray
            Signed lateral (cross-track) distance from the tornado line in km.
            Negative values indicate the left side of the track, positive values the right.
        par : dict
            Dictionary of tornado parameters. Must contain:
            - 'Rmax_m' : float
                Radius of maximum tangential wind in meters.

        Returns
        -------
        vmax_ms : ndarray
            Maximum wind speed in meters per second (m/s) at distance r_pm_km from the tornado path.
        '''
        Vt, Vr, Vf = HazardFootprintGenerator.calc_v_from_EFscale(S)
        r_pm_m = r_pm_km * 1e3
        Rmax = par['Rmax_m']
        vmax_ms = HazardFootprintGenerator.model_To_analytical(r_pm_m, Rmax, Vr, Vt, Vf)
        vmax_ms[vmax_ms == 0] = np.nan
        return vmax_ms


    ## STORM SURGE FUNCTIONS ##
    @staticmethod
    def model_SS_Bathtub(I_trigger, src, topo_z):
        '''
        Simple Bathtub model to estimate storm surge (SS) height over a grid.

        This method computes the storm surge height at each grid cell
        using a "bathtub" approach: starting from the coastline, the surge
        height is reduced inland according to topography.

        Parameters
        ----------
        I_trigger : ndarray, shape (nx, ny)
            Grid of triggering intensity (windspeed, m/s) along 
            the coastline.
        src : object
            Source object containing:
            - `grid`: grid coordinates with `nx`, `ny`, `x`, `y`
            - `SS_char`: storm surge source coordinates
            - `par['SS']['bathy']`: bathymetry decay parameter
        topo_z : ndarray, shape (nx, ny)
            Elevation grid of the terrain.

        Returns
        -------
        I_SS : ndarray, shape (nx, ny)
            Storm surge height (m) at each grid cell.
        '''
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


    ## HEATWAVE FUNCTIONS ##
    @staticmethod
    def pdf_T_advectivemodel(T, mu0, lat):
        '''
        Compute a probability density function (PDF) of temperature using a simple advective 
        three-Gaussian model.

        Parameters
        ----------
        T : ndarray
            Array of temperature values (°C) where the PDF is evaluated.
        mu0 : float
            Mean surface temperature (°C) at the location of interest.
        lat : float
            Latitude (°) of the location. Determines the relative weights and shifts 
            of warm/cold advective anomalies according to Tamarin-Brodsky et al. (2022).

        Returns
        -------
        pdf : ndarray
            Temperature probability density function evaluated at `T`.
        sigma_pdf : float
            Effective standard deviation of the PDF, including contributions from 
            advective warm/cold anomalies.
        components : dict
            Dictionary with individual Gaussian components and smoothing parameter:
            - 'pdf0' : Gaussian centered at `mu0`
            - 'pdf_c' : Gaussian for cold advection
            - 'pdf_w' : Gaussian for warm advection
            - 'sigma_hat' : smoothing standard deviation used for all three components
        
        References
        ----------
        Tamarin-Brodsky et al. (2022), A Simple Model for Interpreting Temperature Variability and Its 
        Higher-Order Changes. J. Climate 35, 387-403.
        '''
        if np.abs(lat) <= 40.:                  # WARNING: only defined down to 30° degree lat. in article
            dT_warm, dT_cold = 3.3, 4.9         # pp. 395-396, fig.7d-f
            w_warm, w_cold = .25, .17
        elif np.abs(lat) >= 55:
            dT_warm, dT_cold = 5.8, 2.5
            w_warm, w_cold = .1, .24
        else:  # 40-55° range
            dT_warm, dT_cold = 4, 4.2
            w_warm, w_cold = 1/3, 1/3
        mu_warm = mu0 + dT_warm
        mu_cold = mu0 - dT_cold
        w0 = 1. - w_warm - w_cold
        sigma_hat = .25 * (dT_warm + dT_cold)                        # for smoothed pdf, weakly multimodal
        pdf0 = w0 * norm.pdf(T, mu0, sigma_hat)
        pdf_c = w_cold * norm.pdf(T, mu_cold, sigma_hat)
        pdf_w = w_warm * norm.pdf(T, mu_warm, sigma_hat)
        pdf = pdf0 + pdf_c + pdf_w
        sigma_pdf = np.sqrt(sigma_hat**2 + dT_warm*dT_cold*(1-w0))   # eq.16 (in supplement)
        components = {'pdf0': pdf0, 'pdf_c': pdf_c, 'pdf_w': pdf_w, 'sigma_hat': sigma_hat}
        return pdf, sigma_pdf, components

    @staticmethod
    def sample_T_advectivemodel(mu0_stoch, lat, seed = None):
        '''
        Sample stochastic temperatures from the 3-Gaussian advective PDF.

        Parameters
        ----------
        mu0_stoch : ndarray
            Array of mean surface temperatures (°C) at the location(s). Each entry is
            treated independently to generate a stochastic temperature sample.
        lat : float
            Latitude (°) of the region.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        T_stoch : ndarray
            Array of length Nsim containing stochastic temperature realizations (°C).
        '''
        if seed is not None:
            np.random.seed(seed)
        Nsim = len(mu0_stoch)
        Ti = np.linspace(-30, 50, 801)
        T_stoch = np.empty(Nsim)
        for i, T0_sampled in enumerate(mu0_stoch):
            pdf, _, _ = HazardFootprintGenerator.pdf_T_advectivemodel(Ti, T0_sampled, lat)
            pdf = pdf / pdf.sum()
            T_stoch[i] = np.random.choice(Ti, p = pdf)
        return T_stoch

    @staticmethod
    def sample_T_daily(T_adv_stoch, sigma_daily, Ndays=30, rho=0.7, seed=None):
        '''
        Generate a daily temperature time series for a month around a given stochastic monthly mean,
        with temporal correlation.

        The daily temperatures are generated using a first-order autoregressive
        [AR(1)] process. This introduces temporal autocorrelation between consecutive
        days, which is essential for representing multi-day heatwaves.

        Autocorrelation means that temperature on a given day is statistically
        dependent on the temperature of the previous day. A positive autocorrelation
        (rho > 0) increases the persistence of warm or cold anomalies, allowing
        sequences of consecutive hot days to occur.

        Mathematically, the daily temperature evolves as:

            T_d = rho * T_{d-1}
                + (1 - rho) * T_adv_stoch
                + sqrt(1 - rho^2) * ε_d

        where:
            - T_adv_stoch is the stochastic monthly temperature (equilibrium mean),
            - rho is the lag-1 autocorrelation coefficient (0 ≤ rho < 1),
            - ε_d is Gaussian white noise with standard deviation sigma_daily.

        The scaling factor sqrt(1 - rho^2) ensures that the stationary daily
        temperature variance is equal to sigma_daily², independently of rho.

        Parameters
        ----------
        T_adv_stoch : float
            Stochastic monthly temperature (°C) for the month.
        sigma_daily : float
            Standard deviation of daily fluctuations around T_adv_stoch (°C).
        Ndays : int
            Number of days in the month (default=30).
        rho : float
            Temporal correlation coefficient (0=independent, 1=perfectly correlated).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        T_daily : ndarray
            Array of length Ndays containing daily temperatures (°C) for the month.
        '''
        if seed is not None:
            np.random.seed(seed)
        T_daily = np.empty(Ndays)
        T_daily[0] = T_adv_stoch
        for d in range(1, Ndays):
            epsilon = np.random.normal(0, sigma_daily)
            T_daily[d] = rho*T_daily[d-1] + (1-rho)*T_adv_stoch + np.sqrt(1-rho**2)*epsilon
        return T_daily

    @staticmethod
    def get_HW_atloc(T_daily, T_th, Dt_HW):
        '''
        Identify heatwave events and extract their durations and indices.

        A heatwave is defined as a contiguous sequence of days with
        T_daily >= T_th lasting at least Dt_HW days.

        Parameters
        ----------
        T_daily : ndarray
            Daily temperatures (°C).
        T_th : float
            Heatwave temperature threshold (°C).
        Dt_HW : int
            Minimum number of consecutive days to define a heatwave.

        Returns
        -------
        events : list of tuple
            List of (start_index, end_index) pairs for each heatwave,
            where indices are inclusive and zero-based.
        durations : list of int
            Durations (in days) of all detected heatwaves.
        '''
        is_hot = T_daily >= T_th
        durations = []
        events = []
        start = None
        for i, hot in enumerate(is_hot):
            if hot and start is None:
                start = i
            elif not hot and start is not None:
                length = i - start
                if length >= Dt_HW:
                    durations.append(length)
                    events.append((start, i - 1))
                start = None
        if start is not None:
            length = len(T_daily) - start
            if length >= Dt_HW:
                durations.append(length)
                events.append((start, len(T_daily) - 1))
        return events, durations

    @staticmethod
    def get_HW_footprint(T_map_daily, Tth = 35., Dt = 3):
        '''
        Identify the spatial heatwave footprint over a month from daily temperature fields.

        A heatwave at a given grid cell is defined as a contiguous sequence of days
        during which the daily temperature exceeds a threshold Tth for at least
        Dt consecutive days. The heatwave footprint is the spatial extent of
        all grid cells that experience at least one such heatwave during the month.

        Parameters
        ----------
        T_map_daily : ndarray
            Daily temperature field with shape (Ndays, nx, ny), where Ndays
            is the number of days in the month and (nx, ny) define the spatial grid.
        Tth : float, optional
            Heatwave temperature threshold (°C). Default is 35°C.
        Dt : int, optional
            Minimum number of consecutive days above Tth required to define
            a heatwave event. Default is 3 days.

        Returns
        -------
        HW_fp_maxT : ndarray
            Two-dimensional array with shape (nx, ny) containing the maximum
            temperature (°C) reached during heatwave days at each grid cell.
        HW_duration : int
            Total heatwave duration in days (i.e, event size), defined as the number of days during
            which at least one grid cell is in a heatwave state.
        '''
        Ndays, nx, ny = T_map_daily.shape
        HW_fp = np.zeros_like(T_map_daily, dtype = bool)
        is_hot = T_map_daily >= Tth
        for i in range(nx):
            for j in range(ny):
                start = None
                for t in range(Ndays):
                    if is_hot[t, i, j] and start is None:
                        start = t
                    elif not is_hot[t, i, j] and start is not None:
                        if t - start >= Dt:
                            HW_fp[start:t, i, j] = True
                        start = None
                if start is not None and Ndays - start >= Dt:
                    HW_fp[start:Ndays, i, j] = True

#        HW_fp_tcollapse = HW_fp.any(axis = 0)
        HW_fp_maxT = np.full((nx, ny), np.nan)
        for i in range(nx):
            for j in range(ny):
                if HW_fp[:, i, j].any():
                    HW_fp_maxT[i, j] = T_map_daily[HW_fp[:, i, j], i, j].max()
        HW_duration = HW_fp.any(axis=(1, 2)).sum()
        return HW_fp_maxT, HW_duration

    @staticmethod
    def model_HW_threshold(src, atmoLayer, path_HW_stochset, path_HW_stochset_data, \
                        Nsim = 10000, Tmin = 10, Tmax = 40, plot_HWprocess = False):
        '''
        Generate a stochastic catalog of heatwave (HW) footprint events based on
        large-scale advective temperature anomalies and daily correlated temperature
        variability.

        The function performs Monte Carlo simulations of heatwave events by combining:
        (i) yearly-scale advective temperature anomalies,
        (ii) daily-scale stochastic temperature variability, and
        (iii) spatial temperature fields. Heatwave events are identified using a
        temperature threshold and minimum duration criterion, and qualifying events
        are stored as spatial footprint arrays on disk.

        Parameters
        ----------
        src : dict
            Source dictionary containing model parameters in ``src.par['HW']``.
            Required keys include:
            - ``T_th`` : float
                Heatwave temperature threshold (°C).
            - ``Dt_da`` : int
                Minimum duration (days) to define a heatwave.
            - ``Dt_max_da`` : int
                Maximum duration (days) of simulated daily temperatures.
            - ``sigmaT_daily`` : float
                Standard deviation of daily temperature variability (°C).
            - ``sigmaT_yearly`` : float
                Standard deviation of yearly temperature variability (°C).
            - ``corrT`` : float
                Temporal correlation coefficient of daily temperatures.
            - ``lat_deg`` : float
                Latitude (degrees) used in the advective temperature model.

        atmoLayer : class
            Atmospheric environmental layer.

        path_HW_stochset : str
            Path where diagnostic figures of heatwave processes are saved.

        path_HW_stochset_data : str
            Path where heatwave footprint arrays (``.npy`` files) are stored.

        Nsim : int, optional
            Number of Monte Carlo simulations to perform. Default is 10000.

        Tmin : float, optional
            Minimum temperature used for histogram binning in diagnostic plots (°C).
            Default is 10.

        Tmax : float, optional
            Maximum temperature used for histogram binning in diagnostic plots (°C).
            Default is 40.

        plot_HWprocess : bool, optional
            If True, diagnostic plots illustrating the heatwave generation process
            are produced and saved for each detected event. Default is False.

        Returns
        -------
        None
            Heatwave footprint events are written to disk as ``.npy`` files.
        '''
        T_th_HW, Dt_HW = src.par['HW']['T_th'], src.par['HW']['Dt_da']
        Ndays = src.par['HW']['Dt_max_da']
        dayi = np.arange(src.par['HW']['Dt_max_da'])+1
        Ti = np.arange(Tmin, Tmax, 1)
        Tmin_compute = T_th_HW - src.par['HW']['sigmaT_daily']
        T0 = np.max(atmoLayer.T)

        T0_sim = np.random.normal(T0, src.par['HW']['sigmaT_yearly'], Nsim)
        Tadv_sim = HazardFootprintGenerator.sample_T_advectivemodel(T0_sim, src.par['HW']['lat_deg'])
        DTadv_sim = Tadv_sim - T0

        nx, ny = atmoLayer.T.shape
        catalog_hazFootprints_HW = {}
        k = 1
        for sim in range(Nsim):
            if sim % 1000 == 0:
                print(f'{sim}/{Nsim}', end = '\r', flush = True)
            if Tadv_sim[sim] > Tmin_compute:
                T_map_mean = atmoLayer.T + DTadv_sim[sim]
                T_daily_stoch = HazardFootprintGenerator.sample_T_daily(Tadv_sim[sim], src.par['HW']['sigmaT_daily'], \
                                            Ndays = Ndays, rho = src.par['HW']['corrT'])    
                dT_daily_stoch = T_daily_stoch - Tadv_sim[sim]

                T_map_daily_stoch = np.empty((Ndays, nx, ny))
                for t in range(Ndays):
                    T_map_daily_stoch[t] = T_map_mean + dT_daily_stoch[t]

                HW_ti_stoch, _ = HazardFootprintGenerator.get_HW_atloc(T_daily_stoch, T_th_HW, Dt_HW)
                HW_fp_stoch, HW_S_stoch = HazardFootprintGenerator.get_HW_footprint(T_map_daily_stoch, Tth = T_th_HW, Dt = Dt_HW)
                if HW_S_stoch >= Dt_HW:
                    evID = 'HW' + str(k)
                    catalog_hazFootprints_HW[evID] = HW_fp_stoch
                    np.save(f'{path_HW_stochset_data}/HW_fp_event_{evID}_{HW_S_stoch}.npy', HW_fp_stoch)
                    k += 1
                    if plot_HWprocess:
                        plt.rcParams['font.size'] = '16'
                        fig, ax = plt.subplots(1,3, figsize=(20,6))
                        ax[0].hist(Tadv_sim, bins = Ti, color = 'darkgrey')
                        ax[0].axvline(Tadv_sim[sim], color = 'black', linestyle = 'solid')
                        ax[0].axvline(T_th_HW, color = 'darkred', linestyle = 'dashed')
                        ax[0].set_xlabel('$T$ (°C)')
                        ax[0].set_ylabel('Density')
                        ax[0].set_title('Baseline+advection variability', pad = 20)
                        ax[0].spines['right'].set_visible(False)
                        ax[0].spines['top'].set_visible(False)
                        ax[1].plot(dayi, T_daily_stoch, color = 'black')
                        for start, end in HW_ti_stoch:
                            ax[1].axvspan(dayi[start] - .5, dayi[end] + .5, color='darkred', alpha = .2)
                        ax[1].axhline(Tadv_sim[sim], color = 'black', linestyle = 'dotted')
                        ax[1].axhline(T_th_HW, color = 'darkred', linestyle = 'dashed')
                        ax[1].set_ylim(T_th_HW-5, T_th_HW+5)
                        ax[1].set_xlabel('Day of the month')
                        ax[1].set_ylabel('$T$ (°C)')
                        ax[1].set_title(f'Daily temp. at {Tadv_sim[sim]:.1f}°C', pad = 20)
                        ax[1].spines['right'].set_visible(False)
                        ax[1].spines['top'].set_visible(False)
#                        ax[2].contourf(atmoLayer.grid.xx, atmoLayer.grid.yy, HW_fp_stoch.astype(int), cmap = 'Reds', alpha = .5, levels = [.5, 1.5])
                        ax[2].contourf(atmoLayer.grid.xx, atmoLayer.grid.yy, HW_fp_stoch, cmap = 'Reds')
                        ax[2].set_xlabel('$x$ (km)')
                        ax[2].set_ylabel('$y$ (km)')
                        ax[2].set_title(f'Heatwave fp. {evID} ({HW_S_stoch}da)', pad = 20)
                        ax[2].set_aspect(1)
                        fig.tight_layout()
                        plt.savefig(f'{path_HW_stochset}/char_{evID}.jpg')
                        plt.close()



    ###################################
    ## INTENSITY FOOTPRINT GENERATOR ##
    ###################################
    def generate(self):
        print('generating footprints for:', end=' ')
        for ID in self.src.par['perils']:
            indperil = np.where(self.stochset['ID'] == ID)[0]
            Nev_peril = len(indperil)
            print(ID, end='... ')

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

            elif ID == 'HW':
                self._run_HW(Nev_peril, self.atmoLayer)

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
                        I_sym_t = self.calc4TC_I_v_ms(track_S[j], r, self.src.par['TC'])
                        I_t[:, :, j], *_ = self.add_v_forward(
                            self.src.par['TC']['vforward_m/s'], I_sym_t, track_x, track_y, self.src.grid, j)
                    self.catalog_hazFootprints[evID] = np.nanmax(I_t, axis=2)

            elif ID == 'To':
                Tocoord = self.evchar[get_peril_evID(self.evchar['evID']) == 'To'].reset_index(drop=True)
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    Tocoord_evID = Tocoord[Tocoord['evID'] == evID].reset_index(drop=True)
                    r2To = self.calc_lateral_distance_signed(self.src.grid.xx, self.src.grid.yy, Tocoord_evID['x'], Tocoord_evID['y'])
                    vmax_flat = self.calc4To_I_v_ms(S, r2To.flatten(), self.src.par['To'])
                    self.catalog_hazFootprints[evID] = vmax_flat.reshape(np.shape(self.src.grid.xx))

            elif ID == 'WS':
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    S = self.stochset['S'][indperil].values[i]
                    nx, ny = np.shape(self.src.grid.xx)
                    S_cst = np.repeat(S, nx*ny)
                    self.catalog_hazFootprints[evID] = S_cst.reshape(nx, ny)

            ## secondary perils ##
            elif ID == 'SS':
                pattern = re.compile(r'TC(\d+)')
                for i in range(Nev_peril):
                    evID = self.stochset['evID'][indperil].values[i]
                    evID_trigger = re.search(pattern, evID).group()
                    I_trigger = self.catalog_hazFootprints[evID_trigger]
                    self.catalog_hazFootprints[evID] = self.model_SS_Bathtub(I_trigger, self.src, self.topo_z)


        print('catalogue completed')
        return self.catalog_hazFootprints

    ## TC CASE ##
    def get_TC_timeshot(self, evID, t):
        '''
        Compute a single time-step (snapshot) of a tropical cyclone (TC) event.

        This method calculates both the symmetric and asymmetric wind fields
        at time index `t` along the cyclone track, as well as the tangential
        and total wind velocity components.

        Parameters
        ----------
        evID : str
            Event ID of the tropical cyclone.
        t : int
            Time index along the cyclone track (0 ≤ t < npt), where npt is
            the number of points along the track.

        Returns
        -------
        tuple of ndarray
            I_sym_t : ndarray
                Symmetric wind field at time step `t` (storm without motion effects).
            I_asym_t : ndarray
                Asymmetric wind field at time step `t` (storm including motion effects).
            vtot_x : ndarray
                Total wind velocity component along x-axis.
            vtot_y : ndarray
                Total wind velocity component along y-axis.
            vtan_x : ndarray
                Tangential velocity component along x-axis.
            vtan_y : ndarray
                Tangential velocity component along y-axis.
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
        I_sym_t = self.calc4TC_I_v_ms(track_S[t], r, self.src.par['TC'])
        I_asym_t, vtot_x, vtot_y, vtan_x, vtan_y = self.add_v_forward(
            self.src.par['TC']['vforward_m/s'], I_sym_t, track_x, track_y, self.src.grid, t)

        return I_sym_t, I_asym_t, vtot_x, vtot_y, vtan_x, vtan_y

    ## HW CASE ##
    def _load_HW_footprints(self, path_HW_data):
        pattern = re.compile(r"HW_fp_event_(HW\d+)_(\d+)\.npy")
        results = []
        for fname in os.listdir(path_HW_data):
            match = pattern.match(fname)
            if match:
                evID = match.group(1)
                S_raw = int(match.group(2))
                full_path = os.path.join(path_HW_data, fname)
                HW_fp = np.load(full_path)
                results.append({'evID': evID, 'S_raw': S_raw, 'HW_fp': HW_fp})
        df = pd.DataFrame([{'evID': r['evID'], 'S_raw': r['S_raw']} for r in results])
        return results, df              

    def _run_HW(self, Nev_peril, atmoLayer_T):
        path_HW_stochset = 'figs/HW_stochset_tmp/'
        Nsim = 100000   # WARNING: hard-coded, high enough to get reasonable estimates of rate(Si)

        Si = self.src.par['HW']['Si_da']
        evIDi = [f"HW{i+1}" for i in range(len(Si))]

        cache_files = {evID: self._cache_path(evID) for evID in evIDi}
        all_cached = (not self.force_recompute and all(os.path.exists(path) for path in cache_files.values()) and os.path.exists(path_HW_stochset))
        if all_cached:
            self.rate_HW = np.load(os.path.join(self.cache_dir, f'rates_HW.npy'))
            for evID, path in cache_files.items():
                self.catalog_hazFootprints[evID] = np.load(path)
            print('(loading from cache)')
        else:
            print('(computing)')
            os.makedirs(path_HW_stochset, exist_ok = True)
            path_HW_stochset_data = os.path.join(path_HW_stochset, 'data')
            os.makedirs(path_HW_stochset_data, exist_ok = True)
            HazardFootprintGenerator.model_HW_threshold(self.src, atmoLayer_T, path_HW_stochset, path_HW_stochset_data, \
                            Nsim = Nsim, plot_HWprocess = True)
            
            print('Fetching footprints from potential footprints')
            HW_footprints, HW_fp_metadata = self._load_HW_footprints(path_HW_stochset + 'data')
            # Pool largest footprint per Si - for Tutorial 2
            Si2evID = dict(zip(Si, evIDi))
            HW_fp_metadata['S'] = HW_fp_metadata['S_raw'].apply(lambda s: GenMR_utils.map2upperTail(s, Si))    
            HW_fp_metadata['fp_S'] = [np.sum(~np.isnan(HW_footprints[i]['HW_fp'])) for i in HW_fp_metadata.index]
                                    # fp_S as footprint spatial extent, proxy to temperature amplitude above threshold
            lbdi = (HW_fp_metadata.groupby('S').size().reindex(Si, fill_value=0) / Nsim)
            lbdi = lbdi.rename(index = Si2evID)

            self.rate_HW = lbdi.values
            cache_file = os.path.join(self.cache_dir, f'rates_HW.npy')
            np.save(cache_file, lbdi)

            for i in range(Nev_peril):
                df_Si = HW_fp_metadata[HW_fp_metadata['S'] == Si[i]]
                indmax = df_Si['fp_S'].idxmax()
                self.catalog_hazFootprints[evIDi[i]] = HW_footprints[indmax]['HW_fp']
                cache_file = self._cache_path(evIDi[i])
                np.save(cache_file, HW_footprints[indmax]['HW_fp'])



class DynamicHazardFootprintGenerator:
    '''
    Generate dynamical hazard footprints for a given stochastic event set.

    This class computes dynamic hazard footprints (e.g., landslides, fluvial floods, wildfires) 
    over a spatial grid, using stochastic events and environmental layers.

    Parameters
    ----------
    stochset : pandas.DataFrame
        Stochastic event set.
    src : Src
        Source object defining peril sources and their characteristics.
    soilLayer : class
        Environmental layer object representing soil characteristics.
    urbLandLayer : class
        Environmental layer object representing urban land-use
    '''
    def __init__(self, stochset, src, soilLayer, urbLandLayer):
        self.stochset = stochset
        self.src = src
        self.soil = soilLayer
        self.urb = urbLandLayer
        self.grid = urbLandLayer.grid
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
            elif ID == 'WF':
                self._run_WF(indperil, Nev_peril)

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


    def _run_WF(self, indperil, Nev_peril):
        frame_path = 'figs/WF_CA_frames/'
        if os.path.exists(frame_path) and not self.force_recompute:
            print('Loading from cache potential footprints')
        else:
            print('Computing potential footprints')
            frame_plot = True    #False
            WF_CA = CellularAutomaton_WF(self.src, self.urb, frame_plot)
            WF_CA.run()
        print('Fetching footprints from potential footprints')
        WF_CA_footprints, WF_CA_fp_metadata = self.load_WF_CA_footprints(frame_path + 'data')
        for item in WF_CA_footprints:
            item['burnt_area_ha'] = item['burnt_area_cells'] * (self.grid.w ** 2) * 100
        WF_CA_fp_metadata['burnt_area_ha'] = WF_CA_fp_metadata['burnt_area_cells'] * (self.grid.w ** 2) * 100

        WF_pool = WF_CA_footprints.copy()
        for i in range(Nev_peril):
            evID = self.stochset['evID'][indperil].values[i]
            S = self.stochset['S'][indperil].values[i]
            cache_file = self._cache_path(evID)
            print(f'{evID} (fetched from cache)')
            # fetch a matching footprint S from CA catalogue
            diffs = np.array([abs(r['burnt_area_ha'] - S) for r in WF_pool])
            min_diff = np.min(diffs)
            candidate_idxs = np.where(diffs == min_diff)[0]
            chosen_idx = np.random.choice(candidate_idxs)                      # add rdm seed ?
            self.catalog_hazFootprints[evID] = WF_pool[chosen_idx]['WF_fp']
            _ = WF_pool.pop(chosen_idx)
            np.save(cache_file, self.catalog_hazFootprints[evID])

    def load_WF_CA_footprints(self, path_WF_CA_data):
        pattern = re.compile(r"WF_fp_event_(\d+)_(\d+)\.npy")
        results = []
        for fname in os.listdir(path_WF_CA_data):
            match = pattern.match(fname)
            if match:
                event_id = int(match.group(1))
                burnt_area_cells = int(match.group(2))
                full_path = os.path.join(path_WF_CA_data, fname)
                WF_fp = np.load(full_path)
                results.append({'WF_CA_id': event_id, 'burnt_area_cells': burnt_area_cells, 'WF_fp': WF_fp})
        results = sorted(results, key=lambda d: d['burnt_area_cells'], reverse = True)    
        df = pd.DataFrame([{'WF_CA_id': r['WF_CA_id'], 'burnt_area_cells': r['burnt_area_cells']}
            for r in results])
        return results, df


## FLUVIAL FLOOD CASE ##
class CellularAutomaton_FF:
    '''
    Cellular Automaton model for flood (FF) propagation.

    Parameters
    ----------
    I_RS : ndarray
        Rainfall intensity (m) at FF source
    src : Src
        Hazard source object containing flood source parameters and catchment information.
    grid : class
        Spatial grid object with attributes such as nx, ny, x, y, and cell width.
    topoLayer_z : ndarray
        Topography elevation field (m) corresponding to the spatial grid.
    movie : dict
        Dictionary containing movie settings:
            - 'create' (bool): whether to save CA frames.
            - 'path' (str): folder path to store frames; hardcoded to 'figs/FF_CA_frames/'.
    '''
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
    Cellular Automaton model for landslide (LS) propagation.

    Parameters
    ----------
    soilLayer : class
        Soil layer object containing topography, soil thickness, and related properties.
    wetness : ndarray
        Field of soil wetness or saturation across the spatial grid.
    movie : dict
        Dictionary containing movie settings:
            - 'create' (bool): whether to save CA frames.
            - 'path' (str): folder path to store frames; hardcoded to 'figs/LS_CA_frames/'.
    kmax : int, optional
        Maximum number of CA iterations; default is 20.
    '''
    def __init__(self, soilLayer, wetness, movie, kmax = 20):
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


## WILDFIRE CASE ##
class CellularAutomaton_WF:
    '''
    Cellular Automaton model for wildfire (WF) propagation.

    Parameters
    ----------
    src : class
        Source object containing wildfire parameters, including spread ratio and related settings.
    urbLandLayer : class
        Urban land layer object containing land-use map, building types, and grid information.
    frame_plot : bool
        Flag indicating whether to generate CA frame plots during simulation.
    '''
    def __init__(self, src, urbLandLayer, frame_plot):
        self.src = src
        self.urbLandLayer = urbLandLayer
        self.frame_plot = frame_plot

        self.grid = copy.copy(self.urbLandLayer.grid)
        self.landuse_S = copy.copy(self.urbLandLayer.S)

        self.indForest = np.where(self.landuse_S.flatten() == 1)[0]

        self.indForest2Grass = np.random.choice(
            self.indForest,
            size=int(len(self.indForest) * self.src.par['WF']['ratio_grass']),
            replace=False
        )

        landuse_S4WF_flat = self.landuse_S.flatten()
        landuse_S4WF_flat[self.indForest2Grass] = 0

        # add wood buildings to forest state:
        self.indwoodBldg = np.where(self.urbLandLayer.bldg_type.flatten() == 'W')[0]
        landuse_S4WF_flat[self.indwoodBldg] = 1

        self.landuse_S4WF = landuse_S4WF_flat.reshape(self.landuse_S.shape)

        self.path_WF_CA = 'figs/WF_CA_frames'
        os.makedirs(self.path_WF_CA, exist_ok=True)
        self.path_WF_CA_data = os.path.join(self.path_WF_CA, 'data')
        os.makedirs(self.path_WF_CA_data, exist_ok=True)

        self.S = np.zeros(self.landuse_S4WF.shape)
        self.S[self.landuse_S4WF == 1] = 1

        self.k = 1  # WF event counter
        self.i = 0  # iteration

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.src.par['WF']['nsim']:
            raise StopIteration

        if self.i % 1000 == 0:
            print(self.i, '/', self.src.par['WF']['nsim'])

        # LOADING (long-term tree growth)
        S_flat = self.S.flatten()
        landuse_S4WF_flat = self.landuse_S4WF.flatten()

        self.indForest = np.where(self.landuse_S.flatten() == 1)[0]
        indForest_notree = self.indForest[np.where(S_flat[self.indForest] == 0)[0]]

        if len(indForest_notree) > 0:
            new_tree_xy = np.random.choice(indForest_notree, size=self.src.par['WF']['rate_newtrees'])
            S_flat[new_tree_xy] = 1
            landuse_S4WF_flat[new_tree_xy] = 1

        # TRIGGERING (lightning)
        if np.random.random(1) <= self.src.par['WF']['p_lightning']:
            lightning_xy = np.random.choice(self.indForest, size=1)  # to limit number of simulations needed
            WF_fp = np.zeros(self.S.shape)
            WF_fp[:, :] = np.nan

            if S_flat[lightning_xy] == 1:
                self.S = S_flat.reshape(self.S.shape)
                self.landuse_S4WF = landuse_S4WF_flat.reshape(self.S.shape)

                S_clumps = measure.label(self.S, connectivity=1)
                clump_WF = S_clumps.flatten()[lightning_xy]
                indWF = S_clumps == clump_WF

                WF_fp[indWF] = 5
                self.S[indWF] = 0
                self.landuse_S4WF[indWF] = 0

                burntArea_cells = np.sum(WF_fp == 5)

                WF_Smin = 1e4   # hardcoded here, could be sizeDistr['WF']['Smin'] as additional input par.
                if burntArea_cells >= WF_Smin:
                    if self.frame_plot:
                        plt.rcParams['font.size'] = '14'
                        _, ax = plt.subplots(1, 1, figsize=(7, 7))
                        ax.contourf(self.grid.xx, self.grid.yy,
                                    GenMR_env.ls.hillshade(self.urbLandLayer.topo.z, vert_exag=.1),
                                    cmap='gray', alpha=.1)
                        ax.pcolormesh(self.grid.xx, self.grid.yy, self.landuse_S4WF,
                                    cmap=GenMR_utils.col_S, vmin=-1, vmax=5, alpha=.5)
                        ax.pcolormesh(self.grid.xx, self.grid.yy, WF_fp,
                                    cmap=GenMR_utils.col_S, vmin=-1, vmax=5)

                        plt.savefig(f'{self.path_WF_CA}/WF_fp_event_{self.k}_{burntArea_cells}.jpg', dpi=300)
                        plt.close()

                    np.save(f'{self.path_WF_CA_data}/WF_fp_event_{self.k}_{burntArea_cells}.npy', WF_fp)
                    self.k += 1
                    
                # repopulate wood buildings for next WF (i.e., independent events)
                landuse_S4WF_flat = self.landuse_S4WF.flatten()
                landuse_S4WF_flat[self.indwoodBldg] = 1
                self.landuse_S4WF = landuse_S4WF_flat.reshape(self.landuse_S.shape)
                self.S[self.landuse_S4WF == 1] = 1

        # move iteration forward
        self.i += 1

        # return current state if desired
        return self.S, self.landuse_S4WF

    def run(self):
        for _ in self:
            pass



###################
# LOSS FOOTPRINTS #
###################

def vuln_f(I, peril):
    '''
    Calculate the mean damage ratio (MDR) given hazard intensity for different perils.

    Parameters
    ----------
    I : float or ndarray
        Hazard intensity, interpreted according to the peril type:
        - 'AI' or 'Ex': overpressure (kPa)
        - 'EQ': peak ground acceleration (m/s²)
        - 'FF' or 'SS': inundation depth (m)
        - 'LS': landslide thickness (m)
        - 'VE': volcanic ash thickness (m)
        - 'TC', 'To', 'WS': wind speed (m/s)
        - 'WF': wildfire state (1 for burnt, 0 otherwise)
    peril : str
        Type of peril ('AI', 'Ex', 'EQ', 'FF', 'SS', 'LS', 'VE', 'TC', 'To', 'WS', 'WF').

    Returns
    -------
    MDR : float or ndarray
        Mean damage ratio corresponding to the given hazard intensity and peril.
    '''
    if peril == 'AI' or peril == 'Ex':   # I = overpressure (kPa)
        mu = np.log(20)
        sig = .4
        MDR = .5 * (1 + scipy.special.erf((np.log(I) - mu)/(sig * np.sqrt(2))))
    if peril == 'EQ':                    # I = peak ground acceleration (m/s2)
        mu = np.log(6)
        sig = .6
        MDR = .5 * (1 + scipy.special.erf((np.log(I) - mu)/(sig * np.sqrt(2))))
    if peril == 'FF' or peril == 'SS':   # I = inundation depth (m)
        c = .45
        MDR = c * np.sqrt(I)
        MDR[MDR > 1] = 1
    if peril == 'LS':                    # I = landslide thickness (m)
        c1 = -1.671
        c2 = 3.189
        c3 = 1.746
        MDR = 1 - np.exp(c1*((I+c2)/c2 - 1)**c3)
    if peril == 'VE':                    # I = ash thickness (m)
        g_earth = 9.81                   # (m/s^2)
        rho_ash = 900                    # (kg/m3)  (dry ash)
        I_kPa = rho_ash * g_earth * I * 1e-3
        mu = 1.6
        sig = .4
        MDR = .5 * (1 + scipy.special.erf((np.log(I_kPa) - mu)/(sig * np.sqrt(2))))
    if peril == 'TC' or peril == 'To' or peril == 'WS':   # I = wind speed (m/s)
        v_thresh = 25.7 # 50 kts
        v_half = 74.7
        vn = (I - v_thresh) / (v_half - v_thresh)
        vn[vn < 0] = 0
        MDR = vn**3 / (1+vn**3)
    if peril == 'WF':                    # I = 5 (burnt) or 0
        MDR = np.zeros_like(I)
        MDR[I == 5] = 1
    return MDR

class RiskFootprintGenerator:
    '''
    Generate risk footprints (damage and loss) for a set of hazard footprints.

    Parameters
    ----------
    catalog_hazFootprints : dict
        Dictionary of hazard footprints with event IDs as keys and 2D arrays as values.
    urbLandLayer : object
        Urban land layer containing building values (attribute `bldg_value`).
    evtable : pandas.DataFrame
        Event table with at least a column 'evID'; a 'loss' column will be added.
    '''
    def __init__(self, catalog_hazFootprints, urbLandLayer, evtable):
        self.catalog_hazFootprints = catalog_hazFootprints
        self.expo_value = urbLandLayer.bldg_value.astype(float, copy=True)
        self.ELT = evtable.copy()
        self.ELT['loss'] = np.nan
        self.evIDs_wFp = list(catalog_hazFootprints.keys())
        self.catalog_dmgFootprints = {}
        self.catalog_lossFootprints = {}

        target_shape = self.catalog_hazFootprints[self.evIDs_wFp[0]].shape
        for evID in self.evIDs_wFp:
            if self.catalog_hazFootprints[evID].shape != target_shape:
                raise ValueError(
                    f'Hazard footprint for {evID} has shape '
                    f'{self.catalog_hazFootprints[evID].shape}, expected {target_shape}'
                )
        self.hazfp_stack = np.stack(
            [self.catalog_hazFootprints[evID] for evID in self.evIDs_wFp], axis=0
        )
        dmg_stack = np.zeros_like(self.hazfp_stack, dtype=float)
        loss_stack = np.zeros_like(self.hazfp_stack, dtype=float)

        for i, evID in enumerate(tqdm(self.evIDs_wFp, desc='Computing MDR & Loss')):
            peril = evID[:2]
            hazfp = self.hazfp_stack[i]
            MDR = vuln_f(hazfp, peril)
            MDR[np.isnan(self.expo_value)] = np.nan
            MDR[MDR == 0] = np.nan

            dmg_stack[i] = MDR

            loss = MDR * self.expo_value
            loss_stack[i] = loss

            self.ELT.loc[self.ELT['evID'] == evID, 'loss'] = np.nansum(loss)

        for i, evID in enumerate(self.evIDs_wFp):
            self.catalog_dmgFootprints[evID] = dmg_stack[i]
            self.catalog_lossFootprints[evID] = loss_stack[i]



############
# PLOTTING #
############

def plot_src(src, hillshading_z = '', file_ext = '-'):
    '''
    Plot the spatial locations of peril sources on a 2D grid.

    Optionally overlays hillshading of the terrain and saves the figure.

    Parameters
    ----------
    src : object
        An instance of the `Src` class containing peril source definitions and grid.
    hillshading_z : ndarray or str, optional
        2D array of topographic elevation values for hillshading (default is '').
    file_ext : str, optional
        If not '-', saves the figure with this file extension (e.g., 'jpg', 'pdf').

    Returns
    -------
    None
        Displays the plot and optionally saves it to file.
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

    if 'To' in src.par['perils']:
        h_ai =ax[0].scatter(src.To_char['x0'], src.To_char['y0'], color = GenMR_utils.col_peril('To'), s=20, marker = 'o', clip_on = False)
        handles.append(h_ai)
        labels.append('Vortex line seed: Tornado (To)')
        # CS conditional on To
        if 'CS' in src.par['perils']:
            for i in range(src.par['To']['N']):
                circle = Circle((src.To_char['x0'][i], src.To_char['y0'][i]), src.par['CS']['R_km'], fill = False, color = GenMR_utils.col_peril('CS'), \
                                 linewidth = 1, linestyle = 'dashed')
                h_cs = ax[0].add_patch(circle)
            handles.append(h_cs)
            labels.append('Super-cell seed: Convective storm (CS)')

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


def plot_hazFootprints(catalog_hazFootprints, grid, topoLayer_z, plot_Imin, plot_Imax, nstoch = 5, file_ext = '-'):
    '''
    Plot stochastic hazard footprints for multiple events and perils on a spatial grid.

    Parameters
    ----------
    catalog_hazFootprints : dict
        Dictionary of hazard footprints keyed by event ID. Each value is a 2D array of intensities.
    grid : object
        Grid object with attributes `xx`, `yy`, `xmin`, `xmax`, `ymin`, `ymax`.
    topoLayer_z : ndarray
        2D array of topographic elevations for hillshading.
    plot_Imin : dict
        Dictionary of minimum intensity values for each peril (keyed by peril code).
    plot_Imax : dict
        Dictionary of maximum intensity values for each peril (keyed by peril code).
    nstoch : int, optional
        Number of stochastic realizations to plot per peril (default 5).
    file_ext : str, optional
        File extension to save figure (e.g., 'png', 'pdf'). If '-', figure is not saved (default '-').
    '''
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
        Imin = plot_Imin[perils[i]]
        Imax = plot_Imax[perils[i]]
        for j in range(nplot):
            I_plt = np.copy(catalog_hazFootprints[evID_shuffled[j]])
            I_plt[I_plt >= Imax] = Imax
            I_plt[I_plt < Imin] = np.nan
            ax[i,j].contourf(grid.xx, grid.yy, I_plt, cmap = 'Reds', levels = np.linspace(Imin, Imax, 100))
            ax[i,j].contourf(grid.xx, grid.yy, GenMR_env.ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
            ax[i,j].plot([grid.xmin + grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmin + grid.xbuffer, grid.xmin + grid.xbuffer],
                       [grid.ymin + grid.ybuffer, grid.ymin + grid.ybuffer, grid.ymax - grid.ybuffer, grid.ymax - grid.ybuffer, grid.ymin + grid.ybuffer], \
                        linestyle='dotted', color='black')

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


def plot_vulnFunctions():
    '''
    Plot vulnerability (MDR) functions for different perils.
    '''
    pi_kPa = np.linspace(1e-6, 50, 100)
    MDR_blast = vuln_f(pi_kPa, 'AI')      # or 'Ex'
    PGAi = np.linspace(1e-6, 15, 100)     # m/s2
    MDR_EQ = vuln_f(PGAi, 'EQ')
    hwi = np.linspace(1e-6, 7, 1000)      # m
    MDR_flood = vuln_f(hwi, 'FF')         # or 'SS'
    hsi = np.linspace(1e-6, 7, 100)       # m
    MDR_LS = vuln_f(hsi, 'LS')
    hai = np.linspace(1e-6, 2, 100)       # m
    MDR_VE = vuln_f(hai, 'VE')
    g_earth = 9.81                   # [m/s^2]
    rho_ash = 900                    # [kg/m3]  (dry ash)
    pi_VE_kPa = rho_ash * g_earth * hai * 1e-3
    vi = np.linspace(0, 100, 100)      # m/s
    MDR_WS = vuln_f(vi, 'WS')

    plt.rcParams['font.size'] = '18'
    fig, ax = plt.subplots(2,3, figsize = (20,12))
    ax[0,0].plot(pi_kPa, MDR_blast, color = 'black')
    ax[0,0].set_title('Blast (AI, Ex)', pad = 20)
    ax[0,0].set_xlabel('Overpressure $P$ [kPa]')
    ax[0,0].set_ylabel('MDR')
    ax[0,0].set_ylim(0,1.01)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].grid()

    ax[0,1].plot(PGAi, MDR_EQ, color = 'black')
    ax[0,1].set_title('Earthquake (EQ)', pad = 20)
    ax[0,1].set_xlabel('PGA [m/s$^2$]')
    ax[0,1].set_ylabel('MDR')
    ax[0,1].set_ylim(0,1.01)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].grid()

    ax[0,2].plot(hwi, MDR_flood, color = 'black')
    ax[0,2].set_title('Flooding (FF, SS)', pad = 20)
    ax[0,2].set_xlabel('Inundation depth $h$ [m]')
    ax[0,2].set_ylabel('MDR')
    ax[0,2].set_ylim(0,1.01)
    ax[0,2].spines['right'].set_visible(False)
    ax[0,2].spines['top'].set_visible(False)
    ax[0,2].grid()

    ax[1,0].plot(hsi, MDR_LS, color = 'black')
    ax[1,0].set_title('Landslide (LS)', pad = 20)
    ax[1,0].set_xlabel('Deposited height $h$ [m]')
    ax[1,0].set_ylabel('MDR')
    ax[1,0].set_ylim(0,1.01)
    ax[1,0].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)
    ax[1,0].grid()

    ax[1,1].plot(pi_VE_kPa, MDR_VE, color = 'black')
    ax[1,1].set_title('Volcanic eruption (VE)', pad = 20)
    ax[1,1].set_xlabel('Ash load $P$ [kPa]')
    ax[1,1].set_ylabel('MDR')
    ax[1,1].set_ylim(0,1.01)
    ax[1,1].spines['right'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].grid()
    ax2 = ax[1,1].twiny()
    ax2.set_xlabel('Ash thickness $h$ [m]')
    ax2.plot(hai, MDR_VE, color = 'white', alpha = 0)
    ax2.spines['right'].set_visible(False)

    ax[1,2].plot(vi, MDR_WS, color = 'black')
    ax[1,2].set_title('Wind (TC, To, WS)', pad = 20)
    ax[1,2].set_xlabel('Maximum wind speed $v_{max}$ [m/s]')
    ax[1,2].set_ylabel('MDR')
    ax[1,2].set_ylim(0,1.01)
    ax[1,2].spines['right'].set_visible(False)
    ax[1,2].spines['top'].set_visible(False)
    ax[1,2].grid()

    fig.tight_layout()
    plt.savefig('figs/vulnFunctions.jpg', dpi = 300)
    plt.pause(1)
    plt.show()
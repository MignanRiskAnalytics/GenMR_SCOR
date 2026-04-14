"""
GenMR Utility Functions
=======================

This module provides miscellaneous utility functions for input/output management, 
computing, and plotting within the GenMR package. These functions support 
core GenMR workflows by streamlining data handling, computation, and visualisation tasks.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.1.2
:Date: 2026-04-14
:License: AGPL-3
"""


import numpy as np

import os
import json
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
from matplotlib.colors import ListedColormap

from shapely.geometry import Point

import networkx as netx
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve



##################
## INPUT/OUTPUT ##
##################
def init_io():
    '''
    Initialise required I/O directories for GenMR.

    Creates the following directories if they do not already exist:

    - io/
    - figs/
    - movs/
    '''
    os.makedirs('io', exist_ok=True)
    os.makedirs('figs', exist_ok=True)
    os.makedirs('movs', exist_ok=True)


def save_dict2json(data, filename = 'par_tmpsave'):
    '''
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        The dictionary to save.
    filename : str, optional
        The name of the JSON file (default is 'par_tmpsave'). The file is saved in the 'io/' directory.

    Returns
    -------
    None
        This function writes the dictionary to a JSON file and does not return a value.
    '''
    file = open('io/' + filename + '.json', 'w')
    json.dump(data, file)


def save_class2pickle(data, filename = 'envLayer_tmpsave'):
    '''
    Save a class instance to a pickle file.

    Parameters
    ----------
    data : object
        The class instance to save.
    filename : str, optional
        The name of the pickle file (default is 'envLayer_tmpsave'). 
        The file is saved in the 'io/' directory.

    Returns
    -------
    None
        This function writes the class instance to a pickle file and does not return a value.
    '''
    with open(os.path.join('io', filename + '.pkl'), 'wb') as file:
        pickle.dump(data, file)


def load_json2dict(filename):
    '''
    Load a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    filename : str
        Path to the JSON file, relative to the current working directory.

    Returns
    -------
    data : dict
        The dictionary obtained from the JSON file.
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = json.load(file)
    return data
    

def load_pickle2class(filename):
    '''
    Load a pickle file and reconstruct the original Python object.

    Parameters
    ----------
    filename : str
        Path to the pickle file, relative to the current working directory.

    Returns
    -------
    data : object
        The Python object loaded from the pickle file.
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = pickle.load(file)
    return data



########################
## ARRAY MANIPULATION ##
########################
def xy_to_rc(x, y, grid):
    '''
    Convert digital template coordinates (x, y) to grid indices (row, col).

    Parameters
    ----------
    x : float
        x-coordinate (km), corresponding to grid row axis.
    y : float
        y-coordinate (km), corresponding to grid column axis.
    grid : RasterGrid
        Grid object with attributes ``x`` and ``y`` as 1D coordinate arrays,
        constructed with ``indexing='ij'`` so that rows correspond to x
        and columns correspond to y.

    Returns
    -------
    row : int
        Row index of the nearest grid cell to ``x``.
    col : int
        Column index of the nearest grid cell to ``y``.
    '''
    row = np.argmin(np.abs(grid.x - x))  # row = x index
    col = np.argmin(np.abs(grid.y - y))  # col = y index
    return (row, col)


def incrementing(xmin, xmax, xbin, method):
    '''
    Return evenly spaced values within a given interval in linear or 
    logarithmic space, or repeat a value a specified number of times.

    Parameters
    ----------
    xmin : float
        The minimum value of the interval.
    xmax : float
        The maximum value of the interval.
    xbin : float or int
        Step size for 'lin' or 'log' methods, or the number of repetitions for 'rep'.
    method : str
        The spacing method. Options are:
        - 'lin' : linear spacing
        - 'log' : logarithmic spacing
        - 'rep' : repeat `xmax` `xbin` times

    Returns
    -------
    xi : numpy.ndarray
        The array of incremented values according to the selected method.
    '''
    if method == 'lin':
        xi = np.arange(xmin, xmax + xbin, xbin)
    if method == 'log':
        xi = 10**np.arange(np.log10(xmin), np.log10(xmax) + xbin, xbin)
    if method == 'rep':
        xi = np.repeat(xmax, xbin)
    return xi


def partitioning(IDs, w, n):
    '''
    Partition weighted IDs into a 1D array of length `n` using cumulative probability.

    Parameters
    ----------
    IDs : array-like
        A list or array of identifiers.
    w : array-like
        The weights associated with each ID. Must be non-negative and typically normalized.
    n : int
        The number of partitions or samples to generate.

    Returns
    -------
    vec_IDs : numpy.ndarray
        A sorted 1D array of length `n` containing IDs selected according to their weights.
    '''
    indsort = np.argsort(w)
    cumPr = np.cumsum(w[indsort])
    midpt = np.arange(1/n, 1+1/n, 1/n) - 1/n/2
    vec_IDs = np.zeros(n).astype(int)
    for i in range(n):
        vec_IDs[i] = IDs[indsort][np.argwhere(cumPr > midpt[i])[0]]
    return np.sort(vec_IDs)


def pooling(m, f, method = 'max'):
    '''
    Downscale a 2D matrix by applying pooling (max, min, or mean) 
    over non-overlapping f x f blocks.

    Parameters
    ----------
    m : numpy.ndarray
        The 2D matrix to be pooled.
    f : int
        The pooling factor, defining the size of each block.
    method : str, optional
        The pooling method to apply (default is 'max'). Options are:
        - 'max'  : maximum value in each block
        - 'min'  : minimum value in each block
        - 'mean' : mean value in each block

    Returns
    -------
    m_pool : numpy.ndarray
        The pooled matrix of shape (ceil(nx/f), ceil(ny/f)), 
        where nx and ny are the dimensions of the input matrix.
    '''
    nx, ny = m.shape
    xpart = int(np.ceil(nx/float(f)))
    ypart = int(np.ceil(ny/float(f)))
    m_pad = np.full((xpart*f, ypart*f), np.nan)
    m_pad[:nx,:ny] = np.copy(m)
    shape_pool = (xpart, f, ypart, f)
    if method == 'max':
        m_pool = np.nanmax(m_pad.reshape(shape_pool), axis = (1,3))
    if method == 'min':
        m_pool = np.nanmin(m_pad.reshape(shape_pool), axis = (1,3))
    if method == 'mean':
        m_pool = np.nanmean(m_pad.reshape(shape_pool), axis = (1,3))
    return m_pool


def zero_boundary_2d(arr, nx, ny):
    '''
    Set the outer boundary of a 2D array to zero along all four edges.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array to be modified.
    nx : int
        The number of rows to zero out at the top and bottom edges.
    ny : int
        The number of columns to zero out at the left and right edges.

    Returns
    -------
    arr : numpy.ndarray
        The input array with its boundary regions set to zero.
    '''
    arr[:nx,:] = 0
    arr[-nx:,:] = 0
    arr[:,:ny] = 0
    arr[:,-ny:] = 0
    return arr


def get_neighborhood_ind(i, j, grid_shape, r_v, method = 'Moore'):
    '''
    Get the indices of neighboring cells around a focal cell in a 2D grid, 
    based on a specified neighborhood type and radius of vision.

    Parameters
    ----------
    i : int
        Row index of the focal cell.
    j : int
        Column index of the focal cell.
    grid_shape : tuple of int
        The shape of the 2D grid as (nx, ny).
    r_v : int
        The radius of vision (neighborhood radius).
    method : str, optional
        The neighborhood definition (default is 'Moore'). Options are:
        - 'Moore'          : all cells within a square radius `r_v`, excluding the center.
        - 'vonNeumann'     : cross-shaped neighborhood with Manhattan distance `r_v`.
        - 'White_etal1997' : circular neighborhood based on Euclidean distance.

    Returns
    -------
    neigh_ind : list of numpy.ndarray
        A list containing two arrays:
        - neigh_ind[0] : row indices of neighboring cells.
        - neigh_ind[1] : column indices of neighboring cells.
        Both arrays exclude the focal cell `(i, j)`.
    '''
    nx, ny = grid_shape
    # rv_box neighborhood
    indx = range(i - r_v, i + r_v + 1)
    indy = range(j - r_v, j + r_v + 1)
    # cut at grid borders
    indx_k = np.array([np.nan if (k < 0 or k > (nx - 1)) else k for k in indx])
    indy_k = np.array([np.nan if (k < 0 or k > (ny - 1)) else k for k in indy])
    indx_cut = ~np.isnan(indx_k)
    indy_cut = ~np.isnan(indy_k)
    ik, jk = [indx_k[indx_cut].astype('int'), indy_k[indy_cut].astype('int')]
    # mask
    mask = np.ones((2*r_v + 1, 2*r_v + 1), dtype = bool)
    nx_mask, ny_mask = mask.shape
    i0 = int(np.floor(nx_mask/2))
    j0 = int(np.floor(ny_mask/2))
    mask[i0,j0] = 0
    if method == 'Moore':
        mask_cut = mask[np.ix_(indx_cut, indy_cut)]
    if method == 'vonNeumann':
        mask = np.zeros((nx_mask, ny_mask), dtype = bool)
        mask[i0,:] = 1
        mask[:,j0] = 1
        mask[i0,j0] = 0
        mask_cut = mask[np.ix_(indx_cut, indy_cut)]
    if method == 'White_etal1997':
        ikk, jkk = np.meshgrid(ik, jk, indexing='ij')
        nx_mask, ny_mask = [len(ik), len(jk)]
        mask_cut = np.zeros((nx_mask, ny_mask), dtype = bool)
#        ic = int(np.floor(nx_mask/2))
#        jc = int(np.floor(nx_mask/2))
        rad = np.sqrt((ikk - ik[i0])**2 + (jkk - jk[j0])**2)
        mask_cut[rad <= r_v] = 1
        mask_cut[i0,j0] = 0
    neigh_ind = [np.meshgrid(ik,jk)[i].flatten()[mask_cut.flatten()] for i in range(2)]
    return neigh_ind


def get_ind_aspect2moore(ind_old):
    '''
    Convert neighborhood indices defined by an aspect-angle system into 
    standard Moore neighborhood indices.

    Parameters
    ----------
    ind_old : int or array-like
        Index or array of indices derived from the aspect-angle convention (0-8),
        where orientation is computed as `np.round(aspect * 8 / 360).astype(int)`.

    Returns
    -------
    ind_new : int or numpy.ndarray
        The corresponding Moore neighborhood indices, following the ordering 
        used in `get_neighborhood_ind()`.

    Notes
    -----
    The aspect-angle system assigns indices as follows:

          7 6 5
          0   4
          1 2 3

    Moore neighborhood indices (used in `get_neighborhood_ind`) follow this pattern:

          0 1 2
          3   4
          5 6 7

    The conversion maps aspect-angle indices to Moore indices.
    '''
    ind_new = np.array([3,5,6,7,4,2,1,0,3])
    return ind_new[ind_old]


def get_S_ceil(size, Si):
    '''
    Map event size to the ceiling size class in Si.

    Returns the smallest value in Si that is greater than or equal to `size`.
    Values outside the range of Si are clipped to the nearest endpoint.
    '''
    Si = np.asarray(Si)
    size = np.asarray(size)
    idx = np.searchsorted(Si, size, side='left')
    idx = np.clip(idx, 0, len(Si) - 1)
    return Si[idx]

def get_S_floor(size, Si):
    '''
    Map event size to the floor size class in Si.

    Returns the largest value in Si that is less than or equal to `size`.
    Values outside the range of Si are clipped to the nearest endpoint.
    '''
    Si = np.asarray(Si)
    size = np.asarray(size)
    idx = np.searchsorted(Si, size, side='right') - 1
    idx = np.clip(idx, 0, len(Si) - 1)
    return Si[idx]


def get_val_grid2loc(loc_coords, fp, grid):
    '''
    Estimate the values at loc_coords given meshed values in grid.
    '''
    cell_coords = np.column_stack((grid.xx.ravel(), grid.yy.ravel()))
    tree = cKDTree(cell_coords)
    _, idx = tree.query(loc_coords)
    val_flat = fp.ravel()[idx]
    return val_flat

def def_distance_kernel(d0, dx):
    '''
    Build a 2-D unnormalised exponential decay kernel.

    Parameters
    ----------
    d0 : float
        Spatial decay length (km). The kernel drops to ``1/e`` at this distance from the centre.
    dx : float
        Grid spacing (km).

    Returns
    -------
    kernel : np.ndarray, shape (2R+1, 2R+1)
        2-D exponential decay kernel, where ``R = int(4 * d0 / dx)``.
    '''
    R = int(4 * d0 / dx) # kernel size: cover ~4 decay lengths
    x = np.arange(-R, R + 1)
    y = np.arange(-R, R + 1)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2) * dx
    kernel = np.exp(-dist / d0)
    return kernel

def gen_convmap4fp(fp, d0, dx):
    '''
    Convolve a 2-D footprint field with an exponential decay kernel.

    NaN values in ``fp`` are treated as zero prior to convolution.

    Parameters
    ----------
    fp : np.ndarray, shape (ny, nx)
        Input footprint field.
    d0 : float
        Spatial decay length passed to ``def_distance_kernel`` (km).
    dx : float
        Grid spacing (km).

    Returns
    -------
    result : np.ndarray, shape (ny, nx)
        Smoothed map.
    '''
    kernel = def_distance_kernel(d0, dx)
    fp_clean = np.nan_to_num(fp, nan=0.)
    result = fftconvolve(fp_clean, kernel, mode='same')
    return result

def rasterize_pts2grid(coords, grid):
    '''
    Rasterise point locations onto a 2-D grid.

    Each point is assigned to its nearest grid cell (nearest-neighbour).
    Cells with one or more points are accumulated (count of points per cell); all others remain 0.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Point coordinates as (x, y) pairs, in the same units as the grid.
    grid : object
        Grid object with attributes ``xx`` and ``yy`` (2-D coordinate
        arrays, km).

    Returns
    -------
    field : np.ndarray, shape (ny, nx)
        Binary raster with 1 at occupied cells and 0 elsewhere.
    '''
    field = np.zeros_like(grid.xx, dtype = float)
    for x, y in coords:
        d = (grid.xx - x)**2 + (grid.yy - y)**2
        idx = np.unravel_index(np.argmin(d), grid.xx.shape)
        field[idx] += 1
    return field

def gen_convmap4pts(coords, d0, grid):
    '''
    Convolve point locations with an exponential decay kernel in a grid.

    Points are first rasterised onto the grid (see ``rasterize_pts2grid``),
    then convolved with an exponential decay kernel of decay length ``d0``.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Point coordinates as (x, y) pairs (km).
    d0 : float
        Spatial decay length passed to ``def_distance_kernel`` (km).
    grid : object
        Grid.

    Returns
    -------
    result : np.ndarray, shape (ny, nx)
        Smoothed point-source map.
    '''
    field4pts = rasterize_pts2grid(coords, grid)
    kernel = def_distance_kernel(d0, grid.w)
    result = fftconvolve(field4pts, kernel, mode='same')
    return result


######################
## SAMPLING METHODS ##
######################
def sample_points_in_polygon(poly, n):
    '''
    Uniform random sampling inside a shapely Polygon.
    '''
    minx, miny, maxx, maxy = poly.bounds
    points = []
    while len(points) < n:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            points.append((x, y))
    return np.array(points)



########################
# NETWORK MANIPULATION #
########################
def get_net_coord(net):
    '''
    Extract node and edge coordinates from a NetworkX graph for visualization.

    Parameters
    ----------
    net : networkx.Graph
        A NetworkX graph whose nodes contain a 'pos' attribute storing (x, y) coordinates.

    Returns
    -------
    node_x : list of float
        The x-coordinates of all nodes.
    node_y : list of float
        The y-coordinates of all nodes.
    edge_x : list of float
        The x-coordinates of edges, arranged as 
        `[x0, x1, None, x0, x1, None, ...]` for plotting line segments.
    edge_y : list of float
        The y-coordinates of edges, arranged as 
        `[y0, y1, None, y0, y1, None, ...]` for plotting line segments.
    '''
    pos = netx.get_node_attributes(net, 'pos')
    node_x = [xx for xx, yy in pos.values()]
    node_y = [yy for xx, yy in pos.values()]
    edge_x = [xx for n0, n1 in net.edges for xx in (pos[n0][0], pos[n1][0], None)]
    edge_y = [yy for n0, n1 in net.edges for yy in (pos[n0][1], pos[n1][1], None)]
    return node_x, node_y, edge_x, edge_y


def graph_to_full_adjacency(g, n_nodes):
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in g.edges():
        A[i, j] = 1
        A[j, i] = 1
    return A


#######################
# PHYSICAL PARAMETERS #
#######################

g_earth = 9.81                     # (m/s^2)
R_earth = 6371.                    # (km)
A_earth = 4 * np.pi * R_earth**2   # (km^2)

A_CONUS = 8080464.                 # (km^2)
A_IT = 301230.                     # (km^2)
A_JP = 377915.                     # (km^2)
A_US_CA = 423970.                  # (km^2)
A_US_FL = 170312.                  # (km^2)

rho_wat = 1000.   # (kg/m^3)
rho_atm = 1.15    # (kg/m^3)

month_labels = ['January','February','March','April','May','June','July','August','September','October','November','December']
month_labels_short = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def get_ndays_inMonth(leap_year = False):
    '''
    Return number of days for each month in a 1D array.

    Parameters
    ----------
    months : ndarray
        1D array of month numbers (1–12)
    leap_year : bool, optional
        If True, February has 29 days (default: False)

    Returns
    -------
    days : ndarray
        Number of days per month
    '''
    months = np.arange(1,13)
    days = np.full_like(months, 31)
    days[np.isin(months, [4, 6, 9, 11])] = 30
    days[months == 2] = 29 if leap_year else 28
    if np.any((months < 1) | (months > 12)):
        raise ValueError('months must be in the range 1–12')
    return days

def fetch_A0(level):
    '''
    Fetch the predefined area value corresponding to a specified geographical level.

    Parameters
    ----------
    level : str
        The geographical level. Options include:
        - 'global'
        - 'CONUS'
        - 'IT'
        - 'JP'
        - 'US_CA'
        - 'US_FL'

    Returns
    -------
    A0 : float
        The area value associated with the specified level.
    '''
    levels = ['global', 'CONUS', 'IT', 'JP', 'US_CA', 'US_FL']
    As = [A_earth, A_CONUS, A_IT, A_JP, A_US_CA, A_US_FL]
    ind = level == levels
    A0 = As[ind]
    return A0

# Tornado Enhanced Fujita (EF) scale: lower bounds considered in m/s, converted from mph
mph2ms = .44704
map_EF2vmax = {
    3: np.round(136 * mph2ms),
    4: np.round(166 * mph2ms),
    5: np.round(201 * mph2ms)
}




##################
## RISK METRICS ##
##################

def calc_EP(lbd):
    '''
    Calculate the cumulative event frequency and the exceedance probability 
    for a series of event rates.

    Parameters
    ----------
    lbd : array-like
        An array of event rates (lambda) for individual events.

    Returns
    -------
    EFi : numpy.ndarray
        Cumulative event frequencies, computed as the cumulative sum of the event rates.
    EPi : numpy.ndarray
        Exceedance probabilities corresponding to each cumulative frequency, calculated as
        `1 - exp(-EFi)` (following Mignan, 2024, eq. 3.22).
    '''
    nev = len(lbd)
    EFi = np.zeros(nev)
    for i in range(nev):
        EFi[i] = np.sum(lbd[0:i+1])
    EPi = 1 - np.exp(- EFi)                               # Mignan (2024:eq. 3.22)
    return EFi, EPi


def calc_riskmetrics_fromELT(ELT, q_VAR):
    '''
    Calculate key risk metrics from an Event Loss Table (ELT), including 
    Average Annual Loss (AAL), Value-at-Risk (VaR), and Tail Value-at-Risk (TVaR).

    Parameters
    ----------
    ELT : pandas.DataFrame
        Event Loss Table containing at least the following columns:
        - 'L'   : Loss for each event
        - 'lbd' : Event rate (lambda) for each event
    q_VAR : float
        Confidence level for Value-at-Risk (e.g., 0.95 for 95% VaR)

    Returns
    -------
    ELT : pandas.DataFrame
        Input ELT augmented with cumulative event frequency ('EF') and 
        exceedance probability ('EP') columns.
    AAL : float
        Average Annual Loss, computed as the sum of lbd * L (Mignan, 2024:eq. 3.18).
    VaRq_interp : float
        Interpolated Value-at-Risk at the given confidence level.
    TVaRq_interp : float
        Interpolated Tail Value-at-Risk at the given confidence level.
    VaRq : float
        Discrete VaR from the ELT.
    TVaRq : float
        Discrete TVaR from the ELT.
    '''
    AAL = np.sum(ELT['lbd'] * ELT['L'])                   # Mignan (2024:eq. 3.18)
    ELT = ELT.sort_values(by = 'L', ascending = False)    # losses in descending order
    EFi, EPi = calc_EP(ELT['lbd'].values)
    ELT['EF'], ELT['EP'] = [EFi, EPi]
    # VaR_q and TVaR_q
    p = 1 - q_VAR
    ELT_asc = ELT.sort_values(by = 'L')                    # losses in ascending order
    VaRq = ELT_asc['L'][ELT_asc['EP'] < p].iloc[0]         # Mignan (2024:eq. 3.23)
    TVaRq = np.sum(ELT_asc['L'][ELT_asc['L'] > VaRq]) / len(ELT_asc['L'][ELT_asc['L'] > VaRq])   # derived from Mignan (2024:eq. 3.24)

    L_hires = 10**np.linspace(np.log10(ELT_asc['L'].min()+1e-6), np.log10(ELT_asc['L'].max()), num = 1000)
    EP_hires = np.interp(L_hires, ELT_asc['L'], ELT_asc['EP'])
    VaRq_interp = L_hires[EP_hires < p][0]
    TVaRq_interp = np.sum(L_hires[L_hires > VaRq_interp]) / len(L_hires[L_hires > VaRq_interp])

    return ELT, AAL, VaRq_interp, TVaRq_interp, VaRq, TVaRq






####################
# PLOTTING OPTIONS #
####################

# terrain color map
n_water, n_land = [50,200]
col_water = plt.cm.terrain(np.linspace(0, 0.17, n_water))
col_land = plt.cm.terrain(np.linspace(0.25, 1, n_land))
col_terrain = np.vstack((col_water, col_land))
cmap_z = plt_col.LinearSegmentedColormap.from_list('cmap_z', col_terrain)
class norm_z(plt_col.Normalize):
    # from https://stackoverflow.com/questions/40895021/python-equivalent-for-matlabs-demcmap-elevation-appropriate-colormap
    # col_val = n_water/(n_water+n_land)
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.2, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel
        self.col_val = col_val
        plt_col.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    
def marker_peril(peril):
    '''
    Return a marker symbol corresponding to a given point-source peril.

    Parameters
    ----------
    peril : str
        The type of point-source peril. Options include:
        - 'AI' : Asteroid impact
        - 'Ex' : Explosion
        - 'VE' : Volcanic eruption

    Returns
    -------
    marker : str
        The symbol used to represent the peril:
        - '+' : for 'AI' or 'Ex'
        - '^' : for 'VE'
        - ''  : empty string if the peril is unrecognized
    '''
    marker = ''
    if peril == 'AI' or peril == 'Ex':
        marker = '+'
    if peril == 'VE':
        marker = '^'
    return(marker)
    

def col_peril(peril):
    '''
    Return a color code corresponding to a given point-source or natural peril.

    Parameters
    ----------
    peril : str
        The type of peril. Options include:
        - 'AI'  : Asteroid impact
        - 'EQ'  : Earthquake
        - 'LS'  : Landslide
        - 'VE'  : Volcanic eruption
        - 'FF'  : Fluvial flood
        - 'SS'  : Storm surge
        - 'RS'  : Rainstorm
        - 'TC'  : Tropical cyclone
        - 'To'  : Tornado
        - 'WS'  : Windstorm
        - 'Ex'  : Explosion

    Returns
    -------
    col : str
        The hexadecimal color code assigned to the peril:
        - '#663399' : 'AI' (Rebeccapurple)
        - '#CD853F' : 'EQ', 'LS', 'VE' (Peru)
        - '#20B2AA' : 'FF', 'SS' (MediumSeaGreen)
        - '#4169E1' : 'RS', 'TC', 'To', 'WS' (RoyalBlue)
        - '#FF8C00' : 'Ex' (Safety Orange)
    '''
    col_peril_extra = '#663399'    #Rebeccapurple
    col_peril_geophys = "#CD853F"  #Peru
    col_peril_hydro = "#20B2AA"    #MediumSeaGreen
    col_peril_meteo = "#4169E1"    #RoyalBlue
    col_peril_tech = "#FF8C00"     #'Safety Orange'
    if peril == 'AI':
        col = col_peril_extra
    if peril == 'EQ' or peril == 'LS' or peril == 'VE':
        col = col_peril_geophys
    if peril == 'FF' or peril == 'SS':
        col = col_peril_hydro
    if peril == 'CS' or peril == 'RS' or peril == 'TC' or peril == 'To' or peril == 'WS':
        col = col_peril_meteo
    if peril == 'Ex':
        col = col_peril_tech
    if peril == 'Ex' or peril == 'Li':
        col = 'red'
    return col


def cmap_peril(peril):
    '''
    Create a two-color matplotlib colormap transitioning from white to the 
    color associated with a given peril.

    Parameters
    ----------
    peril : str
        The type of peril for which to generate the colormap. 
        Passed to `col_peril(peril)` to get the corresponding color.

    Returns
    -------
    cmap_peril : matplotlib.colors.LinearSegmentedColormap
        A two-color colormap transitioning from white to the color representing the peril.
    '''
    colors = [(1,1,1), col_peril(peril)]
    cmap_peril = plt_col.LinearSegmentedColormap.from_list('peril_col', colors, N = 2)
    return cmap_peril


colors = [(0/255.,127/255.,191/255.),        # -1 - water mask
          (236/255., 235/255., 189/255.),    # 0 - grassland (fall green)
          (34/255.,139/255.,34/255.),        # 1 - forest
          (131/255.,137/255.,150/255.),      # 2 - built, residential
          (10/255.,10/255.,10/255.),         # 3 - built, industry
          (230/255.,230/255.,230/255.),      # 4 - built, commercial
          (255/255.,215/255.,0/255.),        # 5 - crop, wheat
          (255/255.,140/255.,0/255.),        # 6 - crop, maize
          (255/255.,0/255.,0/255.)           # 7 - wildfire
         ]          
#col_S = plt_col.LinearSegmentedColormap.from_list('col_S', colors, N = 7)
col_S = plt_col.ListedColormap(colors, name='col_S')

colors = [(0, 100/255., 0),                  # 0 - stable FS (dark green)
          (255/255.,215/255.,0/255.),        # 1 - critical FS (gold)
          (178/255.,34/255.,34/255.)]        # 2 - unstable FS (darkred)
col_FS = plt_col.LinearSegmentedColormap.from_list('col_FS', colors, N = 3)


def col_state_h(h, h0):
    '''
    Assign integer state codes to a height matrix based on erosion and 
    landslide conditions relative to a reference height.

    Parameters
    ----------
    h : numpy.ndarray
        The height matrix.
    h0 : float
        The reference height for scaling the erosion/landslide thresholds.

    Returns
    -------
    h_plot : numpy.ndarray
        An array of the same shape as `h`, where each element is an integer 
        code representing the state:

        - 0 : erosion +++ (scarp)
        - 1 : intact
        - 2 : erosion ++
        - 3 : erosion +
        - 4 : landslide +
        - 5 : landslide ++
    '''
    h_plot = np.copy(h)
    h_plot[h == 0] = 0                               # erosion +++ (scarp)
    h_plot[h == h0] = 1                              # intact
    h_plot[np.logical_and(h > 0, h <= h0/2)] = 2     # erosion ++
    h_plot[np.logical_and(h > h0/2, h < h0)] = 3     # erosion +
    h_plot[np.logical_and(h > h0, h <= 2*h0)] = 4    # landslide +
    h_plot[h > 2*h0] = 5                             # landslide ++
    return h_plot


colors = [(105/255,105/255,105/255),        # 0 - scarp / erosion +++ (dimgrey)
#          (236/255,235/255,189/255),        # 1 - intact (fall green)
          (216/255,228/255,188/255),        # 1 - intact (fall green)
          (195/255,176/255,145/255),        # 2 - erosion ++ (khaki)
          (186/255,135/255,89/255),         # 3 - erosion + (deer)
          (155/255,118/255,83/255),         # 4 - landslide + (dirt)
          (131/255,105/255,83/255)]         # 5 - landslide ++ (pastel brown)
col_h = plt_col.LinearSegmentedColormap.from_list('col_h', colors, N = 6)


col_industrialZone = {
    'industrial harbor': 'royalblue',
    'riverside industrial park': 'skyblue',
    'city industrial park': 'tan'
}
col_commercialZone = {
    'waterfront commercial district': 'darkblue',
    'riverside commercial district': 'blue',
    'city commercial district': 'pink'
}

cmap_mask = ListedColormap(['none', 'lime'])


def get_edge_width(edge_bw, wmin = .5, wmax = 5.):
    '''
    Normalize edge betweenness values to a specified width range for graph plotting.

    Parameters
    ----------
    edge_bw : array-like
        Edge betweenness values for each edge in the graph.
    wmin : float, optional
        Minimum width value (default is 0.5).
    wmax : float, optional
        Maximum width value (default is 5.0).

    Returns
    -------
    widths : ndarray
        Array of scaled edge widths, with same shape as ``edge_bw``.
    '''
    edge_bw = np.array(edge_bw)
    return wmin + (edge_bw - edge_bw.min()) / (edge_bw.max() - edge_bw.min()) * (wmax - wmin)









## DEPRECATED ## - to remove in future revision

#_ROOT = os.path.abspath(os.path.dirname(__file__))
#def get_data(filename):
#    '''
#    DEPRECATED - Return path to package data file.
#    '''
#    return os.path.join(_ROOT, 'data', filename)

#def add0s_iter(i):
#    if i < 10:
#        i_str = '0000' + str(i)
#    elif i < 100:
#        i_str = '000' + str(i)
#    elif i < 1000:
#        i_str = '00' + str(i)
#    elif i < 10000:
#        i_str = '0' + str(i)
#    else:
#        i_str = str(i)
#    return i_str
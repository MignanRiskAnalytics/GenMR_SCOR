"""
GenMR Utility Functions
=======================

This module provides miscellaneous utility functions for input/output management, 
computing, and plotting within the GenMR_Basic package. These functions support 
core GenMR workflows by streamlining data handling, computation, and visualisation tasks.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-10-20
:License: AGPL-3
"""


import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as plt_col



##################
## INPUT/OUTPUT ##
##################
def init_io():
    '''
    Initialise required I/O directories for GenMR.
    Creates the following directories if they do not already exist:
      - io/
      - figs/
      - figs/CellAut_steps/
      - movs/
    '''
    os.makedirs('io', exist_ok=True)
    os.makedirs('figs/CellAut_steps', exist_ok=True)
    os.makedirs('movs', exist_ok=True)


_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(filename):
    '''
    Return path to package data file.
    '''
    return os.path.join(_ROOT, 'data', filename)


def save_dict2json(data, filename = 'par_tmpsave'):
    '''
    Save a dictionary in a json file.
    
    Args:
        data (dict): A dictionary
        filename (str, optional): The name of the json file
        
    Returns:
        A json file
    '''
    file = open('io/' + filename + '.json', 'w')
    json.dump(data, file)


def save_class2pickle(data, filename = 'envLayer_tmpsave'):
    '''
    Save a class instance in a pickle file.
    
    Args:
        data (class): A class instance
        filename (str, optional): The name of the pickle file
        
    Returns:
        A pickle file
    '''
    with open(os.path.join('io', filename + '.pkl'), 'wb') as file:
        pickle.dump(data, file)


def load_json2dict(filename):
    '''
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = json.load(file)
    return data
    

def load_pickle2class(filename):
    '''
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = pickle.load(file)
    return data


def add0s_iter(i):
    if i < 10:
        i_str = '0000' + str(i)
    elif i < 100:
        i_str = '000' + str(i)
    elif i < 1000:
        i_str = '00' + str(i)
    elif i < 10000:
        i_str = '0' + str(i)
    else:
        i_str = str(i)
    return i_str



#######################
## LIST MANIPULATION ##
#######################
def flatten_list(nestedlist):
    '''
    Return a flatten list from a nested list.
    '''
    return [item for sublist in nestedlist for item in sublist]



########################
## ARRAY MANIPULATION ##
########################
def incrementing(xmin, xmax, xbin, method):
    '''
    Return evenly spaced values within a given interval in linear 
    or log space, or repeat xmax xbin times if method = 'rep'.
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
    Return a 1D array of length n of IDs based on their weights w
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
    Return a downscaled matrix by applying pooling (min, mean or max) on a matrix.
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


def get_neighborhood_ind(i, j, grid_shape, r_v, method = 'Moore'):
    '''
    Get the indices of the neighboring cells, depending on method and radius of vision
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
        ic = int(np.floor(nx_mask/2))
        jc = int(np.floor(nx_mask/2))
        rad = np.sqrt((ikk - ik[i0])**2 + (jkk - jk[j0])**2)
        mask_cut[rad <= r_v] = 1
        mask_cut[i0,j0] = 0
    return [np.meshgrid(ik,jk)[i].flatten()[mask_cut.flatten()] for i in range(2)]



#######################
# PHYSICAL PARAMETERS #
#######################
g_earth = 9.81                     # [m/s^2]
R_earth = 6371.                    # [km]
A_earth = 4 * np.pi * R_earth**2   # [km^2]
rho_wat = 1000.   # [kg/m^3]
rho_atm = 1.15    # [kg/m^3]



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
    Return marker for given point source peril.
    '''
    marker = ''
    if peril == 'AI':
        marker = '+'
    if peril == 'VE':
        marker = '^'
    return(marker)
    

def col_peril(peril):
    col_peril_extra = '#663399'    #Rebeccapurple
    col_peril_geophys = "#CD853F"  #Peru
    col_peril_hydro = "#20B2AA"    #MediumSeaGreen
    col_peril_meteo = "#4169E1"    #RoyalBlue
    if peril == 'AI':
        col = col_peril_extra
    if peril == 'EQ' or peril == 'LS' or peril == 'VE':
        col = col_peril_geophys
    if peril == 'FF' or peril == 'SS':
        col = col_peril_hydro
    if peril == 'RS' or peril == 'WS':
        col = col_peril_meteo
    return col


def cmap_peril(peril):
    #white, col_peril
    colors = [(1,1,1), col_peril(peril)]
    cmap_peril = plt_col.LinearSegmentedColormap.from_list('peril_col', colors, N = 2)
    return cmap_peril


colors = [(0/255.,127/255.,191/255.),        # -1 - water mask
          (236/255., 235/255., 189/255.),    # 0 - grassland (fall green)
          (34/255.,139/255.,34/255.),        # 1 - forest
          (131/255.,137/255.,150/255.),      # 2 - built, residential
          (10/255.,10/255.,10/255.),         # 3 - built, industry
          (230/255.,230/255.,230/255.)]      # 4 - built, commercial
col_S = plt_col.LinearSegmentedColormap.from_list('col_S', colors, N = 6)


colors = [(0, 100/255., 0),                  # 0 - stable FS (dark green)
          (255/255.,215/255.,0/255.),        # 1 - critical FS (gold)
          (178/255.,34/255.,34/255.)]        # 2 - unstable FS (darkred)
col_FS = plt_col.LinearSegmentedColormap.from_list('col_FS', colors, N = 3)


def col_state_h(h, h0):
    h_plot = np.copy(h)
    h_plot[h == 0] = 0                               # erosion +++ (scarp)
    h_plot[h == h0] = 1                              # intact
    h_plot[np.logical_and(h > 0, h <= h0/2)] = 2     # erosion ++
    h_plot[np.logical_and(h > h0/2, h < h0)] = 3     # erosion +
    h_plot[np.logical_and(h > h0, h <= 2*h0)] = 4    # landslide +
    h_plot[h > 2*h0] = 5                             # landslide ++
    return h_plot

colors = [(105/255,105/255,105/255),        # 0 - scarp / erosion +++ (dimgrey)
          (236/255,235/255,189/255),        # 1 - intact (fall green)
          (195/255,176/255,145/255),        # 2 - erosion ++ (khaki)
          (186/255,135/255,89/255),         # 3 - erosion + (deer)
          (155/255,118/255,83/255),         # 4 - landslide + (dirt)
          (131/255,105/255,83/255)]         # 5 - landslide ++ (pastel brown)
col_h = plt_col.LinearSegmentedColormap.from_list('col_h', colors, N = 6)



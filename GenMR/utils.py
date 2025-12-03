"""
GenMR Utility Functions
=======================

This module provides miscellaneous utility functions for input/output management, 
computing, and plotting within the GenMR package. These functions support 
core GenMR workflows by streamlining data handling, computation, and visualisation tasks.

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.1.1
:Date: 2025-12-03
:License: AGPL-3
"""


import numpy as np

import os
import json
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as plt_col

import networkx as netx



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
    Load a JSON file and return its contents as a dictionary.
    
    Args:
        filename (str): Path to the JSON file, relative to the current working directory.
        
    Returns:
        data (dict): The dictionary obtained from the JSON file.
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = json.load(file)
    return data
    

def load_pickle2class(filename):
    '''
    Load a pickle file and reconstruct the original Python object.
    
    Args:
        filename (str): The path to the pickle file, relative to the current working directory.
        
    Returns:
        data (object): The Python object loaded from the pickle file.
    '''
    wd = os.getcwd()
    file = open(wd + filename, 'rb')
    data = pickle.load(file)
    return data



#######################
## LIST MANIPULATION ##
#######################
#def flatten_list(nestedlist):
#    '''
#    Return a flatten list from a nested list.
#    '''
#    return [item for sublist in nestedlist for item in sublist]



########################
## ARRAY MANIPULATION ##
########################
def incrementing(xmin, xmax, xbin, method):
    '''
    Return evenly spaced values within a given interval in linear or 
    logarithmic space, or repeat a value a specified number of times.
    
    Args:
        xmin (float): The minimum value of the interval.
        xmax (float): The maximum value of the interval.
        xbin (float or int): The step size (for 'lin' or 'log') or the 
            number of repetitions (for 'rep').
        method (str): The spacing method. Options are:
            'lin' - linear spacing
            'log' - logarithmic spacing
            'rep' - repeat xmax xbin times
        
    Returns:
        xi (numpy.ndarray): The array of incremented values according to 
            the selected method.
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
    Partition weighted IDs into a 1D array of length n.
    
    Args:
        IDs (array-like): A list or array of identifiers.
        w (array-like): The weights associated with each ID. Must be 
            non-negative and typically normalized.
        n (int): The number of partitions or samples to generate.
        
    Returns:
        vec_IDs (numpy.ndarray): A sorted 1D array of length n 
            containing IDs selected according to their weights.
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
    Downscale a matrix by applying pooling (max, min, or mean) over 
    non-overlapping f x f blocks.
    
    Args:
        m (numpy.ndarray): The 2D matrix to be pooled.
        f (int): The pooling factor, defining the size of each block.
        method (str, optional): The pooling method to apply. Options are:
            'max'  - maximum value in each block (default)
            'min'  - minimum value in each block
            'mean' - mean value in each block
        
    Returns:
        m_pool (numpy.ndarray): The pooled matrix of shape 
            (ceil(nx/f), ceil(ny/f)), where nx and ny are the 
            dimensions of the input matrix.
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
    
    Args:
        arr (numpy.ndarray): The 2D array to be modified.
        nx (int): The number of rows to zero out at the top and bottom.
        ny (int): The number of columns to zero out at the left and right.
        
    Returns:
        arr (numpy.ndarray): The input array with its boundary regions set to zero.
    '''
    arr[:nx,:] = 0
    arr[-nx:,:] = 0
    arr[:,:ny] = 0
    arr[:,-ny:] = 0
    return arr


def get_neighborhood_ind(i, j, grid_shape, r_v, method = 'Moore'):
    '''
    Get the indices of neighboring cells around a focal location, based on a 
    specified neighborhood type and radius of vision.
    
    Args:
        i (int): Row index of the focal cell.
        j (int): Column index of the focal cell.
        grid_shape (tuple): The shape of the 2D grid as (nx, ny).
        r_v (int): The radius of vision (neighborhood radius).
        method (str, optional): The neighborhood definition. Options are:
            'Moore'          - all cells within a square radius r_v, excluding center
            'vonNeumann'     - cross-shaped radius r_v (Manhattan distance)
            'White_etal1997' - circular neighborhood based on Euclidean distance
        
    Returns:
        neigh_ind (list of numpy.ndarray): A list containing two arrays:
            neigh_ind[0] - the row indices of neighboring cells  
            neigh_ind[1] - the column indices of neighboring cells
            Both arrays exclude the focal cell (i, j).
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
    
    Args:
        ind_old (int or array-like): Index or array of indices derived from 
            the aspect-angle convention (0-8), where orientation is computed as 
            np.round(aspect * 8 / 360).astype(int).
    
    Returns:
        ind_new (numpy.ndarray or int): The corresponding Moore neighborhood 
            indices, following the ordering used in get_neighborhood_ind().
    
    Note:
        The aspect angle directs towards index np.round(aspect*8/360).astype('int').
        It therefore takes the form:  765
                                      0 4
                                      123
        while Moore indices take the form: 012 (see get_neighborhood_ind() function).
                                           3 4
                                           567
    '''
    ind_new = np.array([3,5,6,7,4,2,1,0,3])
    return ind_new[ind_old]



########################
# NETWORK MANIPULATION #
########################
def get_net_coord(net):
    '''
    Extract node and edge coordinates from a NetworkX graph for visualization.
    
    Args:
        net (networkx.Graph): A NetworkX graph whose nodes contain a 
            'pos' attribute storing (x, y) coordinates.
            
    Returns:
        node_x (list): The x-coordinates of all nodes.
        node_y (list): The y-coordinates of all nodes.
        edge_x (list): The x-coordinates of edges, arranged as 
            [x0, x1, None, x0, x1, None, ...] for plotting line segments.
        edge_y (list): The y-coordinates of edges, arranged as 
            [y0, y1, None, y0, y1, None, ...] for plotting line segments.
    '''
    pos = netx.get_node_attributes(net, 'pos')
    node_x = [xx for xx, yy in pos.values()]
    node_y = [yy for xx, yy in pos.values()]
    edge_x = [xx for n0, n1 in net.edges for xx in (pos[n0][0], pos[n1][0], None)]
    edge_y = [yy for n0, n1 in net.edges for yy in (pos[n0][1], pos[n1][1], None)]
    return node_x, node_y, edge_x, edge_y



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

def fetch_A0(level):
    '''
    Fetch the predefined area value corresponding to a specified geographical level.
    
    Args:
        level (str): The geographical level. Options include:
            'global', 'CONUS', 'IT', 'JP', 'US_CA', 'US_FL'
        
    Returns:
        A0 (float): The area value associated with the specified level.
    '''
    levels = ['global', 'CONUS', 'IT', 'JP', 'US_CA', 'US_FL']
    As = [A_earth, A_CONUS, A_IT, A_JP, A_US_CA, A_US_FL]
    ind = level == levels
    A0 = As[ind]
    return A0



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
    
    Args:
        peril (str): The type of point-source peril. Options include:
            'AI'  - Asteroid impact
            'Ex'  - Explosion
            'VE'  - Volcanic eruption
        
    Returns:
        marker (str): The symbol used to represent the peril:
            '+' for 'AI' or 'Ex'
            '^' for 'VE'
            '' (empty string) if peril is unrecognized
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
    
    Args:
        peril (str): The type of peril. Options include:
            'AI'  - Asteroid impact
            'EQ'  - Earthquake
            'LS'  - Landslide
            'VE'  - Volcanic eruption
            'FF'  - Fluvila flood
            'SS'  - Storm surge
            'RS'  - Rainstorm
            'TC'  - Tropical cyclone
            'WS'  - Windstorm
            'Ex'  - Explosion
        
    Returns:
        col (str): The hexadecimal color code assigned to the peril:
            '#663399' (Rebeccapurple) for 'AI'
            '#CD853F' (Peru) for 'EQ', 'LS', 'VE'
            '#20B2AA' (MediumSeaGreen) for 'FF', 'SS'
            '#4169E1' (RoyalBlue) for 'RS', 'TC', 'WS'
            '#FF8C00' (Safety Orange) for 'Ex'
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
    if peril == 'RS' or peril == 'TC' or peril == 'WS':
        col = col_peril_meteo
    if peril == 'Ex':
        col = col_peril_tech
    return col


def cmap_peril(peril):
    '''
    Create a two-color matplotlib colormap transitioning from white to the 
    color associated with a given peril.
    
    Args:
        peril (str): The type of peril for which to generate the colormap. 
            Passed to col_peril(peril) to get the corresponding color.
        
    Returns:
        cmap_peril (matplotlib.colors.LinearSegmentedColormap): A two-color 
            colormap transitioning from white to the peril color.
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
          (255/255.,0/255.,0/255.)]          # 5 - wildfire
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
    
    Args:
        h (numpy.ndarray): The height matrix.
        h0 (float): The reference height for scaling the erosion/landslide thresholds.
        
    Returns:
        h_plot (numpy.ndarray): An array of the same shape as `h`, where each element 
            is an integer code representing the state:
                0 - erosion +++ (scarp)
                1 - intact
                2 - erosion ++
                3 - erosion +
                4 - landslide +
                5 - landslide ++
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
    'inland industrial park': 'tan'
}



##################
## RISK METRICS ##
##################

def calc_EP(lbd):
    '''
    Calculate the cumulative event frequency and the exceedance probability 
    for a series of event rates.
    
    Args:
        lbd (array-like): An array of event rates (lambda) for individual events.
        
    Returns:
        EFi (numpy.ndarray): Cumulative event frequencies, computed as the 
            cumulative sum of the event rates.
        EPi (numpy.ndarray): Exceedance probabilities corresponding to each 
            cumulative frequency, calculated as 1 - exp(-EFi) 
            (following Mignan, 2024:eq. 3.22).
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
    
    Args:
        ELT (pandas.DataFrame): Event Loss Table containing at least the columns:
            'L'   - loss for each event
            'lbd' - event rate (lambda) for each event
        q_VAR (float): Confidence level for Value-at-Risk (e.g., 0.95 for 95% VaR)
        
    Returns:
        ELT (pandas.DataFrame): The input ELT augmented with cumulative event 
            frequency ('EF') and exceedance probability ('EP') columns.
        AAL (float): Average Annual Loss, computed as the sum of lbd * L 
            (Mignan 2024, eq. 3.18).
        VaRq_interp (float): Interpolated Value-at-Risk at the given confidence level.
        TVaRq_interp (float): Interpolated Tail Value-at-Risk at the given confidence level.
        VaRq (float): Discrete VaR from the ELT.
        TVaRq (float): Discrete TVaR from the ELT.
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
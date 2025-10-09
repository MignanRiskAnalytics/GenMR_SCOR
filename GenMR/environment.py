"""
GenMR Virtual Environment Generator
===================================

This module defines the environmental layers and objects used in the GenMR digital template. 
Environmental layers are implemented as classes, and associated environmental objects are 
defined as classes or methods.

Layers and Related Objects (v1.1.1)
-----------------------------------
* **Topography** — includes tectonic hills, river valleys, and volcanic edifices  
  → properties: slope, aspect
* **Soil** — includes factor of safety
* **Natural land**
* **Urban land** — includes road network  
  → properties: asset value

Planned Additions (v1.1.2)
---------------------------
* Atmosphere
* Power grid
* Population

Author: Arnaud Mignan, Mignan Risk Analytics GmbH
Version: 0.1
Date: 2025-10-09
License: AGPL-3
"""


import copy
import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
ls = plt_col.LightSource(azdeg=45, altdeg=45)

import numpy as np
import pandas as pd

import GenMR.utils as GenMR_utils


#####################
# ENVIRONMENT SETUP #
#####################

class RasterGrid:
    """Define the coordinates (x,y) of the square-pixels of a 2D raster grid.
    
    Notes:
        If x0, xbuffer, ybuffer and/or lat_deg are not provided by the user, 
        they are fixed to xmin, 0, 0 and 45, respectively.

    Attributes:
        par (dict): Input, dictionary with keys ['w', 'xmin', 'x0' (opt.),
                    'xmax', 'ymin', 'ymax', 'xbuffer' (opt.), 'ybuffer' (opt.)]
        w (float): Pixel width in km
        xmin (float): Minimum abcissa of buffer box
        xmax (float): Maximum abcissa of buffer box
        ymin (float): Minimum ordinate of buffer box
        ymax (float): Maximum ordinate of buffer box
        xbuffer (float): Buffer width in the x direction (default is 0.)
        ybuffer (float): Buffer width in the y direction (default is 0.)
        lat_deg (float): Latitude at center of the grid (default is 45.)
        x0 (float): Abscissa of reference N-S coastline (default is xmin)
        x (ndarray(dtype=float, ndim=1)): 1D array of unique abscissas
        y (ndarray(dtype=float, ndim=1)): 1D array of unique ordinates
        xx (ndarray(dtype=float, ndim=2)): 2D array of grid abscissas
        yy (ndarray(dtype=float, ndim=2)): 2D array of grid ordinates
        nx (int): Length of x
        ny (int): Length of y

    Returns: 
        class instance: A new instance of class RasterGrid
    
    Example:
        Create a grid

            >>> grid = RasterGrid({'w': 1, 'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 3})
            >>> grid.x
            array([0., 1., 2.])
            >>> grid.y
            array([0., 1., 2., 3.])
            >>> grid.xx
            array([[0., 0., 0., 0.],
               [1., 1., 1., 1.],
               [2., 2., 2., 2.]])
            >>> grid.yy
            array([[0., 1., 2., 3.],
               [0., 1., 2., 3.],
               [0., 1., 2., 3.]])
    """
    
    def __init__(self, par):
        """
        Initialize RasterGrid
        
        Args:
            par (dict): Dictionary of input parameters with the following keys:
                w (float): Pixel width in km
                xmin (float): Minimum abcissa of buffer box
                xmax (float): Maximum abcissa of buffer box
                ymin (float): Minimum ordinate of buffer box
                ymax (float): Maximum ordinate of buffer box
                xbuffer (float, optional): Buffer width in the x direction (default is 0)
                ybuffer (float, optional): Buffer width in the y direction (default is 0)
                x0 (float, optional): Abscissa of reference N-S coastline (default is xmin)
        """
        
        self.par = par
        self.w = par['w']
        self.xmin = par['xmin']
        self.xmax = par['xmax']
        self.ymin = par['ymin']
        self.ymax = par['ymax']
        if 'xbuffer' in par.keys():
            self.xbuffer = par['xbuffer']
        else:
            self.xbuffer = 0.
        if 'ybuffer' in par.keys():
            self.ybuffer = par['ybuffer']
        else:
            self.ybuffer = 0.
        if 'x0' in par.keys():
            self.x0 = par['x0']
        else:
            self.x0 = self.xmin
        if 'lat_deg' in par.keys():
            self.lat_deg = par['lat_deg']
        else:
            self.lat_deg = 45.
        self.x = np.arange(self.xmin - self.w/2, self.xmax + self.w/2, self.w) + self.w/2
        self.y = np.arange(self.ymin - self.w/2, self.ymax + self.w/2, self.w) + self.w/2
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.nx = len(self.x)
        self.ny = len(self.y)

    def __repr__(self):
        return 'RasterGrid({})'.format(self.par)


def downscale_RasterGrid(grid, factor, appl = 'pooling'):
    '''
    Reduce the resolution of a RasterGrid grid for specific applications.
    appl = topo for topography generation (with outer layer) for later upscaling
         = pooling for max- or mean-pooling
         
    Args:
        xxx
        
    Returns:
        
    '''
    w = grid.w * factor
    if appl == 'topo':
        grid_par_lowres = {'w': w,
                       'xmin': grid.xmin - w/2, 'x0': grid.x0, 'xmax': grid.xmax + w/2,
                       'ymin': grid.ymin - w/2, 'ymax': grid.ymax + w/2,
                       'xbuffer': grid.xbuffer, 'ybuffer': grid.ybuffer, 'lat_deg': grid.lat_deg}
    if appl == 'pooling':
        grid_par_lowres = {'w': w,
                       'xmin': grid.xmin + w/2, 'x0': grid.x0, 'xmax': grid.xmax + w/2,
                       'ymin': grid.ymin + w/2, 'ymax': grid.ymax + w/2,
                       'xbuffer': grid.xbuffer, 'ybuffer': grid.ybuffer, 'lat_deg': grid.lat_deg}
    return RasterGrid(grid_par_lowres)



#################
# PERIL SOURCES #
#################
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
        self.par = par
        self.grid = copy.copy(grid)

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
            return print('No earthquake source initiated in source parameter list')
    
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
        return src_xi, src_yi, src_id, src_L, seg_id, seg_strike, seg_L

    def __repr__(self):
        return 'Src({})'.format(self.par)



##############################
# NATURAL ENVIRONMENT LAYERS #
##############################

def calc_coord_river_dampedsine(grid, par, z = ''):
    '''
    Calculate the (x,y,z) coordinates of the river(s) defined from a damped sine wave.
    '''
    nriv = len(par['riv_y0'])
    river_xi = np.array([])
    river_id = np.array([])
    river_yi = np.array([])
    river_zi = np.array([])
    for riv in range(nriv):
        expdecay = par['riv_A_km'][riv] * np.exp(-par['riv_lbd'][riv] * grid.x)
        yrv_0 = expdecay * np.cos(par['riv_ome'][riv] * grid.x) + par['riv_y0'][riv]
        indy = np.where(grid.y > par['riv_y0'][riv] - 1e-6)[0][0]
        if len(z) != 0:
            zrv_0 = z[:,indy]
            indland = np.where(zrv_0 >= 0)
        else:
            zrv_0 = np.zeros(grid.nx)
            indland = np.arange(grid.nx)
        river_xi = np.append(river_xi, grid.x[indland])
        river_yi = np.append(river_yi, yrv_0[indland])
        river_zi = np.append(river_zi, zrv_0[indland])
        river_id = np.append(river_id, np.repeat(riv, grid.nx)[indland])
    return river_xi, river_yi, river_zi, river_id



####################################
# TECHNOLOGICAL ENVIRONMENT LAYERS #
####################################

# coming soon



#####################################
# SOCIO-ECONOMIC ENVIRONMENT LAYERS #
#####################################

# coming mid 2026



############
# PLOTTING #
############

def plot_src(src, file_ext = '-'):
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
    if 'EQ' in src.par['perils']:
        for src_i in range(len(src.par['EQ']['x'])):
            h_eq, = ax[0].plot(src.par['EQ']['x'][src_i], src.par['EQ']['y'][src_i], color = GenMR_utils.col_peril('EQ'))
        handles.append(h_eq)
        labels.append('Earthquake (EQ)')
    if 'FF' in src.par['perils']:
        river_xi, river_yi, _, river_id = calc_coord_river_dampedsine(src.grid, src.par['FF'])
        for src_i in range(len(src.par['FF']['riv_y0'])):
            indriv = river_id == src_i
            h_ff, = ax[0].plot(river_xi[indriv], river_yi[indriv], color = GenMR_utils.col_peril('FF'))
        handles.append(h_ff)
        labels.append('Fluvial Flood (FF)')
    if 'VE' in src.par['perils']:
        h_ve = ax[0].scatter(src.par['VE']['x'], src.par['VE']['x'], color = GenMR_utils.col_peril('VE'), s=100, marker='^')
        handles.append(h_ve)
        labels.append('Volcanic Eruption (VE)')
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
    ax[0].set_title('Peril source coordinates', size = 14)
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
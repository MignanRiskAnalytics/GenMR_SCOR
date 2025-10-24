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

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-10-20
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import copy

import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
from matplotlib.patches import Patch
ls = plt_col.LightSource(azdeg=45, altdeg=45)

from scipy.interpolate import RegularGridInterpolator

import GenMR.utils as GenMR_utils


#####################
# ENVIRONMENT SETUP #
#####################

class RasterGrid:
    """Define the coordinates (x,y) of the square-pixels of a 2D raster grid.
    
    Notes:
        If x0, xbuffer, ybuffer and/or lat_deg are not provided by the user, 
        they are fixed to xmin, 0, 0 and 45, respectively.

    Args:
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
    
    Args:
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
class EnvLayer_topo:
    '''
    Defines an environmental layer for the topography on a coarser grid before interpolating back to default.
    The state variable is the altitude z(x,y).
    Derived variables include slope and aspect.
    Other characteristics include the coastline coordinates
    
    Returns:
        ... default resolution [m]
    '''
    def __init__(self, src, par):
        self.ID = 'topo'
        self.src = copy.copy(src)
        self.grid = self.src.grid
        self.par = par
        # downscaling
        self.gridlow = downscale_RasterGrid(self.grid, self.par['lores_f'], appl = 'topo')
        # topo construction by intrusion/extrusion
        zx = (self.gridlow.x - self.grid.x0) * self.par['bg_tan(phi)']
        self.z = np.repeat(zx, self.gridlow.ny).reshape(self.gridlow.nx, self.gridlow.ny)
        if self.par['th'] and 'EQ' in self.src.par['perils']:
            self.z += self.model_th_ellipsoid()
        if self.par['vo'] and 'VE' in self.src.par['perils']:
            self.z += self.model_vo_cone()
        if self.par['fr']:
            zfr = self.model_fr_diamondsquare()
            zfr = zfr / np.max(np.abs(zfr)) * np.max(self.z) * self.par['fr_eta']
            self.z += zfr
        if self.par['rv'] and 'FF' in self.src.par['perils']:
            self.z = self.model_rv_dampedsine()
        if self.par['cs']:
            self.z = self.model_cs_compoundbevel()
        # upscaling
        interp = RegularGridInterpolator((self.gridlow.x, self.gridlow.y), self.z)
        self.z = interp((self.grid.xx, self.grid.yy)) * 1e3
        delattr(self, 'gridlow')
        # intrude river channel
        nriv = len(self.src.par['FF']['riv_y0'])
        river_xi, river_yi, river_zi, river_id = self.river_coord
        for riv in range(nriv):
            Q = self.src.par['FF']['Q_m3/s'][riv]
            river_h = Q / (2 * (self.grid.w * 1e3)**2)                       # 2 N-S pixels to facilitate CA flow
            indriv = np.where(river_id == riv)
            river_x = river_xi[indriv]
            river_y = river_yi[indriv]
            for i in range(len(river_x)):
                indx0 = np.where(self.grid.x > river_x[i] - 1e-6)[0]
                y0 = self.grid.y[self.grid.y > river_y[i] - 1e-6][0]
                indy0 = np.where(self.grid.y == y0)[0]
                self.z[indx0[0],indy0] = self.z[indx0[0],indy0] - river_h * 2       # river ordinate
                self.z[indx0[0],indy0-1] = self.z[indx0[0],indy0-1] - river_h * 2   # pixel below river ordinate

    def __repr__(self):
        return 'EnvLayer_topo({},{},{})'.format(repr(self.grid), self.par, self.src)
    
    @property
    def coastline_coord(self):
        if len(self.z) == self.grid.nx:
            xc = [self.grid.x[self.z[:,j] <= 0][-1] for j in range(self.grid.ny)]
            yc = self.grid.y
        else:
            xc = [self.gridlow.x[self.z[:,j] <= 0][-1] for j in range(self.gridlow.ny)]
            yc = self.gridlow.y
        return xc, yc

    @property
    def river_coord(self):
        '''
        Call the function calc_coord_river_dampedsine()
        
        Args:
            Self
            
        Returns:
            ...
        '''
        return calc_coord_river_dampedsine(self.grid, self.src.par['FF'], z = self.z)

    @property
    def slope(self):
        tan_slope, _ = calc_topo_attributes(self.z, self.grid.w)
        slope = np.arctan(tan_slope) * 180 / np.pi
        return slope

    @property
    def aspect(self):
        _, aspect = calc_topo_attributes(self.z, self.grid.w)
        return aspect

    ###############################################
    # layer modifiers, incl. peril source objects #
    ###############################################

    def algo_diamondsquare(self, i, roughness, sig, rng):
        '''
        Generate a fractal topography following the Diamond-Square algorithm
        
        i: iteration integer, determines size of final matrix
        roughness: 0 < < 1, equivalent to 1-H with H: Hurst exponent with Dfractal = 3-H
        m: initial square matrix of size 2^i+1
        sig: standard deviation for random deviations
        '''
        size = 2**i+1
        m = np.zeros((size, size))
        # main loop
        for side_length in 2**np.flip(np.arange(1,i+1)):
            half_side = int(side_length/2)
            #square step
            for col in np.arange(1, size, side_length):
                for row in np.arange(1,size, side_length):
                    avg = np.mean([m[row-1, col-1],                          #upper left
                                   m[row+side_length-1, col-1],              #lower left
                                   m[row-1, col+side_length-1],              #upper right
                                   m[row+side_length-1, col+side_length-1]]) #lower right

                    m[row+half_side-1, col+half_side-1] = avg + rng.normal(0, sig, 1)
            #diamond step
            for row in np.arange(1, size, half_side):
                for col in np.arange((col+half_side)%side_length, size, side_length):
                    avg = np.mean([m[(row-half_side+size) % size-1, col-1],  #above
                                   m[(row+half_side) % size-1, col-1],       #below
                                   m[row-1, (col+half_side) % size-1],       #right
                                   m[row-1, (col-half_side) % size-1]])      #left

                    m[row-1, col-1] = avg + rng.normal(0, sig, 1)
                    #handle the edges by wrapping around to the other side of the array
                    if row == 0:
                        m[size-1-1, col-1] = avg
                    if col == 0: 
                        m[row-1, size-1-1] = avg
            #reduce standard deviation of random deviation by roughness factor
            sig = sig*roughness
        return m

    def model_th_ellipsoid(self):
        zth = np.zeros((self.gridlow.nx, self.gridlow.ny))
        flt_x, flt_y, flt_id, _, seg_id, seg_strike, seg_L = self.src.EQ_char
        n_seg = len(np.unique(seg_id))
        xc, yc, zc = [np.zeros(n_seg), np.zeros(n_seg), np.zeros(n_seg)]
        for seg in range(n_seg):
            indseg = seg_id == seg
            indflt = int(flt_id[indseg][0])
            xi = flt_x[indseg]
            yi = flt_y[indseg]
            z_toptrace = self.src.par['EQ']['z_km'][indflt]
            W = self.src.par['EQ']['w_km'][indflt]
            dip = self.src.par['EQ']['dip_deg'][indflt] * np.pi / 180
            mec = self.src.par['EQ']['mec'][indflt]
            L = seg_L[seg]
            strike = seg_strike[seg] * np.pi / 180
            # ellipsoid axes
            Le = L/np.sqrt(2) # / 2
            We = W/np.sqrt(2) # / 2
            Pe = We/2
            # ellipsoid centroid
            xc_toptrace = np.median(xi)
            yc_toptrace = np.median(yi)
            W_toptrace = W * np.cos(dip)
            W_vert = W * np.sin(dip)
            sign = (xi[1] - xi[0]) / np.abs(xi[1] - xi[0])
            sign2 = strike/np.abs(strike)
            xc[seg] = xc_toptrace + sign*sign2 * np.cos(strike) * W_toptrace / 2
            yc[seg] = yc_toptrace - sign * np.sin(strike) * W_toptrace / 2            
            zc[seg] = z_toptrace - W_vert / 2 + self.par['th_Dz_km']
            # ellipsoid
            for i in range(self.gridlow.nx):
                for j in range(self.gridlow.ny):                
                    # rotate xi, yi, xc, yc
                    x_rot = self.gridlow.x[i] * np.sin(strike) + self.gridlow.y[j] * np.cos(strike)
                    y_rot = -self.gridlow.x[i] * np.cos(strike) + self.gridlow.y[j] * np.sin(strike)
                    xc_rot = xc[seg] * np.sin(strike) + yc[seg] * np.cos(strike)
                    yc_rot = -xc[seg] * np.cos(strike) + yc[seg] * np.sin(strike)  
                    zth_tmp = Pe * np.sqrt(1 - (x_rot-xc_rot)**2/Le**2 - (y_rot-yc_rot)**2/We**2) +zc[seg]
                    if not np.isnan(zth_tmp):
                        if zth[i,j] < zth_tmp:
                            zth[i,j] = zth_tmp
        return zth

    def model_vo_cone(self):
        zvo = np.zeros((self.gridlow.nx, self.gridlow.ny))
        for src_i in range(len(self.src.par['VE']['x'])):
            r = np.sqrt((self.gridlow.xx - self.src.par['VE']['x'][src_i])**2 + \
                        (self.gridlow.yy - self.src.par['VE']['y'][src_i])**2).reshape(self.gridlow.nx, \
                                                                                       self.gridlow.ny)
            indr = np.where(r <= .5 * self.par['vo_w_km'][src_i])
            zvo[indr] = zvo[indr] + (.5 * self.par['vo_w_km'][src_i] - r[indr]) * \
                        self.par['vo_h_km'][src_i] / (.5 * self.par['vo_w_km'][src_i])
        return zvo
    
    def model_fr_diamondsquare(self):
        # topography parameters
        H = 3 - self.par['fr_Df']         # Hurst exponent = roughness max at H = 0
        itr_i = np.arange(3,15)
        nfrac_i = 2**itr_i + 1
        nmax = max([self.gridlow.nx, self.gridlow.ny])
        itr = itr_i[nfrac_i >= nmax][0]   # true fractal -> iter=Inf. Here gives resolution of system    
        l = 2**itr + 1
        rng = np.random.RandomState(self.par['fr_seed'])
        zfr = self.algo_diamondsquare(itr, 1 - H, 1, rng)
        zfr_cropped = zfr[0:self.gridlow.nx, 0:self.gridlow.ny]
        return zfr_cropped

    def model_rv_dampedsine(self):
        nriv = len(self.src.par['FF']['riv_y0'])
        for riv in range(nriv):
            zrv = np.zeros((self.gridlow.nx, self.gridlow.ny))
            # river valley contour            
            expdecay = self.src.par['FF']['riv_A_km'][riv] * \
                        np.exp(-self.src.par['FF']['riv_lbd'][riv] * self.gridlow.x)
            yrv_0 = expdecay * np.cos(self.src.par['FF']['riv_ome'][riv] * self.gridlow.x) \
                        + self.src.par['FF']['riv_y0'][riv]
            yrv_N = self.src.par['FF']['riv_y0'][riv] + (expdecay + self.gridlow.w/2)
            yrv_S = self.src.par['FF']['riv_y0'][riv] - (expdecay + self.gridlow.w/2)
            yrv_N[yrv_N > self.gridlow.ymax] = self.gridlow.ymax
            yrv_S[yrv_S < self.gridlow.ymin] = self.gridlow.ymin
            # river valley z(W-E profile)
            ind = np.where(self.gridlow.y >= self.src.par['FF']['riv_y0'][riv] - 1e-6)[0][0]
            indtmp = self.z[:,ind] >= 0
            x_coastline = self.gridlow.x[indtmp][0]
            zrv_0 = (self.gridlow.x - x_coastline) * self.par['rv_tan(phiWE)']
            # river valley z(x,y)
            for i in range(self.gridlow.nx):
                yS = self.gridlow.y[self.gridlow.y <= yrv_S[i]][-1]
                y0 = self.gridlow.y[self.gridlow.y >= yrv_0[i]][0]
                yN = self.gridlow.y[self.gridlow.y >= yrv_N[i]][0]
                indy0 = np.where(self.gridlow.y == y0)[0]
                zrv[i,indy0] = zrv_0[i]
                zrv[i,indy0-1] = zrv_0[i]
                zrv[i,np.logical_and(self.gridlow.y >= yS, self.gridlow.y < y0)] = zrv_0[i]
                zrv[i,np.logical_and(self.gridlow.y <= yN, self.gridlow.y > y0)] = zrv_0[i]
                zrv[i,self.gridlow.y < yS] = zrv_0[i] + \
                        np.abs(self.gridlow.y[self.gridlow.y < yS] - yS) * self.par['rv_tan(phiNS)']
                zrv[i,self.gridlow.y > yN] = zrv_0[i] + \
                        np.abs(self.gridlow.y[self.gridlow.y > yN] - yN) * self.par['rv_tan(phiNS)']
            for i in range(self.gridlow.nx):
                for j in range(self.gridlow.ny):
                    if zrv[i,j] >= self.z[i,j]:
                        zrv[i,j] = self.z[i,j]
                    else:
                        zrv[i,j] = zrv[i,j] + self.par['rv_eta'] * self.z[i,j]
        return zrv

    def model_cs_compoundbevel(self):
        xc, _ = self.coastline_coord
        zcs = np.copy(self.z)
        for j in range(self.gridlow.ny):
            indcs = np.logical_and(self.gridlow.x >= xc[j], self.gridlow.x < xc[j] + self.par['cs_w_km'])
            indinland = self.gridlow.x >= xc[j] + self.par['cs_w_km']
            zcs_max = np.min([self.par['cs_tan(phi)'] * self.par['cs_w_km'], self.z[indinland, j][0]])
            zcs[indcs,j] = np.linspace(0, zcs_max, np.sum(indcs)) + self.par['cs_eta'] * self.z[indcs,j]
        return zcs


def calc_topo_attributes(z, w):
    z = np.pad(z*1e-3, 1, 'edge')   # from m to km
    # 3x3 kernel method to get dz/dx, dz/dy
    dz_dy, dz_dx = np.gradient(z)
    dz_dx = dz_dx[1:-1,1:-1] / w
    dz_dy = (dz_dy[1:-1,1:-1] / w) * (-1)
    tan_slope = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.arctan(tan_slope) * 180 / np.pi
    aspect = 180 - np.arctan(dz_dy/dz_dx)*180/np.pi + 90 * (dz_dx + 1e-6) / (np.abs(dz_dx) + 1e-6)
    return tan_slope, aspect

    
class EnvLayer_soil:
    '''
    Defines an environmental layer for the soil
    '''
    def __init__(self, topo, par):
        self.ID = 'soil'
        self.topo = copy.copy(topo)
        self.par = par
        self.grid = self.topo.grid
        self.h = np.repeat(self.par['h0_m'], self.grid.nx * self.grid.ny).reshape(self.grid.nx, self.grid.ny)
        self.hw = self.par['wat_h_m']
        if self.par['corr'] == 'remove_unstable':
            # fix h = 0 (scarp) for unstable soil FS<1
            self.h[self.FS_value <= 1] = 0

    @property
    def FS_value(self):
        '''
        Return the factor of safety
        '''
        val = calc_FS(self.topo.slope, self.h, self.wetness, self.par)
        return val
    
    @property
    def FS_state(self):
        FS = np.copy(self.FS_value)
        FS_code = np.zeros((self.grid.nx, self.grid.ny))
        FS_code[FS > 1.5] = 0                                 # stable
        FS_code[np.logical_and(FS > 1, FS <= 1.5)] = 1        # critical
        FS_code[FS <= 1] = 2                                  # unstable
        return FS_code

    @property
    def wetness(self):
        '''
        '''
        wetness = np.ones((self.grid.nx, self.grid.ny))
        indno0 = np.where(self.h != 0)
        wetness[indno0] = self.hw / self.h[indno0]            # hw a scalar for now, possible grid in future
        wetness[wetness > 1] = 1                              # max saturation
        return wetness

        
def calc_FS(slope, h, w, par):
    '''
    Calculates the factor of safety using Eq. 3 of Pack et al. (1998).
        
    Reference:
        Pack RT, Tarboton DG, Goodwin CN (1998), The SINMAP Approach to Terrain Stability Mapping. 
        Proceedings of the 8th Congress of the International Association of Engineering Geology, Vancouver, BC, 
        Canada, 21 September 1998
    '''
    FS = (par['Ceff_Pa'] / (par['rho_kg/m3'] * GenMR_utils.g_earth * h) + np.cos(slope * np.pi/180) * \
         (1 - w * GenMR_utils.rho_wat / par['rho_kg/m3']) * np.tan(par['phieff_deg'] * np.pi/180)) / \
         np.sin(slope * np.pi/180)
    return FS


class EnvLayer_natLand:
    '''
    Defines an environmental layer for the (natural) land classification: water, forest, grassland
    '''
    def __init__(self, soil, par):
        self.ID = 'natLand'
        self.soil = copy.copy(soil)
        self.grid = self.soil.grid
        self.topo = self.soil.topo
        self.src = self.soil.topo.src
        self.par = par
        # class: -1 = water mask, 0 = grassland, 1 = forest, ...
        self.S = np.zeros((self.grid.nx, self.grid.ny))
        self.hW = np.zeros((self.grid.nx, self.grid.ny))
        # define vegetation
        indforest = np.logical_and(self.soil.h >= 0, \
                                   np.logical_and(self.topo.z >= 0, self.topo.z < par['ve_treeline_m']))
        self.S[indforest] = 1
        self.S[self.topo.z < 0] = -1
        # make river channel as 2 N-S bins for smooth flow
        if 'FF' in self.src.par['perils']:
            nriv = len(self.src.par['FF']['riv_y0'])
            river_xi, river_yi, river_zi, river_id = self.topo.river_coord
            for riv in range(nriv):
                indriv = np.where(river_id == riv)
                river_x = river_xi[indriv]
                river_y = river_yi[indriv]
                Q = self.src.par['FF']['Q_m3/s'][riv]
                river_h = Q / (2 * (self.grid.w * 1e3)**2)
                for i in range(len(river_x)):
                    indx0 = np.where(self.grid.x > river_x[i] - 1e-6)[0]
                    y0 = self.grid.y[self.grid.y > river_y[i] - 1e-6][0]
                    indy0 = np.where(self.grid.y == y0)[0]
                    self.S[indx0[0],indy0] = -1
                    self.S[indx0[0],indy0-1] = -1
                    self.hW[indx0[0],indy0] = river_h                      # fill river channel
                    self.hW[indx0[0],indy0-1] = river_h                    # fill river channel


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
        labels.append('Fault segment: Earthquake (EQ)')
    if 'FF' in src.par['perils']:
        river_xi, river_yi, _, river_id = calc_coord_river_dampedsine(src.grid, src.par['FF'])
        for src_i in range(len(src.par['FF']['riv_y0'])):
            indriv = river_id == src_i
            h_ff, = ax[0].plot(river_xi[indriv], river_yi[indriv], color = GenMR_utils.col_peril('FF'))
        handles.append(h_ff)
        labels.append('River: Fluvial Flood (FF)')
    if 'VE' in src.par['perils']:
        h_ve = ax[0].scatter(src.par['VE']['x'], src.par['VE']['x'], color = GenMR_utils.col_peril('VE'), s=100, marker='^')
        handles.append(h_ve)
        labels.append('Volcano: Volcanic Eruption (VE)')
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


def plot_EnvLayer_attr(envLayer, attr, hillshading_z = '', file_ext = '-'):
    '''
    Plot the attribute of an environmental layer.
    
    Note:
        The attr argument should be the string version of the variable name, 
        e.g., z becomes 'z', slope becomes 'slope'.
    
    Args:
        envLayer (class): An instance of environmental layer class
        attr (str): The identifier of the attribute
        hillshading_z (ndarray(dtype=float, ndim=2), optional): The elevation grid array
        file_ext (str, optional): String representing the figure format ('jpg', 'pdf', etc., '-' by default)
    
    Returns:
        A plot
    '''
    fig, ax = plt.subplots(1,1)
    alpha = 1.
    if len(hillshading_z) != 0:
        plt.contourf(envLayer.grid.xx, envLayer.grid.yy, ls.hillshade(hillshading_z, vert_exag=.1), cmap='gray')
        alpha = .5
    if envLayer.ID == 'topo':
        if attr == 'z':
            z_plot = np.copy(envLayer.z)
            z_plot[z_plot < envLayer.par['plt_zmin_m']] = envLayer.par['plt_zmin_m']
            z_plot[z_plot > envLayer.par['plt_zmax_m']] = envLayer.par['plt_zmax_m']
            img = plt.contourf(envLayer.grid.xx, envLayer.grid.yy, z_plot, norm = \
                                     GenMR_utils.norm_z(sealevel = 0, vmax = envLayer.par['plt_zmax_m']), \
                                     cmap = GenMR_utils.cmap_z, levels = np.arange(envLayer.par['plt_zmin_m'], \
                                                            envLayer.par['plt_zmax_m']+100, 100), alpha = .8)
            fig.colorbar(img, ax = ax, fraction = .04, pad = .04, label = 'z (m)')
        elif attr == 'slope':
            img = plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.slope, cmap = 'inferno', alpha = alpha)
            fig.colorbar(img, ax = ax, fraction = .04, pad = .04, label = 'slope ($^\circ$)')
        elif attr == 'aspect':
            img = plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.aspect, cmap = 'coolwarm', alpha = alpha)
            fig.colorbar(img, ax = ax, fraction = .04, pad = .04, label = 'aspect ($^\circ$)')
        else:
            return print('No match found for attribute identifier in topo layer.')
    if envLayer.ID == 'soil':
        if attr == 'h':
            legend_h = [Patch(facecolor=(105/255,105/255,105/255, .5), edgecolor='black', label='h=0 (scarp)'),
                        Patch(facecolor=(216/255,228/255,188/255, .5), edgecolor='black', label='h=$h_0$ (soil)')]
            h_state = GenMR_utils.col_state_h(envLayer.h, envLayer.par['h0_m'])
            plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, h_state, cmap = GenMR_utils.col_h, \
                                         vmin=0, vmax=5, alpha = alpha)
            plt.legend(handles=legend_h, loc='upper left')
        elif attr == 'FS':
            legend_FS = [Patch(facecolor=(0, 100/255., 0, .5), edgecolor='black', label='>1.5 (stable)'),
                        Patch(facecolor=(255/255.,215/255.,0/255., .5), edgecolor='black',label='[1,1.5] (critical)'),
                        Patch(facecolor=(178/255.,34/255.,34/255., .5), edgecolor='black', label='<1 (unstable)')]
            plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.FS_state, cmap = GenMR_utils.col_FS, \
                                         vmin=0, vmax=2, alpha = alpha)
            plt.legend(handles=legend_FS, loc='upper left')
        else:
            return print('No match found for attribute identifier in soil layer.')
    if envLayer.ID == 'natLand':
        if attr == 'S':
            legend_S = [Patch(facecolor=(0/255.,127/255.,191/255.,.5), edgecolor='black', label='Water'),
                        Patch(facecolor=(236/255., 235/255., 189/255.,.5), edgecolor='black', label='Grassland'),
                        Patch(facecolor=(34/255.,139/255.,34/255.,.5), edgecolor='black', label='Forest')]
            plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.S, cmap = GenMR_utils.col_S, \
                                         vmin=-1, vmax=4, alpha = alpha)
            plt.legend(handles=legend_S, loc='upper left')   
        else:
            return print('No match found for attribute identifier in land layer.')
    if envLayer.ID == 'urbLand':
        if attr == 'S':
            legend_S = [Patch(facecolor=(0/255.,127/255.,191/255.,.5), edgecolor='black', label='Water'),
                        Patch(facecolor=(236/255., 235/255., 189/255.,.5), edgecolor='black', label='Grassland'),
                        Patch(facecolor=(34/255.,139/255.,34/255.,.5), edgecolor='black', label='Forest'),
                        Patch(facecolor=(131/255.,137/255.,150/255.,.5), edgecolor='black', label='Residential'),
                        Patch(facecolor=(10/255.,10/255.,10/255.,.5), edgecolor='black', label='Industrial'),
                        Patch(facecolor=(230/255.,230/255.,230/255.,.5), edgecolor='black', label='Commercial')]
            plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.S, cmap = GenMR_utils.col_S, \
                                         vmin=-1, vmax=4, alpha = alpha)
            plt.legend(handles=legend_S, loc='upper left')
        elif attr == 'roadNet':
            plt.plot(envLayer.roadNet_coord[2], envLayer.roadNet_coord[3], color='darkred', lw = 1)
        elif attr == 'bldg_value':
            img = plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.expo_value, cmap = 'inferno_r', alpha = .5)
            fig.colorbar(img, ax = ax, fraction = .04, pad = .04, label = 'Value [$]')
        elif attr == 'built_yr':
            img = plt.pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.built_yr, cmap = 'inferno_r', alpha = .5,\
                                vmin = envLayer.par['city_yr0'], vmax = np.nanmax(envLayer.built_yr))
            fig.colorbar(img, ax = ax, fraction = .04, pad = .04, label = 'Year built')
        else:
            return print('No match found for attribute identifier in land layer.')
    plt.xlabel('$x$ (km)')
    plt.ylabel('$y$ (km)')
    plt.title('Layer:' + envLayer.ID + ' with attribute:' + attr, size = 14)
    ax.set_aspect(1)
    if file_ext != '-':
        plt.savefig('figs/DigitalTemplate_envLayer_' + envLayer.ID + '_' + attr + '.' + file_ext)


def plot_EnvLayers(envLayers, file_ext = '-'):
    '''
    Plot the listed environmental layers for a maximum of 3 attributes/properties
    per layer.
        
    Args:
        envLayers (list): The list of class instances of environmental layers
        save_as (str, optional): String representing the figure format ('jpg' or 'pdf', '-' by default)
    
    Returns:
        A plot
    '''
    nLayers = len(envLayers)
    fig, ax = plt.subplots(nLayers, 3, figsize=(20, 6*nLayers), squeeze = False)
    plt.subplots_adjust(wspace = .25, hspace = .1)
    
    topo_bool = False
    IDs = ''
    for i in range(nLayers):
        envLayer = envLayers[i]
        ## TOPOGRAPHY LAYER ##
        if envLayer.ID == 'topo':
            IDs = IDs + '_topo'
            topo_bool = True
            topo_z = envLayer.z
            topo_xx = envLayer.grid.xx
            topo_yy = envLayer.grid.yy
            z_plot = np.copy(envLayer.z)
            z_plot[z_plot < envLayer.par['plt_zmin_m']] = envLayer.par['plt_zmin_m']
            z_plot[z_plot > envLayer.par['plt_zmax_m']] = envLayer.par['plt_zmax_m']
            ax[i,0].contourf(envLayer.grid.xx, envLayer.grid.yy, ls.hillshade(envLayer.z, vert_exag=.1), cmap='gray')
            img0 = ax[i,0].contourf(envLayer.grid.xx, envLayer.grid.yy, z_plot, norm = \
                                     GenMR_utils.norm_z(sealevel = 0, vmax = envLayer.par['plt_zmax_m']), \
                                     cmap = GenMR_utils.cmap_z, levels = np.arange(envLayer.par['plt_zmin_m'], \
                                                            envLayer.par['plt_zmax_m']+100, 100), alpha = .8)
            if 'EQ' in envLayer.src.par['perils']:
                for src_i in range(len(envLayer.src.par['EQ']['x'])):
                    ax[i,0].plot(envLayer.src.par['EQ']['x'][src_i], envLayer.src.par['EQ']['y'][src_i], \
                                         color = 'yellow', linestyle = 'dashed')
            if 'FF' in envLayer.src.par['perils']:
                river_xi, river_yi, _, river_id = envLayer.river_coord
                for src_i in range(len(envLayer.src.par['FF']['riv_y0'])):
                            indriv = river_id == src_i
                            ax[i,0].plot(river_xi[indriv], river_yi[indriv], color = 'yellow', linestyle = 'dashed')
            if 'VE' in envLayer.src.par['perils']:
                ax[i,0].scatter(envLayer.src.par['VE']['x'], envLayer.src.par['VE']['x'], \
                                        facecolors = 'none', edgecolors = 'yellow', s=100, marker = '^')
            if envLayer.par['cs']:
                coast_x, coast_y = envLayer.coastline_coord
                ax[i,0].plot(coast_x, coast_y, color = 'yellow', linestyle = 'dashed')
            ax[i,0].set_xlabel('$x$ (km)')
            ax[i,0].set_ylabel('$y$ (km)')
            ax[i,0].set_title('TOPOGRAPHY: altitude z', size = 14)
            ax[i,0].set_aspect(1)
            fig.colorbar(img0, ax = ax[i,0], fraction = .04, pad = .04, label = 'z (m)')

            ax[i,1].contourf(envLayer.grid.xx, envLayer.grid.yy, ls.hillshade(envLayer.z, vert_exag=.1), cmap='gray')
            img1 = ax[i,1].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.slope, cmap = 'inferno', alpha = .5)
            ax[i,1].set_xlabel('$x$ (km)')
            ax[i,1].set_title('Slope', size = 14)
            ax[i,1].set_aspect(1)
            fig.colorbar(img1, ax = ax[i,1], fraction = .04, pad = .04, label = 'slope ($^\circ$)')

            ax[i,2].contourf(envLayer.grid.xx, envLayer.grid.yy, ls.hillshade(envLayer.z, vert_exag=.1), cmap='gray')
            img2 = ax[i,2].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.aspect, cmap = 'coolwarm', alpha = .5)
            ax[i,2].set_xlabel('$x$ (km)')
            ax[i,2].set_title('Aspect', size = 14)
            ax[i,2].set_aspect(1)
            fig.colorbar(img2, ax = ax[i,2], fraction = .04, pad = .04, label = 'aspect ($^\circ$)')

        ## SOIL LAYER ##
        if envLayer.ID == 'soil':
            IDs = IDs + '_soil'
            legend_h = [Patch(facecolor=(105/255,105/255,105/255, .5), edgecolor='black', label='h=0 (scarp)'),
                        Patch(facecolor=(216/255,228/255,188/255, .5), edgecolor='black', label='h=$h_0$ (soil)')]
            h_state = GenMR_utils.col_state_h(envLayer.h, envLayer.par['h0_m'])
            legend_FS = [Patch(facecolor=(0, 100/255., 0, .5), edgecolor='black', label='>1.5 (stable)'),
                        Patch(facecolor=(255/255.,215/255.,0/255., .5), edgecolor='black',label='[1,1.5] (critical)'),
                        Patch(facecolor=(178/255.,34/255.,34/255., .5), edgecolor='black', label='<1 (unstable)')]
            
            if topo_bool:
                ax[i,0].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
            ax[i,0].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, h_state, cmap = GenMR_utils.col_h, \
                                         vmin=0, vmax=5, alpha = .5)
            ax[i,0].set_xlabel('$x$ (km)')
            ax[i,0].set_ylabel('$y$ (km)')
            ax[i,0].set_title('SOIL: thickness h', size = 14)
            ax[i,0].set_aspect(1)
            ax[i,0].legend(handles=legend_h, loc='upper left')
                
            if topo_bool:
                ax[i,1].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
            ax[i,1].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.FS_state, cmap = GenMR_utils.col_FS, \
                                         vmin=0, vmax=2, alpha = .5)
            ax[i,1].set_xlabel('$x$ (km)')
            ax[i,1].set_ylabel('$y$ (km)')
            ax[i,1].set_title('Factor of safety', size = 14)
            ax[i,1].set_aspect(1)
            ax[i,1].legend(handles=legend_FS, loc='upper left')
            
            ax[i,2].set_axis_off()

        ## NATURAL LAND LAYER ##
        if envLayer.ID == 'natLand':
            IDs = IDs + '_natLand'
            legend_S = [Patch(facecolor=(0/255.,127/255.,191/255.,.5), edgecolor='black', label='Water'),
                        Patch(facecolor=(236/255., 235/255., 189/255.,.5), edgecolor='black', label='Grassland'),
                        Patch(facecolor=(34/255.,139/255.,34/255.,.5), edgecolor='black', label='Forest')]
            if topo_bool:
                ax[i,0].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
            ax[i,0].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.S, cmap = GenMR_utils.col_S, \
                                         vmin=-1, vmax=4, alpha = .5)
            ax[i,0].set_xlabel('$x$ (km)')
            ax[i,0].set_ylabel('$y$ (km)')
            ax[i,0].set_title('NATURAL LAND', size = 14)
            ax[i,0].set_aspect(1)
            ax[i,0].legend(handles=legend_S, loc='upper left')
            
            ax[i,1].set_axis_off()
            ax[i,2].set_axis_off()

        ## URBAN LAND LAYER ##
        if envLayer.ID == 'urbLand':
            IDs = IDs + '_urbLand'
            legend_S = [Patch(facecolor=(0/255.,127/255.,191/255.,.5), edgecolor='black', label='Water'),
                        Patch(facecolor=(236/255., 235/255., 189/255.,.5), edgecolor='black', label='Grassland'),
                        Patch(facecolor=(34/255.,139/255.,34/255.,.5), edgecolor='black', label='Forest'),
                        Patch(facecolor=(131/255.,137/255.,150/255.,.5), edgecolor='black', label='Residential'),
                        Patch(facecolor=(10/255.,10/255.,10/255.,.5), edgecolor='black', label='Industrial'),
                        Patch(facecolor=(230/255.,230/255.,230/255.,.5), edgecolor='black', label='Commercial')]
            if topo_bool:
                ax[i,0].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
            ax[i,0].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.S, cmap = GenMR_utils.col_S, \
                                         vmin=-1, vmax=4, alpha = .5)
            ax[i,0].set_xlabel('$x$ (km)')
            ax[i,0].set_ylabel('$y$ (km)')
            ax[i,0].set_title('URBAN LAND: state S', size = 14)
            ax[i,0].set_aspect(1)
            ax[i,0].legend(handles=legend_S, loc='upper left')
            
            if topo_bool:
                ax[i,1].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
#            ax[i,1].scatter(envLayer.roadNet_coord[0], envLayer.roadNet_coord[1], color = 'white', edgecolors='black')
#            ax[i,1].plot(envLayer.roadNet_coord[2], envLayer.roadNet_coord[3], color='white', lw = 1, \
#                         path_effects=[pe.Stroke(linewidth = 1.5, foreground='black'), pe.Normal()])
            ax[i,1].plot(envLayer.roadNet_coord[2], envLayer.roadNet_coord[3], color='darkred', lw = 1)
            ax[i,1].set_xlim(envLayer.grid.xmin, envLayer.grid.xmax)
            ax[i,1].set_ylim(envLayer.grid.ymin, envLayer.grid.ymax)
            ax[i,1].set_xlabel('$x$ (km)')
            ax[i,1].set_ylabel('$y$ (km)')
            ax[i,1].set_title('Road network', size = 14)
            ax[i,1].set_aspect(1)

            if topo_bool:
                ax[i,2].contourf(topo_xx, topo_yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray')
            img = ax[i,2].pcolormesh(envLayer.grid.xx, envLayer.grid.yy, envLayer.bldg_value, cmap = 'inferno_r', alpha = .5)
            ax[i,2].set_xlim(envLayer.grid.xmin, envLayer.grid.xmax)
            ax[i,2].set_ylim(envLayer.grid.ymin, envLayer.grid.ymax)
            ax[i,2].set_xlabel('$x$ (km)')
            ax[i,2].set_ylabel('$y$ (km)')
            ax[i,2].set_title('Building value', size = 14)
            ax[i,2].set_aspect(1)
            fig.colorbar(img, ax = ax[i,2], fraction = .04, pad = .04, label = 'Value ($)')
    if file_ext != '-':
        plt.savefig('figs/DigitalTemplate_envLayers' + IDs + '.' + file_ext)
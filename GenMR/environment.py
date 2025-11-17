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
:Version: 1.1.1
:Date: 2025-11-14
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

import networkx as netx

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

    The ``appl`` parameter controls how the downscaling is performed:
    - ``topo``: for topography generation (with outer layer) for later upscaling.
    - ``pooling``: for max- or mean-pooling.

    Args:
        grid (RasterGrid): Input raster grid.
        factor (int): Downscaling factor.
        appl (str, optional): Application mode. Defaults to 'pooling'.

    Returns:
        RasterGrid: Downscaled raster grid.
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
        flt_x, flt_y, flt_id, seg_id, seg_strike, seg_L = (
            self.src.EQ_char[k] for k in ['x', 'y', 'fltID', 'segID', 'strike', 'segL']
        )
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
#        FS_code = np.zeros((self.grid.nx, self.grid.ny))
#        FS_code[FS > 1.5] = 0                                 # stable
#        FS_code[np.logical_and(FS > 1, FS <= 1.5)] = 1        # critical
#        FS_code[FS <= 1] = 2                                  # unstable
        FS_code = get_FS_state(FS)
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

def get_FS_state(FS):
        FS_code = np.zeros_like(FS)
        FS_code[FS > 1.5] = 0                                 # stable
        FS_code[np.logical_and(FS > 1, FS <= 1.5)] = 1        # critical
        FS_code[FS <= 1] = 2                                  # unstable
        return FS_code


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

class EnvObj_roadNetwork():
    '''
    Generate a road network based on the CA described in ...
    '''
    def __init__(self, topo, land, par):
        self.topo = copy.copy(topo)
        self.land = copy.copy(land)
        self.par = par
        self.net = netx.Graph()
        self.net.add_node(0, pos = self.par['city_seed'])
        self.grid = self.topo.grid
        self.Rmax = self.grid.w * self.par['road_Rmax']
        self.mask = np.zeros((self.grid.nx, self.grid.ny), dtype = bool)
        self.S = np.zeros((self.grid.nx, self.grid.ny))     # 0: empty, 1:node, 2:node neighbor, 3:available for node
        self.id = np.full((self.grid.nx, self.grid.ny), -1, dtype = int)
        self.eps = np.random.random(self.grid.nx * self.grid.ny).reshape(self.grid.nx, self.grid.ny)
        self.nodeID = 0
        
        self.mask[self.land.S == -1] = 1
        self.eps[self.mask] = 0
        self.eps[self.topo.slope > self.par['road_maxslope']] = 0
        self.eps[np.logical_or(self.grid.x < self.grid.xmin+self.grid.xbuffer,
                                self.grid.x > self.grid.xmax-self.grid.xbuffer),:] = 0
        self.eps[:,np.logical_or(self.grid.y < self.grid.ymin+self.grid.ybuffer,
                                  self.grid.y > self.grid.ymax-self.grid.ybuffer)] = 0
        
        indxC = np.where(self.grid.x == self.par['city_seed'][0])[0][0]
        indyC = np.where(self.grid.y == self.par['city_seed'][1])[0][0]
        self.S[indxC, indyC] = 1
        self.id[indxC, indyC] = 0
        i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(indxC, indyC, self.S.shape, 1)
        self.S[i_neighbor, j_neighbor] = 2

    def __iter__(self):
        return self
    
    def __next__(self):
        rng = np.random.RandomState(self.par['rdm_seed'])
        i2, j2 = np.where(self.S == 2)
        for k in range(len(i2)):
            i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i2[k], j2[k], self.S.shape, 1)
            S_neighbor = self.S[i_neighbor, j_neighbor]
            ind3 = np.where(S_neighbor == 0)[0]
            self.S[i_neighbor[ind3], j_neighbor[ind3]] = 3
        i3, j3 = np.where(self.S == 3)
        i_node, j_node = np.where(np.logical_and(self.eps == np.max(self.eps[i3, j3]), self.S == 3))
        #new node
        self.nodeID += 1
        self.net.add_node(self.nodeID, pos = (self.grid.x[i_node[0]], self.grid.y[j_node[0]]))
        self.S[i_node, j_node] = 1
        self.id[i_node, j_node] = self.nodeID
        i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_node[0], j_node[0], self.S.shape, 1)
        S_neighbor = self.S[i_neighbor, j_neighbor]
        ind3 = np.where(S_neighbor == 0)[0]
        self.S[i_neighbor, j_neighbor] = 2
        #new edge
        if self.nodeID == 0:
            self.net.add_edge(0, self.nodeID)    
        else:
            # find nodes within Rmax
            i1, j1 = np.where(self.S == 1)
            d2nodes = np.sqrt((self.grid.x[i1] - self.grid.x[i_node])**2 + \
                              (self.grid.y[j1] - self.grid.y[j_node])**2)
            ind2test = np.where(np.logical_and(d2nodes > 1e-6, d2nodes < self.par['road_Rmax']))[0]
            list_nodes = self.id[self.S == 1][ind2test]
            ntest = len(ind2test)
            Lpath = np.zeros(ntest)
            for k in range(ntest):
                testNet = self.net
                testNet.add_edge(list_nodes[k], self.nodeID)
                # get shortest path 
                sp = netx.astar_path(testNet, source = 0, target = self.nodeID)
                Lpath[k] = len(sp)
            # choose path longer than X if possible
            indX = np.where(Lpath >= self.par['road_X'])[0]
            if len(indX) != 0:
                indL = np.where(Lpath == np.min(Lpath[indX]))[0]
            else:
                indL = np.where(Lpath == np.max(Lpath))[0]
            self.net.add_edge(list_nodes[indL[0]], self.nodeID)
            

class EnvLayer_urbLand:
    '''
    Generate a city for land use state classification. Model that combines the SLEUTH city growth CA, 
    the road network growth CA of ref, and the land use state transformation method of ...
    '''
    def __init__(self, natLand, par):
        self.ID = 'urbLand'
        self.grid = copy.copy(natLand.grid)
        self.topo = copy.copy(natLand.topo)
        self.par = par
        # class: -1 = water mask, 0 = grassland, 1 = forest + built: 2 = residential, 3 = industrial, 4 = commercial
        self.S = np.copy(natLand.S)
        self.year = self.par['city_yr0']
        # SLEUTH parameters
        self.urban_yes = np.logical_and(self.S == 1, self.topo.slope < self.par['SLEUTH_maxslope'])
        self.disp_val = int((self.par['SLEUTH_disp'] * .005) * np.sqrt(self.grid.nx**2 + self.grid.ny**2))
        roadg_val = int(self.par['SLEUTH_roadg'] / 100 * (self.grid.nx + self.grid.ny) / 16)
        self.max_roadsearch = 4 * roadg_val * (1 + roadg_val)
        # init road network
        topo_low = copy.copy(self.topo)
        topo_low.grid = downscale_RasterGrid(self.grid, self.par['lores_f'], appl = 'pooling')
        topo_low.z = GenMR_utils.pooling(self.topo.z, self.par['lores_f'], method = 'mean')
        natLand_low = copy.copy(natLand)
        natLand_low.grid = downscale_RasterGrid(self.grid, self.par['lores_f'], appl = 'pooling')
        natLand_low.S = GenMR_utils.pooling(natLand.S, self.par['lores_f'], method = 'min')
        self.roadNet = EnvObj_roadNetwork(topo_low, natLand_low, self.par)
        # init sublayers
        self.built = np.zeros((self.grid.nx,self.grid.ny), dtype = int)         # 0: not built, 1: built
        self.roads = np.zeros((self.grid.nx,self.grid.ny), dtype = int)         # 0: no road, 1: road
        self.built_type = np.full((self.grid.nx,self.grid.ny), -1, dtype = int) # 0: commercial, 1: industry, 2: residential
        self.built_type[natLand.S == -1] = 3                                    # 3: water (+4: roads)
        self.built[np.where(self.grid.x > self.par['city_seed'][0]-1e-6)[0][0], 
                   np.where(self.grid.y > self.par['city_seed'][1]-1e-6)[0][0]] = 1
        self.built_yr = np.full((self.grid.nx,self.grid.ny), np.nan)
        self.built_yr[self.built == 1] = self.year

    def __iter__(self):
        return self

    def __next__(self):
        self.year += 1
        built_new = np.zeros((self.grid.nx,self.grid.ny))
        slope = self.topo.slope
        # spontaneous growth
        for k in range(self.disp_val):
            nroad_px = np.sum(self.roads == 1)
            if nroad_px != 0:
                i_road, j_road = np.where(self.roads == 1)
                rdm = np.random.choice(np.arange(nroad_px))
                i_rdm, j_rdm = [i_road[rdm], j_road[rdm]]
                if self.urban_yes[i_rdm,j_rdm] and np.random.random(1) < \
                        self.calc_Pr_urbanise(slope[i_rdm,j_rdm], self.par) \
                        and i_rdm > 1 and i_rdm < self.grid.nx-2 and j_rdm > 1 and j_rdm < self.grid.ny-2:
                    self.built[i_rdm,j_rdm] = 1           # step (i)
                    # new spreading center growth
                    if np.random.random(1) < self.par['SLEUTH_breed'] / 100:
                        i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_rdm, j_rdm, self.built.shape, 1)
                        urban_yes_neighbor = self.urban_yes[i_neighbor, j_neighbor]
                        built_neighbor = self.built[i_neighbor, j_neighbor]
                        slope_neighbor = slope[i_neighbor, j_neighbor]
                        indbuild = np.logical_and(urban_yes_neighbor, built_neighbor != 1)
                        nbuild = np.sum(indbuild)
                        if nbuild > 0:
                            for l in range(np.min([nbuild, 2])):
                                indnew = np.random.choice(np.where(indbuild)[0], 1)[0]
                                if np.random.random(1) < self.calc_Pr_urbanise(slope_neighbor[indnew], self.par):
                                    i_new = i_neighbor[indnew]
                                    j_new = j_neighbor[indnew]
                                    self.built[i_new,j_new] = 1          # step (ii)
                                    built_new[i_new,j_new] = 1
        # edge growth
        i_built, j_built = np.where(self.built == 1)
        nbuilt = len(i_built)
        for k in range(nbuilt):
            if self.urban_yes[i_built[k], j_built[k]] and np.random.random(1) < self.par['SLEUTH_spread'] / 100 \
                    and i_built[k] > 1 and i_built[k] < self.grid.nx - 2 \
                    and j_built[k] > 1 and j_built[k] < self.grid.ny - 2:
                i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_built[k], j_built[k], self.built.shape, 1)
                built_neighbor = self.built[i_neighbor, j_neighbor]
                slope_neighbor = slope[i_neighbor, j_neighbor]
                indempty = np.where(built_neighbor == 0)[0]
                nempty = len(indempty)
                if nempty > 0:
                    n_builtneighbors = np.array([np.sum(self.built[ \
                                        GenMR_utils.get_neighborhood_ind(i_neighbor[indempty][l], \
                                        j_neighbor[indempty][l], self.built.shape, 1)]) for l in range(nempty)])

                    valid_cell = np.where(n_builtneighbors > 1)[0]
                    for l in range(np.min([len(valid_cell), 2])):
                        indnew = np.random.choice(valid_cell, 1)[0]
                        if np.random.random(1) < self.calc_Pr_urbanise(slope_neighbor[indempty][indnew], self.par):
                            self.built[i_neighbor[indempty][indnew], j_neighbor[indempty][indnew]] = 1  # step (iii)
                            built_new[i_neighbor[indempty][indnew], j_neighbor[indempty][indnew]] = 1
        # Insert road network growth within SLEUTH    
        for rr in range(self.par['road_growth']):
               next(self.roadNet)
        # transfer road network to SLEUTH grid
        node_x, node_y, edge_x, edge_y = GenMR_utils.get_net_coord(self.roadNet.net)
        nedge = int(len(edge_x)/3)
        for i in range(nedge):
            k = i*3
            if edge_x[k+1] - edge_x[k] < 0:
                graph_road_x = np.arange(edge_x[k+1], edge_x[k], self.grid.w / 10)
                graph_road_y = edge_y[k] + (graph_road_x - edge_x[k]) * \
                                (edge_y[k+1] - edge_y[k]) / (edge_x[k+1] - edge_x[k])
            if edge_x[k+1] - edge_x[k] == 0:
                graph_road_x = np.repeat(edge_x[k+1], 200)
                graph_road_y = np.linspace(edge_y[k], edge_y[k+1], 200)
            if edge_x[k+1] - edge_x[k] > 0:
                graph_road_x = np.arange(edge_x[k], edge_x[k+1], self.grid.w / 10)
                graph_road_y = edge_y[k] + (graph_road_x - edge_x[k]) * \
                                (edge_y[k+1] - edge_y[k]) / (edge_x[k+1] - edge_x[k])
            for l in range(len(graph_road_x)):
                ix = np.where(self.grid.x > graph_road_x[l]-1e-6)[0]
                iy = np.where(self.grid.y > graph_road_y[l]-1e-6)[0]
                self.roads[ix[0],iy[0]] = 1
            if i == 0:
                road_coords = np.array([graph_road_x, graph_road_y, np.repeat(i, len(graph_road_x))])
            else:
                road_coords = np.append(road_coords, [graph_road_x, graph_road_y, \
                                                      np.repeat(i, len(graph_road_x))], axis=1)
        self.built_type[self.roads == 1] = 4
        # road-influenced growth
        i_newbuilt, j_newbuilt = np.where(built_new == 1)
        n_newbuilt = len(i_newbuilt)
        kk = 0
        for k in range(n_newbuilt):
            loc_rdm = np.random.choice(np.arange(n_newbuilt), 1)[0]
            d2road = np.sqrt((road_coords[0,:] - self.grid.x[i_newbuilt][loc_rdm])**2 + \
                             (road_coords[1,:] - self.grid.y[j_newbuilt][loc_rdm])**2)
            mind2road = np.min(d2road)
            if mind2road <= self.max_roadsearch * self.grid.w and kk < self.par['SLEUTH_breed']:
                kk += 1
                loc_roadpixel = np.where(d2road == mind2road)[0][0]
                loc_roadrdm = loc_roadpixel + np.random.choice([-1,1],1)[0] * \
                                    int(np.random.random(1)* self.par['SLEUTH_disp'] * 200)
                if loc_roadrdm < 0:
                    loc_roadrdm = 0
                if loc_roadrdm >= len(road_coords[0,:]):
                    loc_roadrdm = len(road_coords[0,:])-1
                x_road = road_coords[0,loc_roadrdm]
                y_road = road_coords[1,loc_roadrdm]
                i_road = np.where(self.grid.x > x_road-1e-6)[0][0]
                j_road = np.where(self.grid.y > y_road-1e-6)[0][0]
                i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_road, j_road, self.built.shape, 1)
                built_neighbor = self.built[i_neighbor, j_neighbor]
                slope_neighbor = slope[i_neighbor, j_neighbor]
                indempty = np.where(built_neighbor == 0)[0]
                nempty = len(indempty)
                if nempty > 0:
                    indnew = np.random.choice(indempty, 1)[0]
                    if np.random.random(1) < self.calc_Pr_urbanise(slope_neighbor[indnew], self.par):
                        self.built[i_neighbor[indnew], j_neighbor[indnew]] = 1    # step (iv)
                        i_neighbor2, j_neighbor2 = GenMR_utils.get_neighborhood_ind(i_neighbor[indnew], \
                                                                        j_neighbor[indnew], self.built.shape, 1)
                        built_neighbor = self.built[i_neighbor2, j_neighbor2]
                        slope_neighbor = slope[i_neighbor2, j_neighbor2]
                        indempty = np.where(built_neighbor == 0)[0]
                        nempty = len(indempty)
                        if nempty >= 2:
                            indnew = np.random.choice(indempty, 2)
                            self.built[i_neighbor2[indnew], j_neighbor2[indnew]] = 1    # step (iv)
        # remove what grew into the mask
        self.built[~self.urban_yes] = 0
        # urban use
        self.built_type = self.get_S_urban(self.built, self.built_type)
        # update final built type: change road to C,I or H
        m_kd = self.transform_landUse()
        indroad = np.where(np.logical_and(self.built == 1, self.built_type == 4))[0]
        nroad = len(indroad)
        #state0_rdm = np.random.choice([0,1,1,1,1,2,2,2,2,2,2,2], nroad)
        state0_rdm = np.repeat(2, nroad)
        newstate_road = np.zeros(nroad, dtype = int)
        newstate_road[:] = -1
        for k in range(nroad):
            i_road, j_road = np.where(np.logical_and(self.built == 1, self.built_type == 4))
            i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_road[k], j_road[k], \
                                                                      self.built.shape, 6, 'White_etal1997')
            neighbor_k = self.built_type[i_neighbor, j_neighbor]
            neighbor_d = self.get_d4mkd(i_road[k], j_road[k], i_neighbor, j_neighbor)
            newstate_road[k] = self.get_state_built(state0_rdm[k], neighbor_k, neighbor_d, m_kd)
        self.built_type[np.logical_and(self.built == 1, self.built_type == 4)] = newstate_road
        # land use state
        self.S[self.built_type == 2] = 2  # housing
        self.S[self.built_type == 1] = 3  # industry
        self.S[self.built_type == 0] = 4  # commercial
        # built attributes
        self.built_yr[np.logical_and(self.S >= 2, np.isnan(self.built_yr))] = self.year

    @property
    def roadNet_coord(self):
        node_x, node_y, edge_x, edge_y = GenMR_utils.get_net_coord(self.roadNet.net)
        return node_x, node_y, edge_x, edge_y
    
    @property
    def bldg_type(self):
        val = np.full((self.grid.nx, self.grid.ny), np.nan, dtype = object)
        val[self.S == 2] = np.where(np.random.random(np.sum(self.S == 2)) >= self.par['res_wood2brick_ratio'], 'M', 'W')
        val[self.S == 3] = 'S'
        val[self.S == 4] = 'RC'
        return val

    @property
    def bldg_roofpitch(self):
        val = np.full((self.grid.nx, self.grid.ny), np.nan, dtype = object)
        val[self.S == 2] = 'H'
        val[self.S == 3] = 'L'
        val[self.S == 4] = 'M'
        return val

    @property
    def bldg_value(self):
        c1 = [24.1, 30.8, 33.6]
        c2 = [.385, .325, .357]
        val = np.full((self.grid.nx, self.grid.ny), np.nan)
        val[self.S == 2] = c1[0] * self.par['GPD_percapita_USD'] **c2[0] * (self.grid.w*1e3)**2
        val[self.S == 3] = c1[1] * self.par['GPD_percapita_USD'] **c2[1] * (self.grid.w*1e3)**2
        val[self.S == 4] = c1[2] * self.par['GPD_percapita_USD'] **c2[2] * (self.grid.w*1e3)**2
        return val
    
    @property
    def infiltration(self):
        # to add - function of built, forest, grassland -> to be used in FF model
        return None

    def calc_Pr_urbanise(self, slope, par):
        expo = par['SLEUTH_slope'] /100 /2.
        pr = ((par['SLEUTH_maxslope'] - np.round(slope)) / par['SLEUTH_maxslope'])**expo
        if slope >= par['SLEUTH_maxslope']:
            pr = 0
        return pr
    
    def transform_landUse(self):
        # White et al. 1997 functions
        nk = 5       # state 0=C, 1=I, 2=H, 3=W, 4=R
        nd = 18      # distances
        m_kd = np.zeros(shape=(3,nk,nd)) # potential to transform to Commercial, Industry, Housing
        m_kd[0,0,:] = [98,98,98,98,38,19,-20,-21,-21,-20,-20,-21,-21,-21,-21,-20,-20,-20]
        m_kd[0,2,:] = [12,8,7,5,5,4,4,3,2,3,2,2,3,2,3,2,3,2]
        m_kd[0,4,0:2] = [98,97]
        m_kd[1,1,:] = [98,97,98,41,15,5,6,5,5,6,7,3,4,0,0,0,0,0]
        m_kd[1,2,:] = [0,0,1,1,2,2,3,4,5,6,7,7,7,7,7,6,6,5]
        m_kd[1,3,0:7] = [56,50,43,35,24,15,6]
        m_kd[2,0,:] = [-20,0,22,21,18,18,16,15,14,13,11,10,8,7,7,5,5,5]
        m_kd[2,1,:] = [-31,-27,-21,-8,-1,3,4,6,6,6,5,6,6,6,6,6,6,6]
        m_kd[2,2,:] = [34,27,23,21,15,14,11,10,9,7,7,6,4,3,4,3,4,3]
        m_kd[2,3,0:3] = [43,23,8]
        m_kd[2,4,:] = [-5,-1,3,3,4,4,4,4,4,3,4,3,3,3,3,3,3,4]
        return m_kd

    def get_d4mkd(self, ic, jc, i_test, j_test):
        r = np.sqrt((i_test-ic)**2 + (j_test-jc)**2)
        d = [1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8), 3, np.sqrt(10), np.sqrt(13), 4, np.sqrt(17), \
             np.sqrt(18), np.sqrt(20), 5, np.sqrt(26), np.sqrt(29), np.sqrt(32), np.sqrt(34), 6]
        neighbor_d = np.concatenate([np.where(d == r[i])[0] for i in range(112)]).ravel()
        return neighbor_d

    def get_state_built(self, state0, neighbor_k, neighbor_d, m_kd):
        alpha = 1.5
        v = 1 + (-np.log(np.random.random(3)))**alpha
        H = np.identity(3)
        indnoNaN = np.where(neighbor_k != -1)[0]
        if len(indnoNaN) != 0:
            prC = v[0] * (1 + np.sum(m_kd[0, neighbor_k[indnoNaN], neighbor_d[indnoNaN]])) + H[state0,0]*4000
            prI = v[1] * (1 + np.sum(m_kd[1, neighbor_k[indnoNaN], neighbor_d[indnoNaN]])) + H[state0,1]*2000  #3000
            prH = v[2] * (1 + np.sum(m_kd[2, neighbor_k[indnoNaN], neighbor_d[indnoNaN]])) + H[state0,2]*4000
            pr = [prC, prI, prH]
            new_state = np.where(pr == np.max(pr))[0]
        else:
            new_state = state0
        return new_state

    def get_S_urban(self, built, built_type):
        '''
        '''
        m_kd = self.transform_landUse()
        i_built, j_built = np.where(built == 1)
        indinit = np.where(built_type[i_built, j_built] == -1)[0]
    #    built_type[i_built[indinit], j_built[indinit]] = np.random.choice([0,1,1,1,1,2,2,2,2,2,2,2], len(indinit))
        built_type[i_built[indinit], j_built[indinit]] = np.repeat(2, len(indinit))
        # all built environment follows White et al. 1997 distribution 
        for k in range(len(i_built)):
            i_neighbor, j_neighbor = GenMR_utils.get_neighborhood_ind(i_built[k], j_built[k], built.shape, \
                                                                      6, 'White_etal1997')
            neighbor_k = built_type[i_neighbor, j_neighbor]
            neighbor_d = self.get_d4mkd(i_built[k], j_built[k], i_neighbor, j_neighbor)
            state0 = built_type[i_built[k], j_built[k]]
            if state0 < 4:   # water (3) state fixed - roads (4) too in loop
                built_type[i_built[k], j_built[k]] = self.get_state_built(state0, neighbor_k, neighbor_d, m_kd)
        return built_type



#####################################
# SOCIO-ECONOMIC ENVIRONMENT LAYERS #
#####################################

# coming mid 2026



############
# PLOTTING #
############

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
        plt.savefig('figs/envLayer_' + envLayer.ID + '_' + attr + '.' + file_ext)


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
        plt.savefig('figs/envLayers' + IDs + '.' + file_ext)
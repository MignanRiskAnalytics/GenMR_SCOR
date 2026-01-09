"""
GenMR Virtual Environment Generator
===================================

This module defines the environmental layers and objects used in the GenMR digital template. 
Environmental layers are implemented as classes, and associated environmental objects are 
defined as classes or methods.

Layers and Related Objects (v1.1.1)
-----------------------------------
* **Topography** - includes tectonic hills, river valleys, and volcanic edifices  
  - properties: slope, aspect
* **Soil**
  - properties: factor of safety
* **Natural land**
* **Urban land** - includes road network  
  - properties: asset value

Layers and Related Objects (v1.1.2)
-----------------------------------
* **Atmosphere**
  - properties: freezing level, tropopause

Planned Additions (v1.1.2)
---------------------------
* Power grid
* Population

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 1.1.2
:Date: 2026-01-09
:License: AGPL-3
"""

import numpy as np
import pandas as pd

import copy

import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
ls = plt_col.LightSource(azdeg=45, altdeg=45)

from shapely.geometry import Polygon, Point, LineString

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

from functools import cached_property
from dataclasses import dataclass
import networkx as netx

import GenMR.utils as GenMR_utils


#####################
# ENVIRONMENT SETUP #
#####################

class RasterGrid:
    """
    Define the coordinates (x, y) of the square pixels of a 2D raster grid.

    Notes
    -----
    If `x0`, `xbuffer`, `ybuffer`, and/or `lat_deg` are not provided by the user, 
    they are set to defaults: `x0 = xmin`, `xbuffer = 0`, `ybuffer = 0`, `lat_deg = 45`.

    Parameters
    ----------
    par : dict
        Dictionary containing the input parameters:
        - w (float)        : Pixel width in km
        - xmin (float)     : Minimum abscissa of buffer box
        - xmax (float)     : Maximum abscissa of buffer box
        - ymin (float)     : Minimum ordinate of buffer box
        - ymax (float)     : Maximum ordinate of buffer box
        - xbuffer (float, optional) : Buffer width in x direction (default 0)
        - ybuffer (float, optional) : Buffer width in y direction (default 0)
        - x0 (float, optional)      : Abscissa of reference N-S coastline (default xmin)
        - lat_deg (float, optional) : Latitude at center of grid (default 45)

    Examples
    --------
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
        self.xmin_nobuffer = self.xmin + self.xbuffer
        self.xmax_nobuffer = self.xmax - self.xbuffer
        self.ymin_nobuffer = self.ymin + self.ybuffer
        self.ymax_nobuffer = self.ymax - self.ybuffer
        self.x = np.arange(self.xmin - self.w/2, self.xmax + self.w/2, self.w) + self.w/2
        self.y = np.arange(self.ymin - self.w/2, self.ymax + self.w/2, self.w) + self.w/2
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.nx = len(self.x)
        self.ny = len(self.y)

    def __repr__(self):
        return 'RasterGrid({})'.format(self.par)


def downscale_RasterGrid(grid, factor, appl = 'pooling'):
    '''
    Reduce the resolution of a RasterGrid for specific applications.

    The ``appl`` parameter controls how the downscaling is performed:
    - ``topo``    : for topography generation (with outer layer) for later upscaling.
    - ``pooling`` : for max- or mean-pooling.

    Parameters
    ----------
    grid : RasterGrid
        Input raster grid.
    factor : int
        Downscaling factor; the width of each new pixel will be `factor * grid.w`.
    appl : str, optional
        Application mode. Options are ``'topo'`` or ``'pooling'``. Default is ``'pooling'``.

    Returns
    -------
    RasterGrid
        Downscaled raster grid with modified pixel width and adjusted boundaries.
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

## TOPOGRAPHY ##
class EnvLayer_topo:
    '''
    Define an environmental layer for the topography on a coarser grid before interpolating back to default.
    The state variable is the altitude z(x,y).
    
    Parameters
    ----------
    src : object
        Source object containing the original RasterGrid and perils information.
    par : dict
        Dictionary of parameters controlling topography generation. Keys include:
        - lores_f : int
            Downscaling factor for the raster grid.
        - bg_tan(phi) : float
            Background slope factor for altitude construction.
        - th : bool
            Include earthquake-related topography (EQ).
        - vo : bool
            Include volcanic eruption cone (VE).
        - fr : bool
            Include fractal diamondsquare topography.
        - fr_eta : float
            Scaling factor for fractal topography.
        - rv : bool
            Include river valley damped sine topography (FF).
        - cs : bool
            Include compound bevel topography.

    Attributes
    ----------
    ID : str
        Identifier of the environmental layer ('topo').
    src : object
        Copy of the source object passed as parameter.
    grid : RasterGrid
        Default-resolution grid associated with the source.
    par : dict
        Parameter dictionary passed at initialization.
    z : numpy.ndarray
        2D array of altitude values in meters at the default resolution.
    gridlow : RasterGrid
        Coarser downscaled grid used temporarily during topography construction (deleted after upscaling).
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
        '''
        Compute the coordinates of the coastline based on the topography.

        Returns
        -------
        xc : list of float
            x-coordinates of the coastline points, taken as the last grid cell 
            in each column where elevation `z <= 0`.
        yc : ndarray of float
            y-coordinates of the grid columns corresponding to the coastline.
        '''
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
        Compute the coordinates and identifiers of rivers using a damped sine model.

        Returns
        -------
        river_xi : ndarray of float
            x-coordinates of river points.
        river_yi : ndarray of float
            y-coordinates of river points.
        river_zi : ndarray of float
            Elevation values at river points.
        river_id : ndarray of int
            Identifier for each river segment.
        '''
        return calc_coord_river_dampedsine(self.grid, self.src.par['FF'], z = self.z)

    @property
    def slope(self):
        '''
        Compute the slope of the topography in degrees.

        Uses a 3x3 kernel finite-difference method to estimate gradients 
        of the elevation field `z(x, y)`.

        Returns
        -------
        slope : ndarray of float
            2D array of slope values in degrees at each grid point.
        '''
        tan_slope, _ = calc_topo_attributes(self.z, self.grid.w)
        slope = np.arctan(tan_slope) * 180 / np.pi
        return slope

    @property
    def aspect(self):
        '''
        Compute the aspect (orientation) of the topography in degrees.

        Aspect is calculated from the x- and y- gradients of the elevation 
        field `z(x, y)` using a 3x3 kernel finite-difference method.

        Returns
        -------
        aspect : ndarray of float
            2D array of aspect values in degrees at each grid point.
            Values indicate the downslope direction.
        '''
        _, aspect = calc_topo_attributes(self.z, self.grid.w)
        return aspect

    ###############################################
    # layer modifiers, incl. peril source objects #
    ###############################################

    def algo_diamondsquare(self, i, roughness, sig, rng):
        '''
        Generate a fractal topography using the Diamond-Square algorithm.

        The algorithm creates a 2D height matrix with fractal properties, 
        controlled by the roughness parameter. The final matrix size is 
        (2^i + 1) × (2^i + 1).

        Parameters
        ----------
        i : int
            Number of iterations; determines the size of the final matrix as 2^i + 1.
        roughness : float
            Roughness parameter (0 < roughness < 1). Equivalent to 1-H, 
            where H is the Hurst exponent (fractal dimension D = 3-H).
        sig : float
            Standard deviation of the random deviations added at each step.
        rng : numpy.random.Generator
            Random number generator used to create Gaussian noise for the surface.

        Returns
        -------
        m : ndarray, shape (2^i + 1, 2^i + 1)
            2D array representing the generated fractal topography.
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
        '''
        Generate a topography perturbation due to earthquake fault segments using ellipsoidal shapes.

        Each fault segment is represented as a 3D ellipsoid based on its geometry
        (length, width, strike, dip) and mechanical parameters.

        Returns
        -------
        zth : ndarray, shape (gridlow.nx, gridlow.ny)
            2D array representing the additional topography induced by 
            all earthquake fault segments. Values are added to the existing 
            topography during intrusion/extrusion.
        '''
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
                    with np.errstate(invalid='ignore', divide='ignore'):
                        zth_tmp = Pe * np.sqrt(1 - (x_rot-xc_rot)**2/Le**2 - (y_rot-yc_rot)**2/We**2) +zc[seg]
                    if not np.isnan(zth_tmp):
                        if zth[i,j] < zth_tmp:
                            zth[i,j] = zth_tmp
        return zth

    def model_vo_cone(self):
        '''
        Generate topography perturbations for volcanic eruptions using conical shapes.

        Each volcanic eruption source is represented as a truncated cone,
        with height and base width defined in the parameters.

        Returns
        -------
        zvo : ndarray, shape (gridlow.nx, gridlow.ny)
            2D array of elevation contributions from all volcanic cones.
            Values are additive to the existing topography.
        '''
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
        '''
        Generate fractal topography using the Diamond-Square algorithm.

        The fractal surface is created on the downscaled grid (`gridlow`) 
        with roughness determined by the Hurst exponent derived from the 
        fractal dimension parameter `fr_Df`.

        Returns
        -------
        zfr_cropped : ndarray, shape (gridlow.nx, gridlow.ny)
            2D array representing the fractal elevation field (unitless or in km
            depending on scaling). Cropped to match the low-resolution grid size.
        '''
        # topography parameters
        H = 3 - self.par['fr_Df']         # Hurst exponent = roughness max at H = 0
        itr_i = np.arange(3,15)
        nfrac_i = 2**itr_i + 1
        nmax = max([self.gridlow.nx, self.gridlow.ny])
        itr = itr_i[nfrac_i >= nmax][0]   # true fractal -> iter=Inf. Here gives resolution of system    
#        l = 2**itr + 1
        rng = np.random.RandomState(self.par['fr_seed'])
        zfr = self.algo_diamondsquare(itr, 1 - H, 1, rng)
        zfr_cropped = zfr[0:self.gridlow.nx, 0:self.gridlow.ny]
        return zfr_cropped

    def model_rv_dampedsine(self):
        '''
        Generate river valley topography using a damped sine function along the x-axis.

        Each river is modeled with an exponentially decaying sinusoidal profile
        in the north-south direction and a linear slope along the west-east direction.
        River depth is blended with the existing topography using a scaling factor.

        Returns
        -------
        zrv : ndarray, shape (gridlow.nx, gridlow.ny)
            2D array representing the elevation adjustments for all river valleys.
            Values are additive to the existing low-resolution topography.
        '''
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
        '''
        Generate a compound bevel topography along the coastline.

        For each column in the grid, the topography is adjusted near the coastline 
        using a linear slope (bevel) over a specified width (`cs_w_km`) and blended 
        with the existing elevation using a scaling factor (`cs_eta`). The slope 
        does not exceed the maximum defined by `cs_tan(phi)` or the inland elevation.

        Returns
        -------
        zcs : ndarray, shape (gridlow.nx, gridlow.ny)
            2D array representing the adjusted elevation with compound bevels 
            along the coastline.
        '''
        xc, _ = self.coastline_coord
        zcs = np.copy(self.z)
        for j in range(self.gridlow.ny):
            indcs = np.logical_and(self.gridlow.x >= xc[j], self.gridlow.x < xc[j] + self.par['cs_w_km'])
            indinland = self.gridlow.x >= xc[j] + self.par['cs_w_km']
            zcs_max = np.min([self.par['cs_tan(phi)'] * self.par['cs_w_km'], self.z[indinland, j][0]])
            zcs[indcs,j] = np.linspace(0, zcs_max, np.sum(indcs)) + self.par['cs_eta'] * self.z[indcs,j]
        return zcs


def calc_topo_attributes(z, w):
    '''
    Compute slope and aspect of a 2D topography grid using finite differences.

    The topography is first converted from meters to kilometers. Gradients 
    are computed with a 3x3 kernel (central differences) using `np.gradient`.
    Slope is expressed in degrees, and aspect is calculated following standard 
    geographic conventions.

    Parameters
    ----------
    z : ndarray, shape (nx, ny)
        2D array of elevation values in meters.
    w : float
        Grid spacing (pixel width) in km.

    Returns
    -------
    tan_slope : ndarray, shape (nx, ny)
        Tangent of the slope at each grid point (unitless).
    aspect : ndarray, shape (nx, ny)
        Aspect angle at each grid point in degrees, measured clockwise from north.
    '''
    z = np.pad(z*1e-3, 1, 'edge')   # from m to km
    # 3x3 kernel method to get dz/dx, dz/dy
    dz_dy, dz_dx = np.gradient(z)
    dz_dx = dz_dx[1:-1,1:-1] / w
    dz_dy = (dz_dy[1:-1,1:-1] / w) * (-1)
    tan_slope = np.sqrt(dz_dx**2 + dz_dy**2)
#    slope = np.arctan(tan_slope) * 180 / np.pi
    aspect = 180 - np.arctan(dz_dy/dz_dx)*180/np.pi + 90 * (dz_dx + 1e-6) / (np.abs(dz_dx) + 1e-6)
    return tan_slope, aspect



## ATMOSPHERE ##
class EnvLayer_atmo:
    '''
    Environmental layer representing atmospheric properties over a raster grid.

    The state variable is near-surface air temperature `T`. IN CONSTRUCTION

    Parameters
    ----------
    topo : EnvLayer_topo
        The topography layer used as reference for the atmosphere layer.
    par : dict
        Dictionary of parameters for the atmosphere layer. Expected keys include:
        - 'month' : int
            Calendar month (1 = January, 12 = December). Used to compute the seasonal phase.

    Attributes
    ----------
    ID : str
        Identifier of the layer ('atmo').
    topo : EnvLayer_topo
        Reference topography layer.
    par : dict
        Parameter dictionary provided at initialization.
    grid : RasterGrid
        Grid associated with the topography layer.
    T0 : float
        Reference near-surface air temperature at z=0.
    T : ndarray, shape (grid.nx, grid.ny)
        Near-surface air temperature at z(x,y).
    '''
    def __init__(self, topo, par):
        self.ID = 'atmo'
        self.topo = copy.copy(topo)
        self.par = par
        self.grid = self.topo.grid
        self.T0, _, _ = self.calc_T0_EBCM(self.par['lat_deg'], self.par['month'], phase = np.pi) # phase hardcoded
        topo_z_corr = self.topo.z.copy() * 1e-3  # m to km
        topo_z_corr[topo_z_corr < 0.] = 0        # water surface at z=0
        self.T = self.calc_T_z(topo_z_corr, self.T0, lapse_rate = self.par['lapse_rate_degC/km'])

    @property
    def z_tropopause(self):
        return self.calc_z_tropopause(self.par['lat_deg'])
    @property
    def z_freezinglevel(self):
        return self.calc_z_freeze(self.T0, self.par['lapse_rate_degC/km'])
    
    
    @staticmethod
    def calc_T0_EBCM(lat, mon, phase = np.pi):
        '''
        Calculate the zonal and seasonal surface temperature in a simple Energy Balance Climate Model (EBCM).

        This function implements a 1-D diffusive EBM with constant albedo, solved analytically 
        following North et al. (1981). It includes:

        - Global annual-mean temperature (T0)
        - Annual-mean latitudinal deviations (T_zonal)
        - Approximate seasonal variation

        Parameters
        ----------
        lat : float or ndarray
            Latitude in degrees (−90° to 90°). Positive for Northern Hemisphere.
        mon : int
            Calendar month (1 = January, 12 = December). Used to compute the seasonal phase.
        phase : float, optional
            Phase shift of the seasonal cycle in radians. Default is π, 
            which aligns Northern Hemisphere summer with month ~6 (June).

        Returns
        -------
        T : float or ndarray
            Temperature at the given latitude and month (°C), including seasonal variation.
        T_zonal : float or ndarray
            Annual-mean latitudinal temperature (zonal) at the given latitude (°C).
        T0 : float
            Global annual-mean temperature (°C).

        References
        ----------
        North et al. (1981), Energy Balance Climate Models. Rev. Geophys. Space Phys. 19(1), 91-121
        '''

        S0 = 1366  # Solar constant (W/m2): irradiance on flat surface perp. to the Sun’s rays at mean Earth–Sun dist.
        A = 211    # (W/m2) - value from Graves et al. (1993:tab.3) as used in North & Stevens (2006)
        B = 1.90   # (W/m2/°C) - value from Graves et al. (1993:tab.3) as used in North & Stevens (2006)
        ap = .7    # Earth coalbedo = 1 - albedo
        S2 = -.477
        D = .649   # (W/m2/°C)

        Q = S0/4.  # because of ratio of a sphere’s area to that of a disk (Adisk = π R**2, Asphere=4π R**2)

        x = np.sin(np.radians(lat))

        # global mean
        T0 = (Q * ap - A) / B               # eq.8, also eq.31

        # zonal deviation (analytic)
        P2 = .5 * (3 * x**2 - 1)            # second Legendre polynomial
        T2 = (Q * ap * S2) / (B + 6*D)      # deriv. from combining eqs.28 and 30 -> T2=Q*H2/(B+6D) (delta_kronecker=0)

        T_zonal = T0 + T2 * P2              # eq.32, see also eq.25 - no seasonality implemented

        # temporal effect (seasonality)
        t = (mon - .5)/12                   # time as fraction of year [0,1]
        P1 = x                              # first Legendre polynomial
        T = T0 + T0 * np.cos(2*np.pi * t - phase) * P1 + T2 * P2      # eq.168

        return T, T_zonal, T0


    @staticmethod
    def calc_T_z(z, T0, lapse_rate = 6.5):
        '''
        Compute atmospheric temperature at height z above the surface.

        Parameters
        ----------
        z : float or ndarray
            Altitude above the surface (km)
        T0 : float or ndarray
            Surface temperature (°C) at z = 0
        lapse_rate : float, optional
            Temperature decrease with altitude (°C/km). Default is 6.5°C/km (average value).

        Returns
        -------
        T : float or ndarray
            Temperature at altitude z (°C)
        '''
        T_z = T0 - lapse_rate * z
        return T_z

        
    @staticmethod
    def calc_z_tropopause(lat, method = 'Mateus_etal2022'):
        '''
        Calculate the tropopause height as a function of latitude using empirical models.

        This function provides approximate tropopause height based on two different
        published methods:

        - `'Son_etal2011'`: Uses the minimum tropopause height from Son et al. (2011)
          and applies a correction to approximate the mean height.
        - `'Mateus_etal2022'`: Uses the sigmoid fit provided by Mateus et al. (2022),
          removing seasonal (day-of-year) dependence
          as its impact is negligible.

        Parameters
        ----------
        lat : float or ndarray
            Latitude in degrees (-90° to 90°). Positive for Northern Hemisphere.
        method : str, optional
            Choice of empirical method. Options are:
            - `'Son_etal2011'` : Uses Son et al. (2011) data with correction.
            - `'Mateus_etal2022'` : Uses Mateus et al. (2022) empirical model (default).

        Returns
        -------
        z_tropopause : float or ndarray
            Tropopause height in kilometers at the specified latitude(s).

        References
        ----------
        Son et al. (2011), The fine-scale structure of the global tropopause derived from 
          COSMIC GPS radio occultation measurements. J. Geophys. Res. 116, D20113, doi:10.1029/2011JD016030
        Mateus et al. (2022), Global Empirical Models for Tropopause Height Determination.
          Remote Sensing 14, 4303, doi: 10.3390/rs14174303
        '''
        if method == 'Son_etal2011':
            z_corr = 5.    # km - to be consistent with 'Mateus_etal2022'
            z_min_tropopause = 7.5 + 2.5 * np.cos(2*np.radians(lat))
            z_tropopause = z_min_tropopause + z_corr
        if method == 'Mateus_etal2022':
            # param. values for northern hemisphere, PVU=3.5 (tab.3)
            a0, a1, a2, a3, a4, a5 = 8.499, 7.823, 26.73, -1.58, .098, -.126
            #doy = (mon-.5) / 12 * 365.
            z_tropopause = a0 + a1 / (1 + np.exp(-(lat - a2)/a3))**a4 #+ a5 * np.cos(2*np.pi * (doy))
        return z_tropopause   # km

    
    @staticmethod
    def calc_z_freeze(T0, lapse_rate):
        '''
        Calculate freezing level in kilometres.

        Parameters
        ----------
        T0 : float or ndarray
            Surface temperature (°C) at z = 0
        lapse_rate : float
            Temperature decrease with altitude (°C/km)

        Returns
        -------
        T : float or ndarray
            Temperature at altitude z (°C)
        '''
        z_freeze = T0 / lapse_rate
        return z_freeze



## SOIL ##
class EnvLayer_soil:
    '''
    Environmental layer representing soil properties over a raster grid.

    The state variable is soil thickness `h`. Additional characteristics 
    include water height `hw` and optional corrections for unstable soil.

    Parameters
    ----------
    topo : EnvLayer_topo
        The topography layer used as reference for the soil layer.
    par : dict
        Dictionary of parameters for the soil layer. Expected keys include:
        - 'h0_m' : float
            Reference soil thickness in meters.
        - 'wat_h_m' : float
            Water height in meters.
        - 'corr' : str
            Correction method for unstable soil, e.g., 'remove_unstable'.

    Attributes
    ----------
    ID : str
        Identifier of the layer ('soil').
    topo : EnvLayer_topo
        Reference topography layer.
    par : dict
        Parameter dictionary provided at initialization.
    grid : RasterGrid
        Grid associated with the topography layer.
    h : ndarray, shape (grid.nx, grid.ny)
        Soil thickness in meters.
    hw : float
        Water height in meters.
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
        Compute the factor of safety (FS) for slope stability.

        Returns
        -------
        FS : numpy.ndarray
            2D array of FS values computed using soil thickness, wetness, slope, and soil parameters.
        '''
        val = calc_FS(self.topo.slope, self.h, self.wetness, self.par)
        return val
    
    @property
    def FS_state(self):
        '''
        Determine the categorical FS state from FS values.

        Returns
        -------
        FS_code : numpy.ndarray
            2D array of integers representing soil stability:
            - 0: stable (FS > 1.5)
            - 1: critical (1 < FS <= 1.5)
            - 2: unstable (FS <= 1)
        '''
        FS = np.copy(self.FS_value)
        FS_code = get_FS_state(FS)
        return FS_code

    @property
    def wetness(self):
        '''
        Compute the soil wetness ratio for each grid cell.

        Returns
        -------
        wetness : numpy.ndarray
            2D array of wetness ratio (dimensionless) in [0,1].
        '''
        wetness = np.ones((self.grid.nx, self.grid.ny))
        indno0 = np.where(self.h != 0)
        wetness[indno0] = self.hw / self.h[indno0]            # hw a scalar for now, possible grid in future
        wetness[wetness > 1] = 1                              # max saturation
        return wetness

        
def calc_FS(slope, h, w, par):
    '''
    Calculate the Factor of Safety (FS) for slope stability using the SINMAP approach 
    (Pack et al., 1998).

    Parameters
    ----------
    slope : numpy.ndarray
        2D array of slope angles in degrees.
    h : numpy.ndarray
        2D array of soil thickness (m).
    w : numpy.ndarray
        2D array of wetness ratio (dimensionless, typically between 0 and 1).
    par : dict
        Dictionary of soil and material parameters:
        - Ceff_Pa : float
            Effective cohesion (Pa).
        - rho_kg/m3 : float
            Soil density (kg/m³).
        - phieff_deg : float
            Effective friction angle (degrees).
        - (optional) Other constants: g_earth, rho_wat used in calculations.

    Returns
    -------
    FS : numpy.ndarray
        2D array of Factor of Safety values for each grid cell.
        Values > 1 indicate stability; values < 1 indicate potential failure.

    References
    ----------
    Pack RT, Tarboton DG, Goodwin CN (1998), The SINMAP Approach to Terrain Stability Mapping. 
    Proceedings of the 8th Congress of the International Association of Engineering Geology, Vancouver, BC, 
    Canada, 21 September 1998
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        FS = (par['Ceff_Pa'] / (par['rho_kg/m3'] * GenMR_utils.g_earth * h) + np.cos(slope * np.pi/180) * \
             (1 - w * GenMR_utils.rho_wat / par['rho_kg/m3']) * np.tan(par['phieff_deg'] * np.pi/180)) / \
             np.sin(slope * np.pi/180)
    return FS

def get_FS_state(FS):
    '''
    Assign a stability code based on the Factor of Safety (FS).

    Parameters
    ----------
    FS : numpy.ndarray
        Array of Factor of Safety values for each grid cell.

    Returns
    -------
    FS_code : numpy.ndarray
        Array of integer stability codes corresponding to FS:
        - 0 : stable (FS > 1.5)
        - 1 : critical (1 < FS <= 1.5)
        - 2 : unstable (FS <= 1)
    '''
    FS_code = np.zeros_like(FS)
    FS_code[FS > 1.5] = 0                                 # stable
    FS_code[np.logical_and(FS > 1, FS <= 1.5)] = 1        # critical
    FS_code[FS <= 1] = 2                                  # unstable
    return FS_code



## NATURAL LAND ##
class EnvLayer_natLand:
    '''
    Defines an environmental layer for natural land classification, including water, forest, and grassland.

    Parameters
    ----------
    soil : EnvLayer_soil
        Soil layer object containing topography, soil thickness, and wetness information.
    par : dict
        Dictionary of parameters controlling land classification. Keys include:
        - ve_treeline_m : float
            Elevation threshold (m) above which forest does not grow.

    Attributes
    ----------
    ID : str
        Identifier of the layer, set to 'natLand'.
    soil : EnvLayer_soil
        Copy of the input soil layer.
    grid : RasterGrid
        Grid object defining the spatial domain and resolution.
    topo : EnvLayer_topo
        Copy of the topography layer associated with the soil layer.
    src : object
        Source object containing original RasterGrid and perils information.
    par : dict
        Dictionary of parameters controlling land classification.
    S : numpy.ndarray
        Integer array of land class codes:
        - -1 : water
        - 0  : grassland
        - 1  : forest
    hW : numpy.ndarray
        River height (m) in river cells, zeros elsewhere.
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
    Calculate the (x, y, z) coordinates of river channels defined by damped sine waves.

    Parameters
    ----------
    grid : RasterGrid
        Raster grid object containing `x` and `y` coordinates of the spatial domain.
    par : dict
        Dictionary containing river parameters. Keys include:
        - riv_y0 : list or array
            Reference y-coordinate of each river source.
        - riv_A_km : list or array
            Amplitude of the damped sine wave (km).
        - riv_lbd : list or array
            Exponential decay coefficient of the wave.
        - riv_ome : list or array
            Angular frequency of the sine wave.
    z : numpy.ndarray or str, optional
        Elevation array of shape (nx, ny). If provided, only points with z >= 0 are included.
        Default is empty string, which treats all points as land (z = 0).

    Returns
    -------
    river_xi : numpy.ndarray
        X-coordinates of river points.
    river_yi : numpy.ndarray
        Y-coordinates of river points.
    river_zi : numpy.ndarray
        Z-coordinates (elevation) of river points.
    river_id : numpy.ndarray
        Integer array indicating the river index for each point.
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

## ROAD NETWORK ##
class EnvObj_roadNetwork():
    '''
    Generate a road network on a raster grid using a cellular automaton (CA) approach.
    Based on the approach developed by Koenig & Bauriedel (2009).

    Parameters
    ----------
    topo : EnvLayer_topo
        Environmental topography layer object containing elevation `z` and slope.
    land : EnvLayer_natLand
        Natural land classification layer, indicating vegetation, water, and land types.
    par : dict
        Dictionary of parameters controlling road network generation. Key entries include:
        - city_seed : tuple(float, float)
            Coordinates of the initial city node (x, y).
        - road_Rmax : float
            Maximum connection distance in units of grid cell width.
        - road_maxslope : float
            Maximum slope (degrees) allowed for road construction.

    Attributes
    ----------
    topo : EnvLayer_topo
        Copy of the input topography layer.
    land : EnvLayer_natLand
        Copy of the input land layer.
    par : dict
        Copy of the input parameter dictionary.
    grid : RasterGrid
        Raster grid associated with the topography layer.
    net : networkx.Graph
        Graph object storing nodes and edges of the road network.
    Rmax : float
        Maximum Euclidean distance for road connections.
    mask : numpy.ndarray
        Boolean array marking unbuildable locations (e.g., water, steep slopes, buffer areas).
    S : numpy.ndarray
        Status array of each grid cell: 0 = empty, 1 = node, 2 = node neighbor, 3 = available for node.
    id : numpy.ndarray
        Grid cell IDs for nodes (-1 for empty).
    eps : numpy.ndarray
        Random number array for stochastic CA processes, masked for invalid locations.
    nodeID : int
        Current maximum node ID in the network.

    References
    ----------
    Koenig R, Bauriedel C (2009), Generating settlement structures: a method for urban planning and analysis supported by cellular automata. 
    Environment and Planning B: Planning and Design, 36, 602-624
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
            

@dataclass
class CriticalInfrastructure:
    '''
    Represents a critical infrastructure facility or asset.

    Parameters
    ----------
    name : str
        Name or identifier of the infrastructure.
    zone_type : str
        Type of industrial zone.
    area : float
        Area of the facility.
    centroid : tuple of float
        Coordinates (x, y) of the facility's centroid.
    polygon : shapely.geometry.Polygon
        Polygon representing the infrastructure footprint.
    distance_to_coast : float
        Minimum distance from the infrastructure to the nearest coastline [km].
    distance_to_river : float
        Minimum distance from the infrastructure to the nearest river [km].
    Ex_S_kton : float
        Explosive or hazard-related characteristic, e.g., explosive stock in kilotons.
    '''
    name: str
    zone_type: str
    area: float
    centroid: tuple
    polygon: Polygon
    distance_to_coast: float
    distance_to_river: float
    Ex_S_kton: float



## URBAN LAND ##
class EnvLayer_urbLand:
    '''
    Generate an urban land use environmental layer.

    This model combines:
    - The SLEUTH city growth cellular automaton (CA), based on Clarke et al. (1997); Candau (2002)
    - A road network growth CA, based on Koenig & Bauriedel (2009)
    - Land use state transformation methods, based on White et al. (1997)

    Parameters
    ----------
    natLand : EnvLayer_natLand
        Natural land layer providing initial land cover, grid, and topography.
    par : dict
        Dictionary of parameters controlling urban growth and SLEUTH CA behavior.
        Keys include:
        - city_yr0 : int, initial year of the city
        - city_seed : tuple(float, float), coordinates of the initial city center
        - SLEUTH_maxslope : float, maximum slope allowed for urban expansion
        - SLEUTH_disp : float, dispersion parameter
        - SLEUTH_roadg : float, road gravity parameter
        - lores_f : int, downscaling factor for grids used in road CA

    Attributes
    ----------
    ID : str
        Layer identifier, here 'urbLand'.
    grid : RasterGrid
        The computational grid for the urban layer.
    topo : EnvLayer_topo
        Topography layer used for slope calculations and urban constraints.
    par : dict
        Parameters controlling urban growth and CA rules (same as input `par`).
    S : ndarray
        Land use classification grid.
        -1: water
         0: grassland
         1: forest
         2: residential
         3: industrial
         4: commercial
    year : int
        Current simulation year.
    urban_yes : ndarray
        Boolean mask indicating which forest cells are eligible for urbanization.
    disp_val : int
        Dispersion value for SLEUTH CA.
    max_roadsearch : int
        Maximum search radius for road network growth.
    roadNet : EnvObj_roadNetwork
        Road network sublayer initialized on a coarsened grid.
    built : ndarray
        Boolean/int grid indicating whether a cell has built structures (0: no, 1: yes).
    roads : ndarray
        Grid indicating presence of roads (0: none, 1: road).
    built_type : ndarray
        Grid indicating type of built-up area:
        0: commercial, 1: industry, 2: residential, 3: water, etc.
    built_yr : ndarray
        Grid storing the year each cell was built.

    References
    ----------
    Candau JT (2002), Temporal calibration sensitivity of the SLEUTH urban growth model. Master Thesis, University of California Santa Barbara, 130 pp.
    Clarke KC, Hoppen S, Gaydos L (1997), A self-modifying cellular automaton model of historical urbanization in the San Francisco Bay area.
    Environment and Planning B: Planning and Design, 24, 247-261. 
    Koenig R, Bauriedel C (2009), Generating settlement structures: a method for urban planning and analysis supported by cellular automata. 
    Environment and Planning B: Planning and Design, 36, 602-624. 
    White R, Engelen G, Uljee I (1997), The use of constrained cellular automata for high-resolution modelling of urban land-use dynamics. 
    Environment and Planning B: Planning and Design, 24, 323-343. 
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
        '''
        Return the coordinates of the road network nodes and edges for plotting.

        Returns
        -------
        node_x : list of float
            X-coordinates of all road network nodes.
        node_y : list of float
            Y-coordinates of all road network nodes.
        edge_x : list of float
            X-coordinates of edges, formatted for plotting line segments 
            as [x0, x1, None, x0, x1, None, ...].
        edge_y : list of float
            Y-coordinates of edges, formatted similarly to `edge_x`.
        '''
        node_x, node_y, edge_x, edge_y = GenMR_utils.get_net_coord(self.roadNet.net)
        return node_x, node_y, edge_x, edge_y
    
    @property
    def bldg_type(self):
        '''
        Assign building construction types based on urban land class and probability rules.

        Returns
        -------
        val : ndarray(dtype=object, shape=(nx, ny))
            Array storing building types for each grid cell:
            - 'M' : masonry (residential)
            - 'W' : wood (residential)
            - 'S' : steel (industrial)
            - 'RC': reinforced concrete (commercial)
            Cells not built or outside urban areas are `nan`.
        '''
        val = np.full((self.grid.nx, self.grid.ny), np.nan, dtype = object)
        val[self.S == 2] = np.where(np.random.random(np.sum(self.S == 2)) >= self.par['bldg_RES_wood2brick'], 'M', 'W')
        val[self.S == 3] = 'S'
        val[self.S == 4] = 'RC'
        return val

    @property
    def bldg_roofpitch(self):
        '''
        Assign a roof pitch type to buildings based on urban land class. Not used yet in damage assessment.

        Returns
        -------
        val : ndarray(dtype=object, shape=(nx, ny))
            Array storing roof pitch type for each grid cell:
            - 'H' : high-pitch roof (residential)
            - 'L' : low-pitch roof (industrial)
            - 'M' : medium-pitch roof (commercial)
            Cells not built or outside urban areas are `nan`.
        '''
        val = np.full((self.grid.nx, self.grid.ny), np.nan, dtype = object)
        val[self.S == 2] = 'H'
        val[self.S == 3] = 'L'
        val[self.S == 4] = 'M'
        return val

    @property
    def bldg_value(self):
        '''
        Estimate the replacement/building value for each urban cell according to Huizinga et al. (2017)

        Returns
        -------
        val : ndarray(dtype=float, shape=(nx, ny))
            Array of building values (USD) per grid cell based on:
            - Residential (S==2)
            - Industrial (S==3)
            - Commercial (S==4)
            Computed as: c1 * (GDP_percapita_USD)**c2 * (cell_area_m2)
            Cells not built or outside urban areas are `nan`.

        References
        ----------
        Huizinga J, de Moel H, Szewczyk W (2017), Global Flood Depth-Damage Functions. Methodology and the Database with Guidelines. 
        JRC Technical Reports. EUR 28552 EN.
        '''
        c1 = [24.1, 30.8, 33.6]
        c2 = [.385, .325, .357]
        val = np.full((self.grid.nx, self.grid.ny), np.nan)
        val[self.S == 2] = c1[0] * self.par['GPD_percapita_USD'] **c2[0] * (self.grid.w*1e3)**2
        val[self.S == 3] = c1[1] * self.par['GPD_percapita_USD'] **c2[1] * (self.grid.w*1e3)**2
        val[self.S == 4] = c1[2] * self.par['GPD_percapita_USD'] **c2[2] * (self.grid.w*1e3)**2
        return val
    
#    @property
#    def infiltration(self):
#        # to add - function of built, forest, grassland -> to be used in FF model
#        return None

    def calc_Pr_urbanise(self, slope, par):
        '''
        Calculate the probability of urbanisation based on local slope using the SLEUTH model.

        Parameters
        ----------
        slope : float or ndarray
            Local slope [degrees] at a grid cell or array of slopes.
        par : dict
            Dictionary of SLEUTH parameters:
            - 'SLEUTH_maxslope' : float
                Maximum slope that can be urbanized [degrees].
            - 'SLEUTH_slope' : float
                Slope influence coefficient [%].

        Returns
        -------
        pr : float or ndarray
            Probability of urbanisation for the given slope(s), in the range [0, 1].
            Returns 0 if slope exceeds 'SLEUTH_maxslope'.
        '''
        expo = par['SLEUTH_slope'] /100 /2.
        if slope >= par['SLEUTH_maxslope']:
            pr = 0
        else:
            pr = ((par['SLEUTH_maxslope'] - np.round(slope)) / par['SLEUTH_maxslope'])**expo
        return pr
    
    def transform_landUse(self):
        '''
        Return the land-use transition potential matrix from White et al. (1997).

        This function constructs the **mₖd matrix**, which encodes the potentials
        for converting existing land-use classes into new urban land-use types
        (Commercial, Industrial, Housing) as a function of distance classes.

        The formulation follows White et al. (1997), where each entry
        ``m_kd[k, current_state, d]`` represents the potential for land of type
        ``current_state`` to transform into target class ``k`` at distance index ``d``.

        Notes
        -----
        - Number of target land-use classes: ``nk = 5``  
        (`0 = C`, `1 = I`, `2 = H`, `3 = W`, `4 = R`)
        - Number of distance classes: ``nd = 18``  
        - Only the 3 *urbanising* target classes have potentials:
        ``k = 0 → Commercial``, ``k = 1 → Industrial``, ``k = 2 → Housing``.

        Returns
        -------
        m_kd : ndarray, shape (3, 5, 18)
            Transition potential matrix where:
            - Axis 0 → target land-use class (C, I, H)
            - Axis 1 → current land-use class
            - Axis 2 → distance classes (0-17)

            Values are integers representing conversion strengths, with
            positive values indicating attraction and negative values indicating
            repulsion for conversion.

        References
        ----------
        White R, Engelen G, Uljee I (1997), The use of constrained cellular automata for high-resolution modelling of urban land-use dynamics. 
        Environment and Planning B: Planning and Design, 24, 323-343. 
        '''
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
        '''
        Return the distance-class indices for a set of test coordinates.

        This computes distance classes used in the White et al. (1997) land-use
        transition potentials. Each `(i_test[k], j_test[k])` pair is assigned to one
        of 18 discrete distance bins based on its Euclidean distance from the
        reference cell `(ic, jc)`.

        Parameters
        ----------
        ic, jc : int
            Indices of the reference grid cell.
        i_test, j_test : array_like
            Arrays of test cell indices for which distance classes will be computed.
            Must be equal-length and indexable.

        Returns
        -------
        neighbor_d : ndarray of int
            Flattened array of distance-class indices in the range `[0, 17]`.  
            For each test coordinate, the returned value is the index of the
            corresponding distance bin.
        '''
        r = np.sqrt((i_test-ic)**2 + (j_test-jc)**2)
        d = [1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8), 3, np.sqrt(10), np.sqrt(13), 4, np.sqrt(17), \
             np.sqrt(18), np.sqrt(20), 5, np.sqrt(26), np.sqrt(29), np.sqrt(32), np.sqrt(34), 6]
        neighbor_d = np.concatenate([np.where(d == r[i])[0] for i in range(112)]).ravel()
        return neighbor_d

    def get_state_built(self, state0, neighbor_k, neighbor_d, m_kd):
        '''
        Determine the new built-type state based on transition potentials.

        Computes the Commercial/Industrial/Housing (C/I/H) transition potentials
        following White et al. (1997), using:
        - neighbor land-use classes ``neighbor_k``
        - distance classes ``neighbor_d``
        - the transition matrix ``m_kd`` (shape: 3 x 5 x 18)

        Parameters
        ----------
        state0 : int
            Current land-use state of the cell (0=C, 1=I, 2=H, 3=W, 4=R).
        neighbor_k : ndarray of int
            Land-use types of neighbors. ``-1`` marks invalid neighbors.
        neighbor_d : ndarray of int
            Distance-class indices of the neighbors (0-17).
        m_kd : ndarray
            Transition potential matrix of shape (3, 5, 18), as returned by
            :meth:`transform_landUse`.

        Returns
        -------
        new_state : int
            The selected new state among:
            - 0 → Commercial
            - 1 → Industrial
            - 2 → Housing

            If no valid neighbors exist, the function returns ``state0``.
        '''
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
        Classify built land-use states using the White et al. (1997) land-use
        transition model.

        Parameters
        ----------
        built : ndarray of shape (nx, ny)
            Binary grid where 1 indicates a built cell and 0 indicates
            non-built.
        built_type : ndarray of shape (nx, ny)
            Integer grid encoding the preliminary building type:
            -1 : unclassified
             0 : commercial
             1 : industry
             2 : housing
             3 : water
             4 : road

        Returns
        -------
        built_type : ndarray of shape (nx, ny)
            Updated building type grid following land-use transitions from
            White et al. (1997).
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


    def get_industrialZones(self):
        '''
        Extract and classify industrial zones from the current land-use grid.

        Industrial areas (state ``3``) are grouped into connected components using
        a von Neumann neighbourhood. Each component is converted into a polygon
        via contour extraction, and the resulting polygon is classified based on
        proximity to the coastline or river.

        Returns
        -------
        list of dict
            One dictionary per detected industrial zone. Each dictionary contains:
            - ``state`` : str  
            Always ``"industrial"``.
            - ``zone_type`` : str  
            One of ``"industrial harbor"``, ``"riverside industrial park"``, or ``"inland industrial park"``.
            - ``vertices`` : (N, 2) ndarray  
            Closed polygon vertex coordinates.
            - ``area`` : float  
            Polygon area.
            - ``distance_to_coast`` : float  
            Minimum distance to coastline.
            - ``distance_to_river`` : float  
            Minimum distance to river.
            - ``closed`` : bool  
            Always ``True``.
        '''
        xx = self.grid.xx
        yy = self.grid.yy
        states = self.S

        coast_x, coast_y = self.topo.coastline_coord
        coastline = LineString(np.column_stack([coast_x, coast_y]))

        riv_x, riv_y, _, _ = self.topo.river_coord
        river = LineString(np.column_stack([riv_x, riv_y]))

        buffer = self.grid.w * 2

        vonNeumann_struct = np.array([[0,1,0],
                                      [1,1,1],
                                      [0,1,0]], dtype=bool)

        industrial_mask = (states == 3)

        labeled, n_components = label(industrial_mask, structure=vonNeumann_struct)

        polygons = []
        plt.figure(figsize=(4,4))

        for comp_id in range(1, n_components + 1):
            component_mask = (labeled == comp_id).astype(float)
            cs = plt.contour(xx, yy, component_mask, levels=[0.5])
            if not cs.collections:
                continue

            paths = cs.collections[0].get_paths()
            if not paths:
                continue
            exterior_path = max(paths, key=lambda p: len(p.vertices))
            vertices = exterior_path.vertices.copy()

            if not np.allclose(vertices[0], vertices[-1]):
                vertices = np.vstack([vertices, vertices[0]])

            poly = Polygon(vertices)
            area = poly.area
            dist_coast = poly.distance(coastline)
            dist_river = poly.distance(river)

            if dist_coast <= buffer:
                zone_type = 'industrial harbor'
            elif dist_river <= buffer:
                zone_type = 'riverside industrial park'
            else:
                zone_type = 'inland industrial park'

            polygons.append({
                "state": "industrial",
                "zone_type": zone_type,
                "vertices": vertices,
                "area": area,
                "distance_to_coast": dist_coast,
                "distance_to_river": dist_river,
                "closed": True,
            })

        plt.close()
        return polygons

    @cached_property
    def industrialZones(self):
        """Cached version of industrial zones."""
        return self.get_industrialZones()

    @cached_property
    def CI_refinery(self):
        '''
        Identify and construct the critical infrastructure object representing  
        the main coastal refinery.

        This property selects the largest polygon classified as an
        **industrial harbor** zone from :attr:`industrialZones`, and returns a
        :class:`CriticalInfrastructure` instance located at its centroid.

        Returns
        -------
        CriticalInfrastructure
            An object describing the refinery, including its name, zone type,
            polygon geometry, area, centroid coordinates, and distances to
            coastline and river.
        '''
        zones = self.industrialZones
        harbor_polys = [p for p in zones if p["zone_type"] == "industrial harbor"]

        if len(harbor_polys) == 0:
            raise ValueError("No industrial harbor polygons found.")

        largest = max(harbor_polys, key=lambda p: p["area"])
        poly = Polygon(largest["vertices"])
        centroid = (poly.centroid.x, poly.centroid.y)

        return CriticalInfrastructure(
            name="CI_refinery",
            zone_type=largest["zone_type"],
            area=largest["area"],
            centroid=centroid,
            polygon=poly,
            distance_to_coast=largest["distance_to_coast"],
            distance_to_river=largest["distance_to_river"],
            Ex_S_kton=None,
        )




#####################################
# SOCIO-ECONOMIC ENVIRONMENT LAYERS #
#####################################

# coming mid 2026



############
# PLOTTING #
############

lgd_industrialZone = [
    mpatches.Patch(
        facecolor=GenMR_utils.col_industrialZone['industrial harbor'], 
        edgecolor='black',
        label='Industrial Harbor'
    ),
    mpatches.Patch(
        facecolor=GenMR_utils.col_industrialZone['riverside industrial park'], 
        edgecolor='black',
        label='Riverside Industrial Park'
    ),
    mpatches.Patch(
        facecolor=GenMR_utils.col_industrialZone['inland industrial park'], 
        edgecolor='black',
        label='Inland Industrial Park'
    )
]

def plot_EnvLayer_attr(envLayer, attr, hillshading_z = '', file_ext = '-'):
    '''
    Plot a specific attribute of an environmental layer.

    This function handles plotting for multiple environmental layers including
    topography, soil, natural land, and urban land. Optionally, hillshading
    can be applied for topography visualization. Legends are added for categorical
    variables, and continuous variables display colorbars.

    Parameters
    ----------
    envLayer : object
        Instance of an environmental layer class (e.g., EnvLayer_topo, EnvLayer_soil, EnvLayer_natLand, EnvLayer_urbLand).
    attr : str
        Name of the attribute to plot. Examples:
        - Topography: 'z', 'slope', 'aspect'
        - Soil: 'h', 'FS'
        - Natural/Urban Land: 'S', 'roadNet', 'bldg_value', 'built_yr', 'industrialZones'
    hillshading_z : ndarray, optional
        2D array of elevation values for hillshading overlay. Default is empty string, meaning no hillshading.
    file_ext : str, optional
        File extension for saving the figure (e.g., 'jpg', 'pdf'). Default is '-' meaning the figure is not saved.

    Returns
    -------
    None
        Displays the plot. Optionally saves the figure if `file_ext` is not '-'.

    Notes
    -----
    - The attribute argument `attr` must match the string name of the variable inside the `envLayer` object.
    - Legends and colormaps are automatically selected based on the layer type and attribute.
    - For `urban` layers, industrial zones are plotted as polygons with colors corresponding to their zone type.
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
                                         vmin=-1, vmax=5, alpha = alpha)
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
                                         vmin=-1, vmax=5, alpha = alpha)
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
        elif attr == 'industrialZones':
            for poly in envLayer.industrialZones:
                patch = MplPolygon(
                        poly['vertices'],
                        closed=True,
                        color=GenMR_utils.col_industrialZone.get(poly['zone_type'], 'gray'),
                        alpha=1.
                )
                ax.add_patch(patch)
            ax.set_xlim(envLayer.grid.xmin, envLayer.grid.xmax)
            ax.set_ylim(envLayer.grid.ymin, envLayer.grid.ymax)
            labels_industrialZone = [h.get_label() for h in lgd_industrialZone]
            ax.legend(lgd_industrialZone, labels_industrialZone, loc='upper left')
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
    Plot multiple environmental layers with up to three attributes per layer.

    This function creates a multi-panel figure where each row corresponds
    to an environmental layer. It supports topography, soil, natural land,
    and urban land layers. The function automatically handles legends, 
    color maps, and optional hillshading for topography.

    Parameters
    ----------
    envLayers : list of objects
        List of environmental layer instances (e.g., EnvLayer_topo, EnvLayer_soil,
        EnvLayer_natLand, EnvLayer_urbLand).
    file_ext : str, optional
        File extension to save the figure ('jpg', 'pdf', etc.). Default is '-' 
        which means the figure is not saved.

    Returns
    -------
    None
        Displays the multi-panel plot. Optionally saves the figure if `file_ext`
        is not '-'.

    Notes
    -----
    - For topography layers, the three columns correspond to: altitude `z`, slope, and aspect.
    - For soil layers, the three columns correspond to: thickness `h`, factor of safety, and an empty panel.
    - For natural land layers, the first column shows land classes (`S`), the other two columns are empty.
    - For urban land layers, the three columns correspond to: state `S`, road network, and building value.
    - Legends, color maps, and hillshading are applied automatically depending on the layer type.
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
                                         vmin=-1, vmax=5, alpha = .5)
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
                                         vmin=-1, vmax=5, alpha = .5)
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
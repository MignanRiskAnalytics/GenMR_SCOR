"""
GenMR Peril Implementation
==========================

This module provides functions to define and implement perils in the GenMR digital template.
This includes tools for full catastrophe risk modelling, such as event loss table (ELT) definition,
intensity and loss footprint computation... TO DEVELOP/REWRITE

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-11-04
:License: AGPL-3
"""

import numpy as np
import pandas as pd

from GenMR import dynamics as GenMR_dynamics
from GenMR import utils as GenMR_utils



###############
# EVENT TABLE #
###############



def gen_stochSet(src, par):
    '''
    '''
    ev_stoch = pd.DataFrame(columns = ['evID', 'loc', 'S', 'S2', 'w', 'lbd', 'Ai_'])
    ev_coord = pd.DataFrame(columns = ['evID', 'x', 'y', 'z'])
    for ID in src.par['perils']:
        if ID in par['primary']:
            evID_peril = [ID + str(i+1) for i in range(par[ID]['stoch_n'])]
            Si = GenMR_utils.incrementing(par[ID]['Smin'], par[ID]['Smax'], par[ID]['Sbin'], par[ID]['Si_method'])
            Si_n = len(Si)
            Si_ind = np.arange(Si_n)
            if par[ID]['stoch_w'] == 'uniform':
                qi = np.repeat(1./Si_n, Si_n)
            Si_ind_vec = GenMR_utils.partitioning(Si_ind, qi, par[ID]['stoch_n'])
            Si_vec = Si[Si_ind_vec]
            wi = 1 / np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
            wi_vec = [wi[Si_ind == i][0] for i in Si_ind_vec]
            if par[ID]['stoch_xy'] == 'rdmPoint_inGrid':
                evID_coord = evID_peril
                evID_loc = [src.par[ID]['object'] + str(i + 1) for i in range(par[ID]['stoch_n'])]
                evID_x, evID_y = pop_rdm_points(par[ID]['stoch_n'], src.grid)
                evID_z = np.repeat(np.nan, par[ID]['stoch_n'])
            if par[ID]['stoch_xy'] == 'fixPoint_inSrc':
                evID_coord = evID_peril
                evID_x, evID_y, evID_loc = pop_fix_points(par[ID]['stoch_n'], src.par[ID])
                evID_z = np.repeat(np.nan, par[ID]['stoch_n'])
            if par[ID]['stoch_xy'] == 'rdmLine_inEQsrc':
                evID_coord, evID_x, evID_y , evID_z, evID_loc = pop_EQ_floatingRupture(evID_peril, Si_vec, src)
            if par[ID]['stoch_xy'] == 'rdmTrack_inGrid':
                evID_loc = [src.par[ID]['object'] + str(i + 1) for i in range(par[ID]['stoch_n'])]
                evID_coord_lores = np.repeat(evID_peril, src.par[ID]['npt'] + 1)
                evID_x_lores, evID_y_lores = pop_rdm_tracks(par[ID]['stoch_n'], src.grid, src.par[ID])
                evID_x, evID_y, evID_coord = get_highres(evID_x_lores, evID_y_lores, evID_coord_lores, src.par[ID])
                evID_z = np.repeat(np.nan, len(evID_x))
            if par[ID]['stoch_xy'] == 'fullArea_inGrid':
                evID_loc = [src.par[ID]['object'] + str(i + 1) for i in range(par[ID]['stoch_n'])]
                evID_coord = np.repeat(evID_peril, 5)
                evID_x, evID_y = pop_grid_area(par[ID]['stoch_n'], src.grid)
                evID_z = np.repeat(np.nan, par[ID]['stoch_n'] * 5)
            if 'S2min' in par[ID]:
                S2i = GenMR_utils.incrementing(par[ID]['S2min'], par[ID]['S2max'], par[ID]['S2bin'], par[ID]['S2i_method'])
                S2i_vec = S2i[Si_ind_vec]
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc, \
                                                              'S': Si_vec, 'S2': S2i_vec, 'w': wi_vec})], ignore_index=True)
                ev_coord = pd.concat([ev_coord, pd.DataFrame({'evID': evID_coord, 'x': evID_x, \
                                                              'y': evID_y, 'z': evID_z})], ignore_index=True)
            else:
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc, \
                                                              'S': Si_vec, 'w': wi_vec})], ignore_index=True)
                ev_coord = pd.concat([ev_coord, pd.DataFrame({'evID': evID_coord, 'x': evID_x, \
                                                              'y': evID_y, 'z': evID_z})], ignore_index=True)

        if ID in par['secondary']:
            ID_LS_mem = 0
            if ID == 'FF' and 'RS' in par['primary']:
                evID_peril = ['FF' + str(i + 1 + ID_LS_mem) for i in range(par['RS']['stoch_n'])]
                evID_trigger = ['RS' + str(i + 1) for i in range(par['RS']['stoch_n'])]
                evID_loc = np.repeat(src.par[ID]['object'], par['RS']['stoch_n'])
                ev_stoch_trigger = ev_stoch[np.isin(ev_stoch['evID'], evID_trigger)]
                Si_vec = GenMR_dynamics.calc_S_RS2FF(ev_stoch_trigger['S'], src.par['FF'])
                S2i_vec = ev_stoch_trigger['S2']
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc, \
                                    'S': Si_vec, 'S2': S2i_vec, 'Ai_': evID_trigger})], ignore_index=True)
            if ID == 'LS' and 'EQ' in par['primary']:
                evID_peril = ['LS' + str(i + 1 + ID_LS_mem) for i in range(par['EQ']['stoch_n'])]
                evID_trigger = ['EQ' + str(i + 1) for i in range(par['EQ']['stoch_n'])]
                evID_loc = np.repeat(src.par[ID]['object'], par['EQ']['stoch_n'])
                ID_LS_mem = len(evID_peril)
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc,\
                                                              'Ai_': evID_trigger})], ignore_index=True)
            if ID == 'LS' and 'RS' in par['primary']:
                evID_peril = ['LS' + str(i + 1 + ID_LS_mem) for i in range(par['RS']['stoch_n'])]
                evID_trigger = ['RS' + str(i + 1) for i in range(par['RS']['stoch_n'])]
                evID_loc = np.repeat(src.par[ID]['object'], par['RS']['stoch_n'])
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc, \
                                                              'Ai_': evID_trigger})], ignore_index=True)
            if ID == 'SS' and 'WS' in par['primary']:
                evID_peril = ['SS' + str(i + 1 + ID_LS_mem) for i in range(par['WS']['stoch_n'])]
                evID_trigger = ['WS' + str(i + 1) for i in range(par['WS']['stoch_n'])]
                evID_loc = np.repeat(src.par[ID]['object'], par['WS']['stoch_n'])
                ev_stoch = pd.concat([ev_stoch, pd.DataFrame({'evID': evID_peril, 'loc': evID_loc, \
                                                              'Ai_': evID_trigger})], ignore_index=True)
    return ev_stoch.reset_index(drop = True), ev_coord.reset_index(drop = True)


def pop_rdm_points(N, grid):
    '''
    Return random uniform coordinates for N points in the grid
    '''
    x_rdm = grid.xmin + np.random.random(N) * (grid.xmax - grid.xmin)
    y_rdm = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    return x_rdm, y_rdm

def pop_fix_points(N, par):
    '''
    Return fixed coordinates for N events 
    '''
    nloc = len(par['x'])
    srcID = np.array([par['object'] + str(i + 1) for i in range(nloc)])
    src_rdm_ind = np.random.choice(np.arange(nloc), N)
    evID_loc = srcID[src_rdm_ind]
    evID_x = np.array(par['x'])[src_rdm_ind]
    evID_y = np.array(par['y'])[src_rdm_ind]
    return evID_x, evID_y, evID_loc

def pop_rdm_tracks(N, grid, par, extrabuffer_f = 3):
    '''
    Return coordinates of N tracks following pseudo-random walk.
    
    Notes:
        For this version of the digital template, a very simple 
        approach used with track oriented W-E on average.
    '''
    track_y0 = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    track_x0 = grid.xmin - extrabuffer_f*grid.xbuffer
    li = (grid.xmax - grid.xmin + extrabuffer_f*grid.xbuffer) / par['npt']
    alphai =  par['phi_sd'] * np.random.randn(N * par['npt'], ).reshape(N, par['npt'])
    track_evID = np.array([])
    track_x = np.array([])
    track_y = np.array([])
    for i in range(N): 
        dx = li * np.cos(alphai[i,:])
        dy = li * np.sin(alphai[i,:])
        x = track_x0 + np.insert(np.cumsum(dx), 0, 0)
        y = track_y0[i] + np.insert(np.cumsum(dy), 0, 0)
        track_x = np.append(track_x, x)
        track_y = np.append(track_y, y)
    return track_x, track_y

def gen_rdmcoord_tracks(N, grid, npt, max_deviation, rdm_seed = None):
    '''
    Return coordinates of N storm tracks, defined as straight lines
    subject to random deviation (below max_deviation) along y at npt points.
    '''
    if rdm_seed is not None:
        np.random.seed(rdm_seed)

    ID = np.repeat(np.arange(N)+1, npt)
    x = np.tile(np.linspace(grid.xmin, grid.xmax, npt), N)
    ystart = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    yend = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    y = np.linspace(ystart, yend, npt, axis = 1).flatten()
    deviation = np.random.uniform(-max_deviation, max_deviation, size = N*npt)
    y += deviation
    return x, y, ID

def pop_grid_area(N, grid):
    '''
    '''
    box_x = np.tile([grid.xmin, grid.xmax, grid.xmax, grid.xmin, grid.xmin], N)
    box_y = np.tile([grid.ymin, grid.ymin, grid.ymax, grid.ymax, grid.ymin], N)
    return box_x, box_y


def pop_EQ_floatingRupture(evIDi, Si, src):
    '''
    '''
    nRup = len(Si)
    li = calc_EQ_M2L(Si)
    flt_x, flt_y, flt_id, flt_L, seg_id, seg_strike, seg_L = src.EQ_char    
    indflt = GenMR_utils.partitioning(np.arange(len(flt_L)), flt_L / np.sum(flt_L), nRup)  # longer faults visited more often
    Rup_loc = np.zeros(nRup, dtype=object)
    Rup_coord = pd.DataFrame(columns = ['evID', 'x', 'y', 'z'])
    i = 0
    while i < nRup:
        flt_target = np.random.choice(indflt, 1)
        indID = flt_id == flt_target
        src_x = flt_x[indID]
        src_y = flt_y[indID]
        src_L = flt_L[flt_target]
        src_z = src.par['EQ']['z_km'][indflt[i]]
        init = np.floor((src_L - li[i]) / src.par['EQ']['bin_km'])
        if src_L >= li[i]:
            u = np.ceil(np.random.random(1) * init).astype(int)[0]                         # random rupture start loc
            Rup_x = src_x[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_y = src_y[u:(u + li[i] / src.par['EQ']['bin_km']).astype(int)]
            Rup_loc[i] = src.par['EQ']['object'] + str(flt_target[0] + 1)
            Rup_coord = pd.concat([Rup_coord, pd.DataFrame({'evID': np.repeat(evIDi[i], len(Rup_x)), \
                            'x': Rup_x, 'y': Rup_y, 'z': np.repeat(src_z, len(Rup_x))})], ignore_index=True)
            i += 1
    return Rup_coord['evID'], Rup_coord['x'], Rup_coord['y'], Rup_coord['z'], Rup_loc

def calc_EQ_M2L(M):
    '''
    '''
    return 10**((M - 4.49) / 1.49)    # for floating rupture computations

def get_highres(x0, y0, id0, par):
    '''
    '''
    x_hires = np.array([])
    y_hires = np.array([])
    id_hires = np.array([])
    evID = np.unique(id0)
    for i in range(len(evID)):
        indev = np.where(id0 == evID[i])[0]
        x_ev = x0[indev]
        y_ev = y0[indev]
        for seg in range(len(x_ev) - 1):
            dx = x_ev[seg + 1] - x_ev[seg]
            dy = y_ev[seg + 1] - y_ev[seg]
            sign1 = dx / np.abs(dx)
            sign2 = dy / np.abs(dy)
            L = np.sqrt(dx**2 + dy**2)
            strike = np.arctan(dx/dy) * 180 / np.pi
            npt = int(np.round(L / par['bin_km']))
            seg_xi = np.zeros(npt)
            seg_yi = np.zeros(npt)
            seg_xi[0] = x_ev[seg]
            seg_yi[0] = y_ev[seg]
            for k in range(1, npt):
                seg_xi[k] = seg_xi[k-1] + sign1 * sign2 * par['bin_km'] * np.sin(strike * np.pi / 180)
                seg_yi[k] = seg_yi[k-1] + sign1 * sign2 * par['bin_km'] * np.cos(strike * np.pi / 180)
            x_hires = np.append(x_hires, np.append(seg_xi, x_ev[seg + 1]))
            y_hires = np.append(y_hires, np.append(seg_yi, y_ev[seg + 1]))
            id_hires = np.append(id_hires, np.repeat(evID[i], len(seg_xi)+1))
    return x_hires, y_hires, id_hires




#####################
# HAZARD FOOTPRINTS #
#####################

# TO ADD...






###################
# LOSS FOOTPRINTS #
###################







############
# PLOTTING #
############




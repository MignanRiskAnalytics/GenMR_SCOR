"""
GenMR Dynamics Modelling
========================

This module ...

:Author: Arnaud Mignan, Mignan Risk Analytics GmbH
:Version: 0.1
:Date: 2025-11-04
:License: AGPL-3
"""

import numpy as np
import pandas as pd


###################################
## PERIL ONE-TO-ONE INTERACTIONS ##
###################################


def calc_S_RS2FF(S_RS, par):
    '''
    # flow Q [m3/s] = RS [m/s] * A catchment [m2]
    '''
    S_FF = S_RS * 1e-3 / 3600 * par['A_km2'] * 1e6
    return np.round(S_FF)


def calc_S_WS2SS(v_max):
    '''
    Empirical relationship from Lin et al. (2010) - New York
    vmax: max wind speed [m/s] during storm passage
    '''
    S_SS = .031641 * v_max - .00075537 * v_max**2 + 3.1941e-5 * v_max**3
    return np.round(S_SS, decimals = 3)
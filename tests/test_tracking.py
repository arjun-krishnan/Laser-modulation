# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:25:55 2026

@author: arjun
"""

import numpy as np
from laser_modulator.components import Modulator, Laser
from laser_modulator.tracking import lsrmod_track
import scipy.constants as const

def test_tracking_zero_laser_power(tmp_path):
    mod_file = tmp_path / "mod_params.dat"
    with open(mod_file, "w") as f:
        f.write("START\nE0: 1492\nWL: 800e-9\nNPERIOD: 6\nPERIODLEN: 0.15\nEND\n")
    my_mod = Modulator(str(mod_file), plot=False)
    
    # Create a dummy Laser with EXACTLY ZERO pulse energy
    laser_file = tmp_path / "laser_params.dat"
    with open(laser_file, "w") as f:
        f.write("START\nWL: 800e-9\nE: 0.0\nEND\n")  # Pulse energy E is 0!
    my_laser = Laser(str(laser_file))
    
    N_e = 100
    e_bunch = np.zeros((6, N_e))
    
    gamma_0 = 1492 / 0.511
    p_z0 = gamma_0 * const.m_e * const.c
    
    initial_momentum = p_z0 * (1 + np.random.normal(0, 7e-4, N_e))
    e_bunch[5,:] = initial_momentum
    
    energy_start = np.copy(e_bunch[5,:])
    
    e_bunch = lsrmod_track(my_mod, my_laser, e_bunch, plot_track=False, disp_Progress=False)
    
    energy_end = e_bunch[5,:]
    
    np.testing.assert_allclose(energy_start, energy_end, atol=1e-10)
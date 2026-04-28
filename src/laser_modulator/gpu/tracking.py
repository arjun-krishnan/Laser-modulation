# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:08:06 2026

@author: arjun
"""

import cupy as cp
import numpy as np
from time import time
import sys
import scipy.constants as const

# Natural constants
c = const.c
e_charge = const.e
m_e = const.m_e

def lsrmod_track_gpu(Mod, Lsr, e_bunch, Lsr2=None, tstep=1e-12, zlim=None, plot_track=False, disp_Progress=True):
    
    if zlim == None:
        zlim = Mod.len
        
    # Ensure bunch is on the GPU
    if not isinstance(e_bunch, cp.ndarray):
        bunch = cp.asarray(e_bunch, dtype=cp.float64)
    else:
        bunch = cp.copy(e_bunch)
        
    N_e = len(bunch[0])
    z_0 = cp.mean(bunch[2])
    bunch[2] -= z_0
    z_mean = cp.mean(bunch[2]) 
    
    progressrate = 10
    progress = 0 
    t = 0.0
    
    max_steps = int(zlim / (3e8 * tstep) * 1.2)

    starttime = time()
    
    # Pre-allocate arrays in GPU memory
    EE_history = cp.zeros(max_steps, dtype=cp.float64)
    ZZ_history = cp.zeros(max_steps, dtype=cp.float64)
    dZZ_history = cp.zeros(max_steps, dtype=cp.float64)
    
    # Pre-allocate tracking arrays (for last 6 electrons) in GPU memory
    track_x_history = cp.zeros((max_steps, 6), dtype=cp.float64)
    track_z_history = cp.zeros((max_steps, 6), dtype=cp.float64)
    
    step = 0
    
    while z_mean < zlim: # and step < max_steps:
        if disp_Progress:
            # We add .get() here to safely pull the scalar to the CPU for the print statement
            if progress < (z_mean.get()) / zlim * progressrate:
                elapsed = time() - starttime
                sys.stdout.write('\r Progress: ' + str(progress) + '/' + str(progressrate) + " \t ETA: " + 
                                 str(np.round(elapsed/60 * (progressrate / (progress+0.01) - 1), 2)) + " mins ")
                sys.stdout.flush()
                progress += 1
        
        z = cp.copy(bunch[2])
        z_mean = cp.mean(z)
        ZZ_history[step] = z_mean
        
        Efield_x_vec = Lsr.E_field(bunch[0], bunch[1], bunch[2], t)
        if Lsr2 != None: 
            Efield_x_vec += Lsr2.E_field(bunch[0], bunch[1], bunch[2], t)
        EE_history[step] = Efield_x_vec[0]

        try:
            # Handle if the B_func is your 2D version or 1D version
            if Mod.is2d:
                Bfield_y_vec = Mod.B_func(cp.stack([bunch[0], z], axis=1)).T + Efield_x_vec / c
            else:
                Bfield_y_vec = Mod.B_func(z) + Efield_x_vec / c
        except:
            Bfield_y_vec = Efield_x_vec / c
        
        p_field = bunch[3:]
        p_vec = cp.sqrt(cp.sum(p_field**2, axis=0))
        gamma_vec = cp.sqrt((p_vec / m_e / c) ** 2 + 1)
        
        dp_x_vec = (Efield_x_vec - p_field[2] * Bfield_y_vec / m_e / gamma_vec) * e_charge * tstep
        dp_y_vec = cp.zeros(N_e, dtype=cp.float64)
        dp_z_vec = p_field[0] * Bfield_y_vec / m_e / gamma_vec * e_charge * tstep   
        
        # GPU arrays need to be stacked rather than added as a Python list
        p_new = bunch[3:] + cp.vstack((dp_x_vec, dp_y_vec, dp_z_vec))
        p_vec_new = cp.sqrt(cp.sum(p_new**2 , axis=0))
        gamma_vec_new = cp.sqrt((p_vec_new / m_e / c)**2 + 1)   
                       
        spatial_new = bunch[0:3,:] + p_new / m_e / gamma_vec_new * tstep       
        bunch[0:3] = spatial_new
        bunch[3:] = p_new
        
        track_x_history[step] = bunch[0, -6:]
        track_z_history[step] = bunch[2, -6:]
        
        dZZ_history[step] = t * c - z_mean
        
        t += tstep
        step += 1
  

    if disp_Progress:
            print('\nProgress: '+str(progress)+'/'+str(progressrate))

    if plot_track == True:
        # We use .get() to return standard CPU numpy arrays for plotting
        return bunch.get(), track_x_history[:step].get(), track_z_history[:step].get()
    
    endtime = time()
    print("\nRuntime:  " , np.round(endtime-starttime,2) , " sec")
    
    return bunch.get()
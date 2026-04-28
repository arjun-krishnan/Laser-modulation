# -*- coding: utf-8 -*-
"""
Electron Tracking Module

This module contain the functions for tracking the ensemble of electrons
through the given magnetic lattice (laser modulator or a chicane)
"""
import numpy as np
import sys
from time import time
import scipy.constants as const
from scipy import special


##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2
Z0 = 376.73                     # impedance of free space in Ohm
epsilon_0 = const.epsilon_0     # vacuum permittivity
mu0 = const.mu_0                # vacuum permeability

def lsrmod_track(Mod, Lsr, e_bunch, Lsr2=None, tstep=1e-12, zlim=None, plot_track=False, disp_Progress=True):
    
    if zlim == None:
        zlim = Mod.len
        
    N_e = len(e_bunch[0])
    bunch = np.copy(e_bunch)
    z_0 = np.mean(bunch[2])
    bunch[2] -= z_0
    z_mean = np.mean(bunch[2]) 
    
    progressrate = 10
    progress = 0 
    t = 0
    
    max_steps = int(zlim / (3e8 * tstep) * 1.2)

    starttime = time()
    EE_history = np.zeros(max_steps)
    ZZ_history = np.zeros(max_steps)
    dZZ_history = np.zeros(max_steps)
    
    # Pre-allocate tracking arrays (for last 6 electrons)
    track_x_history = np.zeros((max_steps, 6))
    track_z_history = np.zeros((max_steps, 6))
    
    step = 0
    
    while z_mean < zlim: # and step < max_steps:
        if disp_Progress:
            if progress < (z_mean) / zlim * progressrate:
                elapsed = time() - starttime
                sys.stdout.write('\r Progress: ' + str(progress) + '/' + str(progressrate) + " \t ETA: " + 
                                 str(np.round(elapsed/60 * (progressrate / (progress+0.01) - 1), 2)) + " mins ")
                sys.stdout.flush()
                progress += 1
    
        z = np.copy(bunch[2])
        z_mean = np.mean(z)
        ZZ_history[step] = z_mean
        
        Efield_x_vec = Lsr.E_field(bunch[0],bunch[1],bunch[2],t)
        if Lsr2 != None: 
            Efield_x_vec += Lsr2.E_field(bunch[0],bunch[1],bunch[2],t)
        EE_history[step] = Efield_x_vec[0]

        try:
            Bfield_y_vec = Mod.B_func(z) + Efield_x_vec / c
        except:
            Bfield_y_vec = Efield_x_vec / c
        
        p_field = bunch[3:]
        p_vec = np.sqrt(np.sum(p_field**2, axis=0))
        gamma_vec = np.sqrt((p_vec / m_e / c) ** 2 + 1)
        dp_x_vec = (Efield_x_vec - p_field[2] * Bfield_y_vec / m_e / gamma_vec) * e_charge * tstep
        dp_y_vec = np.zeros(N_e)
        dp_z_vec = p_field[0] * Bfield_y_vec / m_e / gamma_vec * e_charge * tstep   
        p_new = bunch[3:] + [dp_x_vec , dp_y_vec , dp_z_vec]
        p_vec_new = np.sqrt(np.sum(p_new**2 , axis=0))
        gamma_vec_new = np.sqrt((p_vec_new / m_e / c)**2 + 1)    
                       
        spatial_new = bunch[0:3,:] + p_new / m_e / gamma_vec_new * tstep       
        bunch[0:3] = spatial_new
        bunch[3:] = p_new
        
        
        track_x_history[step] = bunch[0, -6:]
        track_z_history[step] = bunch[2, -6:]
        
        dZZ_history[step] = t * c - z_mean
        
        t += tstep
        step += 1
  

    if disp_Progress:
            print('Progress: '+str(progress)+'/'+str(progressrate))

    if plot_track == True:
        return(bunch, track_x_history[:step], track_z_history[:step])
    
    endtime = time()
    print("\nRuntime:  " , np.round(endtime-starttime,2) , " sec")
    
    return bunch  
        


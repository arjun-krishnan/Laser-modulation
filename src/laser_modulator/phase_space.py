# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:59:33 2024

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from numba import jit
#from SPEED_functions import coord_change

##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2

def define_bunch(Test=False, N=1e4, slicelength=8e-6, E0=1.492e9*e_charge, dE=7e-4, R56_dE=0.0007, R51_dx=4e-4, R52_dxp=4e-5):
    N_e = int(N) # number of electrons
    
    ##### electron parameter #####
    e_E = E0   # electron energy in J
    energyspread = dE 
       
    alphaX = 8.811383e-01 #1.8348
    alphaY = 8.972460e-01 #0.1999
    betaX = 13.546
    betaY = 13.401
    emitX = 1.6e-8
    emitY = 1.6e-9   
    Dx    = 0.0894
    Dxprime = -4.3065e-9 
    
    if(Test):
        #slicelength = 4e-6 
        N_e = int(1e4)
        energyspread = 0e-4
        emitX = 0
        emitY = 0  
        Dx = 0
        Dxprime = 0
    
    # Generating electron distribution according to beam parameters    
    CS_inv_x = np.abs(np.random.normal(loc=0, scale=emitX * np.sqrt(2 * np.pi), size=N_e))
    CS_inv_y = np.abs(np.random.normal(loc=0, scale=emitY * np.sqrt(2 * np.pi), size=N_e))
    phase_x = np.random.rand(N_e) * 2 * np.pi
    phase_y = np.random.rand(N_e) * 2 * np.pi
    
    # generate random electron parameters according to beam parameters
    elec0 = np.zeros((6, N_e))
    elec0[4] = (np.random.rand(1, N_e) - 0.5) * slicelength  # / c
    elec0[5] = np.random.normal(loc=0, scale=energyspread, size=N_e)  # /e_m/c**2
    elec0[0] = np.sqrt(CS_inv_x * betaX) * np.cos(phase_x) + elec0[5, :] * Dx
    elec0[1] = -np.sqrt(CS_inv_x / betaX) * (alphaX * np.cos(phase_x) + np.sin(phase_x)) + elec0[5, :] * Dxprime
    elec0[2] = np.sqrt(CS_inv_y * betaY) * np.cos(phase_y)
    elec0[3] = -np.sqrt(CS_inv_y / betaY) * (alphaY * np.cos(phase_y) + np.sin(phase_y))
    
    
    # Adding particles with only dE, dx and dxp to calculate the R56, R51 and R52
    for i in range(6):
        if i == 5:
            elec0[i][-2] , elec0[i][-1] = 0.0, R56_dE
        else:
            elec0[i][-2] , elec0[i][-1] = 0.0, 0.0
        if i == 1:
            elec0[i][-4] , elec0[i][-3] = 0.0, R52_dxp
        else:
            elec0[i][-4] , elec0[i][-3] = 0.0, 0.0
        if i == 0:
            elec0[i][-6] , elec0[i][-5] = 0.0, R51_dx
        else:
            elec0[i][-6] , elec0[i][-5] = 0.0, 0.0
        
    #changing to parameter style: [x,y,z,px,py,pz] in laboratory frame
    elec = coord_change(elec0,e_E)
    np.save("e_dist.npy",elec)
    return(elec)

def coord_change(elec_dummy,e_E):
    elec = np.zeros((6,len(elec_dummy[0])))
    elec[0,:] = elec_dummy[0,:]
    elec[1,:] = elec_dummy[2,:]
    elec[2,:] = elec_dummy[4,:]
    p_elecs   = np.sqrt(((1 + elec_dummy[5,:]) * e_E)**2 - m_e**2 * c**4) / c
    elec[5,:] = p_elecs / (np.sqrt(1 / np.cos(elec_dummy[1,:])**2 + np.tan(elec_dummy[3,:])**2))
    elec[4,:] = elec[5,:] * np.tan(elec_dummy[3,:])
    elec[3,:] = elec[5,:] * np.tan(elec_dummy[1,:])
    return(elec)     

def calc_phasespace(bunch,e_E,plot=False):
    p = np.sqrt(np.sum(bunch[3:]**2 , axis=0))
    E = np.sqrt(m_e**2 * c**4 + p**2 * c**2)
    dEE = E/e_E - 1
    z = np.copy(bunch[2,:]) 
    
    if plot:
        plt.figure()
        plt.plot((z-np.mean(z)) * 1e6 , dEE ,',')
        plt.xlabel('z ($\mu m$)')
        plt.ylabel('$\Delta E/E_0$')
        plt.tight_layout()
    return(z,dEE)    

#@jit(parallel = True)
def calc_bn(tau0, wl, printmax = True):
    wl = np.asarray(wl).reshape(-1,)
    bn = np.zeros(len(wl))
    for i in range(len(wl)):
        z = np.sum(np.exp(-1j * 2 * np.pi * (tau0 / wl[i])))
        bn[i] = abs(z) / len(tau0)
    
    index = np.argmax(bn)
    wl_max = wl[index]
    if printmax:
        print("Maximum bunching factor is", np.round(max(bn),4) , " at " , np.round(wl_max*1e9,2) , " nm")
    return(np.array(bn))


def plot_slice(z, wl, slice_len=0, n_slice=40):
    if slice_len != 0:
        n_slice = int((max(z) - min(z)) / slice_len)

    zz = np.linspace(min(z), max(z), n_slice)
    bn, z_slice = [], []
    
    for i in range(1,len(zz)):
        z1, z2 = zz[i - 1], zz[i]
        z_slice.append(np.mean([z1, z2]))
        slice_zz = z[(z >= z1) * (z < z2)]
        #print(len(slice_zz))
        if len(slice_zz) == 0:
            bn.append(0)
        else:
            bn.append(max(calc_bn(slice_zz, wl, printmax = False)))
        i += 1
    
    z_slice = np.array(z_slice) - np.mean(z_slice)
    bn      = np.array(bn)
    
    plt.figure()
    plt.plot(z_slice, bn)
    plt.xlabel('s (m)')
    plt.ylabel('Bunching Factor')
    return(z_slice, bn)
    

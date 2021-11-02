# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:44:29 2020

@author: Arjun
"""

import os
#Changing the working directory to the source directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from lsrmod_functions import *



##### natural constants #####
c=const.c           # speed of light
e_charge=const.e    # electron charge
m_e=const.m_e       # electron mass in eV/c^2
Z0=376.73           # impedance of free space in Ohm
epsilon_0= const.epsilon_0 # vacuum permittivity

e_E=1.492e9*e_charge    # electron energy in J
e_gamma=e_E/m_e/c**2    # Lorentz factor

        
##### Simulation parameter #####
slicelength=30e-6    # length of simulated bunch slice in m
tstep=5e-12         # timestep in s
N_e=int(1e5)        # number of electrons

bunch_test=define_bunch(Test=True)
bunch_init=define_bunch(Test=False,N=N_e,slicelength=slicelength)
elec=np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl= 800e-9   # wavelength of the laser
l1_sigx= 0.7e-3 # sigma width at the focus
l1_fwhm=40e-15  # pulse length 
l1_E= 2e-3      # pulse energy

##### defining modulator 1 #####
mod1= Modulator(periodlen=0.20,periods=9,laser_wl=l1_wl,e_gamma=e_gamma)
l1= Laser(wl=l1_wl,sigx=l1_sigx,sigy=l1_sigx,pulse_len=l1_fwhm,pulse_E=l1_E,focus=mod1.len/2,M2=1.0,pulsed=False,phi=0e10)

##### defining Laser 2 #####
l2_wl= 400e-9
l2_sigx= 0.85e-3
l2_fwhm=40e-15
l2_E= 1e-3
##### defining modulator 2 #####
mod2= Modulator(periodlen=0.20,periods=9,laser_wl=l2_wl,e_gamma=e_gamma)
l2= Laser(wl=l2_wl,sigx=l2_sigx,sigy=l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=mod2.len/2,M2=1.0,pulsed=False,phi=0e10)


#### Test Tracking through Modulators
elec_test= lsrmod_track(mod1,l1,bunch_test,tstep=tstep)
z,dE=calc_phasespace(elec_test,plot=False)
A11=(max(dE))
elec_test= lsrmod_track(mod2,l2,bunch_test,tstep=tstep)
z,dE=calc_phasespace(elec_test,plot=False)
A22=(max(dE))

print("A1= ",A11,"\t A2= ",A22)

#Calculating the best choice of R56 for this specific A1 and A2
r56_1,r56_2=calc_R56(A11, A22,dE=7e-4,K=l1_wl/l2_wl,m=21,n=-1)

#defining laser noise values
l1.pulsed=False
l1.phi=0e10
l2.pulsed=False
l2.phi=0e10

print("\n\nTracking through Modulator 1...")
elec_M1= lsrmod_track(mod1,l1,elec,tstep=tstep)

elec_C1=chicane_track(elec_M1,R56=r56_1)
#calc_phasespace(elec_C1)

print("\n\nTracking through Modulator 2...")
elec_M2= lsrmod_track(mod2,l2,elec_C1,tstep=tstep)
#calc_phasespace(elec_M2)

elec_C2=chicane_track(elec_M2,R56=r56_2)

plt.figure()
z,dE=calc_phasespace(elec_C2,plot=True)

plt.figure()
wl=np.linspace(19e-9,20e-9,501)
b=calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl,b)

plot_slice(z, wl, n_slice=50)

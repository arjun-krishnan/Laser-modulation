# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:47:00 2024

@author: arjun
"""
from io_functions import read_file
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from os.path import splitext
import pprint

##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2
Z0 = 376.73                     # impedance of free space in Ohm
epsilon_0 = const.epsilon_0     # vacuum permittivity
mu0 = const.mu_0                # vacuum permeability

class Laser:
    def __init__(self,filename):
        params = read_file(filename)
        
        # Assigning default values
        default_values = {
            'WL': 800e-9,          # Default wavelength in meters
            'SIG_X': 1.0e-3,       # Default sigma width of horizontal focus in meters
            'SIG_Y': 1.0e-3,       # Default sigma width of vertical focus in meters
            'T_FWHM': 45e-15,      # Default FWHM pulse length in seconds
            'E': 2.5e-3,           # Default pulse energy in joules
            'M2': 1.0,             # Default M^2 value of the laser
            'X0': 0.0,             # Default X offset
            'Z_OFFSET': 0.0,       # Default Z offset
            'FOCUS': 1.0,          # Default focus parameter
            'PULSED': False,       # Default pulsed laser flag
            'PHI': 0.0             # Default spectral phase of the laser
        }
        # Update the params dictionary with default values
        params = {key: params.get(key, default_values[key]) for key in default_values}

        self.wl     = params['WL']            # Wavelength in meters
        self.sigx   = params['SIG_X']         # Sigma width of horizontal focus in meters
        self.sigy   = params['SIG_Y']         # Sigma width of vertical focus in meters
        self.pulse_len = params['T_FWHM']     # FWHM pulse length in seconds
        self.E      = params['E']             # Pulse energy in joules
        self.M2     = params['M2']            # M^2 value of the laser
        self.X0     = params['X0']            # X offset
        self.Z_offset = params['Z_OFFSET']    # Z offset
        self.focus  = params['FOCUS']         # Focus parameter
        self.pulsed = params['PULSED']        # Pulsed laser flag
        self.phi    = params['PHI']           # Spectral phase of the laser
        
        self.P_max = self.E / (0.94 * self.pulse_len)
        I0 = (2 * self.P_max) / (np.pi * 4 * self.sigx * self.sigy)     # Peak intensity
        self.E0 = np.sqrt(2*Z0*I0)   
        
        
        self.k = 2 * np.pi / self.wl                                    # Wavenumber in 1/m
        self.omega = 2 * np.pi * c / self.wl                            # Angular frequency in rad/s
        self.sigz = np.sqrt(2) * self.pulse_len * c / 2.3548            # Sigma width of pulse length in meters
        self.zRx = np.pi * (2 * self.sigx)**2 / (self.M2 * self.wl)     # Horizontal Rayleigh length in meters
        self.zRy = np.pi * (2 * self.sigy)**2 / (self.M2 * self.wl)     # Vertical Rayleigh length in meters
        
        self.beamsize_x = lambda z: self.sigx * (np.sqrt(1 + z**2 / (self.zRx**2)))   # Horizontal beam size at position z in meters
        self.beamsize_y = lambda z: self.sigy * (np.sqrt(1 + z**2 / (self.zRy**2)))   # Vertical beam size at position z in meters
        
#        self.E0 = 2**-0.25 * np.pi**-0.75 * np.sqrt(Z0 * self.E / (self.sigx * self.sigy * self.sigz / c)) * 1.2   # Factor to make the modulation amplitude equal to elegant simulations
        
        print(filename + " parameters :")
        pprint.pprint(params, sort_dicts=False)
        print()
        
    def E_field(self,X,Y,Z,T):
        Zdif_x = Z - self.focus                   # Distance of electron to focus (mod1_center)
        Zdif_y = Z - self.focus
        X = X - self.X0
        Z_laser = c * T - self.Z_offset              # Position of the laser pulse center
        R_x = Zdif_x * (1 + (self.zRx / Zdif_x)**2)
        R_y = Zdif_y * (1 + (self.zRy / Zdif_y)**2)
        central_E_field = self.E0 * self.sigx / self.beamsize_x(Zdif_x)
        
        if self.pulsed:
            offaxis_pulsed_factor = np.exp(-(Y / self.beamsize_y(Zdif_y))**2 - (X / self.beamsize_x(Zdif_x))**2 - ((Z - Z_laser) / (2 * self.sigz))**2)
        else:
            offaxis_pulsed_factor = np.exp(-(Y / self.beamsize_y(Zdif_y))**2 - (X / self.beamsize_x(Zdif_x))**2)
        
        phase = np.cos(self.k * Z + (self.phi * (const.c * T - Z)**2) - self.omega * T - self.k / 2 * X**2 / R_x - self.k / 2 * Y**2 / R_y) # + np.arctan(Zdif/l1_zRx))  # include this for Gouy phase shift
        return central_E_field * offaxis_pulsed_factor * phase

class Modulator:
    def __init__(self, filename, plot=True):
        
        if splitext(filename)[-1] == '.txt':
            df = pd.read_csv(filename,sep='\t')
            self.l = np.array(df['z']) / 1000 
            self.b = np.array(df['By']) 
            self.len = self.l[-1]        
            self.B_func = interp1d(self.l,self.b)
            if plot == True:
                plt.figure()
                plt.plot(self.l, self.b)
                plt.xlabel('z (m)')
                plt.ylabel('B (T)')
            return
        
        params = read_file(filename)
        
        # Assigning default values
        default_values = {
          'E0': 1492,          # Default energy in GeV
          'WL': 800e-9,          # Default M1 value
          'NPERIOD': 9, 
          'PERIODLEN': 0.25
          }
        
        params = {key: params.get(key, default_values[key]) for key in default_values}
        
        print(filename + " parameters :")
        pprint.pprint(params, sort_dicts=False)
        print()
        
        self.wl = params['WL']
        self.E0 = params['E0']
        e_gamma = self.E0 / 0.511
        self.periodlen = params['PERIODLEN']
        self.periods = params['NPERIOD'] 
        self.len = self.periods * self.periodlen
        self.center = self.len / 2
        self.K = np.sqrt(4 * self.wl * e_gamma**2 / self.periodlen - 2)
        self.Bmax = 2 * np.pi * self.K * m_e * c / (e_charge * self.periodlen)
        
        '''
        s = np.linspace(0,self.len,1000)
        B = self.Bmax * np.sin(2 * np.pi * s / self.periodlen)
        i = 0
        while(s[i] < self.periodlen):
            if(s[i] < self.periodlen / 2):
                B[i] *= 0.25
            else:
                B[i] *= 0.75
            i += 1
            
        i = -1
        while(-s[i] + self.len < self.periodlen):
            if(-s[i] + self.len < self.periodlen / 2):
                B[i] *= 0.25
            else:
                B[i] *= 0.75
            i -= 1
        '''
        
        s = np.linspace(0, self.len, 1000)
        B = self.Bmax * np.sin(2 * np.pi * s / self.periodlen)
        for i in range(len(s)):
            if s[i] < self.periodlen:
                B[i] *= 0.25 if s[i] < self.periodlen / 2 else 0.75
            elif self.len - s[i] < self.periodlen:
                B[i] *= 0.25 if self.len - s[i] < self.periodlen / 2 else 0.75
        
        if plot:
            plt.figure()
            plt.plot(s, B)
            plt.xlabel('z (m)')
            plt.ylabel('B (T)')
            
        self.B_func = interp1d(s,B) 
        
class SPEED_Lattice:
    def __init__(self, filename, plot=True):
        
        if splitext(filename)[-1] == '.txt':
            df = pd.read_csv(filename,sep='\t')
            self.l = np.array(df['z']) / 1000 
            self.b = np.array(df['By']) 
            self.len = self.l[-1]        
            self.B_func = interp1d(self.l,self.b)
            if plot == True:
                plt.figure()
                plt.plot(self.l, self.b)
                plt.xlabel('z (m)')
                plt.ylabel('B (T)')
            return
        
        params = read_file(filename)
        
        # Assigning default values
        default_values = {
          'E0': 1492,          # Default energy in GeV
          'M1': 800e-9,          # Default M1 value
          'M2': 400e-9,          # Default M2 value
          'RAD': 200e-9,         # Default radius
          'C1': 300,          # Default Chicane 1
          'C2': 500,          # Default Chicane 2
          'PERIOD': 9, 
          'PERIODLEN': 0.25
          }
        # Update the params dictionary with default values
        params = {key: params.get(key, default_values[key]) for key in default_values}
        
        print(filename + " parameters :")
        pprint.pprint(params, sort_dicts=False)
        print()
        
        M1 = params['M1']
        M2 = params['M2']
        Rad = params['RAD']
        IC1 = params['C1']      # power supply for chicane 1 ("Danfysik")
        IC2 = params['C2']
        
        self.E0 = params['E0']
        
        self.len = 5.74925
        self.windings = 48 # upper plus lower coil
        self.gap = 0.05 
        self.period = params['PERIODLEN']
        
        IM1 = self.wl2B(M1)         # Current for the Modulator 1 coils
        IM2 = self.wl2B(M2)
        IR1 = self.wl2B(Rad)
        

        ID1 = IM1/2         # Delta electronica power supplies
        ID2 = IM2/2
        ID3 = IM2/2
        ID4 = IR1/2

        # coil current
        curr=[ -ID1 , IM1 , -IM1 ,  IM1 , -IM1 , IM1 , -IM1 , IM1 , \
               -IC1 , -IC1, -ID1, IC1 , IC1 , IC1 , IC1 , -ID2 , -IC1 , -IC1 , \
               IM2 , -IM2 , IM2 , -IM2 , IM2 , -IM2 , IM2 , \
              -IC2 , -ID3 , IC2 , IC2 , -ID4 , -IC2 , \
               IR1 , -IR1 , IR1 , -IR1 , IR1 , -IR1 , ID4 ]

        factor = mu0 * self.windings / self.gap * 1.19     #The factor 1.19 is the correction term to match the CST simulaitons
        
        b0 = np.zeros(len(curr))
        for m in range(len(curr)):
            b0[m] = factor * curr[m]
                
        yoke = 0.08
        edge = 0.02    # softness parameter for magnet edge
        drift = 0.5   # before and after magnetic structure
        dl = 0.0005   # longitudinal interval
            
        nm = len(curr)
        magnet = self.period / 2
        magnet2 = magnet / 2
        yoke2 = yoke / 2
        len_range = nm * magnet + 2 * drift
        
        nl = int(len_range / dl)
        self.l = np.zeros(nl)
        for k in range(nl):
            self.l[k] = (k - 0.5) * dl
        
        self.b = np.zeros(nl)
        
        for m in range(nm):
            l1 = drift + (m - 1) * magnet  # Magnet center
            for k in range(nl):
                self.b[k] = self.b[k] + b0[m] / (np.exp((l1 + magnet2 - yoke2 - self.l[k]) / edge) + 1)
        
            for k in range(nl):
                self.b[k] = self.b[k] + b0[m] / (np.exp((self.l[k] - magnet2 - yoke2 - l1) / edge) + 1)
        
        if plot:
            plt.figure()
            plt.plot(self.l, self.b)
            plt.xlabel('z (m)')
            plt.ylabel('B (T)')
        
        self.B_func = interp1d(self.l, self.b)
        
    def wl2B(self,fund_wl):
        if fund_wl == 0:
            I = 0
        else:
            e_gamma = self.E0/0.511
            K = np.sqrt(4 * fund_wl * e_gamma**2 / self.period - 2)
            B = 2 * np.pi * K * m_e * c / (e_charge * self.period)
            factor = mu0 * self.windings / self.gap
            I = 1.14 * B / factor  # Current for the first modulator. The factor 1.14 is the correction term to match the CST simulations
            
            return(I)
        
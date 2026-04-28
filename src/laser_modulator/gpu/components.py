# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:42:47 2026

@author: arjun
"""

import cupy as cp
import numpy as np
import pandas as pd
import scipy.constants as const
from cupyx.scipy.interpolate import PchipInterpolator as cupy_interp1d
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pprint
from .utils import read_file 

c = const.c                     
e_charge = const.e              
m_e = const.m_e                 
Z0 = 376.73                     
epsilon_0 = const.epsilon_0     
mu0 = const.mu_0                

class LaserGPU:
    def __init__(self, filename=None, **kwargs):
        params = read_file(filename) if filename else {}
        params.update(kwargs) # Allow programmatic overrides
        
        default_values = {
            'WL': 800e-9,          
            'SIG_X': 1.0e-3,       
            'SIG_Y': 1.0e-3,       
            'T_FWHM': 45e-15,      
            'E': 2.5e-3,           
            'M2': 1.0,             
            'X0': 0.0,             
            'Z_OFFSET': 0.0,       
            'FOCUS': 1.0,          
            'PULSED': True,       
            'PHI': 0.0,
            'D2': 0.0,
            'D3': 0.0
        }
        params = {key: params.get(key, default_values[key]) for key in default_values}

        if filename:
            print(f"{filename} parameters (GPU) :")
            pprint.pprint(params, sort_dicts=False)
            print()

        # Convert to CuPy types
        self.wl = cp.float64(params['WL'])
        self.sigx = cp.float64(params['SIG_X'])
        self.sigy = cp.float64(params['SIG_Y'])
        self.pulse_len = cp.float64(params['T_FWHM'])
        self.E = cp.float64(params['E'])
        self.focus = cp.float64(params['FOCUS'])
        self.X0 = cp.float64(params['X0'])
        self.Z_offset = cp.float64(params['Z_OFFSET'])
        self.M2 = cp.float64(params['M2'])
        self.D2 = cp.float64(params['D2'])
        self.D3 = cp.float64(params['D3'])
        self.pulsed = params['PULSED']
        self.phase = cp.float64(params['PHI'])
        
        self.P_max = 0.94 * self.E / self.pulse_len
        I0 = cp.float64((2 * self.P_max) / (cp.pi * 4 * self.sigx * self.sigy))  
        self.E0 = cp.sqrt(2 * Z0 * I0)  

        self.k = cp.float64(2 * cp.pi / self.wl)
        self.omega0 = cp.float64(2 * cp.pi * c / self.wl)
        self.sigz = cp.sqrt(cp.float64(2)) * self.pulse_len * c / cp.float64(2.3548)
        
        self.zRx = cp.pi * (cp.float64(2) * self.sigx)**2 / (self.M2 * self.wl)
        self.zRy = cp.pi * (cp.float64(2) * self.sigy)**2 / (self.M2 * self.wl)

        self.beamsize_x = lambda z: self.sigx * cp.sqrt(1 + (z / self.zRx) ** 2)
        self.beamsize_y = lambda z: self.sigy * cp.sqrt(1 + (z / self.zRy) ** 2)
        
        self.t_array, self.E_temporal = self._precompute_temporal_profile_gpu()

    def _precompute_temporal_profile_gpu(self):
        """GPU-accelerated temporal profile calculation"""
        # Calculate sigma_t using GPU math
        sigma_t = self.pulse_len / cp.float64(2 * cp.sqrt(2 * cp.log(2)))
        
        if self.D2 != 0 or self.D3 != 0:
            sigma_t_broadened = cp.sqrt(sigma_t**2 + (self.D2 / (2 * sigma_t))**2)
            time_window = 16 * sigma_t_broadened
        else:
            time_window = 12 * sigma_t 

        #time_window = cp.float64(16) * sigma_t
        
        #print(sigma_t, time_window, self.omega0)
        
        Nt = 16384

        t = cp.linspace(-time_window/2, time_window/2, Nt, dtype=cp.float64)
        dt = t[1] - t[0]

        if not self.pulsed:
            E_temporal = 1
            # Add the logic here
        else:
            # Gaussian envelope with carrier frequency
            envelope = cp.exp(-t**2 / (cp.float64(2) * sigma_t**2))
            E_time = envelope * cp.exp(1j * (self.omega0 * t + self.phase))
    
            # FFT on GPU
            E_freq = cp.fft.fftshift(cp.fft.fft(E_time))
            omega = cp.float64(2 * cp.pi) * cp.fft.fftshift(cp.fft.fftfreq(Nt, dt))
    
            # Dispersion terms
            delta_omega = omega - self.omega0
            phase = (cp.float64(0.5) * self.D2 * delta_omega**2 + 
                    cp.float64(1/6) * self.D3 * delta_omega**3)
    
            # Apply phase and inverse FFT
            E_freq_phased = E_freq * cp.exp(-1j * phase)
            E_time_phased = cp.fft.ifft(cp.fft.ifftshift(E_freq_phased))

            # Normalize to preserve energy (not peak amplitude)
            original_energy = cp.sum(cp.abs(E_time) ** 2) * dt
            new_energy = cp.sum(cp.abs(E_time_phased) ** 2) * dt
            E_temporal = cp.real(E_time_phased) * cp.sqrt(original_energy / new_energy)
        
        return t, E_temporal 

    def E_field(self, X, Y, Z, T):
        """GPU-accelerated electric field calculation"""
        # Convert all inputs to CuPy arrays if not already
        X = cp.asarray(X, dtype=cp.float64)
        Y = cp.asarray(Y, dtype=cp.float64)
        Z = cp.asarray(Z, dtype=cp.float64)
        T = cp.asarray(T, dtype=cp.float64)

        # Beam parameters relative to focus
        Zdif_x = Z - self.focus
        Zdif_y = Z - self.focus
        X = X - self.X0

        # Spatial envelope
        central_E_field = self.E0 * self.sigx / self.beamsize_x(Zdif_x)
        spatial_factor = cp.exp(
            -(Y / self.beamsize_y(Zdif_y)) ** 2
            - (X / self.beamsize_x(Zdif_x)) ** 2)

        # Temporal profile
        tau = T - (Z + self.Z_offset)/c
        temporal_factor = cp.interp(tau, self.t_array, self.E_temporal)

        # Phase terms (vectorized)
        R_x = Zdif_x * (cp.float64(1) + (self.zRx/Zdif_x)**2)
        R_y = Zdif_y * (cp.float64(1) + (self.zRy/Zdif_y)**2)
        
        phase = cp.cos(
            -self.k * (X**2)/(cp.float64(2)*R_x) -
            self.k * (Y**2)/(cp.float64(2)*R_y) +
            0.5 * cp.arctan(Zdif_x/self.zRx) +
            0.5 * cp.arctan(Zdif_y/self.zRy)
        )
     
        return central_E_field * spatial_factor * temporal_factor * phase


class ModulatorGPU:
    def __init__(self, filename=None, **kwargs):
        params = read_file(filename) if filename else {}
        params.update(kwargs)
        
        default_values = {
            'E0': 1492,
            'WL': 800e-9,
            'PERIODS': 9,
            'PERIODLEN': 0.25,
            'IS2D': False,
            'PLOT': False
        }
        params = {key: params.get(key, default_values[key]) for key in default_values}

        self.E0 = cp.float64(params['E0'])
        self.wl = cp.float64(params['WL'])
        self.periods = int(params['PERIODS'])
        self.periodlen = cp.float64(params['PERIODLEN'])
        self.is2d = params['IS2D']
        plot_flag = params['PLOT']

        e_gamma = self.E0 / cp.float64(0.511)
        self.padding = cp.float64(0.2)  

        self.len = self.periods * self.periodlen + 2 * self.padding 
        self.center = self.len / cp.float64(2)

        self.K = cp.sqrt(4 * self.wl * e_gamma**2 / self.periodlen - 2)
        self.Bmax = 2 * cp.pi * self.K * m_e * c / (e_charge * self.periodlen)

        self.l = cp.linspace(0, float(self.len), 1000)
        self.b = cp.zeros_like(self.l)

        und_start = self.padding
        und_end = self.len - self.padding  


        for i in range(len(self.l)):
            pos = self.l[i]
            if und_start <= pos <= und_end:
                local_s = pos - und_start
                self.b[i] = self.Bmax * cp.sin(2 * cp.pi * local_s / self.periodlen)

                if local_s < self.periodlen:
                    self.b[i] *= cp.float64(0.25) if local_s < self.periodlen / 2 else cp.float64(0.75)
                elif (self.periods * self.periodlen - local_s) < self.periodlen:
                    tail_s = self.periods * self.periodlen - local_s
                    self.b[i] *= cp.float64(0.25) if tail_s < self.periodlen / 2 else cp.float64(0.75)

        if plot_flag:
            plt.figure()
            plt.plot(self.l.get(), self.b.get())
            plt.xlabel('z (m)')
            plt.ylabel('B (T)')
            plt.title('GPU Modulator Profile')
            plt.show()

        self.B_func = cupy_interp1d(self.l, self.b, extrapolate=True)


class LatticeGPU:
    def __init__(self, filename=None, **kwargs):
        params = read_file(filename) if filename else {}
        params.update(kwargs)
        
        default_values = {
            'E0': 1492,
            'L1': 800e-9,
            'L2': 400e-9,
            'H': 4,
            'C1': 800,
            'C2': 800,
            'FIELD_FILE': None, 
            'IS2D': False,
            'PLOT': False
        }
        params = {key: params.get(key, default_values[key]) for key in default_values}

        self.is2d = params['IS2D']
        plot_flag = params['PLOT']
        E0 = params['E0']
        e_gamma = E0 / 0.511
        
        self.len = 5.74925
        windings = 48 
        gap = 0.05 
        period = 0.25
        yoke = 0.08
        edge = 0.02    
        drift = 0.5   
        dl = 0.0005   
        
        field_file = params['FIELD_FILE']
        if field_file is not None:
            df = pd.read_csv(field_file, sep='\t')
            # It's better to process pandas data on numpy first, then move to GPU
            l_cpu = np.array(df['z'], dtype='float64') / 1000 
            b_cpu = np.array(df['By'], dtype='float64') 
            
            self.l = cp.asarray(l_cpu)
            self.b = cp.asarray(b_cpu)
            self.len = cp.float64(self.l[-1]) 

            if plot_flag:
                plt.figure()
                plt.plot(l_cpu, b_cpu)
                plt.xlabel('z (m)')
                plt.ylabel('B (T)')
                plt.title('GPU Lattice Field File')
                plt.show()
                
            if self.is2d:
                df2 = pd.read_csv("fieldfiles/M2_279A_transverse_field.txt", skiprows=3, header=None, sep='\t')
                x1 = np.array(df2[0], dtype='float32')
                B1 = np.array(df2[1], dtype='float32')
                B_norm = (B1/B1[300]).reshape(-1,1)
                B_2D = B_norm * b_cpu # Use CPU array for RegularGridInterpolator
                # RegularGridInterpolator is purely a Scipy/CPU function. 
                # If you need 2D interpolation inside a GPU loop, we will need to rewrite this later!
                self.B_func = RegularGridInterpolator((x1, l_cpu), B_2D, method='linear', bounds_error=False, fill_value=0)
            else:
                self.B_func = cupy_interp1d(self.l, self.b, extrapolate=True)
            return
        
        
        IM1 = 0 if params['L1'] == 0 else 1.14 * (2 * np.pi * np.sqrt(4 * params['L1'] * e_gamma**2 / period - 2) * m_e * c / (e_charge * period)) / (mu0 * windings / gap)
        IM2 = 0 if params['L2'] == 0 else 1.14 * (2 * np.pi * np.sqrt(4 * params['L2'] * e_gamma**2 / period - 2) * m_e * c / (e_charge * period)) / (mu0 * windings / gap)
        IR1 = 0 if params['H'] == 0 else 1.14 * (2 * np.pi * np.sqrt(4 * (params['L1']/params['H']) * e_gamma**2 / period - 2) * m_e * c / (e_charge * period)) / (mu0 * windings / gap)

        ID1, ID2, ID3, ID4 = IM1/2, IM2/2, IM2/2, IR1/2
        IC1, IC2 = params['C1'], params['C2']
    
        curr = [-ID1, IM1, -IM1, IM1, -IM1, IM1, -IM1, IM1, 
                -IC1, -IC1, -ID1, IC1, IC1, IC1, IC1, -ID2, -IC1, -IC1, 
                IM2, -IM2, IM2, -IM2, IM2, -IM2, IM2, 
                -IC2, -ID3, IC2, IC2, -ID4, -IC2, 
                IR1, -IR1, IR1, -IR1, IR1, -IR1, ID4]

        factor = mu0 * windings / gap * 1.19     
        b0 = np.array([factor * c_val for c_val in curr])
                
        nm = len(curr)
        magnet, magnet2, yoke2 = period / 2, period / 4, yoke / 2
        len_range = nm * magnet + 2 * drift
        
        nl = int(len_range / dl)
        l_cpu = np.array([(k - 0.5) * dl for k in range(nl)])
        b_cpu = np.zeros(nl)
        
        for m in range(nm):
            l1 = drift + (m - 1) * magnet 
            b_cpu += b0[m] / (np.exp((l1 + magnet2 - yoke2 - l_cpu) / edge) + 1)
            b_cpu += b0[m] / (np.exp((l_cpu - magnet2 - yoke2 - l1) / edge) + 1)
        
        if plot_flag:
            plt.figure()
            plt.plot(l_cpu, b_cpu)
            plt.xlabel('z (m)')
            plt.ylabel('B (T)')
            plt.show()
        
        # Transfer the final generated arrays to the GPU
        self.l = cp.asarray(l_cpu)
        self.b = cp.asarray(b_cpu)
        self.B_func = cupy_interp1d(self.l, self.b, extrapolate=True)
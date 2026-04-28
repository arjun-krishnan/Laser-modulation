# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:31:58 2026

@author: arjun
"""

import cupy as cp
import scipy.constants as const
import matplotlib.pyplot as plt

# Natural constants
c = const.c
e_charge = const.e
m_e = const.m_e

def define_bunch_gpu(Test=False, N=1e4, slicelength=8e-6, E0=1.492e9*e_charge, 
                    dE=7e-4, lattice='del008', R56_dE=0.0007, 
                    R51_dx=4e-4, R52_dxp=4e-5, seed=None):
    """Generate particle bunch directly on GPU with CuPy"""

    if seed is not None:               
        cp.random.seed(seed)  
        
    N_e = int(N)
    ##### electron parameter #####
    energyspread = dE 
        
    if lattice == 'eehg':
        #These values are suitable for the big EEHG lattice
        alphaX = 8.811383e-01 #1.8348
        alphaY = 8.972460e-01 #0.1999
        betaX = 13.546
        betaY = 13.401
        emitX = 1.6e-8
        emitY = 1.6e-9   
        Dx    = 0.0894
        Dxprime = -4.3065e-9 
    
    elif lattice == 'del008':
        # The lattice parameters at the beginning of U250 del008 model
        alphaX = 1.92938
        alphaY = 0.210161
        betaX = 6.69295
        betaY = 13.5857  
        Dx    = -0.0894
        Dxprime = -4.3065e-9 
        emitX = 1.6e-8
        emitY = 1.6e-9 
        
    elif lattice == 'del21':
        # The lattice parameters at the beginning of U250 del21 model
        alphaX = 1.18911
        alphaY = 0.260189
        betaX = 5.378
        betaY = 11.4688
        Dx    = 0.00377785
        Dxprime = -0.00714072
        emitX = 1.6e-8
        emitY = 1.6e-9 
    
    else:
        print("Unknown input for lattice! Please check!")
        
    if(Test):
        slicelength = slicelength 
        N_e = int(1e5)
        energyspread = 0e-4
        emitX = 0
        emitY = 0  
        Dx = 0
        Dxprime = 0
        
    # =============================================
    # GPU-optimized random number generation
    # =============================================
    # Generate random numbers directly on GPU
    CS_inv_x = cp.abs(cp.random.normal(0, cp.sqrt(2*cp.pi)*emitX, N_e))
    CS_inv_y = cp.abs(cp.random.normal(0, cp.sqrt(2*cp.pi)*emitY, N_e))
    phase_x = cp.random.rand(N_e)*2*cp.pi
    phase_y = cp.random.rand(N_e)*2*cp.pi

    # Initialize array directly on GPU
    elec0 = cp.zeros((6, N_e), dtype=cp.float64)
    
    # =============================================
    # GPU-accelerated coordinate initialization
    # =============================================
    elec0[4] = (cp.random.rand(N_e) - 0.5) * slicelength
    elec0[5] = cp.random.normal(0, energyspread, N_e)
    
    # Beam optics calculations on GPU
    sqrt_betaX = cp.sqrt(CS_inv_x * betaX)
    elec0[0] = sqrt_betaX * cp.cos(phase_x) + elec0[5] * Dx
    elec0[1] = -cp.sqrt(CS_inv_x / betaX) * (alphaX * cp.cos(phase_x) + cp.sin(phase_x)) + elec0[5] * Dxprime
    
    sqrt_betaY = cp.sqrt(CS_inv_y * betaY)
    elec0[2] = sqrt_betaY * cp.cos(phase_y)
    elec0[3] = -cp.sqrt(CS_inv_y / betaY) * (alphaY * cp.cos(phase_y) + cp.sin(phase_y))

    # =============================================
    # Special particles for diagnostics (GPU)
    # =============================================
    # Last 6 particles are special markers
    indices = cp.array([-6, -5, -4, -3, -2, -1])
    
    # Set positions and momenta
    elec0[0, indices] = cp.array([0.0, R51_dx, 0.0, 0.0, 0.0, 0.0], dtype=cp.float64)
    elec0[1, indices] = cp.array([0.0, 0.0, 0.0, R52_dxp, 0.0, 0.0], dtype=cp.float64)
    elec0[5, indices] = cp.array([0.0, 0.0, 0.0, 0.0, 0.0, R56_dE], dtype=cp.float64)

    # =============================================
    # GPU-optimized coordinate transformation
    # =============================================
    elec = cp.zeros((6, N_e), dtype=cp.float64)
    p_elecs = cp.sqrt(((1+elec0[5])*E0)**2 - (m_e**2*c**4)) / c
    tan_vals = cp.tan(elec0[3])
    
    elec[5] = p_elecs / cp.sqrt(1/cp.cos(elec0[1])**2 + tan_vals**2)
    elec[4] = elec[5] * tan_vals
    elec[3] = elec[5] * cp.tan(elec0[1])
    
    # Position coordinates
    elec[0] = elec0[0]
    elec[1] = elec0[2]
    elec[2] = elec0[4]

    # Save directly in GPU format
    cp.save("e_dist_gpu.npy", elec)
    
    return elec


def coord_change(elec_dummy, e_E):
    """Transform dummy coordinates to physical coordinates on the GPU"""
    N = elec_dummy.shape[1]
    elec = cp.zeros((6, N), dtype=cp.float64)
    
    # Map coordinates
    elec[0] = elec_dummy[0]
    elec[1] = elec_dummy[2]
    elec[2] = elec_dummy[4]

    # Compute momentum magnitude
    norm_E = 1.0 + elec_dummy[5]
    p_tot = cp.sqrt((norm_E * e_E * e_charge)**2 - (m_e * c**2)**2) / c

    # Angles
    phi_x = elec_dummy[1]
    phi_y = elec_dummy[3]
    denom = cp.sqrt(1.0 / cp.cos(phi_x)**2 + cp.tan(phi_y)**2)

    elec[5] = p_tot / denom
    elec[4] = elec[5] * cp.tan(phi_y)
    elec[3] = elec[5] * cp.tan(phi_x)

    return elec


def calc_phasespace(bunch, e_E, plot=False):
    """Calculate and optionally plot the phase space (energy spread vs z)"""
    p = cp.sqrt(cp.sum(bunch[3:]**2 , axis=0))
    E = cp.sqrt(m_e**2 * c**4 + p**2 * c**2)
    dEE = E / e_E - 1.0
    
    # REMOVED cp.copy() -> A view is perfectly fine here
    z = bunch[2] 
    
    if plot:
        plt.figure()
        plt.plot((z - cp.mean(z)).get() * 1e6 , dEE.get() ,',')
        plt.xlabel(r'z ($\mu m$)')
        plt.ylabel(r'$\Delta E/E_0$')
        plt.tight_layout()
        plt.show()
        
    return z.get(), dEE.get()   


def calc_bn(tau0, wl, printmax=True):
    """Calculate the bunching factor efficiently on the GPU"""
    wl = cp.asarray(wl, dtype=cp.float64).reshape(-1,)
    
    # Pre-allocate GPU array to prevent PCIe ping-pong
    bn = cp.zeros(len(wl), dtype=cp.float64)
    N = len(tau0)

    for i in range(len(wl)):
        # Calculate directly into the pre-allocated GPU array
        bn[i] = cp.abs(cp.sum(cp.exp(-1j * 2 * cp.pi * (tau0 / wl[i])))) / N

    # argmax is significantly faster than cp.where(bn == cp.max(bn))
    max_idx = int(cp.argmax(bn))
    
    if printmax:
        print(f"Maximum bunching factor is {cp.round(bn[max_idx], 4)} at {cp.round(wl[max_idx]*1e9, 2)} nm")
        
    return bn.get()


def plot_slice(z, wl, slice_len=0, n_slice=40, plot=True):
    """Calculate and plot bunching factor by slicing the bunch longitudinally"""
    z_min, z_max = cp.min(z), cp.max(z)
    
    if slice_len != 0:
        n_slice = int((z_max - z_min) / slice_len)
    
    bin_edges = cp.linspace(z_min, z_max, n_slice + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_indices = cp.digitize(z, bin_edges) - 1 
    valid_mask = (bin_indices >= 0) & (bin_indices < n_slice)

    # Pre-allocate on GPU
    bn = cp.zeros(n_slice, dtype=cp.float64)
    
    for i in range(n_slice):
        idx = (bin_indices == i) & valid_mask
        z_bin = z[idx]
        if z_bin.size > 0:
            # We use [0] because calc_bn returns a CPU numpy array.
            # We immediately push that single float back to the GPU array.
            bn[i] = calc_bn(z_bin, cp.array([wl]), printmax=False)[0]
        else:
            bn[i] = 0.0

    z_slice = bin_centers - cp.mean(bin_centers)

    if plot:
        plt.figure()
        plt.plot(z_slice.get(), bn.get())
        plt.xlabel(r'Slice Position ($\mu m$)')
        plt.ylabel('Bunching Factor')
        plt.show()
    
    return z_slice.get(), bn.get()

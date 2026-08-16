# -*- coding: utf-8 -*-
"""
Updated L2 Sigma Scan Laser Modulation Example

This example demonstrates how to scan the laser sigma (beam size) parameter
to study its effect on modulation and bunching. It uses the updated API with
input files for laser and modulator parameters.

The script:
1. Reads base laser and modulator parameters from input files
2. Generates an electron bunch
3. Performs a scan of laser sigma parameter for Modulator 2
4. Tracks through both modulators for each sigma value
5. Analyzes bunching factor vs sigma
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from pathlib import Path

# Import from laser_modulator package
from laser_modulator.components import Laser, Modulator
from laser_modulator.tracking import lsrmod_track
from laser_modulator.phase_space import define_bunch, calc_phasespace, calc_bn, plot_slice

# Natural constants
c = const.c
e_charge = const.e
m_e = const.m_e

# Get the directory where this script is located
script_dir = Path(__file__).parent
input_dir = script_dir / "input_files"

##### Simulation parameters #####
slicelength = 30e-6    # length of simulated bunch slice in m
tstep = 5e-12          # timestep in s
N_e = int(1e5)         # number of electrons

print("=" * 70)
print("L2 Sigma Scan Laser Modulation Example - Using Input Files")
print("=" * 70)

##### Define electron bunch #####
print("\nGenerating electron bunch...")
e_E = 1.492e9 * e_charge    # electron energy in J
bunch_test = define_bunch(Test=True)
bunch_init = define_bunch(Test=False, N=N_e, slicelength=slicelength, E0=e_E)
elec = np.copy(bunch_init)

##### Define Modulator 1 from LTT file #####
print("\nInitializing Modulator 1...")
mod1_file = input_dir / "Mod1.LTT"
mod1 = Modulator(filename=str(mod1_file))

##### Define Laser 1 from LSR file #####
print("Initializing Laser 1...")
laser1_file = input_dir / "Laser1.LSR"
l1 = Laser(filename=str(laser1_file))

##### Define Modulator 2 from LTT file #####
print("Initializing Modulator 2...")
mod2_file = input_dir / "Mod2.LTT"
mod2 = Modulator(filename=str(mod2_file))

##### Define Laser 2 from LSR file (template for sigma scan) #####
print("Initializing Laser 2 (template)...")
laser2_file = input_dir / "Laser2.LSR"
l2_template = Laser(filename=str(laser2_file))

##### Test tracking through Modulator 1 #####
print("\n" + "=" * 70)
print("Testing Modulator 1 with test bunch...")
print("=" * 70)
elec_test_m1 = lsrmod_track(mod1, l1, bunch_test, tstep=tstep)
z_test_m1, dE_test_m1 = calc_phasespace(elec_test_m1, e_E, plot=False)
A1 = max(dE_test_m1)
print(f"Maximum energy modulation (A1) from Modulator 1: {A1:.6e} (ΔE/E0)")

##### Test tracking through Modulator 2 (baseline) #####
print("\nTesting Modulator 2 with test bunch (baseline)...")
elec_test_m2 = lsrmod_track(mod2, l2_template, bunch_test, tstep=tstep)
z_test_m2, dE_test_m2 = calc_phasespace(elec_test_m2, e_E, plot=False)
A2 = max(dE_test_m2)
print(f"Maximum energy modulation (A2) from Modulator 2: {A2:.6e} (ΔE/E0)")

##### Track full bunch through Modulator 1 #####
print("\n" + "=" * 70)
print("Tracking full bunch through Modulator 1...")
print("=" * 70)
elec_M1 = lsrmod_track(mod1, l1, elec, tstep=tstep)

# Calculate phase space after Modulator 1
z_M1, dE_M1 = calc_phasespace(elec_M1, e_E, plot=False)
print(f"Phase space after Modulator 1:")
print(f"  Energy spread range: {min(dE_M1):.6e} to {max(dE_M1):.6e}")

##### Sigma scan for Modulator 2 #####
print("\n" + "=" * 70)
print("Performing Sigma Scan for Modulator 2...")
print("=" * 70)

# Define sigma range for scanning (in meters)
sigma_array = np.linspace(0.2e-3, 2.0e-3, 10)
bunching_results = []
sigma_results = []

print(f"\nScanning {len(sigma_array)} sigma values...")
print("-" * 70)

for i, sigx in enumerate(sigma_array):
    print(f"Sigma scan [{i+1}/{len(sigma_array)}]: σ = {sigx*1e3:.3f} mm")
    
    # Create laser with modified sigma
    l2_scan = Laser(
        filename=str(laser2_file),
        SIG_X=sigx,
        SIG_Y=sigx
    )
    
    # Test tracking to get modulation amplitude A2
    elec_test_m2_scan = lsrmod_track(mod2, l2_scan, bunch_test, tstep=tstep)
    z_test_m2_scan, dE_test_m2_scan = calc_phasespace(elec_test_m2_scan, e_E, plot=False)
    A2_scan = max(dE_test_m2_scan)
    print(f"    A2 = {A2_scan:.6e}")
    
    # Track through Modulator 2
    elec_M2_scan = lsrmod_track(mod2, l2_scan, elec_M1, tstep=tstep)
    z_M2_scan, dE_M2_scan = calc_phasespace(elec_M2_scan, e_E, plot=False)
    
    # Calculate bunching factor
    wl_scan = np.linspace(19e-9, 20e-9, 501)
    b_scan = calc_bn(z_M2_scan, wl_scan, printmax=False)
    max_bunching = np.max(b_scan)
    
    bunching_results.append(max_bunching)
    sigma_results.append(sigx * 1e3)  # Convert to mm
    
    print(f"    Max bunching factor = {max_bunching:.6f}")
    print()

##### Plot results #####
print("=" * 70)
print("Plotting sigma scan results...")
print("=" * 70)

plt.figure(figsize=(10, 6))
plt.plot(sigma_results, bunching_results, 'o-', linewidth=2, markersize=8)
plt.xlabel("Laser Sigma (mm)", fontsize=12)
plt.ylabel("Maximum Bunching Factor", fontsize=12)
plt.title("Bunching Factor vs Laser Sigma (Modulator 2)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(script_dir / "bunching_vs_sigma.png"), dpi=150)
print(f"Results plot saved to: {script_dir / 'bunching_vs_sigma.png'}")

# Find optimal sigma
optimal_idx = np.argmax(bunching_results)
optimal_sigma = sigma_results[optimal_idx]
max_bunching = bunching_results[optimal_idx]

print(f"\nOptimal results:")
print(f"  Best sigma: {optimal_sigma:.3f} mm")
print(f"  Max bunching factor: {max_bunching:.6f}")

print("\n" + "=" * 70)
print("L2 Sigma Scan completed successfully!")
print("=" * 70)

plt.show()

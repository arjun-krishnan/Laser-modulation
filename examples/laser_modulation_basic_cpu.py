# -*- coding: utf-8 -*-
"""
Updated Basic Laser Modulation Example

This example demonstrates laser-electron interaction simulation using the current
API with input files for laser and modulator parameters.

The script:
1. Reads laser and modulator parameters from LSR and LTT files
2. Generates an electron bunch
3. Tracks electrons through the modulator
4. Analyzes the resulting phase space
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
print("Laser Modulation Basic Example - Using Input Files")
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
print("\nInitializing Modulator 2...")
mod2_file = input_dir / "Mod2.LTT"
mod2 = Modulator(filename=str(mod2_file))

##### Define Laser 2 from LSR file #####
print("Initializing Laser 2...")
laser2_file = input_dir / "Laser2.LSR"
l2 = Laser(filename=str(laser2_file))

##### Test tracking through Modulator 1 #####
print("\n" + "=" * 70)
print("Testing Modulator 1 with test bunch...")
print("=" * 70)
elec_test_m1 = lsrmod_track(mod1, l1, bunch_test, tstep=tstep)
z_test_m1, dE_test_m1 = calc_phasespace(elec_test_m1, e_E, plot=False)
A1 = max(dE_test_m1)
print(f"Maximum energy modulation (A1) from Modulator 1: {A1:.6e} (ΔE/E0)")

##### Test tracking through Modulator 2 #####
print("\nTesting Modulator 2 with test bunch...")
elec_test_m2 = lsrmod_track(mod2, l2, bunch_test, tstep=tstep)
z_test_m2, dE_test_m2 = calc_phasespace(elec_test_m2, e_E, plot=False)
A2 = max(dE_test_m2)
print(f"Maximum energy modulation (A2) from Modulator 2: {A2:.6e} (ΔE/E0)")

##### Track full bunch through Modulator 1 #####
print("\n" + "=" * 70)
print("Tracking full bunch through Modulator 1...")
print("=" * 70)
elec_M1 = lsrmod_track(mod1, l1, elec, tstep=tstep)

# Calculate phase space after Modulator 1
z_M1, dE_M1 = calc_phasespace(elec_M1, e_E, plot=True)
print(f"\nPhase space after Modulator 1:")
print(f"  Energy spread range: {min(dE_M1):.6e} to {max(dE_M1):.6e}")
print(f"  Longitudinal position range: {(max(z_M1) - min(z_M1))*1e6:.2f} µm")

##### Track full bunch through Modulator 2 #####
print("\n" + "=" * 70)
print("Tracking full bunch through Modulator 2...")
print("=" * 70)
elec_M2 = lsrmod_track(mod2, l2, elec_M1, tstep=tstep)

# Calculate phase space after Modulator 2
z_M2, dE_M2 = calc_phasespace(elec_M2, e_E, plot=True)
print(f"\nPhase space after Modulator 2:")
print(f"  Energy spread range: {min(dE_M2):.6e} to {max(dE_M2):.6e}")
print(f"  Longitudinal position range: {(max(z_M2) - min(z_M2))*1e6:.2f} µm")

##### Calculate bunching factor #####
print("\n" + "=" * 70)
print("Calculating bunching factor...")
print("=" * 70)
wl = np.linspace(19e-9, 20e-9, 501)
b = calc_bn(z_M2, wl, printmax=True)

plt.figure(figsize=(10, 6))
plt.plot(wl*1e9, b)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Bunching Factor")
plt.title("Bunching Factor vs Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(script_dir / "bunching_factor.png"), dpi=150)
print(f"Bunching factor plot saved to: {script_dir / 'bunching_factor.png'}")

##### Plot slice analysis #####
print("\n" + "=" * 70)
print("Analyzing slice bunching...")
print("=" * 70)
z_slice, bn_slice = plot_slice(z_M2, wl, n_slice=50)
plt.savefig(str(script_dir / "bunching_slice.png"), dpi=150)
print(f"Bunching slice plot saved to: {script_dir / 'bunching_slice.png'}")

print("\n" + "=" * 70)
print("Simulation completed successfully!")
print("=" * 70)

plt.show()

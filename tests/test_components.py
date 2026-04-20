import os
import numpy as np
from laser_modulator.components import Laser, Modulator, SPEED_Lattice

def test_laser_k_calculation(tmp_path):
    wl = 400e-9
    k_expected = 2 * np.pi / wl
    
    dummy_filename = tmp_path/"test_laser_params.txt"
    with open(dummy_filename, "w") as f:
        f.write("START\n")
        f.write(f"WL: {wl}\n")
        f.write("END\n")
        
    laser = Laser(dummy_filename)
    
    np.testing.assert_allclose(laser.k, k_expected, rtol=1e-7)
    
    
def test_modulator_boundaries_integral(tmp_path):
    dummy_filename = tmp_path/"test_mod_params.LTT"
    with open(dummy_filename, "w") as f:
        f.write("START\n")
        f.write("E0: 1492 #MeV \n WL: 800e-9 \n NPERIOD: 3 \n PERIODLEN: 0.15 \n")
        f.write("END\n")
        
        mod_test = Modulator(dummy_filename, plot=False)
        
        B_start = mod_test.B[0]
        B_end = mod_test.B[-1]
        
        B_integral = np.trapz(mod_test.B, mod_test.s)
        
        np.testing.assert_allclose(B_start, 0.0, atol=1e-8)
        np.testing.assert_allclose(B_end, 0.0, atol=1e-8)
        np.testing.assert_allclose(B_integral, 0.0, atol=1e-8)
        

def test_speed_boundaries_integral(tmp_path):
    dummy_filename = tmp_path/"test_mod_params.LTT"
    with open(dummy_filename, "w") as f:
        f.write("START\n")
        f.write("END\n")
        
    speed_test = SPEED_Lattice(dummy_filename)
    
    B_start = speed_test.b[0]
    B_end = speed_test.b[-1]
    
    B_integral = np.trapz(speed_test.b, speed_test.l)
    
    np.testing.assert_allclose(B_start, 0.0, atol=1e-8)
    np.testing.assert_allclose(B_end, 0.0, atol=1e-8)
    np.testing.assert_allclose(B_integral, 0.0, atol=1e-8)
        
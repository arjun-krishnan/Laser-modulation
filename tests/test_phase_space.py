# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:51:26 2026

@author: arjun
"""

import numpy as np
from laser_modulator.phase_space import calc_bn

def test_calc_bn_perfect_bunching():
    # Create 10 electrons, all perfectly aligned at delay = 0
    tau0_perfect = np.zeros(10) 
    
    wavelengths = np.array([800e-9, 400e-9]) 
    
    bunching_factors = calc_bn(tau0_perfect, wavelengths, printmax=False)
    
    # Assertion: The bunching factor should be exactly 1.0 for both wavelengths
    expected_factors = np.array([1.0, 1.0])
    
    np.testing.assert_allclose(bunching_factors, expected_factors, rtol=1e-7)
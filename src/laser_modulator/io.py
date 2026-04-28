# -*- coding: utf-8 -*-
"""
I/O Module

This module contains functions for reading and writing from/to files.
"""

# src/laser_modulator/io.py

import numpy as np
import pandas as pd
import pathlib
import ast  # Safe evaluation module

def read_file(filename):
    """
    Reads parameter files safely.
    Values between START and END markers are extracted into a dictionary.
    """
    parameters = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    start_reading = False
    for line in lines:
        line = line.strip()

        if line == "START":
            start_reading = True
            continue
        elif line == "END":
            break

        if start_reading and ": " in line:
            key, value = line.split(': ', 1)
            try:
                # Safely evaluate strings to floats, ints, or booleans
                parameters[key.upper()] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If it's a plain string (like a filename), just keep it as a string
                parameters[key.upper()] = value
                
    return parameters

def write_results(bunch, file_path, overwrite=False):
    """
    Writes the particle bunch to a CSV file. 
    Safe for automated batch jobs (no blocking inputs).
    """
    print(f"Writing to {file_path} ...")
    file = pathlib.Path(file_path)
    
    if file.is_file() and not overwrite:
        raise FileExistsError(f"The file {file_path} already exists. Set overwrite=True to replace it.")
        
    # Handle if bunch is a raw NumPy array (from your tracker)
    if isinstance(bunch, np.ndarray):
        # Transpose it so columns are x, px, y, py, z, pz
        header = "x,px,y,py,z,pz"
        np.savetxt(file_path, bunch.T, delimiter=",", header=header, comments='')
        
    # Handle if bunch was already converted to a Pandas DataFrame
    elif isinstance(bunch, pd.DataFrame):
        bunch.to_csv(file_path, index=False)
        
    else:
        raise TypeError("bunch must be a NumPy array or Pandas DataFrame")
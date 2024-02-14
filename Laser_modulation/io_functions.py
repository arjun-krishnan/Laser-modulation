# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:32:18 2024

@author: arjun
"""

import numpy as np
import pathlib

def read_file(filename):
    parameters = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Initialize flag to indicate whether to start reading parameters
        start_reading = False
        for line in lines:
            line = line.strip()

            # Check for START and END markers
            if line == "START":
                start_reading = True
                continue
            elif line == "END":
                break

            # If START marker is found, start reading parameters
            if start_reading:
                key, value = line.split(': ')

                parameters[key.upper()] = eval(value)
                
    return(parameters)



def write_results(bunch,file_path):
    print("Writing to "+file_path+" ...")
    file = pathlib.Path(file_path)
    if file.is_file():
        ch = input("The file already exist! Overwrite? (Y/N)")
        if ch == 'y':
            bunch.to_csv(file_path)
    else:
        bunch.to_csv(file_path)
# Laser-modulation
This python code simulates laser-electron interaction in an undulator field, defined by user.

## Input files
### Lattice Input File
The file <filename>.LTT defines the properties of the undulator magnetic field used in the simulation. The parameters included in this file are:
- E0: Energy of the electron beam in GeV.
- WL: Fundamental wavelength of the undulator in meters.
- NPERIOD: Number of periods in the lattice.
- PERIODLEN: Length of each lattice period in meters.

*Below are only valid when defining **SPEED_Lattice** object*
- M1: First mirror wavelength in meters.
- M2: Second mirror wavelength in meters.
- RAD: Radius of the lattice in meters.
- C1: Chicane 1 parameter.
- C2: Chicane 2 parameter.

### Laser Input File
The file <filename>.LSR defines the properties of the laser used in the simulation. The parameters are:
- WL: Wavelength of the laser in meters.
- SIG_X: Sigma width at focus in meters (X-direction).
- SIG_Y: Sigma width at focus in meters (Y-direction).
- T_FWHM: Full width at half maximum (FWHM) of the temporal profile in seconds.
- E: Pulse energy of the laser in joules.
- FOCUS: Focus distance of the laser from the beginning of the defined magnetic field in meters.
- X0: Initial position in the X-direction.
- Z_OFFSET: Pulse offset in the Z-direction.
- M2: Beam quality parameter.
- PULSED: Boolean indicating whether the laser is pulsed (True/False).
- PHI: Spectral phase of the laser, quantifying the frequency chirp.

## Usage
To use the input files, adjust the parameters to match the desired lattice and laser properties for your simulation.

- For normal undulators, use the **Modulator** object, specifying the filename of the LTT file as a parameter.

- Alternatively, define a lattice using a user-defined field file (.txt). The file should have two columns: the first column indicates the z position in millimeters, and the second column indicates the vertical magnetic field in Tesla, separated by a tab.

- For the special Modulator-Chicane-Modulator-Chicane-Radiator configuration used in the **EEHG** project named **SPEED** at *DELTA, TU Dortmund*, use the **SPEED_Lattice** object. This object also requires the LTT file as input, along with additional parameters. ([link to the paper](https://accelconf.web.cern.ch/ipac2023/pdf/MOPM032.pdf))

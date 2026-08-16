# Updated Laser Modulation Examples

This directory contains updated examples that use the current laser-modulator API with support for input files and GPU acceleration.

## Example Files

### 1. **laser_modulation_basic_updated.py** (CPU Version)
Updated basic example using the modern API with input files.

**Features:**
- Reads laser parameters from LSR files (Laser1.LSR, Laser2.LSR)
- Reads modulator parameters from LTT files (Mod1.LTT, Mod2.LTT)
- Tracks electrons through two modulators sequentially
- Calculates phase space and bunching factors
- Generates visualization plots

**Usage:**
```bash
python laser_modulation_basic_updated.py
```

**Output:**
- Console output showing modulation parameters and tracking progress
- PNG files: `bunching_factor.png`, `bunching_slice.png`

---

### 2. **laser_modulation_basic_gpu.py** (GPU-Accelerated Version)
GPU-accelerated version of the basic example for faster computation.

**Features:**
- All features of the CPU version
- GPU acceleration via CuPy (NVIDIA CUDA)
- Requires: CuPy installed (`pip install cupy-cuda11x`)
- Significantly faster for large electron bunches

**Usage:**
```bash
python laser_modulation_basic_gpu.py
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CuPy: `pip install cupy-cuda11x` (replace 11x with your CUDA version)
- To find your CUDA version: `nvidia-smi`

**Output:**
- Console output with GPU computation info
- PNG files: `bunching_factor_gpu.png`, `bunching_slice_gpu.png`

---

### 3. **laser_modulation_L2sigma_updated.py** (CPU Version - Sigma Scan)
Updated example that performs a parameter scan of the laser sigma (beam size).

**Features:**
- Scans laser sigma parameter from 0.2 mm to 2.0 mm
- Tracks through both modulators for each sigma value
- Analyzes how bunching factor varies with laser beam size
- Identifies optimal sigma for maximum bunching

**Usage:**
```bash
python laser_modulation_L2sigma_updated.py
```

**Output:**
- Console output showing sigma scan progress and results
- PNG file: `bunching_vs_sigma.png` (shows optimal sigma)

**Scan Parameters:**
- Sigma range: 0.2 - 2.0 mm
- Number of points: 10
- Modify `sigma_array = np.linspace(...)` to change scan range

---

### 4. **laser_modulation_L2sigma_gpu.py** (GPU-Accelerated Version - Sigma Scan)
GPU-accelerated sigma scan for even faster parameter optimization.

**Features:**
- All features of the CPU sigma scan
- GPU acceleration for faster computation
- Same output format as CPU version for easy comparison

**Usage:**
```bash
python laser_modulation_L2sigma_gpu.py
```

**Requirements:**
- Same as GPU basic example (NVIDIA GPU + CuPy)

**Output:**
- Console output with GPU computation info
- PNG file: `bunching_vs_sigma_gpu.png`

---

## Input Files

Located in `input_files/` directory:

| File | Purpose | Parameters |
|------|---------|------------|
| `Laser1.LSR` | Laser 1 parameters | Wavelength, beam size, pulse energy, etc. |
| `Laser2.LSR` | Laser 2 parameters | Wavelength, beam size, pulse energy, etc. |
| `Mod1.LTT` | Modulator 1 lattice | Electron energy, undulator wavelength, periods |
| `Mod2.LTT` | Modulator 2 lattice | Electron energy, undulator wavelength, periods |
| `U250.LTT` | SPEED lattice (advanced) | EEHG configuration parameters |

### LSR File Format Example:
```
# Parameters for Laser 1
START
WL: 800e-9              # Wavelength in meters
SIG_X: 0.8e-3           # Horizontal sigma width at focus
SIG_Y: 0.8e-3           # Vertical sigma width at focus
T_FWHM: 45e-15          # Pulse duration (FWHM)
E: 2.1e-3               # Pulse energy in joules
FOCUS: 0.55             # Focus distance from undulator start
X0: 0                   # Initial X offset
Z_OFFSET: 0             # Initial Z offset
M2: 1.0                 # Beam quality parameter
PULSED: False           # Pulsed laser flag
PHI: 0                  # Spectral phase (chirp)
END
```

### LTT File Format Example:
```
# Properties of lattice
START
E0: 1492                # Electron energy in GeV
WL: 800e-9              # Laser wavelength in meters
NPERIOD: 6              # Number of undulator periods
PERIODLEN: 0.15         # Period length in meters
END
```

---

## API Changes from Old Code

### Old API (legacy):
```python
from lsrmod_functions import *

mod1 = Modulator(periodlen=0.20, periods=9, laser_wl=l1_wl, e_gamma=e_gamma)
l1 = Laser(wl=l1_wl, sigx=l1_sigx, sigy=l1_sigx, pulse_len=l1_fwhm, ...)
elec = lsrmod_track(mod1, l1, bunch, tstep=tstep)
```

### New API (current):
```python
from laser_modulator.components import Laser, Modulator
from laser_modulator.tracking import lsrmod_track

# From files:
mod1 = Modulator(filename="Mod1.LTT")
l1 = Laser(filename="Laser1.LSR")

# Or programmatically:
mod1 = Modulator(filename="Mod1.LTT", NPERIOD=9)
l1 = Laser(filename="Laser1.LSR", SIG_X=0.8e-3)

elec = lsrmod_track(mod1, l1, bunch, tstep=tstep)
```

### GPU API:
```python
from laser_modulator.gpu.components import LaserGPU, ModulatorGPU
from laser_modulator.gpu.tracking import lsrmod_track_gpu

mod1_gpu = ModulatorGPU(filename="Mod1.LTT")
l1_gpu = LaserGPU(filename="Laser1.LSR")
elec_gpu = lsrmod_track_gpu(mod1_gpu, l1_gpu, bunch_gpu, tstep=tstep)
```

---

## Simulation Parameters

Default parameters in all examples:
- **Electron bunch size**: 100,000 electrons
- **Slice length**: 30 µm
- **Time step**: 5 ps
- **Electron energy**: 1.492 GeV (DELTA facility)
- **Wavelength range (bunching)**: 19-20 nm
- **Number of wavelength points**: 501

Modify these in the script for different simulations:
```python
slicelength = 30e-6    # Change electron bunch length
tstep = 5e-12          # Change time step
N_e = int(1e5)         # Change number of electrons
```

---

## Performance Notes

### CPU vs GPU Performance:
- **CPU**: Good for small simulations (< 10k electrons) or parameter studies
- **GPU**: Recommended for production runs (100k+ electrons)
- **Speedup**: Typically 10-20x faster with GPU on NVIDIA cards

---

## Troubleshooting

### CuPy Installation Issues:
```bash
# Find your CUDA version
nvidia-smi

# Install corresponding CuPy version
pip install cupy-cuda11x  # Replace 11x with your CUDA version

# Verify installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Memory Errors:
- Reduce `N_e` parameter
- Use CPU version instead
- Check GPU memory: `nvidia-smi`

### Import Errors:
- Ensure `laser_modulator` package is installed
- Install with: `pip install -e .` from project root

---

## Creating Custom Examples

To create your own example:

1. **Prepare input files** in `input_files/` directory
2. **Copy a template** (e.g., `laser_modulation_basic_updated.py`)
3. **Modify parameters**:
   ```python
   mod1 = Modulator(filename="Mod1.LTT")
   l1 = Laser(filename="Laser1.LSR")
   # or
   l1 = Laser(filename="Laser1.LSR", SIG_X=1.0e-3, E=5.0e-3)
   ```
4. **Run tracking** and analyze results
5. **Save plots** for documentation

---

## References

- **EEHG Project**: https://accelconf.web.cern.ch/ipac2023/pdf/MOPM032.pdf
- **DELTA Facility**: TU Dortmund (https://www.delta.tu-dortmund.de/)
- **Laser-Electron Interaction Theory**: See project README.md

---

## Version Info

- **Created**: August 2026
- **API Version**: 2.0 (File-based configuration)
- **GPU Support**: CuPy-based CUDA acceleration
- **Python Version**: 3.8+


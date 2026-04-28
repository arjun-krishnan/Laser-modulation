
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    import warnings
    warnings.warn("CuPy is not installed. GPU features in laser_modulator will be disabled.")

# Only expose the GPU functions if the hardware is actually there
if GPU_AVAILABLE:
    from .components import LaserGPU, ModulatorGPU, LatticeGPU
    from .tracking import lsrmod_track_gpu, chicane_track
    from .utils import define_bunch_gpu, calc_bn, plot_slice
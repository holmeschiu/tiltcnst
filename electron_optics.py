#===================Imports=================
import math
import numpy as np
import scipy as sp
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import time


#===================Functions=================
# Electron wavelength function
def wvlength_pm(kvolt: int) -> float:
    """Electron wavelength. 
    Args:
        kvolt (int): accelerating voltage in kV
    Returns:
        float: wavelength in pm
    """
    return 1.23e3 / math.sqrt(kvolt*1e3 * (1 + 9.78e-7*kvolt*1e3))


# Electron interaction coefficient function
def sigma(kvolt: float) -> float:
    """Interaction coefficient of an electron.  $$\sigma _e = 2 \pi m e \lambda / h^2$$
    Args:
        kvolt (float): accelerating voltage in kV.
    Returns:
        float: interaction coefficient of an electron. 
    """
    return 2.08847e6 * wvlength_pm(kvolt)


# Aperture function 
def aperture(cutf_frequency_inm: float, imge_size: int, pxel_size_nm: float) -> np.ndarray:
    """Objective aperture function.  A 2D mask function.  The origin is in the image center. 

    Args:
        cutf_frequency_inm (float): cutoff frequency in nm^-1
        imge_size (int): image size in pixels
        pxel_size_nm (float): pixel size in nm/pixel 
    Returns:
        np.ndarray: a mask function for the aperture
    """
    # Initializing aperture array
    ap = np.zeros((imge_size, imge_size), dtype=float)
    
    # Validate even-sized image
    if imge_size % 2 == 1:
        print("The input image size should be even. ")
        sys.exit(0)
    
    # Compute center and frequency step
    else:
        cntr_shift_inm = imge_size / 2 / pxel_size_nm
        freq_step_inm = (1/pxel_size_nm) / imge_size

    # Create grid
    y, x = np.meshgrid(np.arange(imge_size), np.arange(imge_size), indexing='ij')
    
    # Compute radial frequency at each pixel
    distances = np.sqrt((x * freq_step_inm - cntr_shift_inm)**2 + (y * freq_step_inm - cntr_shift_inm)**2)
    
    # Create circular mask (aperture)
    ap[distances < cutf_frequency_inm] = 1.0

    return ap
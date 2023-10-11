#!/usr/bin/env python3
# Multislice method.  
#
#   nx, ny, nz:     location of the specimen in pixels
#   pixel_size_nm:  pixel size (nm)
#
#   kvolt:          accelerating voltage (kV)
#   spcm_tilt_rad:  specimen tilting angle (rad)
#   spcm_tilta_rad: specimen tilting axis (rad)
#   beam_tilt_rad:  beam tilting angle (rad)
#   dz_nm:          slice thickness (nm)
#   wvlength_pm:    electron wavelength (pm)
#
#   is_tds:         use thermal diffuse scattering
#
#
#   Exit wave function at each step:
#       psi_n_1(x, y, dz_nm) = exp(6.28318*i*dz_nm / wvlength_pm) * conv(psi_n(x, y, dz_nm), propagator(x, y, dz_nm))
#       propagator(x, y, dz_nm) = -i / wvlength_pm / dz_nm * exp(i*3.14159 / wvlength_pm / dz_nm(x**2 + y**2))
#
#

# Generic imports
import cmath
import numpy as np

# Defined imports
import atomic_potential
import pdb
import molecular_potential_pdb as mp

# Imports for typing 
from typing import Any, Tuple, List
from nptyping import NDArray


# Parameters ==========
def wvlength_pm(kvolt: int) -> float:
    """Electron wavelength. 
    Args:
        kvolt (int): accelerating voltage in kV
    Returns:
        float: wavelength in pm
    """
    return 1.23e3 / cmath.sqrt(kvolt*1e3 * (1 + 9.78e-7*kvolt*1e3))


def sigma(kvolt: float) -> float:
    """Interaction coefficient of an electron.  $$\sigma _e = 2 \pi m e \lambda / h^2$$
    Args:
        kvolt (float): accelerating voltage in kV.
    Returns:
        float: interaction coefficient of an electron. 
    """
    return 2.08847e6 * wvlength_pm(kvolt)


# Entrance ==========
def main():
    pdb_file = "protein.pdb"
    box_size = 1024
    pxel_size_nm = 0.105
    trgt_slice_nm = 0.3
    kvolt = 300.0

    slce_zpixels = trgt_slice_nm // pxel_size_nm
    total_slice_number = box_size // slce_zpixels + 1

    protein = PDB(pdb_file).read_pdb()
    mol_potential = mp.integrate_atomic_potential(protein, box_size, pxel_size_nm)

    deltaZ = slice_zpixels * pxel_size_nm
    box_center = box_size / 2
    propagator_r = np.zeros(box_size, box_size)
    propagator_i = np.zeros(box_size, box_size)
    for m in range(box_size):
        for n in range(box_size):
            rsq_ = (m - box_center)**2 + (n - box_center)**2
            rsq_ = rsq_ * pxel_size_nm**2
            prop_phs = 3.14159 / wvlength_pm(kvolt) / deltaZ * rsq_ * 1e3
            c1 = 1 / wvlength_pm(kvolt) / deltaZ * 1e21

            propagator_r[m, n] = propagator_r[m, n] + c1 * cmath.sin(prop_phs)
            propagator_i[m, n] = propagator_i[m, n] + c1 * cmath.cos(prop_phs)
    
    # Initialize the incident wave function $$\psi _0 (x, y) = 1$$
    psi_r = np.ones(box_size, box_size)
    psi_i = np.zeros(box_size, box_size)

    for slc in range(total_slce_number):
        # Procedure (page 181)
        # Divide the specimen into thin slices and integrate them along z axis
        slce_start = slc * slce_zpixels

        if slc != (total_slice_number - 1):
            slc_end = slce_start + slce_zpixels
        else:
            slc_end = box_size
            
        v_n = np.array(mol_potential[:, :, slce_start:slce_end])
        v_nz = np.sum(v_n, axis=0)      # need to check the axis here

        # Transmission function
        tr_n = cmath.cos(sigma(kvolt)*v_nz)
        ti_n = cmath.sin(sigma(kvolt)*v_nz)

        # Recursively transmit and propagate the wave function through each slice
        # Propagator: Eq. (6.65) on page 165
        t_psi_r = tr_n * psi_r - ti_n * psi_i
        t_psi_i = ti_n * psi_r + tr_n * psi_i

        prpg_n = propagator_r + 1j * propagator_i
        trns_n = t_psi_r + 1j * t_psi_i
        psi_n1 = np.ifft(np.fft(prpg_n) * np.fft(trns_n))

        psi_r = psi_n1.real
        psi_i = psi_n1.imag
            
    # psi_r and psi_i is the final exit wave for the real and imaginary parts, respectively. 
    #
    # Pass the final exit wave function to the TCIF if the specimen is tilted
    # or to the CTF if the specimen is untilted. 


if __name__ == '__main__':
    main()

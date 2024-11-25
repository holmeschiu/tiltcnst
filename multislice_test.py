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
import matplotlib.pyplot as plt
import time

# Defined imports
import atomic_potential
import pdb_1 as pdb
import molecular_potential_pdb as mp
import tcif

# Imports for typing 
from typing import Any, Tuple, List
from nptyping import NDArray


# Function to simulate the electron wavelength
def wvlength_pm(kvolt: int) -> float:
    """Electron wavelength. 
    Args:
        kvolt (int): accelerating voltage in kV
    Returns:
        float: wavelength in pm
    """
    return 1.23e3 / cmath.sqrt(kvolt*1e3 * (1 + 9.78e-7*kvolt*1e3))


# Function to simulate the electron interaction coefficient
def sigma(kvolt: float) -> float:
    """Interaction coefficient of an electron.  $$\sigma _e = 2 \pi m e \lambda / h^2$$
    Args:
        kvolt (float): accelerating voltage in kV.
    Returns:
        float: interaction coefficient of an electron. 
    """
    return 2.08847e6 * wvlength_pm(kvolt)


# # Function to simualte the wave function 
# def simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt):

#     # Initialize variables
#     slce_zpixels = trgt_slice_nm // pxel_size_nm
#     total_slice_number = int(box_size // slce_zpixels + 1)

#     # Read PDB file
#     protein = pdb.PDB(pdb_file)
#     protein.read_pdb()

#     # Check box size
#     min_x = min(atom.x for atom in protein.atoms)
#     max_x = max(atom.x for atom in protein.atoms)
#     min_y = min(atom.y for atom in protein.atoms)
#     max_y = max(atom.y for atom in protein.atoms)
#     min_z = min(atom.z for atom in protein.atoms)
#     max_z = max(atom.z for atom in protein.atoms)

#     box_size_nm = box_size * pxel_size_nm
#     box_size_angstrom = box_size_nm * 10

#     print(f"Box size: {box_size_nm:.2f} nm ({box_size_angstrom:.2f} Å), {box_size} pixels")
#     print(f"PDB structure dimensions: X: {min_x:.2f} Å to {max_x:.2f} Å, Y: {min_y:.2f} Å to {max_y:.2f} Å, Z: {min_z:.2f} Å to {max_z:.2f} Å")

#     if (max_x - min_x) > box_size_angstrom or (max_y - min_y) > box_size_angstrom or (max_z - min_z) > box_size_angstrom:
#         print("Warning: The PDB structure is larger than the box. Consider increasing the box size.")
#     else:
#         print("The PDB structure fits within the box.")

#     # Calculate molecular potential
#     mol_potential = mp.integrate_atomic_potential(protein, box_size, pxel_size_nm)
#     mol_potential *= 1e20

#     # Initialize propagators
#     deltaZ = slce_zpixels * pxel_size_nm
#     box_center = box_size / 2
#     propagator_r = np.zeros((box_size, box_size), dtype=complex)
#     propagator_i = np.zeros((box_size, box_size), dtype=complex)

#     for m in range(box_size):
#         for n in range(box_size):
#             rsq_ = (m - box_center)**2 + (n - box_center)**2
#             rsq_ *= pxel_size_nm**2
#             prop_phs = 3.14159 / wvlength_pm(kvolt) / deltaZ * rsq_ * 1e3
#             c1 = 1 / wvlength_pm(kvolt) / deltaZ * 1e21

#             propagator_r[m, n] += c1 * cmath.sin(prop_phs)
#             propagator_i[m, n] += c1 * cmath.cos(prop_phs)

#     # Initialize the incident wave function
#     psi_r = np.ones((box_size, box_size))
#     psi_i = np.zeros((box_size, box_size))

#     # Process each slice
#     for slc in range(total_slice_number):
#         slce_start = int(slc * slce_zpixels)
#         slce_end = int(slce_start + slce_zpixels) if slc != (total_slice_number - 1) else int(box_size)

#         v_n = np.array(mol_potential[:, :, slce_start:slce_end])
#         v_nz = np.sum(v_n, axis=2)

#         tr_n = np.cos(sigma(kvolt) * v_nz)
#         ti_n = np.sin(sigma(kvolt) * v_nz)

#         # Propagate wave function
#         t_psi_r = tr_n * psi_r - ti_n * psi_i
#         t_psi_i = ti_n * psi_r + tr_n * psi_i

#         prpg_n = propagator_r + 1j * propagator_i
#         trns_n = t_psi_r + 1j * t_psi_i

#         prpg_n /= np.max(np.abs(prpg_n)) + 1e-10
#         trns_n /= np.max(np.abs(trns_n)) + 1e-10

#         psi_n1 = np.fft.ifft2(np.fft.fft2(prpg_n) * np.fft.fft2(trns_n))
#         psi_r, psi_i = psi_n1.real, psi_n1.imag


#     return psi_r, psi_i

# Function to simulate wave function 
def simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt):

    # Precompute constants
    slce_zpixels = int(trgt_slice_nm // pxel_size_nm)
    total_slice_number = int(box_size // slce_zpixels + 1)
    deltaZ = slce_zpixels * pxel_size_nm
    wavelength_inv = 1 / wvlength_pm(kvolt)
    sigma_val = sigma(kvolt)
    c1 = wavelength_inv / deltaZ * 1e21

    # Read PDB file
    protein = pdb.PDB(pdb_file)
    protein.read_pdb()

    # Check box size
    min_x = min(atom.x for atom in protein.atoms)
    max_x = max(atom.x for atom in protein.atoms)
    min_y = min(atom.y for atom in protein.atoms)
    max_y = max(atom.y for atom in protein.atoms)
    min_z = min(atom.z for atom in protein.atoms)
    max_z = max(atom.z for atom in protein.atoms)

    box_size_nm = box_size * pxel_size_nm
    box_size_angstrom = box_size_nm * 10

    print(f"Box size: {box_size_nm:.2f} nm ({box_size_angstrom:.2f} Å), {box_size} pixels")
    print(f"PDB structure dimensions: X: {min_x:.2f} Å to {max_x:.2f} Å, Y: {min_y:.2f} Å to {max_y:.2f} Å, Z: {min_z:.2f} Å to {max_z:.2f} Å")

    if (max_x - min_x) > box_size_angstrom or (max_y - min_y) > box_size_angstrom or (max_z - min_z) > box_size_angstrom:
        print("Warning: The PDB structure is larger than the box. Consider increasing the box size.")
    else:
        print("The PDB structure fits within the box.")

    # Calculate molecular potential
    mol_potential = mp.integrate_atomic_potential(protein, box_size, pxel_size_nm)
    mol_potential *= 1e20

    # Vectorized propagator initialization
    box_center = box_size / 2
    coords = np.arange(box_size) - box_center
    x, y = np.meshgrid(coords, coords)
    rsq_ = (x**2 + y**2) * pxel_size_nm**2
    prop_phs = 3.14159 * wavelength_inv / deltaZ * rsq_ * 1e3
    propagator_r = c1 * np.sin(prop_phs)
    propagator_i = c1 * np.cos(prop_phs)

    # Initialize the incident wave function
    psi_r = np.ones((box_size, box_size))
    psi_i = np.zeros((box_size, box_size))

    # Process each slice
    for slc in range(total_slice_number):
        slce_start = int(slc * slce_zpixels)
        slce_end = int(slce_start + slce_zpixels) if slc != (total_slice_number - 1) else box_size

        v_n = mol_potential[:, :, slce_start:slce_end]
        v_nz = np.sum(v_n, axis=2)

        # Transmission function
        tr_n = np.cos(sigma_val * v_nz)
        ti_n = np.sin(sigma_val * v_nz)

        # Propagate wave function
        t_psi_r = tr_n * psi_r - ti_n * psi_i
        t_psi_i = ti_n * psi_r + tr_n * psi_i

        prpg_n = propagator_r + 1j * propagator_i
        trns_n = t_psi_r + 1j * t_psi_i

        # Normalize values to avoid overflow
        prpg_n /= np.max(np.abs(prpg_n)) + 1e-10
        trns_n /= np.max(np.abs(trns_n)) + 1e-10

        # FFT for propagation
        psi_n1 = np.fft.ifft2(np.fft.fft2(prpg_n) * np.fft.fft2(trns_n))
        psi_r, psi_i = psi_n1.real, psi_n1.imag

    return psi_r, psi_i


# Test
if __name__ == '__main__':
    
    # Start time
    start_time = time.time()
    
    # Parameters for wave function simulation
    pdb_file = '1dat.pdb'
    box_size = 300
    pxel_size_nm = 0.1
    trgt_slice_nm = 0.3
    kvolt = 300.0

    # Calling multislice 
    print('Calling multislice function...')
    psi_r, psi_i = simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt)

    # Parameters for TCIF 
    imge_size = box_size # comes from psi_r shape
    Cs_mm = 2.0
    df1_nm = 1500
    df2_nm = 1500
    alpha_rad = np.deg2rad(5.0)
    beta_rad = 0 # tilt axis is the x-axis
    
    psi_complex = psi_r + 1j * psi_i

    # Calling phase distortion function
    print('Calling w0 function...')
    w0 = tcif.W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)

    # Calling TCIF
    print('Calling TCIF function...')
    amp, phs = tcif.tcif(psi_complex, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)


    # Normalize the amplitude array
    amplitude_normalized = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    
    # Intensity = amplitude^2
    intensity = amplitude_normalized ** 2

    # Power spectrum for normalized amplitudes
    dB = (-10 * np.log(amplitude_normalized)) + 10e-6

    # Perform radial averaging on the intensity data
    spatial_frequencies, radial_average = tcif.radial_averages(intensity, pxel_size_nm)

    # Plotting normalized intensity
    print('Plotting normalized intensities...')
    plt.imshow(intensity, cmap='gray') 
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Intensity of Normalized Amplitudes')
    plt.savefig('intensity.png')
    plt.clf() 

    # Plotting dB
    print('Plotting dB...')
    plt.imshow(dB, cmap='gray') 
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Power Spectrum of Normalized Amplitudes')
    plt.savefig('power_spectrum.png')
    plt.clf()

    # Plotting radial averaged intensity
    print('Plotting radial averaged intensity...')
    plt.plot(spatial_frequencies, radial_average, color = 'black')
    plt.xlabel('Spatial Frequency (nm$^{-1}$)')
    plt.ylabel('Average Intensity')
    plt.title('Radial Average Intensity vs. Spatial Frequency')
    plt.savefig('radial_avg_intensity.png', dpi = 800)
    plt.clf()

    # End time
    end_time = time.time()

    # Time taken
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

#===================Imports=================
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import atomic_potential as ap
import pdb_handler as pdb
import molecular_potential_pdb as mp
from tcif import *
from plotting import * 
from electron_optics import *
from typing import Any, Tuple, List
from nptyping import NDArray
import os
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch


# Function to simulate wave function 
def simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt):

    # Precompute constants
    slce_zpixels = int(trgt_slice_nm // pxel_size_nm)
    # Ensures total number of slices covers the full z-range
    total_slice_number = int(np.ceil(box_size / slce_zpixels)) 
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

    # Box size dimensions
    box_size_nm = box_size * pxel_size_nm
    box_size_angstrom = box_size_nm * 10
    pxel_size_angstrom = pxel_size_nm * 10

    # Printing box size and pdb file dimensions
    print(f"Box size: {box_size_nm:.2f} nm ({box_size_angstrom:.2f} Å), {box_size} pixels")
    print(f"PDB structure dimensions: X: {min_x:.2f} Å to {max_x:.2f} Å, Y: {min_y:.2f} Å to {max_y:.2f} Å, Z: {min_z:.2f} Å to {max_z:.2f} Å")
    print('Pixel size:', pxel_size_nm, 'nm', '(', pxel_size_angstrom, 'Å)')

    # Box size warning
    if (max_x - min_x) > box_size_angstrom or (max_y - min_y) > box_size_angstrom or (max_z - min_z) > box_size_angstrom:
        print("Warning: The PDB structure is larger than the box. Consider increasing the box size.")
    else:
        print("The PDB structure fits within the box.")

    # Calculate molecular potential
    mol_potential = mp.integrate_atomic_potential(protein, box_size, pxel_size_nm)
    print("Molecular potential stats:")
    print(f"  min: {np.min(mol_potential):.3e}")
    print(f"  max: {np.max(mol_potential):.3e}")
    print(f"  mean: {np.mean(mol_potential):.3e}")

    # Scale potentials to prevent instability
    mol_potential *= 1e20
    print("Molecular potential stats scaled:")
    print(f"  min: {np.min(mol_potential):.3e}")
    print(f"  max: {np.max(mol_potential):.3e}")
    print(f"  mean: {np.mean(mol_potential):.3e}")

    # Histogram of the scaled molecular potential
    plt.figure(figsize=(6, 4))
    plt.hist(mol_potential.flatten(), bins=200)
    plt.title("Histogram of Scaled Molecular Potential")
    plt.xlabel("Potential (V)")
    plt.ylabel("Voxel count")
    plt.yscale("log")  # Use log scale to see the tails
    plt.tight_layout()
    plt.savefig('mol_potential_histogram.png', dpi=800)
    plt.clf()

  
    # Flatten potential and find top 10 potential values 
    flat_potential = mol_potential.flatten()
    top_indices = np.argpartition(flat_potential, -10)[-10:]
    top_values = flat_potential[top_indices]
    top_coords = np.array(np.unravel_index(top_indices, mol_potential.shape)).T  


    # === Print atoms near high potential voxels ===
    # Convert voxel indices back to real-world coordinates (in Å)
    voxel_size_ang = pxel_size_nm * 10  # nm → Å
    search_radius = 2.0 # Å

    print("\n=== Atoms Near High Potential Voxels (within 2.0 Å) ===")
    for idx, (vx, vy, vz) in enumerate(top_coords):
        # Convert voxel center to Å using box origin shift
        real_x = (vx - box_size // 2) * voxel_size_ang
        real_y = (vy - box_size // 2) * voxel_size_ang
        real_z = (vz - box_size // 2) * voxel_size_ang

        print(f"\nVoxel {idx}: ({real_x:.2f}, {real_y:.2f}, {real_z:.2f}) Å — Potential: {top_values[idx]:.2e}")

        found = False
        for atom in protein.atoms:
            dx = atom.x - real_x
            dy = atom.y - real_y
            dz = atom.z - real_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance <= search_radius:
                print(f"  → {atom.element:2s} at ({atom.x:.2f}, {atom.y:.2f}, {atom.z:.2f}) Å — Distance: {distance:.2f} Å")
                found = True

        if not found:
            print("  No atoms found within search radius.")


    # Extract atomic coordinates (Å) and convert to voxel indices 
    voxel_size_ang = pxel_size_nm * 10
    atom_x = np.array([atom.x for atom in protein.atoms])
    atom_y = np.array([atom.y for atom in protein.atoms])
    atom_z = np.array([atom.z for atom in protein.atoms])

    atom_vox_x = (atom_x / voxel_size_ang + box_size // 2).astype(int)
    atom_vox_y = (atom_y / voxel_size_ang + box_size // 2).astype(int)
    atom_vox_z = (atom_z / voxel_size_ang + box_size // 2).astype(int)


    # 3D plotting 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms 
    ax.scatter(atom_vox_x, atom_vox_y, atom_vox_z, c='gray', s=5, alpha=0.4)

    # Plot top 10 potential voxels
    x, y, z = top_coords[:, 0], top_coords[:, 1], top_coords[:, 2]
    sc = ax.scatter(x, y, z, c=top_values, cmap='inferno', s=80, edgecolors='black', label='Top Potentials')

    # Plot details
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.set_title('Molecular Potential Outliers and Atomic Structure')
    plt.colorbar(sc, label='Potential Value')
    # plt.show()
    plt.savefig('outliers.png', dpi=800)
    plt.clf()



    # Create output directory for slice plots
    grouped_output_dir = "grouped_mol_potential_slices"
    os.makedirs(grouped_output_dir, exist_ok=True)

    # Vectorized Fresnel propagator initialization
    box_center = box_size / 2
    coords = np.arange(box_size) - box_center
    x, y = np.meshgrid(coords, coords)
    rsq_ = (x**2 + y**2) * pxel_size_nm**2
    prop_phs = np.pi * wavelength_inv / deltaZ * rsq_ * 1e3
    propagator_r = c1 * np.sin(prop_phs)
    propagator_i = c1 * np.cos(prop_phs)

    # Initialize the incident wave function
    psi_r = np.ones((box_size, box_size))
    psi_i = np.zeros((box_size, box_size))

    # Process each slice
    for slc in range(total_slice_number):
        
        # Calculate slice start
        slce_start = slc * slce_zpixels
        # Cap slice size by box size to prevent overshooting
        slce_end = min(slce_start + slce_zpixels, box_size)

        # Prevents overshooting when slce_start is too high
        if slce_start >= box_size:
            break  # Don't try to slice beyond the array bounds

        # Calculate potential of slice (sub-slices)
        v_n = mol_potential[:, :, slce_start:slce_end]

        # Avoids using an empty array
        if v_n.size == 0:
            print(f"Warning: Empty slice at index {slc}, start={slce_start}, end={slce_end}")
            continue

        # Collapses the 3D z-stack into a 2D map for simulation/plotting
        v_nz = np.sum(v_n, axis=2)

        # Plotting each slice
        plt.imshow(v_nz, cmap='magma')
        plt.title(f'Grouped Molecular Potential Slice {slc} ({slce_start}:{slce_end})')
        plt.colorbar(label='Summed Potential (V)')
        plt.tight_layout()
        filename = os.path.join(grouped_output_dir, f'grouped_slice_{slc:02d}.png')
        plt.savefig(filename, dpi=300)
        plt.clf()


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

        # FFT for Fresnel propagation
        psi_n1 = np.fft.ifft2(np.fft.fft2(prpg_n) * np.fft.fft2(trns_n))
        psi_r, psi_i = psi_n1.real, psi_n1.imag

    return psi_r, psi_i


# Test
if __name__ == '__main__':
    
    # Start time
    start_time = time.time()
    
    # Parameters for wave function simulation
    pdb_file = '1dat.pdb'
    box_size = 30
    pxel_size_nm = 0.2 # 0.28
    trgt_slice_nm = 0.3
    kvolt = 200.0

    # Calling multislice 
    print('Calling multislice function...')
    psi_r, psi_i = simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt)

    # Parameters for TCIF 
    imge_size = box_size 
    Cs_mm = 2.0
    df1_nm = 2000
    df2_nm = 2000
    alpha_rad = np.deg2rad(0.0)
    beta_rad = 0 # tilt axis is the x-axis
    
    # Combined wave
    psi_complex = psi_r + 1j * psi_i

    # Plotting exit wave amplitude (real space)
    plt.imshow(np.abs(psi_complex), cmap='gray')
    plt.title("Exit Wave Amplitude")
    plt.colorbar()
    plt.savefig('ex_wv_amp.png', dpi = 800)
    plt.clf()

    # Plotting exit wave phase (real space)
    plt.imshow(np.angle(psi_complex), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.title("Exit Wave Phase")
    plt.colorbar(label="Radians")
    plt.savefig('ex_wv_phase.png', dpi=800)
    plt.clf()

    # Plotting exit wave phase (Fourier space)
    plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(psi_complex))), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.title("Exit Wave Phase")
    plt.colorbar(label="Radians")
    plt.savefig('ex_wv_phase_fspace.png', dpi=800)
    plt.clf()

    # Calling phase distortion function
    w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)

    # Calling TCIF
    print('Calling TCIF function...')
    amp, phs, freqs, nyquist_frequency = tcif(psi_complex, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)

    # Plotting TCIF afflicted phase (Fourier space)
    plt.imshow(phs, cmap='twilight') 
    plt.colorbar(label = 'Radians')
    plt.title('Apoferritin Multislice TCIF Afflicted Phase')
    plt.savefig('msphs.png', dpi = 800)
    plt.clf() 

    # Calculating/plotting absolute phase difference (like in Mariani et al)
    phase_diff = np.abs(phs - np.angle(np.fft.fftshift(np.fft.fft2(psi_complex))))
    plt.imshow(phase_diff,cmap='viridis', vmin=-np.pi, vmax=np.pi)
    plt.title("Absolute Phase Difference")
    plt.colorbar(label="Radians")
    plt.savefig('phs_diff_ms.png', dpi=800)
    plt.clf()

    # End time
    end_time = time.time()

    # Time taken
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

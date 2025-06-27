import numpy as np
import time
import matplotlib.pyplot as plt
import os
from tcif import *
from plotting import save_imshow, save_lineplot, radial_average
from electron_optics import wvlength_pm, sigma
from multislice import simulate_wave_function, mp  

# Simulation parameters
pdb_file = '1dat.pdb'
box_size = 80
pxel_size_nm = 0.2
trgt_slice_nm = 0.3
kvolt = 200.0
Cs_mm = 2.0
df1_nm = 2000
df2_nm = 2000
beta_rad = 0  # x-axis tilt

output_root = 'simulation_outputs'
os.makedirs(output_root, exist_ok=True)

tilt_angles_deg = [0, 10, 20, 30, 60]  

# Loop through tilt angles
for angle_deg in tilt_angles_deg:
    start_time = time.time()
    print(f"\n=== Simulating tilt angle: {angle_deg}° ===")
    alpha_rad = np.deg2rad(angle_deg)

    # Make custom directory to store files
    tilt_dir = os.path.join(output_root, f"tilt_{angle_deg:02d}deg")
    os.makedirs(tilt_dir, exist_ok=True)

    # Run multislice simulation
    psi_r, psi_i, summed_exit_wave, mol_potential = simulate_wave_function(pdb_file, box_size, pxel_size_nm, trgt_slice_nm, kvolt)

    # Plot molecular potential histogram
    plt.figure(figsize=(6, 4))
    plt.hist(mol_potential.flatten(), bins=200)
    plt.title(f"Molecular Potential Histogram — Tilt {angle_deg}°")
    plt.xlabel("Potential (V)")
    plt.ylabel("Voxel count")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(tilt_dir, f'ms_mol_potential_histogram.png'), dpi=300)
    plt.close()

    # Save exit wave amplitude (real space)
    save_imshow(
        data=np.abs(summed_exit_wave),
        title=f'Amplitude (Real Space) — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'ms_amplitude.png'),
        cmap='gray'
    )

    # Save exit wave phase (real space)
    save_imshow(
        data=np.angle(summed_exit_wave),
        title=f'Phase (Real Space) — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'ms_phase_real.png'),
        cmap='viridis',
        colorbar_label='Radians',
        vmin=-np.pi,
        vmax=np.pi
    )

    # Fourier phase
    ex_wv_phs_f = np.angle(np.fft.fftshift(np.fft.fft2(summed_exit_wave)))
    save_imshow(
        data=ex_wv_phs_f,
        title=f'Phase (Fourier Space) — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'ms_phase_fourier.png'),
        cmap='viridis',
        colorbar_label='Radians',
        vmin=-np.pi,
        vmax=np.pi
    )

    # Run TCIF
    print('Running TCIF...')
    w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, box_size, pxel_size_nm)
    amp, phs, freqs, nyquist_frequency = tcif(
        summed_exit_wave, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), box_size, pxel_size_nm
    )

    # Save TCIF amplitude
    save_imshow(
        data=amp,
        title=f'TCIF Amplitude — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'tcif_amplitude.png'),
        cmap='viridis',
        colorbar_label='Amplitude'
    )

    # Save TCIF phase
    save_imshow(
        data=phs,
        title=f'TCIF Phase — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'tcif_phase.png'),
        cmap='viridis',
        colorbar_label='Radians',
        vmin=-np.pi,
        vmax=np.pi
    )

    # Save phase difference
    phase_diff = np.abs(phs - ex_wv_phs_f)
    save_imshow(
        data=phase_diff,
        title=f'Phase Difference — Tilt {angle_deg}°',
        filename=os.path.join(tilt_dir, f'phase_diff.png'),
        cmap='viridis',
        colorbar_label='Radians',
        vmin=-np.pi,
        vmax=np.pi
    )

    # Radial average of amplitude
    rad_amp = radial_average((amp - np.min(amp)) / (np.max(amp) - np.min(amp)))
    spatial_frequencies_amp = np.linspace(0, nyquist_frequency, len(rad_amp))
    save_lineplot(spatial_frequencies_amp, rad_amp,
                filename=os.path.join(tilt_dir, f'tcif_radial_avg_amplitude.png'),
                xlabel='Spatial Frequency (m$^{-1}$)',
                ylabel='Radially Averaged Amplitude',
                title=f'Radial Avg Amplitude — Tilt {angle_deg}°')

    # Radial average of phase difference
    rad_phs_diff = radial_average(phase_diff)
    spatial_frequencies_phs = np.linspace(0, nyquist_frequency, len(rad_phs_diff))
    save_lineplot(spatial_frequencies_phs, rad_phs_diff,
                filename=os.path.join(tilt_dir, f'tcif_radial_avg_phase_diff.png'),
                xlabel='Spatial Frequency (m$^{-1}$)',
                ylabel='Radially Averaged Phase Difference (rad)',
                title=f'Radial Avg Phase Difference — Tilt {angle_deg}°')

    # Closes all figures to save memory
    plt.close('all')
    
    elapsed = time.time() - start_time
    print(f"Done — Tilt {angle_deg}°, Time: {elapsed:.1f} sec")

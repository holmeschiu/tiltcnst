#===================Imports=================
import math
import numpy as np
import scipy as sp
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import time
from proj_3d_to_2d import *
from amorphous_carbon_film import * 
from weak_phase import wk_phs_obj
from plotting import *
from electron_optics import *


#===================Functions=================
# Phase distortion function 
def W_0(Cs_mm: float, wvlength_pm: float, df1_nm: float, df2_nm: float, beta_0_rad: float, imge_size: int, pxel_size_nm: float) -> np.ndarray:
    """Phase distortion function in 2D.  The origin is shifted to the center. 
       beta_0 is defined in Equation 5 in Cter paper (Penczek et al., 2014).

    Args:
        Cs_mm (float):          spherical aberration in mm
        wvlength_pm (float):    electron wavelength in pm
        df1_nm (float):         defocus 1 in nm (long axis)
        df2_nm (float):         defocus 2 in nm (short axis)
        beta_0_rad (float):     angle between defocus 1 and x-axis in radian
        imge_size:              image size in pixels
        pxel_size_nm:           pixel size in nm
    Returns:
        np.ndarray:             distorted phases in radian
    """
    # Creating array to hold values
    ctf = np.zeros((imge_size, imge_size), dtype=float)

    # Frequency step size
    freq_step_inm = (1./pxel_size_nm) / imge_size
    
    # Center shift
    cntr_shift_inm = 1. / (2 * pxel_size_nm)
    
    # Defocus values
    za = (df1_nm + df2_nm)*1e-9 / 2
    zb = (df1_nm - df2_nm)*1e-9 / 2

    kj, ki = np.meshgrid(np.arange(imge_size), np.arange(imge_size), indexing='ij')
    ssqu = ((ki*freq_step_inm - cntr_shift_inm)*1e9)**2 + ((kj*freq_step_inm - cntr_shift_inm)*1e9)**2

    # Calculating alpha_g
    alpha_g = np.arctan2(kj * freq_step_inm - cntr_shift_inm, ki * freq_step_inm - cntr_shift_inm)

    # Defocus term
    W1 = za + zb * np.cos(2*(alpha_g - beta_0_rad))
    W21 = -0.5 * W1 * wvlength_pm*1e-12 * ssqu
    # Cs term
    W22 = 0.25 * (Cs_mm*1e-3) * (wvlength_pm*1e-12)**3 * ssqu**2

    # Adding phase distortion to CTF array
    ctf += (math.pi * 2) * (W21 + W22)
    

    return ctf


# Calculate spatial frequencies function 
def calc_spat_freq(imge_size: int, pxel_size_nm: float) -> np.ndarray:
    """Calculate frequency bins and nyquist frequency
    Args:
        imge_size (int):        image size in pixels
        pxel_size_nm (float):   pixel size in nm
    Returns:
        np.ndarray:             spatial frerquencies in m^-1
    """
    
    # Calculated frequencies in m^-1 and centers the zero frequency
    freqs = np.fft.fftshift(np.fft.fftfreq(imge_size, d=pxel_size_nm * 1e-9))
    
    # Calculate nyqust frequency in m^-1
    nyquist_freq = (1 / (2 * pxel_size_nm * 1e-9))

    return freqs, nyquist_freq


# TCIF function 
def tcif(spec: np.ndarray, W_0: np.ndarray, beta_rad: float, alpha_rad: float, wvlength_pm: float, imge_size: int, pxel_size_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate TCIF in Fourier space. The origin is in the image center.
    Args:
        spec (np.ndarray): specimen exit wave, expected to be a complex 2D array
        W_0 (np.ndarray): phase distortion function, expected to be a 2D array
        beta_rad (float): angle between defocus 1 and x-axis in radian
        alpha_rad (float): angle of tilt in radian
        wvlength_pm (float): electron wavelength in pm
        imge_size (int): image size in pixels
        pxel_size_nm (float): pixel size in nm/pixel
    Returns:
        Tuple[np.ndarray, np.ndarray]: amplitude and phase arrays
    """

    # Check that phase distortion array and specimen size matches image size
    if (W_0.size != imge_size**2) or (spec.size != imge_size**2):
        print("The size of the input array or exit wave does not match the image size.")
        sys.exit()

    # Calculate the spatial frequency grid (in m^-1)
    freqs, nyquist_frequency = calc_spat_freq(imge_size, pxel_size_nm)

    # Shifted FFT of the specimen exit wave
    f_spec = np.fft.fftshift(np.fft.fft2(spec)) 
    
    # Create meshgrid for frequency coordinates
    ki, kj = np.meshgrid(freqs, freqs, indexing='ij')

    # Calculate the squared spatial frequency grid
    ssqu = ki**2 + kj**2

    # Calculate pnx_inm, pny_inm, ppx_inm, ppy_inm for all pixels at once
    factor = 0.5 * (wvlength_pm*1e-12) * math.tan(alpha_rad)

    pnx_inm = ki - factor * math.cos(beta_rad) * ssqu
    pny_inm = kj - factor * math.sin(beta_rad) * ssqu
    ppx_inm = ki + factor * math.cos(beta_rad) * ssqu
    ppy_inm = kj + factor * math.sin(beta_rad) * ssqu


    # Interpolate for the real and imaginary parts of f_spec
    interpr = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.real, method='cubic', bounds_error=False, fill_value=0.0)
    interpi = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.imag, method='cubic', bounds_error=False, fill_value=0.0)

    # Vectorized interpolation for pnx_inm, pny_inm, ppx_inm, ppy_inm - have to stack coordinates for interpolator
    sp_rpn = interpr(np.stack((pnx_inm, pny_inm), axis=-1))
    sp_ipn = interpi(np.stack((pnx_inm, pny_inm), axis=-1))
    sp_rpp = interpr(np.stack((ppx_inm, ppy_inm), axis=-1))
    sp_ipp = interpi(np.stack((ppx_inm, ppy_inm), axis=-1))
    
    # Apply phase modulation
    phmod_cos = np.cos(W_0)
    phmod_sin = np.sin(W_0)

    # Compute Q1r, Q1i, Q2r, Q2i for the entire array at once
    Q1r = phmod_cos * sp_rpn + phmod_sin * sp_ipn
    Q1i = -phmod_sin * sp_rpn + phmod_cos * sp_ipn
    Q2r = phmod_cos * sp_rpp - phmod_sin * sp_ipp
    Q2i = phmod_sin * sp_rpp + phmod_cos * sp_ipp

    # Compute Q3r, Q3i for the entire array at once
    Q3r = -(Q1i - Q2i)
    Q3i = (Q1r - Q2r)

    # Compute amplitude and phase in a vectorized manner
    amp = np.sqrt(Q3r**2 + Q3i**2)
    phs = np.arctan2(Q3i, Q3r)

    return amp, phs, freqs, nyquist_frequency

def add_poisson_noise(image, mean_electrons_per_px=10):
    """
    Apply Poisson noise to simulate electron counting statistics.

    Parameters:
        image (np.ndarray): Real-space image (non-negative intensities).
        mean_electrons_per_px (float): Approx. average counts per pixel.

    Returns:
        np.ndarray: Noisy image.
    """
    # Normalize to [0, 1]
    image = image - np.min(image)
    image = image / np.max(image)

    # Scale to electron counts
    electron_counts = image * mean_electrons_per_px

    # Poisson sample
    noisy_counts = np.random.poisson(electron_counts)

    # Rescale back to [0, 1]
    noisy_image = noisy_counts / mean_electrons_per_px
    return noisy_image


#===================Main usage=================
if __name__ == '__main__':
    
    # Start time
    start_time = time.time()

    # Parameters
    imge_size = 256
    pxel_size_nm = 0.28
    kvolt = 200
    Cs_mm = 2.0
    df1_nm = 2000.0
    df2_nm = 2000.0
    beta_rad = 0.0
    alpha_rad = np.deg2rad(0.0)

    # Phase distortion called
    w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)

    ########################## amorphous carbon film  ##########################################################################
    # Amorphous carbon film simulation parameters:
    # mean_density = 0.6   # depends on your pixel size, etc.
    # mean_density = (pxel_size_nm ** 2)
    # delta_x_A = 0.2            # 0.2 A/pixel 
    # total_thickness_nm = 10.0  # 10 nm film
    # slice_thickness_nm = 1.0   # each slice is 1 nm
    # n_slices = total_thickness_nm / slice_thickness_nm

    # spec = simulate_amorphous_carbon_film(
    #     imge_size, imge_size,
    #     delta_x_A,
    #     total_thickness_nm,
    #     slice_thickness_nm,
    #     mean_density,
    #     carbon_scattering_factor)


    ########################## weak-phase object  ##########################################################################
    # print('Generating weak phase object...')
    # spec, *_ = wk_phs_obj() 

    # # Plotting phase of raw object prior to TCIF applied 
    # print('Plotting raw phase...')
    # raw_phase = np.angle(np.fft.fftshift(np.fft.fft2(spec)))
    # save_imshow(raw_phase, title='Raw Phase', filename='phase_raw.png', colorbar_label='radians', cmap='viridis')


    ########################## ribosome/apoF projection #####################################################################
    # File for the projection
    # mrc_filename = '8tu7_4ang_apix2.mrc' # apoF
    mrc_filename = '/Users/kiradevore/Documents/python_scripts/TCIF/250402_opt_of_250111/ribo_3D_maps/ribo_apix2_res4.mrc' # ribosome
    
    # Angles for projection
    angles = (0, 0, 0)  # Euler angles for projection
    print('Angles: ', angles)
    
    # Generating 2D projection
    print('Generating 2D projection...')
    spec = gen_2d_proj(mrc_filename, angles, axis = 0)

    # Plotting raw phase
    raw_phase = np.angle(np.fft.fftshift(np.fft.fft2(spec)))
    save_imshow(raw_phase, title='Raw Phase', filename='phase_raw.png', colorbar_label='radians', cmap='viridis')

    # Plotting phase distribution 
    plt.hist(raw_phase.flatten(), bins=100, color='#2E6F8E', alpha=0.7)
    plt.xlabel('Phase Values (radians)')
    plt.ylabel('Frequency')
    plt.title('Raw Phase Histogram')
    plt.savefig('phase_hist_raw.png', dpi = 800)
    plt.clf()

    #########################################################################################################################  
    # Plotting 2D projection
    save_imshow(spec, title='2D Projection', filename='projection.png', cmap='gray', colorbar_label='')
    # save_imshow(np.abs(spec) ** 2, title='Weak Phase Object', filename='projection.png', cmap='viridis', colorbar_label='')


    ########################################################################################################################
    # Calling TCIF
    print('Calling TCIF...')
    amp, phs, freqs, nyquist_frequency = tcif(spec, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)

    # Plotting the CTF
    ctf = -2 * np.sin(w0)
    save_imshow(ctf, cmap='gray', title='CTF', filename='CTF.png',
                extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency),
                xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Spatial Frequency (m$^{-1}$)', colorbar_label='radians', invert_yaxis=False)


    ########################################################################################################################
    # Image reconstruction
    print('Reconstructing image...')
    tilt_im = np.abs(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phs)))) 
    
    # Normalizes real - space image
    tilt_im_n = (tilt_im - tilt_im.mean()) / (tilt_im.std())
    
    # Plots real space image
    save_imshow(tilt_im_n, title='Normalized Real Space Image After TCIF', 
                filename='tilted_img_FFT.png', cmap='gray')
   

    # Add Poisson noise (simulate detector shot noise)
    print('Adding Poisson noise...')
    tilt_im_noisy = add_poisson_noise(tilt_im, mean_electrons_per_px=10)
    
    # Normalize noisy image
    tilt_im_noisy_n = (tilt_im_noisy - tilt_im_noisy.mean()) / tilt_im_noisy.std()
    
    # Plots real space noisy image
    save_imshow(tilt_im_noisy_n, title='Normalized Real Space Image After TCIF w/ Poisson Noise', 
                filename='tilted_img_nFFT.png', cmap='gray')



    # # Plots FFT of reconstructed image
    # fft_of_reconstructed_img = np.fft.fftshift(np.abs(np.fft.fft2(tilt_im))**2)
    # fft_of_reconstructed_img = (fft_of_reconstructed_img - fft_of_reconstructed_img.mean()) / (fft_of_reconstructed_img.std())
    # save_imshow(fft_of_reconstructed_img, title='FFT of Reconstructed Image', filename='fft_of_reconstructed_img.png', cmap='gray')


    ########################################################################################################################
    # Normalize the amplitude array
    amplitude_normalized = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    # Plotting amplitude after TCIF is applied
    print('Plotting amplitude...')
    save_imshow(amp, title='Amplitude', filename='amplitude.png', cmap='viridis',
                extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency),
                xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Spatial Frequency (m$^{-1}$)', invert_yaxis=False)

    # Intensity = amplitude^2
    intensity = amp ** 2
    intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    # Plotting normalized intensity
    print('Plotting normalized intensities...')
    save_imshow(intensity_normalized, title='Normalized Intensity', filename='intensity.png', cmap='viridis',
                extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency),
                xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Spatial Frequency (m$^{-1}$)', invert_yaxis=False)


    # Plotting power spectrum for amplitudes
    dB = (-10 * np.log(amp)) + 10e-6
    # dB = np.fft.fftshift(np.fft.fft2(spec))
    # dB = np.abs(dB) ** 2
    # Plotting dB
    print('Plotting dB...')
    save_imshow(dB, title='Power Spectrum', filename='power_spectrum.png', 
                cmap='viridis', colorbar_label='dB', invert_yaxis=False,
                extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency),
                        xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Spatial Frequency (m$^{-1}$)')

    # Plotting phase after TCIF is applied
    print('Plotting phase...')
    save_imshow(phs, title='TCIF Afflicted Phase', filename='phase_TCIF.png', colorbar_label='radians', cmap='viridis')


    # Phase unwrapping
    from skimage.restoration import unwrap_phase

    phs_unwrapped = unwrap_phase(phs)
    save_imshow(phs_unwrapped, title='Unwrapped Phase Map', filename='NEW_phase_unwrapped.png', colorbar_label='radians', cmap='twilight')



    # Plotting phase distribution 
    plt.hist(phs.flatten(), bins=100, color='#2E6F8E', alpha=0.7)
    plt.xlabel('Phase Values (radians)')
    plt.ylabel('Frequency')
    plt.title('TCIF Phase Histogram')
    plt.savefig('phase_hist_TCIF.png', dpi = 800)
    plt.clf()


    ########################################################################################################################
    # Plotting radial averaged amplitude 
    print('Plotting radial averaged amplitude...')
    # Perform radial averaging of amplitudes
    radial_avg_amp = radial_average(amplitude_normalized)
    # Map radial distances to spatial frequencies 
    spatial_frequencies = np.linspace(0, nyquist_frequency, len(radial_avg_amp))
    save_lineplot(spatial_frequencies, radial_avg_amp, filename='radial_avg_amplitude.png',
                  xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Radially Averaged Amplitude',
                  title='Radially Averaged Amplitude vs. Spatial Frequency (m$^{-1}$)')
    

    # Calculate absolute phase difference between TCIF object and raw object
    phase_diff = np.abs(phs - raw_phase)
    # Plotting absolute phase difference
    save_imshow(phase_diff, title='Absolute Phase Difference in Fourier Space', xlabel='Spatial Frequency (m$^{-1}$)',
                ylabel='Spatial Frequency (m$^{-1}$)', filename='phase_diff_heatmap.png', colorbar_label='radians', 
                cmap='viridis', invert_yaxis=False,
                extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency))

    
    # Perform radial averaging on the phase difference array
    radial_avg_phase_diff = radial_average(phase_diff)
    # Compute average of radial average phase difference
    avg_val = np.nanmean(radial_avg_phase_diff)  
    # Plot the radially averaged phase difference
    print('Plotting radially averaged phase difference...')
    # save_lineplot(spatial_frequencies, radial_avg_phase_diff, title='Radially Averaged Phase Difference vs. Spatial Frequency',
    #               xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Radians', linecolor='black',
    #                hline=[avg_val, 'red'], legend='Mean = {:.2f}'.format(avg_val), filename='radial_avg_phase_difference.png')
    
    save_lineplot(spatial_frequencies, radial_avg_phase_diff, title='Radially Averaged Phase Difference vs. Spatial Frequency',
                xlabel='Spatial Frequency (m$^{-1}$)', ylabel='Radians', linecolor='black',
                filename='radial_avg_phase_difference.png')

    
    # End time
    end_time = time.time()

    # Time taken
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
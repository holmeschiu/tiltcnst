#!/usr/bin/env python3
# TCIF (Q(p)). 
# Notes:
#   1. The image is presumed to be squared. 
#   2. Presumed the input specimen exit wave has the same dimension and pixel 
#      size as the TCIF. 
#

import math
import numpy as np
import scipy as sp
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from projection_of_3d_to_2d import generate_2d_projection
from amorphous_carbon_film import * 
from weak_phase import *


def wvlength_pm(kvolt: int) -> float:
    """Electron wavelength. 
    Args:
        kvolt (int): accelerating voltage in kV
    Returns:
        float: wavelength in pm
    """
    return 1.23e3 / math.sqrt(kvolt*1e3 * (1 + 9.78e-7*kvolt*1e3))


def aperture_inm(cutf_frequency_inm: float, imge_size: int, pxel_size_nm: float) -> np.ndarray:
    """Objective aperture function.  A 2D mask function.  The origin is in the image center. 
    Args:
        cutf_frequency_inm (float): cutoff frequency in nm^-1
        imge_size (int): image size in pixels
        pxel_size_nm (float): pixel size in nm/pixel 
    Returns:
        np.ndarray: a mask function for the aperture
    """
    start = np.zeros((imge_size, imge_size), dtype=float)
    
    if imge_size % 2 == 1:
        print("The input image size should be even. ")
        sys.exit(0)
    else:
        cntr_shift_inm = imge_size / 2 / pxel_size_nm
        freq_step_inm = (1/pxel_size_nm) / imge_size

    for i in range(imge_size):
        for j in range(imge_size):
            if math.sqrt((i*freq_step_inm-cntr_shift_inm)**2 + (j*freq_step_inm-cntr_shift_inm)**2) < cutf_frequency_inm:
                start[i, j] = 1.0
            else:
                start[i, j] = 0.0

    return start


# def W_0(Cs_mm: float, wvlength_pm: float, df1_nm: float, df2_nm: float, beta_0_rad: float, imge_size: int, pxel_size_nm: float) -> np.ndarray:
#     """Phase distortion function in 2D.  The origin is in the center. 
#        beta_0 is defined in Equation 5 in Cter paper (Penczek et al., 2014).
#     Args:
#         Cs_mm (float): spherical aberration in mm
#         wvlength_pm (float): electron wavelength in pm
#         df1_nm (float): defocus 1 in nm (long axis)
#         df2_nm (float): defocus 2 in nm (short axis)
#         beta_0_rad (float): angle between defocus 1 and x-axis in radian
#         imge_size: image size in pixels
#         pxel_size_nm: pixel size in nm
#     Returns:
#         np.ndarray: distorted phases in radian
#     """
#     start = np.zeros((imge_size, imge_size), dtype=float)

#     freq_step_inm = (1./pxel_size_nm) / imge_size
#     cntr_shift_inm = 1. / (2 * pxel_size_nm)
#     za = (df1_nm + df2_nm) / 2
#     zb = (df1_nm - df2_nm) / 2

#     for i in range(imge_size):
#         for j in range(imge_size):
            
#             if i*freq_step_inm != cntr_shift_inm:
#                 alpha_g = math.atan((j*freq_step_inm-cntr_shift_inm)/(i*freq_step_inm-cntr_shift_inm))
#             else:
#                 alpha_g = math.atan((j*freq_step_inm-cntr_shift_inm)/(10e-6))
#             ssqu = (i*freq_step_inm - cntr_shift_inm)**2 + (j*freq_step_inm - cntr_shift_inm)**2

#             W1 = za + zb * math.cos(2*(alpha_g - beta_0_rad))
#             W21 = -0.5 * W1 * wvlength_pm * ssqu * 1e-3
#             W22 = 0.25 * Cs_mm * wvlength_pm**3 * ssqu**2 * 1e-3
#             start[i, j] = start[i, j] + 6.28318 * (W21 + W22)

#     return start

# Vectorized phase distortion function 
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
    za = (df1_nm + df2_nm) / 2
    zb = (df1_nm - df2_nm) / 2

    # Creating meshgrid for vectorization
    ki, kj = np.meshgrid(np.arange(imge_size), np.arange(imge_size), indexing='ij')

    # Squared spatial frequency 
    ssqu = (ki*freq_step_inm - cntr_shift_inm)**2 + (kj*freq_step_inm - cntr_shift_inm)**2
    # Calculate alpha_g (angle between defocus 1 and x-axis)
    # if (ki*freq_step_inm - cntr_shift_inm).any() == 0:
    #     alpha_g = math.pi/2
    # else:
    #     alpha_g = np.arctan((kj*freq_step_inm-cntr_shift_inm)/(ki*freq_step_inm-cntr_shift_inm))

    # Calculating alpha_g
    alpha_g = np.arctan2(kj * freq_step_inm - cntr_shift_inm, ki * freq_step_inm - cntr_shift_inm)

    # Defocus term
    W1 = za + zb * np.cos(2*(alpha_g - beta_0_rad))
    W21 = -0.5 * W1 * wvlength_pm * ssqu * 1e-3
    # Cs term
    W22 = 0.25 * Cs_mm * wvlength_pm**3 * ssqu**2 * 1e-3

    # Adding phase distortion to CTF array
    ctf += 6.28318 * (W21 + W22)

    return ctf


def calculate_spatial_frequencies(imge_size: int, pxel_size_nm: float):
    """Calculate frequency bins using np.fft.fftfreq and scale them with the pixel size (nm^-1)
    Args:
        imge_size (int):
        pxel_size_nm (float):
    Returns:
        freqs
    """
    # Frequency step
    f_spacing = 1. / (imge_size * pxel_size_nm)
    print('f_spacing', f_spacing)
    print('pixel_size_', pxel_size_nm)

    # Frequencies in nm^-1
    # freqs = np.fft.fftfreq(imge_size, d=f_spacing)
    # freqs = np.fft.fftfreq(imge_size, d=pxel_size_nm) # fix?????
    freqs = np.linspace(-2.5, 2.5, 256)

    # Shift frequencies to center the zero frequency
    # freqs = np.fft.fftshift(freqs)
    print(freqs)

    return freqs

"""
def tcif(spec: np.ndarray, W_0: np.ndarray, beta_rad: float, alpha_rad: float, wvlength_pm: float, imge_size: int, pxel_size_nm: float) -> [np.ndarray, np.ndarray]:
    # Check
    if (W_0.size != imge_size**2) or (spec.size != imge_size**2):
        print("The size of the input array or exit wave does not match the image size.")
        sys.exit()

    # Arrays to hold output values
    amp = np.zeros((imge_size, imge_size), dtype=float)
    phs = np.zeros((imge_size, imge_size), dtype=float)

    # Calculate the spatial frequency grid (in nm^-1)
    freqs = calculate_spatial_frequencies(imge_size, pxel_size_nm)

    # Shifted FFT of the specimen exit wave
    f_spec = np.fft.fftshift(np.fft.fft2(spec))  

    # Set up interpolators for real and imaginary parts of the FFT of the specimen
    interpr = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.real, method='cubic', bounds_error=False, fill_value=np.float64(0.0))
    interpi = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.imag, method='cubic', bounds_error=False, fill_value=np.float64(0.0))

    for i in range(imge_size):
        for j in range(imge_size):
            # sx and sy are now in the centered frequency grid
            sx = freqs[i]
            sy = freqs[j]
            ssqu = sx**2 + sy**2

            pnx_inm = sx - 0.5 * cmath.cos(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
            pny_inm = sy - 0.5 * cmath.sin(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
            ppx_inm = sx + 0.5 * cmath.cos(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
            ppy_inm = sy + 0.5 * cmath.sin(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3

            # Interpolation on the input exit wave
            # sp_rpn = interpr((pnx_inm.real, pny_inm.real))
            sp_rpn = interpr((pnx_inm, pny_inm))
            # p_ipn = interpi((pnx_inm.real, pny_inm.real))
            sp_ipn = interpi((pnx_inm, pny_inm))
            # sp_rpp = interpr((ppx_inm.real, ppy_inm.real))
            sp_rpp = interpr((ppx_inm, ppy_inm))
            # sp_ipp = interpi((ppx_inm.real, ppy_inm.real))
            sp_ipp = interpi((ppx_inm, ppy_inm))

            # Apply phase modulation
            phmod_cos = cmath.cos(W_0[i, j])
            phmod_sin = cmath.sin(W_0[i, j])
            
            Q1r = phmod_cos * sp_rpn + phmod_sin * sp_ipn
            Q1i = -phmod_sin * sp_rpn + phmod_cos * sp_ipn
            Q2r = phmod_cos * sp_rpp - phmod_sin * sp_ipp
            Q2i = phmod_sin * sp_rpp + phmod_cos * sp_ipp

            Q3r = -(Q1i - Q2i)
            Q3i = (Q1r - Q2r)

            amp[i, j] = amp[i, j] + math.sqrt(Q3r**2 + Q3i**2)
            phs[i, j] = phs[i, j] + math.atan(Q3i / Q3r)

    return amp, phs
"""


# Vectorized TCIF function
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

    # Check
    if (W_0.size != imge_size**2) or (spec.size != imge_size**2):
        print("The size of the input array or exit wave does not match the image size.")
        sys.exit()

    # Calculate the spatial frequency grid (in nm^-1)
    freqs = calculate_spatial_frequencies(imge_size, pxel_size_nm)

    # Shifted FFT of the specimen exit wave
    f_spec = np.fft.fftshift(np.fft.fft2(spec)) 
    
    # Create meshgrid for frequency coordinates
    ki, kj = np.meshgrid(freqs, freqs, indexing='ij')

    # Calculate the squared spatial frequency grid
    ssqu = ki**2 + kj**2

    # Calculate pnx_inm, pny_inm, ppx_inm, ppy_inm for all pixels at once
    factor = 0.5 * wvlength_pm * math.tan(alpha_rad) * 1e-3

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
    # phs = np.arctan(Q3i / Q3r)

    return amp, phs, freqs
    

# def tcif_transform(amp: np.ndarray, phs: np.ndarray) -> np.ndarray:
#     """Inverse Fourier transform of the tilt imaging function. 
#     Args:
#         amp (np.ndarray): amplitudes
#         phs (np.ndarray): phases
#     Returns:
#         np.ndarray: intensities in real space
#     """
#     img_size = amp.shape[0]
#     decom = np.zeros((img_size, img_size), dtype=complex) # used to be decom = np.zeros((img_size, img_size), dtype=float)

#     for m in range(img_size):
#         for n in range(img_size):
#             decom[m, n] = amp[m, n] * (math.cos(phs[m, n]) + 1j * math.sin(phs[m, n]))

#     return (np.fft.ifft(decom))**2


# Function to perform radial averaging
def radial_average(data: np.ndarray) -> np.ndarray:
    """Compute the radial average of a 2D array."""
    
    # Get the center of the image
    center = np.array(data.shape) // 2
    
    # Create a meshgrid of indices for the image
    y, x = np.indices(data.shape)
    
    # Calculate the radial distance from the center for each point
    distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Flatten the data and distances
    data_flat = data.flatten()
    distances_flat = distances.flatten()

    # Get the maximum distance (radius)
    max_distance = int(np.max(distances))

    # Prepare an array to hold the binned averages
    binned_data = np.zeros(max_distance)

    # Loop over each radial distance bin
    for i in range(max_distance):
        
        # Get the data in the current radial bin +/- a small buffer
        mask = (distances_flat >= i - 0.1) & (distances_flat < i + 0.1)
        binned_data[i] = np.mean(data_flat[mask])

    return binned_data



if __name__ == '__main__':
    
    # Start time
    start_time = time.time()

    # Parameters
    imge_size = 256
    pxel_size_nm = 0.2
    kvolt = 200
    Cs_mm = 2.0
    df1_nm = 2000.0
    df2_nm = 2000.0
    beta_rad = 0.0
    alpha_rad = np.deg2rad(10.0)

    # Phase distortion function called
    w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)

    ########################## random object  ##########################################################################
    # Setting seed for testing
    # np.random.seed(1387)

    # Generating test object
    print('Generating object...')
    # spec = np.random.rand(imge_size, imge_size)

    ########################## amrophous carbon film  ##########################################################################
    # # Amorphous carbon film simulation parameters:
    # mean_density = 0.6   # depends on your pixel size, etc.
    # # mean_density = (pxel_size_nm ** 2)
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
    # spec = weak_phase_object(imge_size=imge_size)

    ########################## ribosome/apoF projection #####################################################################
    # Parameters for the projection
    # mrc_filename = '8tu7_4ang_apix2.mrc' # apoF
    
    mrc_filename = 'ribo_apix2_res4.mrc' # ribosome
    angles = (0, 0, 0)  # Euler angles for projection
    print('Angles: ', angles)
    
    # Generating 2D projection
    print('Generating 2D projection...')
    spec = generate_2d_projection(mrc_filename, angles, axis = 0)

    #########################################################################################################################
    # Plotting 2D projection
    # plt.imshow(np.abs(spec) ** 2, cmap='gray')
    plt.imshow(spec, cmap = 'gray')
    plt.title('2D Projection')
    # plt.title('Multislice')
    plt.colorbar()
    plt.savefig('projection.png', dpi = 800)
    plt.clf()

    ########################################################################################################################
    # Calling TCIF
    print('Calling TCIF...')
    amp, phs, freqs = tcif(spec, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)


    ##########
    # Phase wrapping issue?
    from skimage.restoration import unwrap_phase

    phs_unwrapped = unwrap_phase(phs)

    plt.imshow(phs_unwrapped, cmap='hsv')
    plt.colorbar(label='Unwrapped Phase (radians)')
    plt.title('Unwrapped Phase Map')
    plt.savefig('phase_unwrapped.png', dpi = 800)
    plt.clf()
    ##########


    # Image reconstruction
    print('Reconstructing image...')
    tilt_im = np.abs(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phs)))) 

    # Normalizes real - space image
    tilt_im_n = (tilt_im - tilt_im.mean()) / (tilt_im.std())
    
    # Plots real space image
    plt.imshow(tilt_im_n, cmap='gray')
    plt.title('Real Space Image After TCIF')
    plt.colorbar()
    plt.savefig('tilted_img_FFT.png', dpi = 800)
    plt.clf()


    # Plots FFT of reconstructed image
    fft_of_reconstructed_img = np.fft.fftshift(np.abs(np.fft.fft2(tilt_im))**2)
    fft_of_reconstructed_img = (fft_of_reconstructed_img - fft_of_reconstructed_img.mean()) / (fft_of_reconstructed_img.std())
    plt.imshow(fft_of_reconstructed_img, cmap='gray')
    plt.title('FFT of Reconstructed Image')
    plt.colorbar()
    plt.savefig('fft_of_reconstructed_img.png', dpi = 800)
    plt.clf()

    ########
    # Find the DC component 
    center = tuple(np.array(fft_of_reconstructed_img.shape) // 2)

    # Change the DC component to a desired value
    desired_dc_value = 0.1
    fft_of_reconstructed_img[center] = desired_dc_value

    # Save or plot the modified reconstructed image
    plt.imshow(fft_of_reconstructed_img, cmap='gray')
    plt.colorbar()
    plt.title("Modified Image with Custom DC Component")
    plt.savefig("fft_of_reconstructed_img_modified.png", dpi=800)
    plt.clf()

    ########

    # Normalize the amplitude array
    amplitude_normalized = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    # Plotting amplitude
    print('Plotting amplitude...')
    plt.imshow(amp, cmap='gray') 
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Amplitude')
    plt.savefig('amplitude.png', dpi = 800)
    plt.clf() 

    # Plotting phase 
    print('Plotting phase...')
    plt.imshow(phs, cmap='hsv') 
    plt.colorbar(label = 'radians')
    plt.gca().invert_yaxis()
    plt.title('Phase (radians)')
    plt.savefig('phase_TCIF.png', dpi = 800)
    plt.clf() 


    # Examining phase distribution - phase wrapping doesn't seem to be an issue
    plt.hist(phs.flatten(), bins=100, color='green', alpha=0.7)
    plt.xlabel("Phase values (radians)")
    plt.ylabel("Frequency")
    plt.title("Phase Histogram")
    plt.savefig('phase_hist.png', dpi = 800)
    plt.clf()


    # Intensity = amplitude^2
    intensity = amp ** 2
    intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    # Plotting normalized intensity
    print('Plotting normalized intensities...')
    plt.imshow(intensity_normalized, cmap='gray') 
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Normalized Intensity')
    plt.savefig('intensity.png', dpi = 800)
    plt.clf() 

    # Power spectrum for amplitudes
    dB = (-10 * np.log(amp)) + 10e-6
   
    # Plotting dB
    print('Plotting dB...')
    plt.imshow(dB, cmap='gray') 
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Power Spectrum (dB)')
    plt.savefig('power_spectrum.png', dpi = 800)
    plt.clf()

    # dB = (-20 * np.log(intensity)) + 10e-6
    # plt.imshow(dB, cmap='gray') 
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.title('Power Spectrum (dB)')
    # plt.savefig('power_spectrum.png')
    # plt.clf() 

    # Perform radial averaging of amplitudes
    radial_avg_amp = radial_average(amplitude_normalized)
    nyquist_frequency = 1 / (2 * pxel_size_nm)
    # Plotting radial averaged amplitude 
    print('Plotting radial averaged amplitude...')
    # Map radial distances to spatial frequencies - x-axis
    spatial_frequencies = np.linspace(0, nyquist_frequency, len(radial_avg_amp))
    plt.plot(spatial_frequencies, radial_avg_amp, color='black')
    plt.xlabel('Spatial Frequency (nm$^{-1}$)')
    plt.ylabel('Radially Averaged Amplitude')
    plt.title('Radially Averaged Amplitude vs. Spatial Frequency (nm$^{-1}$)')
    plt.savefig('radial_avg_amplitude.png', dpi=800)
    plt.clf()
    

    # Plotting phase of raw object prior to TCIF applied 
    raw_phase = np.angle(np.fft.fftshift(np.fft.fft2(spec)))
    plt.imshow(raw_phase, cmap= 'hsv')
    plt.colorbar(label = 'radians')
    plt.gca().invert_yaxis()
    plt.title('Raw Phase (radians)')
    plt.savefig('phase_raw.png', dpi = 800)
    plt.clf()

    # Calculate absolute phase difference between TCIF object and raw object
    phase_difference = np.abs(phs - raw_phase)
    # Plotting absolute phase difference
    plt.imshow(phase_difference, cmap='hsv', extent=(-nyquist_frequency, nyquist_frequency, -nyquist_frequency, nyquist_frequency))
    plt.colorbar(label='radians')
    plt.xlabel('Spatial Frequency (Horizontal, nm$^{-1}$)')
    plt.ylabel('Spatial Frequency (Vertical, nm$^{-1}$)')
    plt.title('Absolute Phase Difference in Fourier Space')
    plt.savefig('phase_difference_heatmap.png', dpi=800)
    plt.clf()

    # Perform radial averaging on the phase difference array
    radial_avg_phase_diff = radial_average(phase_difference)
    # Compute average of radial average phase difference
    avg_val = np.nanmean(radial_avg_phase_diff)
    print(avg_val)
    print((np.pi) / 2)
    # Generate corresponding radial spatial frequencies
    spatial_frequencies = np.linspace(0, nyquist_frequency, len(radial_avg_phase_diff))
    # spatial_frequencies = np.linspace(0, np.max(freqs), len(radial_avg_amp))
    
    # Plot the radially averaged phase difference
    print('Plotting radially averaged phase difference...')
    plt.plot(spatial_frequencies, radial_avg_phase_diff, color='black')
    plt.axhline(y=avg_val, color='red', linestyle='--', label='Mean = {:.2f}'.format(avg_val))
    plt.xlabel('Spatial Frequency (nm$^{-1}$)')
    plt.ylabel('Radially Averaged Phase Difference')
    plt.title('Radially Averaged Phase Difference vs. Spatial Frequency')
    plt.legend()
    plt.savefig('radial_avg_phase_difference.png', dpi = 800)
    plt.clf()


    # # Perform radial averaging on the phase difference array
    # cos_phs = np.cos(radial_avg_phase_diff)
    # # Plot the radially averaged phase difference
    # print('Plotting radially averaged phase difference...')
    # plt.plot(spatial_frequencies, cos_phs, color='black')
    # plt.xlabel('Spatial Frequency (nm$^{-1}$)')
    # plt.ylabel('cos(Radially Averaged Phase Difference)')
    # plt.title('cos(Radially Averaged Phase Difference) vs. Spatial Frequency')
    # plt.savefig('cos_radial_avg_phase_difference.png', dpi = 800)
    # plt.clf()

    # # Plotting the CTF
    # ctf_plot = -2 * np.sin(w0)
    # plt.imshow(ctf_plot, cmap='gray') 
    # plt.colorbar()
    # plt.title('CTF')
    # plt.gca().invert_yaxis()
    # plt.savefig('CTF.png')
    # plt.clf()  # Clear the figure
    
    # End time
    end_time = time.time()

    # Time taken
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
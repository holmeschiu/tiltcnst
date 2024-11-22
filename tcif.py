#!/usr/bin/env python3
# TCIF (Q(p)). 
# Notes:
#   1. The image is presumed to be squared. 
#   2. Presumed the input specimen exit wave has the same dimension and pixel 
#      size as the TCIF. 
#

import cmath
import math
import numpy as np
import scipy as sp
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import time


def wvlength_pm(kvolt: int) -> float:
    """Electron wavelength. 
    Args:
        kvolt (int): accelerating voltage in kV
    Returns:
        float: wavelength in pm
    """
    return 1.23e3 / cmath.sqrt(kvolt*1e3 * (1 + 9.78e-7*kvolt*1e3))


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


    # if (ki*freq_step_inm - cntr_shift_inm).any() == 0:
    #     alpha_g = math.pi/2
    # else:
    #     alpha_g = np.arctan((kj*freq_step_inm-cntr_shift_inm)/(ki*freq_step_inm-cntr_shift_inm))
    # Calculate alpha_g (angle between defocus 1 and x-axis

    # Calculating alpha_g
    alpha_g = np.arctan2(kj * freq_step_inm - cntr_shift_inm, ki * freq_step_inm - cntr_shift_inm)
    
    # Defocus term
    W1 = za + zb * np.cos(2*(alpha_g - beta_0_rad))
    W21 = -0.5 * W1 * wvlength_pm * ssqu * 1e-3
    # Cs term
    W22 = 0.25 * Cs_mm * wvlength_pm**3 * ssqu**2 * 1e-3

    # Adding phase distortion to CTF array
    ctf += 6.28318 * (np.real(W21) + np.real(W22))

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

    # Frequencies in nm^-1
    freqs = np.fft.fftfreq(imge_size, d=f_spacing)

    # Shift frequencies to center the zero frequency
    freqs = np.fft.fftshift(freqs)

    return freqs


# def tcif(spec: np.ndarray, W_0: np.ndarray, beta_rad: float, alpha_rad: float, wvlength_pm: float, imge_size: int, pxel_size_nm: float):
#     """Calculate TCIF in Fourier space. The origin is in the image center.
#     Args:
#         spec (np.ndarray): specimen exit wave, expected to be a 2D array
#         W0 (np.ndarray): phase distortion function, expected to be a 2D array
#     """
#     # Check
#     if (W_0.size != imge_size**2) or (spec.size != imge_size**2):
#         print("The size of the input array or exit wave does not match the image size.")
#         sys.exit()

#     # Arrays to hold output values
#     amp = np.zeros((imge_size, imge_size), dtype=float)
#     phs = np.zeros((imge_size, imge_size), dtype=float)

#     # Calculate the spatial frequency grid (in nm^-1)
#     freqs = calculate_spatial_frequencies(imge_size, pxel_size_nm)

#     # Shifted FFT of the specimen exit wave
#     f_spec = np.fft.fftshift(np.fft.fft2(spec))  

#     # Set up interpolators for real and imaginary parts of the FFT of the specimen
#     interpr = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.real, method='cubic', bounds_error=False, fill_value=np.float64(0.0))
#     interpi = sp.interpolate.RegularGridInterpolator((freqs, freqs), f_spec.imag, method='cubic', bounds_error=False, fill_value=np.float64(0.0))

#     for i in range(imge_size):
#         for j in range(imge_size):
#             # sx and sy are now in the centered frequency grid
#             sx = freqs[i]
#             sy = freqs[j]
#             ssqu = sx**2 + sy**2

#             pnx_inm = sx - 0.5 * cmath.cos(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
#             pny_inm = sy - 0.5 * cmath.sin(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
#             ppx_inm = sx + 0.5 * cmath.cos(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
#             ppy_inm = sy + 0.5 * cmath.sin(beta_rad) * ssqu * wvlength_pm * cmath.tan(alpha_rad) * 1e-3

#             # Interpolation on the input exit wave
#             # sp_rpn = interpr((pnx_inm.real, pny_inm.real))
#             sp_rpn = interpr((pnx_inm, pny_inm))
#             # p_ipn = interpi((pnx_inm.real, pny_inm.real))
#             sp_ipn = interpi((pnx_inm, pny_inm))
#             # sp_rpp = interpr((ppx_inm.real, ppy_inm.real))
#             sp_rpp = interpr((ppx_inm, ppy_inm))
#             # sp_ipp = interpi((ppx_inm.real, ppy_inm.real))
#             sp_ipp = interpi((ppx_inm, ppy_inm))

#             # Apply phase modulation
#             phmod_cos = cmath.cos(W_0[i, j])
#             phmod_sin = cmath.sin(W_0[i, j])
            
#             Q1r = phmod_cos * sp_rpn + phmod_sin * sp_ipn
#             Q1i = -phmod_sin * sp_rpn + phmod_cos * sp_ipn
#             Q2r = phmod_cos * sp_rpp - phmod_sin * sp_ipp
#             Q2i = phmod_sin * sp_rpp + phmod_cos * sp_ipp

#             Q3r = -(Q1i - Q2i)
#             Q3i = (Q1r - Q2r)

#             amp[i, j] = amp[i, j] + math.sqrt(Q3r**2 + Q3i**2)
#             phs[i, j] = phs[i, j] + math.atan(Q3i / Q3r)

#     return amp, phs

# Vectorized TCIF function
def tcif(spec: np.ndarray, W_0: np.ndarray, beta_rad: float, alpha_rad: float, wvlength_pm: float, imge_size: int, pxel_size_nm: float):
    """Calculate TCIF in Fourier space. The origin is in the image center.
    Args:
        spec (np.ndarray): specimen exit wave, expected to be a 2D array
        W_0 (np.ndarray): phase distortion function, expected to be a 2D array
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
    factor = 0.5 * wvlength_pm * cmath.tan(alpha_rad) * 1e-3
    pnx_inm = ki - factor * cmath.cos(beta_rad) * ssqu
    pny_inm = kj - factor * cmath.sin(beta_rad) * ssqu
    ppx_inm = ki + factor * cmath.cos(beta_rad) * ssqu
    ppy_inm = kj + factor * cmath.sin(beta_rad) * ssqu

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

    return amp, phs
            

def tcif_transform(amp: np.ndarray, phs: np.ndarray) -> np.ndarray:
    """Inverse Fourier transform of the tilt imaging function. 
    Args:
        amp (np.ndarray): amplitudes
        phs (np.ndarray): phases
    Returns:
        np.ndarray: intensities in real space
    """
    img_size = amp.shape[0]
    decom = np.zeros((img_size, img_size), dtype=complex) # used to be decom = np.zeros((img_size, img_size), dtype=float)

    for m in range(img_size):
        for n in range(img_size):
            decom[m, n] = amp[m, n] * (cmath.cos(phs[m, n]) + 1j * cmath.sin(phs[m, n]))

    return (np.fft.ifft(decom))**2

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
        mask = (distances_flat >= i - 0.5) & (distances_flat < i + 0.5)
        binned_data[i] = np.mean(data_flat[mask])

    return binned_data


if __name__ == '__main__':
    
    # Start time
    start_time = time.time()

    # Parameters
    imge_size = 512
    pxel_size_nm = 0.1
    kvolt = 300
    Cs_mm = 2.0
    df1_nm = 1500.0
    df2_nm = 1500.0
    beta_rad = 0.0
    alpha_rad = np.deg2rad(0.0)

    # Phase distortion function called
    w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)

    # Setting seed for testing
    np.random.seed(1387)

    # Generating test object
    print('Generating object...')
    spec = np.random.rand(imge_size, imge_size)

    # Calling TCIF
    print('Calling TCIF...')
    amp, phs = tcif(spec, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)

    # Normalize the amplitude array
    amplitude_normalized = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    
    # Intensity = amplitude^2
    intensity = amplitude_normalized ** 2

    # Power spectrum for normalized amplitudes
    dB = (-10 * np.log(amplitude_normalized)) + 10e-6

    # Perform radial averaging on the intensity data
    radial_avg_intensity = radial_average(intensity)

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
    plt.plot(radial_avg_intensity, color='black')
    plt.title('Radial Averaged Intensity')
    plt.xlabel('Radial Distance (pixels)')
    plt.ylabel('Average Intensity')
    plt.savefig('radial_avg_intensity.png', dpi = 800)
    plt.clf()
    
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
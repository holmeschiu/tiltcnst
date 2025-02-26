#!/usr/bin/env python3

from tcif import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


imge_size = 256
pxel_size_nm = 0.2
kvolt = 300
Cs_mm = 2.0
df1_nm = -500.0
df2_nm = -500.0
beta_rad = 0.0
alpha_rad = np.deg2rad(30.0)

# mrc_filename = '8tu7_2_ang_med.mrc'  
mrc_filename = '8tu7_4ang_apix2.mrc'
angles = (0, 0, 0)  # Euler angles for projection
spec = generate_2d_projection(mrc_filename, angles, axis = 0)

# Plotting 2D projection
plt.imshow(spec, cmap='gray')
plt.title('2D Projection of Apoferritin')
plt.colorbar()
plt.savefig('projection.png', dpi = 800)
plt.clf()

# Phase distortion array
w0 = W_0(Cs_mm, wvlength_pm(kvolt), df1_nm, df2_nm, beta_rad, imge_size, pxel_size_nm)
# Call TCIF
amp, phs = tcif(spec, w0, beta_rad, alpha_rad, wvlength_pm(kvolt), imge_size, pxel_size_nm)

# Image reconstruction
tilt_im = np.abs(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phs)))) # Numpy FFT
# tilt_im = np.abs(idft2d_matrix(np.fft.ifftshift(amp * np.exp(1j * phs)))) # DFT

# Normalizes image
tilt_im = (tilt_im - tilt_im.mean()) / (tilt_im.std())

# Plots real space image
plt.imshow(tilt_im, cmap='gray')
plt.title('Real Space Image After TCIF')
plt.colorbar()
plt.savefig('tilted_img.png', dpi = 800)
plt.clf()

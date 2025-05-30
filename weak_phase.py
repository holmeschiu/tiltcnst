#===================Imports=================
import numpy as np
import matplotlib.pyplot as plt


#===================Functions=================
def wk_phs_obj(imge_size=256, pxel_size_nm=0.2, phase_sigma=80.0, amp_sigma=30.0, phase_strength=1.0) -> np.ndarray:
    """
    Create a weak phase object using real-world units to match TCIF spatial frequencies.
    
    Args:
        imge_size (int): Image size in pixels
        pxel_size_nm (float): Pixel size in nm
        phase_sigma_nm (float): Sigma of Gaussian phase blob in nm
        phase_strength (float): Strength of phase modulation in radians
        
    Returns:
        wavefield (np.ndarray): Complex wavefield
        X, Y (np.ndarray): Real-space grids in nm
        amplitude (np.ndarray)
        phase (np.ndarray)
    """
     
    # x = np.linspace(-2.5, 2.5, imge_size)
    # y = np.linspace(-2.5, 2.5, imge_size)
    # X, Y = np.meshgrid(x, y)

    # Real-space coordinates in nm
    field_width_nm = imge_size * pxel_size_nm
    x = np.linspace(-field_width_nm/2, field_width_nm/2, imge_size)
    y = np.linspace(-field_width_nm/2, field_width_nm/2, imge_size)
    X, Y = np.meshgrid(x, y)

    # Gaussian amplitude
    # amplitude = np.exp(-(X**2 + Y**2) / (2 * amp_sigma**2))
    # Constant amplitude across the entire object
    amplitude = np.ones_like(X)
    
    # Gaussian phase
    # phase = phase_strength * np.exp(-(X**2 + Y**2) / (2 * phase_sigma**2))
    # Random phase 
    random_core = np.random.uniform(-20, 20, size=(imge_size, imge_size))
    phase = phase_strength * random_core
    
    # Complex wavefield
    wavefield = amplitude * np.exp(1j * phase)
    
    return wavefield, X, Y, amplitude, phase


#===================Main usage=================
if __name__ == '__main__':
    # Generate wavefield
    wavefield, X, Y, amplitude, phase = wk_phs_obj()

    # Plot amplitude, phase, and intensity
    plt.figure(figsize=(15, 5))

    extent_nm = [X.min(), X.max(), Y.min(), Y.max()]

    plt.subplot(1, 4, 1)
    plt.title('Amplitude (Real Space)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    im1 = plt.imshow(amplitude, cmap='viridis', extent=extent_nm)
    plt.colorbar(im1)

    plt.subplot(1, 4, 2)
    plt.title('Intensity (Real Space)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    im2 = plt.imshow(np.abs(wavefield)**2, cmap='viridis', extent=extent_nm)
    plt.colorbar(im2)

    plt.subplot(1, 4, 3)
    plt.title('Phase (Real Space)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    im3 = plt.imshow(phase, cmap='viridis', extent=extent_nm)
    plt.colorbar(im3, label='radians')

    plt.subplot(1, 4, 4)
    im4 = plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(wavefield))), cmap='viridis', extent=extent_nm)
    plt.title('Phase (Fourier Space)')
    plt.xlabel('kx (nm$^{-1}$)')
    plt.ylabel('ky (nm$^{-1}$)')
    plt.colorbar(im4, label='radians')

    plt.tight_layout()
    plt.savefig('weak_phase.png', dpi = 800)
    plt.clf()

#===================Imports=================
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


#===================Functions=================
# Function to rotate the volume around ZYZ axes
def rotate_vol(data, angles):
    """
    Rotate a 3D volume using ZYZ Euler angles.

    Args:
        data (np.ndarray): 3D array of the volume data.
        angles (tuple): Tuple of three Euler angles (phi, theta, psi) in degrees.

    Returns:
        np.ndarray: Rotated 3D volume.
    """
    
    # Rotate volume data according to ZYZ Euler angles
    # First rotation - Z - uses "nearest" form of interpolation meaning the pixel value from previous pixel is repeated
    rotated_data = rotate(data, angles[0], axes = (0, 1), reshape = False, mode = "constant")
    
    # Second rotation - Y
    rotated_data = rotate(rotated_data, angles[1], axes = (0, 2), reshape = False, mode = "constant")
    
    # Third rotation - Z
    rotated_data = rotate(rotated_data, angles[2], axes = (0, 1), reshape = False, mode = "constant")
    
    # Returning rotated data
    return rotated_data


# Function to generate 2D projection
def gen_2d_proj(mrc_filename, angles, axis = 0):
    """
    Generate a 2D projection of a 3D density map.

    Args:
        mrc_filename (str): Path to the MRC file.
        angles (tuple): Tuple of three Euler angles (phi, theta, psi) in degrees.

    Returns:
        np.ndarray: 2D projection of the rotated 3D volume.
    """
    
    # Open the MRC file
    with mrcfile.open(mrc_filename, permissive = True) as mrc:
        # Read the data from the MRC file
        data = mrc.data

        # Rotate the volume data according to the given ZYZ Euler angles
        rotated_data = rotate_vol(data, angles)
        
        # Sum along the specified axis to get the projection
        projection = np.sum(rotated_data, axis = axis)
        
        return projection
    

# Function to add cryo-EM-like noise
def add_cryo_em_noise(image: np.ndarray, electron_dose: float = 100, detector_noise_std: float = 0.01, normalize: bool = True) -> np.ndarray:
    
    # Normalize to mimic intensity scaling
    image = image - image.min()
    image = image / image.max()

    # Scale signal to e- counts - bright pixels = more e-
    signal_scaled = image * electron_dose

    # Apply Poisson noise
    noisy_poisson = np.random.poisson(signal_scaled).astype(np.float32)

    # Rescale back to normalized range
    noisy_poisson /= electron_dose

    # Add Gaussian detector noise
    noisy_image = noisy_poisson + np.random.normal(loc=0.0, scale=detector_noise_std, size=image.shape)

    # Normalize to original image statistics
    if normalize:
        orig_mean, orig_std = image.mean(), image.std()
        noisy_image = (noisy_image - noisy_image.mean()) / (noisy_image.std() + 1e-8)
        noisy_image = noisy_image * orig_std + orig_mean

    return noisy_image



if __name__ == '__main__':

    # Input MRC file name
    # mrc_filename = '8tu7_2_ang_med.mrc'
    # mrc_filename = '8tu7_4ang_apix2.mrc'
    mrc_filename = 'ribo_apix2_res4.mrc'

    # Number of projections to generate
    num_of_projections = 1

    # For loop to generate 2D projections 
    for pro in range(num_of_projections):
        
        # Euler Angles - degrees
        # angles = (np.random.uniform(0, 360), np.random.uniform(0, 360), np.random.uniform(0, 360))
        angles = (0, 0, 0) 
        
        # Generating projection 
        proj = gen_2d_proj(mrc_filename, angles, axis = 0)

        # Plotting
        plt.imshow(proj, cmap = 'gray')
        plt.title('2D Projection')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('projection.png', dpi = 800)
        plt.clf()

        # Add realistic cryo-EM noise
        noisy_proj = add_cryo_em_noise(proj, electron_dose=50, detector_noise_std=0.02)

        # Save noisy projection
        plt.imshow(noisy_proj, cmap='gray')
        plt.title('Noisy 2D Projection')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('projection_noisy.png', dpi=800)
        plt.clf()

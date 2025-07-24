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
    

if __name__ == '__main__':

    # Input MRC file name
    # mrc_filename = '8tu7_2_ang_med.mrc'
    # mrc_filename = '8tu7_4ang_apix2.mrc'
    mrc_filename = '/Users/kiradevore/Documents/python_scripts/TCIF/250402_opt_of_250111/ribo_3D_maps/ribo_apix2_res4_box1024.mrc'

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


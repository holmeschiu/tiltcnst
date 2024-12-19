# Libraries
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import time

# Noting start time
start_time = time.time()

# Function to rotate the volume around ZYZ axes
def rotate_volume(data, angles):
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
    rotated_data = rotate(data, angles[0], axes = (0, 1), reshape = False, mode = "nearest")
    
    # Second rotation - Y
    rotated_data = rotate(rotated_data, angles[1], axes = (0, 2), reshape = False, mode = "nearest")
    
    # Third rotation - Z
    rotated_data = rotate(rotated_data, angles[2], axes = (0, 1), reshape = False, mode = "nearest")
    
    # Returning rotated data
    return rotated_data


# Function to generate 2D projection
def generate_2d_projection(mrc_filename, angles, axis = 0):
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

        # Check if data needs to be converted from complex
        if np.iscomplexobj(data):
            # Taking the absolute value for projection
            data = np.abs(data)
        
        # Rotate the volume data according to the given ZYZ Euler angles
        rotated_data = rotate_volume(data, angles)
        
        # Sum along the specified axis to get the projection
        projection = np.sum(rotated_data, axis = axis)
        
        # Save the projection to an image file 
        plt.imshow(projection, cmap = "gray")
        plt.colorbar()
        plt.savefig(f"proj_{pro}.png")
        plt.gca().invert_yaxis()
        plt.clf()

        return projection



# Execution
# Input MRC file name
mrc_filename = "1dat_2_ang_small.mrc" 

# Number of projections to generate
# num_of_projections = int(input("How many projections do you need? "))
num_of_projections = 1

# For loop to generate 2D projections 
for pro in range(num_of_projections):
    
    # Euler Angles - degrees
    # angles = (np.random.uniform(0, 360), np.random.uniform(0, 360), np.random.uniform(0, 360))
    angles = (0, 0, 0) # used to test
    
    # Printing projection number and angles of projection
    print("Phi:", angles[0], "degrees", "\nTheta:", angles[1], "degrees", "\nPsi:", angles[2], "degrees")
    
    # Generating projection 
    generate_2d_projection(mrc_filename, angles, axis = 0)


# Calculating end time of program
end_time = time.time()
# Printing program run time
# print(f"Execution time: {end_time - start_time} seconds")
#===================Imports=================
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


#===================Functions=================
# Function to perform radial averaging
def radial_average(data: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a 2D array.
    Args:
        data (np.ndarray): A 2D NumPy array.
    Returns:
        np.ndarray: A 1D NumPy array containing the radially averaged values
                    for each integer distance from the center up to the maximum radius.
    """
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
        
        # Ensures values are within bins
        if np.any(mask):
            binned_data[i] = np.mean(data_flat[mask])
        # Fill empty bins with nan
        else:
            binned_data[i] = np.nan  # or 0?

    return binned_data


# Function for plotting 2D data
def save_imshow(data: np.ndarray, title: str, xlabel: str = '', ylabel: str = '', 
                filename: str = '', cmap: str = 'gray', colorbar_label: str = '',
                invert_yaxis: bool = True, extent=None, vmin=None, vmax=None):
    """
    Display and save a 2D image with consistent formatting.

    Args:
        data (np.ndarray): 2D array to plot.
        title (str): Plot title.
        filename (str): Output filename (e.g., 'plot.png').
        cmap (str): Colormap (e.g., 'gray', 'hsv').
        colorbar_label (str): Label for the colorbar.
        invert_yaxis (bool): Whether to invert the y-axis.
        extent (list or tuple): Extent for imshow (e.g., used in Fourier space).
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
    """
    plt.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    cb = plt.colorbar()
    if colorbar_label:
        cb.set_label(colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.savefig(filename, dpi=300)
    plt.close()



# Function for line plots
def save_lineplot(x, y, title: str, xlabel: str, ylabel: str,
                  filename: str, linecolor: str = 'black',
                  hline: Tuple[float, str] = None, legend: str = None):
    """
    Save a line plot.

    Args:
        x (array-like): x-axis values.
        y (array-like): y-axis values.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        filename (str): Output filename.
        linecolor (str): Line color.
        hline (Tuple[float, str], optional): y-value and color for a horizontal line.
        legend (str, optional): Legend label for the horizontal line.
    """
    plt.plot(x, y, color=linecolor)
    if hline:
        plt.axhline(y=hline[0], color=hline[1], linestyle='--', label=legend)
        if legend:
            plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

#!/usr/bin/env python3
# Calculate the molecular Coulomb potential. 
#

# Imports
import numpy as np
import sys
from typing import List
import math


# Precomputing constants
TWOPI = 2 * math.pi
SQUPI = math.pi ** 2 


# Function to read coefficients from Kirkland's scattering table
def tblt_coefficients(Z: int) -> List[float]:
    """
    Read coefficients from the scattering table for a given atomic number Z.
    
    Returns:
        List of 12 floats corresponding to Kirkland's a1, b1, ..., c3, d3
    """
    # Opens Kirkland's scattering coefficient table
    try: 
        data = open('fparams.dat', 'r').readlines()
    
    # Exits if file not found
    except FileNotFoundError:
        print("The scattering table file, fparams.dat, was not found.")
        sys.exit(1)
    
    # Extracts coefficients from file
    for line in data:
        if line.startswith('Z'):
            atom_number = int(line.split(',')[0][2:5])
            if atom_number == Z:
                idx = data.index(line)
                a1 = float(data[idx+1].split()[0])
                b1 = float(data[idx+1].split()[1])
                a2 = float(data[idx+1].split()[2])
                b2 = float(data[idx+1].split()[3])
                a3 = float(data[idx+2].split()[0])
                b3 = float(data[idx+2].split()[1])
                c1 = float(data[idx+2].split()[2])
                d1 = float(data[idx+2].split()[3])
                c2 = float(data[idx+3].split()[0])
                d2 = float(data[idx+3].split()[1])
                c3 = float(data[idx+3].split()[2])
                d3 = float(data[idx+3].split()[3])
                break
    
    return [a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3]


# Function to compute the atomic potential (in V)
def atom_potential(atom_number: int, box_size: int, pxel_size_nm: float,
                   ax_nm: float, ay_nm: float, az_nm: float) -> np.ndarray:
    """
    Computes the 3D Coulomb potential of a single atom on a voxel grid using Kirkland's parameterization.

    The potential is based on a combination of exponential and Gaussian terms, as described in Kirkland's textbook (Eq. 5.9) (Eq. C.19), 
    and assumes the atom is positioned at coordinates (ax_nm, ay_nm, az_nm) relative to the center of the grid.

    Args:
        atom_number (int): Atomic number (Z) of the element, used to retrieve scattering parameters.
        box_size (int): Size of the 3D cubic grid (in pixels).
        pxel_size_nm (float): Size of a voxel (pixel) in nanometers.
        ax_nm (float): Atom x-position relative to the grid center (in nanometers).
        ay_nm (float): Atom y-position relative to the grid center (in nanometers).
        az_nm (float): Atom z-position relative to the grid center (in nanometers).

    Returns:
        np.ndarray: A 3D NumPy array of shape (box_size, box_size, box_size) representing the
                    electrostatic potential (in volts) contributed by the atom at each voxel.
    
    """

    tbcof = tblt_coefficients(atom_number)

    # Precomputed constants
    C1 = 2 * SQUPI * 0.0529e-9 * 1.6e-19 * 1e9
    C2 = 2 * (np.pi)**2.5 * 0.0529e-9 * 1.6e-19

    # Grid setup
    coords = (np.arange(box_size) - box_size / 2) * pxel_size_nm
    X, Y, Z = np.meshgrid(coords - ax_nm,
                          coords - ay_nm,
                          coords - az_nm,
                          indexing='ij')

    R = np.sqrt(X**2 + Y**2 + Z**2)
    R[R == 0] = 1e-12  # Avoid divide-by-zero


    # Term 1 - Exponential decay over distance eq C.19 in Kirkland
    t1 = (
        tbcof[0] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[1])) +
        tbcof[2] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[3])) +
        tbcof[4] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[5]))
    ) * C1

    # Term 2 - Gaussian decay over squared distance
    t2 = (
        tbcof[6] * tbcof[7]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[7]) +
        tbcof[8] * tbcof[9]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[9]) +
        tbcof[10] * tbcof[11]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[11])
    ) * C2

    return t1 + t2

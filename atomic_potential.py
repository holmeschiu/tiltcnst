#!/usr/bin/env python3
# Calculates the atomic Coulombic potential using a precomputed grid
#

import numpy as np
import sys
from typing import Tuple
import math

# Constants
TWOPI = 2 * math.pi
SQUPI = math.pi ** 2

# Cache for Kirkland scattering parameters
_fparam_cache = {}

# Function to pull coefficients using cache
def get_coefficients(Z: int):
    """
    Retrieve Kirkland scattering coefficients from fparams.dat, using a cache.
    """
    # Modifies the module-level _fparam_cache variable
    global _fparam_cache

    # If not yet loaded, load the file and populate the cache
    if not _fparam_cache:
        try:
            with open('fparams.dat', 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("The scattering table file, fparams.dat, was not found.")
            sys.exit(1)
        # Loop through each line of file and extract coefficients
        for i, line in enumerate(lines):
            if line.startswith('Z'):
                atom_number = int(line.split(',')[0][2:5])
                a1, b1, a2, b2 = map(float, lines[i+1].split())
                a3, b3, c1, d1 = map(float, lines[i+2].split())
                c2, d2, c3, d3 = map(float, lines[i+3].split())
                _fparam_cache[atom_number] = [a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3]
    
    # Return cached result
    return _fparam_cache[Z]

# Function to precompute grid centered at 0, 0, 0
def precompute_grid(box_size: int, pxel_size_nm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes a 3D grid of voxel coordinates centered on the box.
    """
    coords = (np.arange(box_size) - box_size / 2) * pxel_size_nm
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return X, Y, Z

# Function to compute atomic potential
def atom_potential(atom_number: int,
                   X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                   ax_nm: float, ay_nm: float, az_nm: float) -> np.ndarray:
    """
    Computes the 3D Coulomb potential of a single atom given a fixed grid and the atom's position.

    Args:
        atom_number (int): Atomic number (Z) of the atom.
        X, Y, Z (np.ndarray): Precomputed voxel coordinate grids (in nm).
        ax_nm, ay_nm, az_nm (float): Atom coordinates in nanometers.

    Returns:
        np.ndarray: A 3D array of the atomic potential at each voxel.
    """
    # Gets coefficients
    tbcof = get_coefficients(atom_number)

    # Physical constants
    C1 = 2 * SQUPI * 0.0529e-9 * 1.6e-19 * 1e9  # V·nm
    C2 = 2 * (np.pi)**2.5 * 0.0529e-9 * 1.6e-19  # V·nm³

    # Compute distances from atom to each voxel
    dx = X - ax_nm
    dy = Y - ay_nm
    dz = Z - az_nm
    R2 = dx**2 + dy**2 + dz**2
    R = np.sqrt(R2)
    R[R == 0] = 1e-12  # avoid division by zero

    # Exponential decay term (Eq C.19, Kirkland)
    t1 = (
        tbcof[0] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[1])) +
        tbcof[2] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[3])) +
        tbcof[4] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[5]))
    ) * C1

    # Gaussian decay term (Eq 5.9, Kirkland)
    t2 = (
        tbcof[6] * tbcof[7]**(-1.5) * np.exp(-SQUPI * R2 / tbcof[7]) +
        tbcof[8] * tbcof[9]**(-1.5) * np.exp(-SQUPI * R2 / tbcof[9]) +
        tbcof[10] * tbcof[11]**(-1.5) * np.exp(-SQUPI * R2 / tbcof[11])
    ) * C2

    return t1 + t2

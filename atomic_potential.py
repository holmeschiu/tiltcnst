#!/usr/bin/env python3
# Calculate the molecular Coulomb potential. 
#

import numpy as np
import sys
import cmath
from typing import List


TWOPI = 6.28318
SQUPI = 9.869587728


def tblt_coefficients(Z: int) -> List[float]:
    """Read coefficients from the scattering table. 
    Args:
        Z (int): atomic number. 
    """
    try: 
        data = open('fparams.dat', 'r').readlines()
    except FileNotFoundError:
        print("The scattering table file, fparams.dat, was not found.")
        sys.exit()
    else:
        for line in data:
            if line.startswith('Z'):
                atom_number = int(line.split(',')[0][2:5])
                if atom_number == Z:
                    idx = index(line)
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


def atom_potential(atom_number: int, box_size: int, pxel_size_nm: float) -> np.ndarray:
    """Calculate the atomic potential in volt on grids. Pages 105 and 293.  Eq. (5.9).
    Needs to check the units within the equation. 

    Args:
        a (Atom): atom object
        box_size (int): box size in pixels
        pixel_size_A (float): pixel size in Angstroms
    Returns:
        np.ndarray: atomic potential on grids
    """
    tbcof = tblt_coefficients(atom_number)
    v_grid = np.zeros((box_size, box_size, box_size))
    box_center = box_size / 2

    for i in range(box_size):
        for j in range(box_size):
            for k in range(box_size):
                r_ = sqrt((i-box_center)**2 + (j-box_center)**2 + (k-box_center)**2) * pxel_size_nm

                t1_1 = tbcof[0]/r_ * cmath.exp(-TWOPI*r_*cmath.sqrt(tbcof[1])) * 1e9
                t1_2 = tbcof[2]/r_ * cmath.exp(-TWOPI*r_*cmath.sqrt(tbcof[3])) * 1e9
                t1_3 = tbcof[4]/r_ * cmath.exp(-TWOPI*r_*cmath.sqrt(tbcof[5])) * 1e9
                t1 = 2 * SQUPI * 0.0529e-9 * 1.6e-19 * (t1_1 + t1_2 + t1_3)

                t2_1 = tbcof[6] * tbcof[7]**(-1.5) * cmath.exp(-SQUPI * r_**2 / tbcof[7])
                t2_2 = tbcof[8] * tbcof[9]**(-1.5) * cmath.exp(-SQUPI * r_**2 / tbcof[9])
                t2_3 = tbcof[10] * tbcof[11]**(-1.5) * cmath.exp(-SQUPI * r_**2 / tbcof[11])
                t2 = 2 * (3.14159)**2.5 * 0.0529e-9 * 1.6e-19 * (t2_1 + t2_2 + t2_3)

                v_grid[i, j, k] = t1 + t2

    return v_grid


    
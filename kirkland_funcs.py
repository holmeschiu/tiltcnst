#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Constants
TWOPI = 2 * math.pi
SQUPI = math.pi ** 2
C1 = 2 * SQUPI * 0.0529e-9 * 1.6e-19 * 1e9
C2 = 2 * (math.pi)**2.5 * 0.0529e-9 * 1.6e-19

def tblt_coefficients(Z: int, filepath: str) -> list:
    """
    Retrieve Kirkland scattering parameters for atomic number Z from fparams.dat.

    Args:
        Z (int): Atomic number
        filepath (str): Path to fparams.dat

    Returns:
        list of 12 Kirkland coefficients: a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3
    """
    with open(filepath, 'r') as file:
        data = file.readlines()

    for i, line in enumerate(data):
        if line.startswith('Z') and int(line.split(',')[0][2:5]) == Z:
            a1, b1, a2, b2 = map(float, data[i+1].split())
            a3, b3, c1, d1 = map(float, data[i+2].split())
            c2, d2, c3, d3 = map(float, data[i+3].split())
            return [a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3]
    raise ValueError(f"Atomic number {Z} not found in {filepath}")

def plot_kirkland_terms(fparams_path: str, elements: dict, output_file: str):
    """
    Plot exponential, Gaussian, and total Kirkland potentials for given elements.

    Args:
        fparams_path (str): Path to fparams.dat file
        elements (dict): Mapping of element names to atomic numbers
        output_file (str): Output filename for the saved plot
    """
    R = np.linspace(0.1, 5.0, 1000)  # R in Å

    fig, axs = plt.subplots(len(elements), 3, figsize=(15, 3 * len(elements)))
    fig.suptitle("Kirkland Scaled Atomic Potential Components", fontsize=16)

    for row, (name, Z) in enumerate(elements.items()):
        tbcof = tblt_coefficients(Z, fparams_path)

        # Exponential term (Eq. C.19)
        t1 = (
            tbcof[0] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[1])) +
            tbcof[2] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[3])) +
            tbcof[4] / R * np.exp(-TWOPI * R * np.sqrt(tbcof[5]))
        ) * C1

        # Gaussian term (Eq. 5.9)
        t2 = (
            tbcof[6] * tbcof[7]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[7]) +
            tbcof[8] * tbcof[9]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[9]) +
            tbcof[10] * tbcof[11]**(-1.5) * np.exp(-SQUPI * R**2 / tbcof[11])
        ) * C2

        V_total = t1 + t2

        axs[row, 0].plot(R, t1*1e20, color='blue')
        axs[row, 0].set_title(f"{name} - Exponential Term")
        axs[row, 0].set_ylabel("Potential (V)")
        axs[row, 0].set_xlabel("R (Å)")
        axs[row, 0].grid(True)

        axs[row, 1].plot(R, t2*1e20, color='green')
        axs[row, 1].set_title(f"{name} - Gaussian Term")
        axs[row, 1].set_xlabel("R (Å)")
        axs[row, 1].grid(True)

        axs[row, 2].plot(R, V_total*1e20, color='red')
        axs[row, 2].set_title(f"{name} - Total Potential")
        axs[row, 2].set_xlabel("R (Å)")
        axs[row, 2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print(f"Plot saved to {output_file}")
    plt.savefig(output_file, dpi=800)

if __name__ == '__main__':
    fparams_file = "fparams.dat"  # Ensure this file is in the same directory
    output_img = "kirkland_scaled_potential_components.png"
    elements_to_plot = {
        "Carbon": 6,
        "Nitrogen": 7,
        "Oxygen": 8,
        "Sulfur": 16
    }

    plot_kirkland_terms(fparams_file, elements_to_plot, output_img)

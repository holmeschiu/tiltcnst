#!/usr/bin/env python3
# Computes molecular potential

import pdb_handler as pdb
from pdb_handler import PDB
import numpy as np
from typing import Type
import atomic_potential as ap
from joblib import Parallel, delayed

def integrate_atomic_potential(pdb: Type[PDB], box_size: int, pxel_size_nm: float) -> np.ndarray:
    """
    Computes the full molecular Coulomb potential of all atoms in the protein.

    Args:
        pdb (PDB): A parsed PDB object containing atoms and their coordinates.
        box_size (int): Number of voxels along each box edge.
        pxel_size_nm (float): Size of each voxel in nanometers.

    Returns:
        np.ndarray: The total 3D electrostatic potential grid.
    """
    # Initialize the potential volume
    V_grid = np.zeros((box_size, box_size, box_size), dtype=np.float64)

    # Precompute the coordinate grid (centered at 0,0,0)
    X, Y, Z = ap.precompute_grid(box_size, pxel_size_nm)

    # Loop through all atoms
    for atom in pdb.atoms:
        # Convert Å to nm
        ax = atom.x / 10
        ay = atom.y / 10
        az = atom.z / 10

        # Compute the atom’s contribution to the potential
        V_atom = ap.atom_potential(atom.atomic_number, X, Y, Z, ax, ay, az)

        # Accumulate into the total potential grid
        V_grid += V_atom

    return V_grid



def main():
    pro = PDB("1dat.pdb")
    pro.read_pdb()


if __name__ == '__main__':
    main()

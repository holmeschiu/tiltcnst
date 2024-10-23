#!/usr/bin/env python3

import pdb_1 as pdb
from pdb_1 import PDB
import numpy as np
from typing import Type
import atomic_potential as ap


def integrate_atomic_potential(pdb: Type[PDB], box_size: int, pxel_size_nm: float) -> np.ndarray:
    # used to be V_grid = np.zeros((box_size, box_size, box_size))
    V_grid = np.zeros((box_size, box_size, box_size), dtype=complex)

    for a in pdb.atoms:
        # used to be v = ap.atom_potential(a.atom_number, box_size, pxel_size_nm, a.x/10, a.y/10, a.z/10) 
        v = ap.atom_potential(a.atomic_number, box_size, pxel_size_nm, a.x/10, a.y/10, a.z/10) 
        V_grid = V_grid + v

    return V_grid


def main():
    apof = pdb.PDB("1dat.pdb")
    apof.read_pdb()
    

if __name__ == '__main__':
    main()

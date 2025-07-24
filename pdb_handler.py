#!/usr/bin/env python3
# PDB file handling
#
# Po-Lin Chiu   - 11/13/2023
#
# Imports
import numpy as np
from sklearn.decomposition import PCA

a_attribute = {
    "H":   {"name": "hydrogen",    "atomic_number":   1},
    "C":   {"name": "carbon",      "atomic_number":   6},
    "N":   {"name": "nitrogen",    "atomic_number":   7},
    "O":   {"name": "oxygen",      "atomic_number":   8},
    "P":   {"name": "phosphorus",  "atomic_number":  15}, 
    "S":   {"name": "sulfur",      "atomic_number":  16}
}


class PDB:
    """
    Represents a single atom from a PDB structure.
    Stores coordinates, atom name, residue info, and atomic number.
    """ 
    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.num_atoms = 0

    def read_pdb(self, quiet=False):
        with open(self.filename, "r") as f:
            for line in f:
                
                # Selecting for only atoms
                if line.startswith("ATOM"):
                    atom = Atom()
                    
                    # Extracts serial number 
                    serial_str = line[6:11].strip()
                    
                    # Handles serial number for large PDB with funky formatting
                    try:
                        
                        # Directly converts string to integer for atoms 0 - 99999
                        if serial_str.isdigit():
                            atom.serial = int(serial_str)
                        
                        # If using extended alphanumeric format...
                        elif len(serial_str) == 5 and serial_str[0].isalpha() and serial_str[1:].isdigit():
                            # Convert prefix to number
                            prefix = serial_str[0]
                            digits = serial_str[1:]
                            # Offset each letter block and add digits
                            atom.serial = 100000 + (ord(prefix.upper()) - ord('A')) * 10000 + int(digits)
                        
                        # Fallback: hash string to large integer value
                        else:
                            atom.serial = abs(hash(serial_str)) % 1000000
                            print(f"Warning: Using hashed fallback for atom serial: '{serial_str}' → {atom.serial}")
                    
                    # If parsing doesn't work - give dummy ID 
                    except Exception as e:
                        print(f"Warning: Failed to parse atom serial '{serial_str}': {e}")
                        atom.serial = 999999 

                    # Atom name
                    atom.name = line[12:16].strip()
                    
                    # Takes the first character of the atom name and converts it to uppercase
                    atom.element = atom.name[0].upper()
                    
                    # Amino acid
                    atom.resname = line[17:20]
                    
                    # Chain identifier
                    atom.chain = line[21]
                    
                    # Residue sequence number - residue's index w/in chain
                    atom.resseq = int(line[22:26])

                    # Cartesian coordinates of atom in Angstrom
                    atom.x = float(line[30:38])
                    atom.y = float(line[38:46])
                    atom.z = float(line[46:54])
                    
                    # B-factor - how much an atom is displaced due to thermal motion / model uncertainty
                    atom.bfactor = float(line[60:65])
                    
                    # Occupancy of atom - used for that atoms have partial or alternative positions
                    atom.occupancy = float(line[54:59])
                    
                    # Assign atomic number
                    atom.atomic_number = a_attribute[atom.name[0]]["atomic_number"]
                    self.atoms.append(atom)
        
        # Number of atoms
        self.num_atoms = len(self.atoms)

        # Used to suppress redundant prints to terminal
        if not quiet:
            print("Read number of atoms from the PDB: %d" % self.num_atoms)

            # Print list of unique elements
            self.list_elements()

            # Shift the coordinates according to the gravity center of the molecule
            print("Shift the coordinates according to the gravity center of the molecule.")
        
        # Shift regardless of printing
        self.shift_2origin()

    def write_pdb(self):
        with open(self.filename, "w") as f:
            for atom in self.atoms:
                f.write("ATOM  %5d  %-4s %-3s %-1s %4d    %8.3f%8.3f%8.3f\n" %
                        (atom.serial, atom.name, atom.resname, atom.chain, atom.resseq, atom.x, atom.y, atom.z))
    
    def shift_2origin(self):
        grvt_x = 0.0
        grvt_y = 0.0
        grvt_z = 0.0
        
        # Sum the coordinates of all atoms
        for i in range(self.num_atoms):
            grvt_x += self.atoms[i].x / self.num_atoms
            grvt_y += self.atoms[i].y / self.num_atoms
            grvt_z += self.atoms[i].z / self.num_atoms

        for a in self.atoms:
            a.x = a.x - grvt_x
            a.y = a.y - grvt_y
            a.z = a.z - grvt_z

    def list_elements(self):
        elements = set()
        for atom in self.atoms:
            elements.add(atom.element)
        print("Unique elements in the PDB file:", sorted(elements))
        return elements
    

    # Function to orient the specimen's longest axis along the z-axis
    def orient_thickest_to_z(self, verbose=True):
        """
        Align the longest axis (greatest variance) of the molecule to the Z-axis using PCA.
        Rotates coordinates in-place.
        """
        # Extract atomic coordinates
        coords = np.array([[atom.x, atom.y, atom.z] for atom in self.atoms])
        
        # Mean-center the coordinates
        mean_center = coords.mean(axis=0)
        centered = coords - mean_center

        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(centered)
        principal_axes = pca.components_  # Rows are PC1, PC2, PC3

        # Rotate PC1 to Z-axis
        z_axis = np.array([0, 0, 1])
        longest_axis = principal_axes[0]
        v = np.cross(longest_axis, z_axis)
        c = np.dot(longest_axis, z_axis)
        s = np.linalg.norm(v)

        if s == 0:
            if verbose:
                print("Structure already aligned with Z-axis.")
            return

        # Rodrigues' rotation formula
        vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        # Apply rotation
        rotated = centered @ R.T + mean_center

        # Update atom positions
        for atom, new_coord in zip(self.atoms, rotated):
            atom.x, atom.y, atom.z = new_coord

        if verbose:
            print("Oriented molecule so its longest dimension is aligned with the Z-axis.")

    # Function to write rotated coordinates to a new pdb file
    def write_pdb_as(self, new_filename):
        """
        Writes the atoms to a new PDB file with strict column formatting.
        """
        with open(new_filename, "w") as f:
            for atom in self.atoms:
                f.write(
                    "ATOM  {serial:>5} {name:<4}{resname:>3} {chain:1}"
                    "{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    "{occupancy:6.2f}{bfactor:6.2f}          {element:>2}\n".format(
                        serial=atom.serial,
                        name=atom.name.ljust(4),      # atom name, padded right if short
                        resname=atom.resname,
                        chain=atom.chain,
                        resseq=atom.resseq,
                        x=atom.x,
                        y=atom.y,
                        z=atom.z,
                        occupancy=getattr(atom, "occupancy", 1.00),
                        bfactor=getattr(atom, "bfactor", 20.00),
                        element=atom.element.strip().rjust(2)
                    )
                )
            f.write("END\n")


class Atom:
    def __init__(self):
        self.serial = None
        self.name = None
        self.element = None  
        self.resname = None
        self.chain = None
        self.resseq = None
        self.x = None
        self.y = None
        self.z = None
        # self.bfactor = None
        # self.occupancy = None
        self.atomic_number = None

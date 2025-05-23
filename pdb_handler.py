#!/usr/bin/env python3
# PDB file handling
#
# Po-Lin Chiu   - 11/13/2023
#


a_attribute = {
    "H":   {"name": "hydrogen",    "atomic_number":   1},
    "C":   {"name": "carbon",      "atomic_number":   6},
    "N":   {"name": "nitrogen",    "atomic_number":   7},
    "O":   {"name": "oxygen",      "atomic_number":   8},
    "P":   {"name": "phosphorus",  "atomic_number":  15}, 
    "S":   {"name": "sulfur",      "atomic_number":  16}
}


class PDB:
    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.num_atoms = 0

    def read_pdb(self):
        with open(self.filename, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom = Atom()
                    atom.serial = int(line[6:11])
                    atom.name = line[12:16].strip()
                    atom.element = atom.name[0].upper()
                    atom.resname = line[17:20]
                    atom.chain = line[21]
                    atom.resseq = int(line[22:26])
                    atom.x = float(line[30:38])
                    atom.y = float(line[38:46])
                    atom.z = float(line[46:54])
                    # atom.bfactor = float(line[60:65])
                    # atom.occupancy = float(line[54:59])

                    atom.atomic_number = a_attribute[atom.name[0]]["atomic_number"]
                    self.atoms.append(atom)
        
        # Number of atoms
        self.num_atoms = len(self.atoms)
        print("Read number of atoms from the PDB: %d" % self.num_atoms)

        # Shift the coordinates according to the gravity center of the molecule
        print("Shift the coordinates according to the gravity center of the molecule.")
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

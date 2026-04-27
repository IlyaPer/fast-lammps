from typing import Any

import lammps
import numpy as np

class LammpsExtractor():
    def __init__(self, lammps_instance) -> None:
        self.lammps_instance = lammps_instance
        self.nlocal = lammps_instance.extract_global("nlocal")

    def __get_positions__(self,) -> np.ndarray:
        raw_pos = self.lammps_instance.numpy.extract_atom("x")[:self.nlocal]
        return raw_pos
    
    def __get_atom_types__(self,) -> np.ndarray:
        raw_types = self.lammps_instance.numpy.extract_atom("type")[:self.nlocal]
        return raw_types
    
    def __get_velocities__(self,) -> np.ndarray:
        raw_velocities = self.lammps_instance.numpy.extract_atom("v")[:self.nlocal]
        return raw_velocities
    
    def __get_atom_identificators__(self,) -> np.ndarray:
        raw_ids = self.lammps_instance.numpy.extract_atom("id")[:self.nlocal]
        return raw_ids
    
    def __get_box_size__(self,) -> tuple:
        box = self.lammps_instance.extract_box()
        xlo, xhi = box[0][0], box[1][0]
        ylo, yhi = box[0][1], box[1][1]
        zlo, zhi = box[0][2], box[1][2]
        return xlo, xhi, ylo, yhi, zlo, zhi
    
    def __get_pe_per_atom__(self,) -> tuple:
       return self.lammps_instance.numpy.extract_compute('pe_atom', 1, 1)
    
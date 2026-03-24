from abc import ABC, abstractmethod
import numpy as np
import logging
from ase import Atoms
from ase.io import read
from ase import units
import os
from ase.io import write

logger = logging.getLogger(__name__)


class Extractor(ABC):
    @abstractmethod
    def extract_interesting_regions(self) -> list:
        """Return list of masks with indexes to change"""
        pass

    @abstractmethod
    def check_condition_of_region(self, velocities, masses) -> bool:
        """Check condition of region"""
        pass

    @abstractmethod
    def visualize_interesting_regions(self, positions):
        """Visualizes regions extracted from extract_interesting_regions method"""
        pass


class ExampleLayerExtractor(Extractor):
    def __init__(self):
        super().__init__()

    def check_condition_of_region(self, velocities, masses, threshold=20):

        v_group = velocities
        m_group = masses

        total_mass = np.sum(m_group)
        v_com = np.sum(v_group * m_group[:, None], axis=0) / total_mass

        # Relative velocity (internal motion only)
        v_rel = v_group - v_com

        # Kinetic Energy = 0.5 * m * v^2
        AMU_A2PS2_TO_EV = 1.0364269e-4
        n_atoms = v_group.shape[0]

        ke = 0.5 * np.sum(m_group * np.sum(v_rel**2, axis=1)) * AMU_A2PS2_TO_EV

        current_T = 2 * ke / (3 * n_atoms * units.kB)

        if n_atoms > 0:
            current_T = 2 * ke / (3 * n_atoms * units.kB)
            layer_temp = current_T
        else:
            layer_temp = 0.0

        logging.info(
            f"Количество атомов в слое: {len(m_group)}, температура слоя: {layer_temp}"
        )

        if (layer_temp > threshold):
            return True

        return False

    def extract_interesting_regions(
        self, coordinates, velocities, masses, lattice_constant, lattice_constant_cg, criteria="temp"
    ):

        z = coordinates[:, 2]
        step = lattice_constant*2 + 1e-1
        logging.info(f"Using FCC layer spacing: step = {step}")

        zmax = 40.08
        zmin = z.min()

        layers = []
        logging.error(f"================vvvvvvvvvvvvvvvvvvv===========================")

        for i in range(int((zmax-zmin)//step)+1):
            upper = zmax - i * step
            lower = upper - step

            if lower < zmin:
                lower = zmin

            logging.info(f"LAYER {i}, Collecting atoms from: {lower} to {upper}")

            mask = (z >= lower) & (z <= upper)
            actual_atoms = coordinates[mask]
            
            if np.any(masses[mask] > 200):
                logging.info(f"SKIPPED ALREADY GRAINED ATOMS: min: {masses.min()}, max: {masses.max()}")
                continue


            if not self.check_condition_of_region(velocities[mask], masses[mask], threshold=10):
                continue

            from ase.build import bulk

            au = bulk("Au", "fcc", a=lattice_constant_cg, cubic=True)
            target_atoms = len(actual_atoms) / (4**3)

            target_cells = target_atoms / 4.0
            n = int(round(target_cells))

            nx = int(round(n**0.5))
            if nx < 1:
                nx = 1
            ny = int(round(n / nx))
            if ny < 1:
                ny = 1

            au = bulk("Au", "fcc", a=lattice_constant_cg, cubic=True)
            plane = au.repeat((3, 4, 1))
            positions_of_grained = plane.get_positions()
            mask_bebe = positions_of_grained[:, 2] < lattice_constant_cg / 2

            plane = plane[mask_bebe]
            positions_of_grained = plane.get_positions()
            positions_of_grained[:,2] += (lower+upper)/2


            layers.append((mask, positions_of_grained))
        return layers
    

    def visualize_interesting_regions(self, positions, mode='test'):
        pass
        # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])

class FccCellsExtractor(Extractor):
    def __init__(self):
        super().__init__()

    def check_condition_of_region(self, velocities, masses, threshold=10):

        v_group = velocities
        m_group = masses

        total_mass = np.sum(m_group)
        v_com = np.sum(v_group * m_group[:, None], axis=0) / total_mass

        # Relative velocity (internal motion only)
        v_rel = v_group - v_com

        # Kinetic Energy = 0.5 * m * v^2
        AMU_A2PS2_TO_EV = 1.0364269e-4
        n_atoms = v_group.shape[0]

        ke = 0.5 * np.sum(m_group * np.sum(v_rel**2, axis=1)) * AMU_A2PS2_TO_EV

        current_T = 2 * ke / (3 * n_atoms * units.kB)

        if n_atoms > 0:
            current_T = 2 * ke / (3 * n_atoms * units.kB)
            layer_temp = current_T
        else:
            layer_temp = 0.0

        if (layer_temp > threshold):
            return True

        return False

    def extract_interesting_regions(
        self, lammps_instance
    ):
        """
        Extracts fcc cells from the simulation. Extractor is resistant to fluctuations. The algorithm is described below.
        1) It captures atoms which are already seems like a part of fcc cells. This part is made by Common Neighbour Analysis by LAMMPS itself.
        2) Atoms which seems like fcc are passed to clusterization K-means with fixed number of atoms in one cluster. If a cluster contains less atoms than 12 - it is skipped??
        
        Function returns list of tuple objects. The first member of tuple is list of ids of atoms to be changed, the second member is postion of atoms which are going to approximate target atom.

        :param lammps_instance: the lammps object.
        :type arg1: lammps class
        :param arg2: The second argument.
        :type arg2: str
        :returns: The result of the function.
        :rtype: bool
        :raises SomeException: If a certain condition occurs.
        """

        natoms = lammps.get_natoms(lammps_instance)

        compute 1 all cna/atom 3.08

        data = lmp.extract_compute("1", 1, 1)

        nlocal = L.extract_global("nlocal") # the number of atoms owned by the current processor in a parallel simulation
        raw_ids = L.numpy.extract_atom("id")[:nlocal] 
        raw_pos = L.numpy.extract_atom("x")[:nlocal] 
        raw_vel = L.numpy.extract_atom("v")[:nlocal]
        atom_types = L.numpy.extract_atom("type")[:nlocal]
        masses_types = L.numpy.extract_atom("mass")

        #Clustering with fixed number of clusters?
        return
    

    def visualize_interesting_regions(self, positions, mode='test'):
        pass
        # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])
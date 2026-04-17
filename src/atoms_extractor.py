from abc import ABC, abstractmethod
import numpy as np
import logging
from ase import Atoms
from ase.io import read
from ase import units
import os
from ase.io import write
from sklearn.cluster import MeanShift
from scipy.ndimage import center_of_mass
from ovito.io.lammps import lammps_to_ovito
from ovito.modifiers import CommonNeighborAnalysisModifier
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from k_means_constrained import KMeansConstrained
from ase.build import bulk


logger = logging.getLogger(__name__)


class Extractor(ABC):
    @abstractmethod
    def extract_interesting_regions(self) -> list:
        """Return list of masks with indexes to change"""
        pass

    @abstractmethod
    def check_condition_of_region(self, *_) -> bool:
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

    def visualize_interesting_regions(
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
    

    # def visualize_interesting_regions(self, positions, mode='test'):
    #     pass
        # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])

class FccCellsExtractor(Extractor):
    def __init__(self):
        super().__init__()

    import numpy as np

    def _init_fcc_cells_strict(self, lammps_instance, lattice_constant, expected_atoms_per_cell=32):

        import numpy as np
        
        nlocal = lammps_instance.extract_global("nlocal")
        raw_ids = lammps_instance.numpy.extract_atom("id")[:nlocal]
        raw_pos = lammps_instance.numpy.extract_atom("x")[:nlocal]  # (N, 3)
        
        box = lammps_instance.extract_box()
        xlo, xhi = box[0][0], box[1][0]
        ylo, yhi = box[0][1], box[1][1]
        zlo, zhi = box[0][2], box[1][2]
        
        cell_size = lattice_constant * 2.0 # 2×2×2 FCC = 32 атома
        
        nx = int(np.floor((xhi - xlo) / cell_size))
        ny = int(np.floor((yhi - ylo) / cell_size))
        nz = int(np.floor((zhi - zlo) / cell_size))
        
        used = np.zeros(nlocal, dtype=bool)
        
        mega_cells = {}
        res = []

        box = lammps_instance.extract_box()
        xlo, xhi = box[0][0], box[1][0]
        ylo, yhi = box[0][1], box[1][1]
        zlo, zhi = box[0][2], box[1][2]

        nx = int((xhi - xlo) // cell_size)
        ny = int((yhi - ylo) // cell_size)
        nz = int((zhi - zlo) // cell_size)

        cells = {}

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):

                    x_min = xlo + ix * cell_size
                    x_max = x_min + cell_size

                    y_min = ylo + iy * cell_size
                    y_max = y_min + cell_size

                    z_min = zlo + iz * cell_size
                    z_max = z_min + cell_size

                    cells[(ix, iy, iz)] = (x_min, x_max, y_min, y_max, z_min, z_max)

        return cells

    def visualize_cells(self, lammps_instance, fcc_cells, cell_size=4.08*4, output_prefix='cell', output_dir='./shit'):

        if output_dir is None:
            output_dir = './'
        os.makedirs(output_dir, exist_ok=True)
        # output_dir = './shit'
        
        nlocal = lammps_instance.extract_global("nlocal")
        raw_ids = lammps_instance.numpy.extract_atom("id")[:nlocal]
        raw_pos = lammps_instance.numpy.extract_atom("x")[:nlocal]
        raw_types = lammps_instance.numpy.extract_atom("type")[:nlocal]
        
        symbols = ['Au' for _ in raw_types]
        id_to_idx = {int(raw_ids[i]): i for i in range(nlocal)}
        
        all_atoms_data = []
        cell_counter = 0
        
        for key, ids in fcc_cells.items():
            if key == 'lost':
                continue
            indices = [id_to_idx[int(aid)] for aid in ids if int(aid) in id_to_idx]
            if not indices:
                continue
            positions = raw_pos[indices]
            symbols_cell = [symbols[i] for i in indices]
            atoms_cell = Atoms(symbols=symbols_cell, positions=positions)
            fname = os.path.join(output_dir, f"{output_prefix}_{key[0]}_{key[1]}_{key[2]}.xyz")
            write(fname, atoms_cell)
            cell_label = f"{key[0]},{key[1]},{key[2]}"
            for pos, sym in zip(positions, symbols_cell):
                all_atoms_data.append((pos, sym, cell_label))
            cell_counter += 1
        
        if 'lost' in fcc_cells and fcc_cells['lost']:
            lost_ids = fcc_cells['lost']
            lost_indices = [id_to_idx[int(aid)] for aid in lost_ids if int(aid) in id_to_idx]
            if lost_indices:
                lost_positions = raw_pos[lost_indices]
                lost_symbols = [symbols[i] for i in lost_indices]
                for pos, sym in zip(lost_positions, lost_symbols):
                    all_atoms_data.append((pos, sym, 'lost'))
        
        if all_atoms_data:
            all_positions = np.array([d[0] for d in all_atoms_data])
            all_symbols = [d[1] for d in all_atoms_data]
            all_labels = [d[2] for d in all_atoms_data]
            unique_labels = sorted(set(all_labels))
            label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
            cell_numbers = np.array([label_to_int[lab] for lab in all_labels])
            all_atoms = Atoms(symbols=all_symbols, positions=all_positions)
            all_atoms.arrays['cell_id'] = cell_numbers
            write(os.path.join(output_dir, f"{output_prefix}_all.extxyz"), all_atoms, format='extxyz')
            print(f"[{output_prefix}] Создано {cell_counter} файлов ячеек + all_cells в {output_dir}")
        return output_dir

    def _rand_condition(self, threshold=0.1):

        res = False
        import random

        number = random.random()

        if number < threshold:
            res = True

        return res

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
        self, lammps_instance, lattice_constant, lattice_constant_cg, mass_cg
    ):
        """
        Extracts fcc cells from the simulation. Extractor is resistant to fluctuations. The algorithm is described below.
        1) It captures atoms which are already seems like a part of fcc cells. This part is made by Common Neighbour Analysis by LAMMPS itself.
        2) Atoms which seems like fcc are passed to clusterization K-means with fixed number of atoms in one cluster. If a cluster contains less atoms than 12 - it is skipped??
        
        Function returns list of tuple objects. The first member of tuple is list of ids of atoms to be changed, the second member is postion of atoms which are going to approximate target atom.

        :param lammps_instance: the lammps object.
        :type arg1: lammps class
        :param lattice_constant: Lattice constant which wiil be used as cuttof in CNA and Mean Shift Clustering.
        :type lattice_constant: float
        """

        nlocal = lammps_instance.extract_global("nlocal") # the number of atoms owned by the current processor in a parallel simulation
        raw_ids = lammps_instance.numpy.extract_atom("id")[:nlocal] 
        raw_pos = lammps_instance.numpy.extract_atom("x")[:nlocal] 
        raw_vel = lammps_instance.numpy.extract_atom("v")[:nlocal]
        atom_types = lammps_instance.numpy.extract_atom("type")[:nlocal]
        masses_types = lammps_instance.numpy.extract_atom("mass")
        
        box = lammps_instance.extract_box()
        xlo, xhi = box[0][0], box[1][0]
        ylo, yhi = box[0][1], box[1][1]
        zlo, zhi = box[0][2], box[1][2]

        cell_size = lattice_constant * 2.0

        x_secants = np.arange(xlo, xhi, cell_size)
        y_secants = np.arange(ylo, yhi, cell_size)
        z_secants = np.arange(zlo, zhi, cell_size)

        lammps_instance.command(f"group type1 type 1")

        not_grained = np.where(atom_types == 1)[0]
        logging.info(f'not_grained: {len(not_grained)}, total: {len(atom_types)}')
        fcc_atoms_identificators = not_grained
        already_grained_ids = np.where(atom_types == 2)[0]

        positions = raw_pos[fcc_atoms_identificators]

<<<<<<< HEAD
=======

>>>>>>> 4e67273319cfb7043a360fe1b7d68e836049dcf7
        mega_cells = self._init_fcc_cells_strict(lammps_instance=lammps_instance, lattice_constant=lattice_constant, expected_atoms_per_cell=32)

        import random
        items = list(mega_cells.items())
        # random.shuffle(items)
        logging.error(len(items))

        types = lammps_instance.numpy.extract_atom("type")
        pos = lammps_instance.numpy.extract_atom("x")

        for key, value in items:                
            x_min, x_max, y_min, y_max, z_min, z_max = value
            dx = x_max - x_min
            dy = y_max - y_min
            dz = z_max - z_min

            x_min = max(x_min, xlo)
            x_max = min(x_max, xhi)
            y_min = max(y_min, ylo)
            y_max = min(y_max, yhi)
            z_min = max(z_min, zlo)
            z_max = min(z_max, zhi)

            if dx > 10 or dy > 10 or dz > 10:
                logging.error(f'x_min, x_max, y_min, y_max, z_min, z_max {x_min, x_max, y_min, y_max, z_min, z_max}')
                logging.error(f'{x_min} {x_max} {y_min} {y_max} {z_min} {z_max}')
                continue
            lammps_instance.command(f"region kill block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box")
            # lammps_instance.command("group kill_region region kill")

            if self._rand_condition(0.25):
                

                in_region = (
                    (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
                    (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max) &
                    (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max)
                )

                count_type1 = np.sum((types == 1) & in_region)
                count_type2 = np.sum((types == 2) & in_region)

                total = count_type1 + count_type2

                if total == 0:
                    is_cg = False
                else:
                    is_cg = count_type2 > 0

                if is_cg:
                    typeatom = 1
                    lammps_instance.command(f"lattice fcc {lattice_constant}")
                else:
                    lammps_instance.command(f"lattice fcc {lattice_constant_cg-0.05}")
                    typeatom = 2

                lammps_instance.command(f'delete_atoms region kill')
                lammps_instance.command(f'create_atoms {typeatom} region kill')
                lammps_instance.command(f"region kill delete")
                continue

            lammps_instance.command(f"region kill delete")

        if len(positions) == 0:
            return  
        logging.info(f'collecting fcc: {len(mega_cells)}')
        noise = np.random.normal(0, .1, size=positions.shape)
    
        positions += noise

        #  TODO set velocities!
        return
    

    def visualize_interesting_regions(self, positions, mode='test'):
        pass
        # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])
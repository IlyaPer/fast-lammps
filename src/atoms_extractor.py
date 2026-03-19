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

        logging.info(
            f"Количество атомов в слое: {len(m_group)}, температура слоя: {layer_temp}"
        )

        if (layer_temp > threshold):
            return True

        return False

    def extract_interesting_regions(
        self, coordinates, velocities, masses, lattice_constant, criteria="temp"
    ):

        z = coordinates[:, 2]
        logging.info(f"Positions: from {z.min()} to {z.max()}")
        step = lattice_constant / 2
        logging.info(f"Using FCC layer spacing: step = {step}")

        zmax = z.max()
        zmin = z.min()

        layers = []

        for i in range(int((zmax-zmin)//step)):
            upper = zmax - i * step
            lower = upper - step

            if lower < zmin:
                lower = zmin

            logging.info(f"LAYER {i}, Collecting atoms from: {lower} to {upper}")

            mask = (z >= lower) & (z < upper)
            # mask = z[mask_local]

            if len(mask) < 10:
                logging.warning(f"Layer {i} too small ({len(mask)} atoms), skipping")
                continue

            if len(mask) > 0.9 * len(z):
                logging.error(f"Layer {i} ~ entire box, skipping")
                continue

            if not self.check_condition_of_region(coordinates[mask], velocities[mask], masses[mask]):
                continue

            boundaries_of_the_box =     (
                    coordinates[mask][:, 0].min(),
                    coordinates[mask][:, 0].max(),
                    coordinates[mask][:, 1].min(),
                    coordinates[mask][:, 1].max(),
                    coordinates[mask][:, 2].min(),
                    coordinates[mask][:, 2].max,
            )

            layers.append((mask, boundaries_of_the_box))

        if layers:
            logging.info(f"FIRST LAYER SAMPLE: {layers[0][:10]}")
            logging.info(f"Layer sizes: {[len(l) for l in layers]}")

        return layers

    def visualize_interesting_regions(self, positions, mode='test'):
        pass
        # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])


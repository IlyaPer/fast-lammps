from lammps import lammps
import numpy as np
import argparse
import logging
import src.atoms_extractor as ae

parser = argparse.ArgumentParser(
    prog="Heating_Aurum",
    description="heating aurum",
    epilog="Text at the bottom of help",
)

parser.add_argument("-k", "--scale")
parser.add_argument("-i", "--iteration")
parser.add_argument("-m", "--measure_frequency")
parser.add_argument("-l", "--log_interval")
parser.add_argument("-d", "--delete_layer")

args = parser.parse_args()

logging.basicConfig(
    filename="logs/dynamic_coarse_graining.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SCALE_FACTOR = int(args.scale)
iteration = int(args.iteration)
measure_frequency = int(args.measure_frequency)

sigma_for_gold = 4.08 * (3.6 / 2.33) ** (-1)
SIGMA = sigma_for_gold  # Scaling for sigma
A = SIGMA * (3.6 / 2.33)


def compute_params_CG(scale_factor):
    sigma_for_gold = 4.08 * (3.6 / 2.33) ** (-1)
    SIGMA = sigma_for_gold  # Scaling for sigma

    ATOMIC_UNIT_MASS = 196.196 * 3  # Scaling for masses
    EPSILON = 0.4  # * (scale_factor**3) # Scaling for masses

    A = SIGMA * (3.6 / 2.33) * scale_factor
    return SIGMA, A, EPSILON, ATOMIC_UNIT_MASS

L = lammps()
solver = ae.ExampleLayerExtractor()

SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(SCALE_FACTOR)

logging.info(
    f"Sigma {SIGMA_CG}, Lattice {A_CG}, eps {EPSILON_CG}, Mass {ATOMIC_UNIT_MASS_CG}"
)

L.file('heat_aurum.in') # setup only, no dynamics run


block = f"""
pair_coeff      1 2 {EPSILON_CG} {SIGMA_CG}
pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
mass            2 {ATOMIC_UNIT_MASS_CG}
lattice         fcc {A_CG}
"""

L.commands_string(block)


logging.info("The simulation started successfully.")
iter = 0


while iter < iteration:
    logging.info(
        f"============================= STEP NUMBER {iter} ============================= "
    )

    L.command("reset_atoms id sort yes")
    L.cmd.run(measure_frequency)

    natoms = lammps.get_natoms(L)

    nlocal = L.extract_global("nlocal") # the number of atoms owned by the current processor in a parallel simulation
    raw_ids = L.numpy.extract_atom("id")[:nlocal] 
    raw_pos = L.numpy.extract_atom("x")[:nlocal] 
    raw_vel = L.numpy.extract_atom("v")[:nlocal] 


    idx = np.argsort(raw_ids) # get sorted by id bunch of atoms
    positions = raw_pos[idx]
    velocities = raw_vel[idx]
    natoms = nlocal

    masses = np.repeat([196.196], natoms)

    masks_to_grain = solver.extract_interesting_regions(
        positions,velocities,masses,lattice_constant=A
    )

    if (len(masks_to_grain) == 0) and (iter > 10000):
        logging.info(f"GRAINING OF ALL LAYERS IS COMPLETE. STOP.")
        break

    for i, tuple_info in enumerate(masks_to_grain):
        mask, coordinates_of_region = tuple_info

        string_of_ids = " ".join(map(str, mask))

        L.command(f"group delete_group id {string_of_ids}")
        L.command(f"delete_atoms group delete_group")
        L.command(f"group delete_group delete")
        L.command("reset_atoms id")

        coordinates_of_region_string = " ".join(map(str, coordinates_of_region[i]))
        logging.warning(f"Coordinates fo: {coordinates_of_region_string}")
        L.command(f"lattice fcc {A_CG}")
        L.command(f"region temp block {coordinates_of_region_string} units box")
        L.command(f"create_atoms 2 region temp")
        L.command(f"region temp delete")
        L.command("reset_atoms id")
        break

    L.command("run 0")
    iter += measure_frequency

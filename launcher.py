from lammps import lammps
import numpy as np
import argparse
import logging
import src.atoms_extractor as ae
from src.simulation_launch import compute_params_CG
from src.monitor import MemoryProfiler, create_beautiful_plot
import os

parser = argparse.ArgumentParser(
    prog="Heating_Aurum",
    description="heating aurum",
    epilog="Text at the bottom of help",
)

parser.add_argument("-f", "--file")
parser.add_argument("-k", "--scale")
parser.add_argument("-i", "--iteration")
parser.add_argument("-m", "--measure_frequency")
parser.add_argument("-l", "--log_interval")
parser.add_argument("-d", "--delete_layer")
parser.add_argument("--experiment")
parser.add_argument("--solver")

args = parser.parse_args()

logging.basicConfig(
    filename="logs/dynamic_coarse_graining.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SCALE_FACTOR = int(args.scale) if args.solver == 'layer' else 1
iteration = int(args.iteration)
measure_frequency = int(args.measure_frequency)

sigma_for_gold = 4.08 * (3.6 / 2.33) ** (-1)
SIGMA = sigma_for_gold  # Scaling for sigma
A = SIGMA * (3.6 / 2.33)

SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(SCALE_FACTOR)

logging.info(
    f"Sigma {SIGMA_CG}, Lattice {A_CG}, eps {EPSILON_CG}, Mass {ATOMIC_UNIT_MASS_CG}"
)


L = lammps()

if args.solver == 'layer':
    solver = ae.ExampleLayerExtractor()
if args.solver == 'fcc':
    solver = ae.FccCellsExtractor()

# setup from user
L.file(args.file)

block = f"""
pair_coeff      1 2 0.0 1.0
pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
mass            2 {ATOMIC_UNIT_MASS_CG}
lattice         fcc {A_CG}
# change_box      all z scale 1.2
"""

L.commands_string(block)


logging.info("The simulation started successfully.")
iter = 0


# TODO add Prometheus checkup!

with MemoryProfiler(name="my_script_analysis", track_objects=True, snapshot_interval=20) as profiler:
    while iter < iteration:
        profiler.snapshot(iteration=iter, label=f"step_{iter}")

        L.cmd.run(measure_frequency)

        solver.extract_interesting_regions(L, lattice_constant=A)

        L.command("reset_atoms id")

        L.command("run 0")
        iter += measure_frequency


import glob
metric_files = glob.glob("logs/metrics_*.json")
if metric_files:
    latest_file = max(metric_files, key=os.path.getctime)
    create_beautiful_plot(latest_file, "records/memory_analysis.png")
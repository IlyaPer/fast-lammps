from lammps import lammps
import numpy as np
import argparse
import logging
from src.extractors.extractors import *
from src.utils.approximation import compute_params_CG
from src.utils.utils import LammpsExtractor
from src.modifiers.changer import DynamicChanger
from build.monitor import MemoryProfiler, create_beautiful_plot
import os
# from src.simulation_launch import CrackLayerAnalyzer

parser = argparse.ArgumentParser(
    prog="Heating_Aurum",
    description="heating aurum",
    epilog="Text at the bottom of help",
)

parser.add_argument("-f", "--file")
parser.add_argument("-s", "--scale", type=int, default=2)
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

SCALE_FACTOR = int(args.scale)
iteration = int(args.iteration)
measure_frequency = int(args.measure_frequency)

sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)  # ~0.097
SIGMA = sigma_for_nickel
A = 3.52

SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(SCALE_FACTOR)

logging.info(
    f"Sigma {SIGMA_CG}, Lattice {A_CG}, eps {EPSILON_CG}, Mass {ATOMIC_UNIT_MASS_CG}"
)
block = f'kkkk'

L = lammps()


if args.solver == 'layer':
    pass
    # solver = ExampleLayerExtractor()
if args.solver == 'fcc':
    pass
    # solver = FccCellsExtractor(L,A)

# setup from user
L.file(args.file)

communicator = LammpsExtractor(L)
solver = FccCellsExtractor(communicator, A)
dc = DynamicChanger(solver, A, A_CG, baby_mode=True)


block = f"""
pair_coeff      1 2 20 2.28
pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
mass            2 {ATOMIC_UNIT_MASS_CG}
lattice         fcc {A_CG}
"""

L.commands_string(block)

logging.info("The simulation started successfully.")
iter = 0

with MemoryProfiler(name="accelerate", track_objects=True, snapshot_interval=20) as profiler:
    while iter < iteration:
        profiler.snapshot(iteration=iter, label=f"step_{iter}")

        L.cmd.run(measure_frequency)

       # mean among several snapshots!!! 3-4
       # asssume there are 32 atoms everywhere 
       # Number of snapshots as flag --time_window
        dc.accelerate(L)

        L.command("reset_atoms id")

        L.command("run 0")
        iter += measure_frequency
        logging.info(f'Current iteration: {iter}')


import glob
metric_files = glob.glob("logs/metrics_*.json")
if metric_files:
    latest_file = max(metric_files, key=os.path.getctime)
    create_beautiful_plot(latest_file, "logs/memory_analysis.png")
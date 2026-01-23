from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import write
import numpy as np
from ase import units
import matplotlib.pyplot as plt
from datetime import date
from ase import Atoms
import time
from ase.md import MDLogger
import argparse
import multidimensional_md as mdmd

parser = argparse.ArgumentParser(
                    prog='Heating_Aurum',
                    description='heating aurum',
                    epilog='Text at the bottom of help')

parser.add_argument('-k', '--scale')
parser.add_argument('-i', '--iteration')
parser.add_argument('-m', '--measure_frequency')
parser.add_argument('-l', '--log_interval')

args = parser.parse_args()

K = int(args.scale)
iteration = int(args.iteration)

measure_frequency = int(args.measure_frequency)
log_interval = int(args.log_interval)

BASIC_NUMBER_ATOMS = 8
NUMBER_OF_ATOMS= BASIC_NUMBER_ATOMS // K # Scaling the size!!! Basic number of atoms is 16.


sigma_for_gold = 4.08 * (3.6/2.33)**(-1)
SIGMA= K * sigma_for_gold # Scaling for sigma

ATOMIC_UNIT_MASS = 196.196 * (K**3) # Scaling for masses
EPSILON=0.4 * (K**3) # Scaling for masses
n_layers = NUMBER_OF_ATOMS


A = SIGMA * (3.6/2.33)

au = bulk("Au", "fcc", a=A, cubic=True)

cube = au.repeat((NUMBER_OF_ATOMS, NUMBER_OF_ATOMS, NUMBER_OF_ATOMS))

cube.pbc = (True, True, False)

print(f"Для K={K} vs K=1:")
print(f"Объём: {(A ** 3) * (NUMBER_OF_ATOMS**3)} == {(4.08 ** 3)* (BASIC_NUMBER_ATOMS**3)}")
print(f"Масса: {len(cube) * ATOMIC_UNIT_MASS}  ==  {((BASIC_NUMBER_ATOMS)**3)*4 * 196.196}")
print(f"Количество атомов (должно отличаться): {len(cube)} != {((BASIC_NUMBER_ATOMS)**3)*4}")

write("init_cube.xyz", cube)

cube.calc = (
    LennardJones(sigma=SIGMA, epsilon=EPSILON)
)
layer_thikness = A

cube.set_masses(np.repeat([ATOMIC_UNIT_MASS], len(cube.get_masses()))) # Setting masses of the atoms

pos = cube.get_positions()
all_z = pos[:, 2]
min_z = np.min(all_z)


MASKS_OF_LAYERS = [
    (all_z >= min_z + i*layer_thikness) & (all_z < min_z + (i+1)*layer_thikness)
    for i in range(n_layers)
]

iteration_count = 0

def compute_layer_temperatures_smooth(trajectory, atoms=cube):
    temps = []
    T_atom = np.zeros(len(atoms))
    global iteration_count
    iteration_count += 1

    for i, mask in enumerate(MASKS_OF_LAYERS):
        temp_slice = atoms[mask]

        layer_temp = temp_slice.get_temperature()  # K
        temps.append(layer_temp)

        # всем атомам слоя присваиваем температуру слоя
        T_atom[mask] = layer_temp

    if np.all(np.array(temps) > 250.):
        print(f'\033[91mAll layers > 250 K — stopping after {iteration_count * measure_frequency} steps.\033[0m')
        raise SystemExit

    atoms.set_array("Temperature", T_atom)
    trajectory.append(atoms.copy())
    return temps

def plot_layer_temperatures(atoms=cube):
    temps = []
    z_positions = []

    for i, mask in enumerate(MASKS_OF_LAYERS):
        temp_slice = atoms[mask]
        layer_temp = temp_slice.get_temperature()
        temps.append(layer_temp)
        z_positions.append(min_z + i*layer_thikness)

    return temps

trajectory = []


thermostat = mdmd.MultiGroupLangevinMD(cube,
                                       groups = {'bottom': np.where(MASKS_OF_LAYERS[0])[0],
                                                 'rest': np.where(~MASKS_OF_LAYERS[0])[0]},
                                       temps={'bottom': 300,
                                              'rest': 0},
                                       timestep=5 * units.fs,
                                       friction=0.01 / units.fs)

thermostat.attach(MDLogger(thermostat, cube, f'logs/md_{K}.log', header=True, stress=False, peratom=True, mode="w"), interval=log_interval)


thermostat.attach(
    lambda: compute_layer_temperatures_smooth(trajectory, cube),
    interval=measure_frequency,
)

today = date.today()
curr_time = time.strftime("%H_%M")


try:
    thermostat.run(iteration)
except SystemExit:
    print()
write(f"records/cube_iteration_scale_{K}_{curr_time}_day_{today.day}_.extxyz", trajectory, format="extxyz")
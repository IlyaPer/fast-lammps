from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import write
import numpy as np
from ase import units
from datetime import date
import time
from ase.md import MDLogger
import argparse
import multidimensional_md as mdmd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

parser = argparse.ArgumentParser(
                    prog='Heating_Aurum',
                    description='heating aurum',
                    epilog='Text at the bottom of help')

parser.add_argument('-k', '--scale')
parser.add_argument('-i', '--iteration')
parser.add_argument('-m', '--measure_frequency')
parser.add_argument('-l', '--log_interval')
parser.add_argument('-d', '--delete_layer')

args = parser.parse_args()

K = int(args.scale)
iteration = int(args.iteration)

measure_frequency = int(args.measure_frequency)
log_interval = int(args.log_interval)

BASIC_NUMBER_CELLS = 8
NUMBER_OF_CELLS = BASIC_NUMBER_CELLS // K # Scaling the size!!! Basic number of atoms is 16.
from ase.visualize import view

PBC = (True, True, False)

sigma_for_gold = 4.08 * (3.6/2.33)**(-1)
SIGMA= K * sigma_for_gold # Scaling for sigma

ATOMIC_UNIT_MASS = 196.196 * (K**3) # Scaling for masses
EPSILON=0.4 * (K**3) # Scaling for masses
n_layers = NUMBER_OF_CELLS*2
THRESHOLD=250.

A = SIGMA * (3.6/2.33)
au = bulk("Au", "fcc", a=A, cubic=True)

cube = au.repeat((NUMBER_OF_CELLS, NUMBER_OF_CELLS, NUMBER_OF_CELLS))

cube.pbc = PBC



print(f"Для K={K} vs K=1:")
print(f"Объём: {(A ** 3) * (NUMBER_OF_CELLS**3)} == {(4.08 ** 3)* (BASIC_NUMBER_CELLS**3)}")
print(f"Масса: {len(cube) * ATOMIC_UNIT_MASS}  ==  {((BASIC_NUMBER_CELLS)**3)*4 * 196.196}")
print(f"Количество атомов (должно отличаться при k != 1): {len(cube)} != {((BASIC_NUMBER_CELLS)**3)*4}")



cube.calc = (
    LennardJones(epsilon=EPSILON, sigma=SIGMA)
)
layer_thikness = A/2

cube.set_masses(np.repeat([ATOMIC_UNIT_MASS], len(cube.get_masses()))) # Setting masses of the atoms


def create_masks(cube, n_layers, layer_thikness=layer_thikness): #CHECK INDEXING
    pos = cube.get_positions()
    all_z = pos[:, 2].copy()
    min_z = np.min(pos[:, 2])
    max_z = np.max(all_z)

    MASKS_OF_LAYERS = []

    for i in range(n_layers):
        mask = np.isclose(all_z, min_z + layer_thikness * i, atol=layer_thikness-1e-3)
        MASKS_OF_LAYERS.append(mask)
        all_z[mask] = 1e+10

    return MASKS_OF_LAYERS


MASKS_OF_LAYERS = create_masks(cube, n_layers=n_layers, layer_thikness=layer_thikness)
iteration_count = 0

# combmask = MASKS_OF_LAYERS[0]
# for mask in MASKS_OF_LAYERS[1:]:
#     combmask |= mask

# # for k, mask in enumerate(MASKS_OF_LAYERS):

# write(f'init_cube_q.xyz', cube[combmask])

# exit(0) 

class LastlayerLeft(Exception):

    def __init__(self, value, message="Only one layer left. Stopping."):
        self.value = value
        self.message = message
        super().__init__(self.message)

class LayerTooHot(Exception):

    def __init__(self, value, message="Value is too high"):
        self.value = value
        self.message = message
        super().__init__(self.message)


# cube.get_te

def compute_temperatures(trajectory, cube, MASKS_OF_LAYERS, temperatures_by_layer):
    temps = []
    T_atom = np.zeros(len(cube))
    global iteration_count
    iteration_count += 1

    velocities = cube.get_velocities()
    masses = cube.get_masses()

    for mask in MASKS_OF_LAYERS:
        idxs = np.where(mask)[0]
        # layer_temp = cube[idxs].get_temperature()
        # temps.append(layer_temp)

        # T_atom[mask] = layer_temp
        # continue

        v_group = velocities[idxs]
        m_group = masses[idxs][:, None]

        total_mass = np.sum(m_group)
        v_com = np.sum(v_group * m_group, axis=0) / total_mass

        # Relative velocity (internal motion only)
        v_rel = v_group - v_com

        # Kinetic Energy = 0.5 * m * v^2
        ke = 0.5 * np.sum(m_group * v_rel**2)

        # T = 2 * KE / (3 * N * k_B)
        n_atoms = len(idxs)
        if n_atoms > 0:
            current_T = 2 * ke / (3 * n_atoms * units.kB)
            layer_temp = current_T
        else:
            layer_temp = 0.0

        temps.append(layer_temp)

        T_atom[mask] = layer_temp

    temperatures_by_layer.append(temps)

    cube.set_array("Temperature", T_atom)
    trajectory.append(cube.copy())

    if len(MASKS_OF_LAYERS) == 1:
        raise LastlayerLeft(1)
    
    
    temp_highest_group = temps[-1]

    if temp_highest_group > THRESHOLD:
        raise LayerTooHot(cube[~MASKS_OF_LAYERS[-1]].copy())

trajectory = []
temperatures_by_layer = []


groups = {'bottom': np.where(MASKS_OF_LAYERS[0])[0]}
temps = {'bottom': 300}

thermostat = mdmd.MultiGroupLangevinMD(cube,
                                           groups=groups,
                                           temps=temps,
                                           timestep=2 * units.fs,
                                           friction=0.01 / units.fs)

thermostat.run(200)
        
while True:
    thermostat = mdmd.MultiGroupLangevinMD(cube,
                                           groups=groups,
                                           temps=temps,
                                           timestep=2 * units.fs,
                                           friction=0.01 / units.fs)

    thermostat.attach(MDLogger(thermostat, cube, f'logs/md_{K}.log', header=True, stress=False, peratom=True, mode="a"), interval=log_interval)

    thermostat.attach(
        lambda: compute_temperatures(trajectory, cube, MASKS_OF_LAYERS, temperatures_by_layer),
        interval=measure_frequency,
    )

    try:
        thermostat.run(iteration - iteration_count)
    except LastlayerLeft:
        print("\033[0;31m" + f'Last layer left. Break after: {iteration_count*measure_frequency}.' + '\033[0m')
        break
    except LayerTooHot as e:
        cube = e.value
        cube.calc = (
            LennardJones(epsilon=EPSILON, sigma=SIGMA)
        )
        cube.pbc = PBC
        n_layers = n_layers - 1


        MASKS_OF_LAYERS = create_masks(cube, n_layers=n_layers)
        groups = {
            'bottom': np.where(MASKS_OF_LAYERS[0])[0],
            # 'rest': np.where(~MASKS_OF_LAYERS[0])[0]
        }
        print("\033[93m" + ' Highest layer is too hot. Deleting on iteratrion: ' + str(iteration_count*measure_frequency) + '\033[0m')
        continue
    break
    

today = date.today()
curr_time = time.strftime("%H_%M")
write(f"records/cube_iteration_scale_{K}_{curr_time}_day_{today.day}_.extxyz", trajectory, format="extxyz")

for i in temperatures_by_layer:
    if len(i) < NUMBER_OF_CELLS*2:
        for k in range(NUMBER_OF_CELLS*2 - len(i)):
            i.append(0)

matrix_temperatures_by_layer = np.array(temperatures_by_layer)
tempetarures_with_measurment_val = {}

for i, value in enumerate(temperatures_by_layer, 0):
    tempetarures_with_measurment_val[i*measure_frequency] = np.array(temperatures_by_layer[i])

import pandas as pd

tempetarures_with_measurment_val = pd.DataFrame(tempetarures_with_measurment_val).transpose()

fig, ax = plt.subplots(figsize=(5, 8))
im = ax.imshow(tempetarures_with_measurment_val, cmap='bwr')

fig.colorbar(im, ax=ax)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_title("Gradient of the heat spreading")
ax.set_xlabel("Index of the layer")
ax.set_ylabel("Iteration index")
ax.set_yticks(range(len(tempetarures_with_measurment_val)))
ax.set_yticklabels(tempetarures_with_measurment_val.index)
ax.invert_yaxis()

plt.tight_layout() 
plt.savefig(f'records/temperature gradient_{K}_{curr_time}_day_{today.day}_.png')

plt.show()

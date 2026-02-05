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

BASIC_NUMBER_ATOMS = 8
NUMBER_OF_ATOMS= BASIC_NUMBER_ATOMS // K # Scaling the size!!! Basic number of atoms is 16.


sigma_for_gold = 4.08 * (3.6/2.33)**(-1)
SIGMA= K * sigma_for_gold # Scaling for sigma

ATOMIC_UNIT_MASS = 196.196 * (K**3) # Scaling for masses
EPSILON=0.4 * (K**3) # Scaling for masses
n_layers = NUMBER_OF_ATOMS
THRESHOLD=250.



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
    LennardJones(epsilon=EPSILON, sigma=SIGMA)
)
layer_thikness = A

cube.set_masses(np.repeat([ATOMIC_UNIT_MASS], len(cube.get_masses()))) # Setting masses of the atoms


def create_masks(cube, n_layers, layer_thikness=layer_thikness):
    pos = cube.get_positions()
    all_z = pos[:, 2]
    min_z = np.min(all_z)


    MASKS_OF_LAYERS = [
        (all_z >= min_z + i*layer_thikness) & (all_z < min_z + (i+1)*layer_thikness)
        for i in range(n_layers)
    ]
    return MASKS_OF_LAYERS


MASKS_OF_LAYERS = create_masks(cube, n_layers=NUMBER_OF_ATOMS)
iteration_count = 0


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


def compute_temperatures(trajectory, cube, MASKS_OF_LAYERS):
    temps = []
    T_atom = np.zeros(len(cube))
    global iteration_count
    iteration_count += 1

    for _, mask in enumerate(MASKS_OF_LAYERS):
        temp_slice = cube[mask]

        layer_temp = temp_slice.get_temperature()  # K
        temps.append(layer_temp)

        T_atom[mask] = layer_temp

    cube.set_array("Temperature", T_atom)
    trajectory.append(cube.copy())

    if len(MASKS_OF_LAYERS) == 1:
        raise LastlayerLeft(1)
    
    
    last_temp_slice = cube[MASKS_OF_LAYERS[-1]]
    layer_temp = last_temp_slice.get_temperature()
    temp_highest_group = layer_temp


    if temp_highest_group > THRESHOLD:
    
        raise LayerTooHot(cube[~MASKS_OF_LAYERS[-1]].copy())

trajectory = []

groups = {'bottom': np.where(MASKS_OF_LAYERS[0])[0],
          'rest': np.where(~MASKS_OF_LAYERS[0])[0]}
temps = {'bottom': 300,
        'rest': 0}

while True:
    thermostat = mdmd.MultiGroupLangevinMD(cube,
                                           groups=groups,
                                           temps=temps,
                                           timestep=5 * units.fs,
                                           friction=0.01 / units.fs)

    thermostat.attach(MDLogger(thermostat, cube, f'logs/md_{K}.log', header=True, stress=False, peratom=True, mode="a"), interval=log_interval)

    thermostat.attach(
        lambda: compute_temperatures(trajectory, cube, MASKS_OF_LAYERS),
        interval=measure_frequency,
    )

    try:
        thermostat.run(iteration)
    except LastlayerLeft:
        print("\033[0;31m" + f'Last layer left. Break after: {iteration_count*measure_frequency}.' + '\033[0m')
        break
    except LayerTooHot as e:
        cube = e.value
        cube.calc = (
            LennardJones(epsilon=EPSILON, sigma=SIGMA)
        )
        n_layers = n_layers - 1
        MASKS_OF_LAYERS = create_masks(cube, n_layers=n_layers)
        groups = {
            'bottom': np.where(MASKS_OF_LAYERS[0])[0],
            'rest': np.where(~MASKS_OF_LAYERS[0])[0]
        }
        temps = {'bottom': thermostat.all_group_temperatures()['bottom'], 'rest': thermostat.all_group_temperatures()['rest']}
        print("\033[93m" + ' Highest layer is too hot. Deleting.' + '\033[0m')
    

today = date.today()
curr_time = time.strftime("%H_%M")
write(f"records/cube_iteration_scale_{K}_{curr_time}_day_{today.day}_.extxyz", trajectory, format="extxyz")

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
from ase.visualize import view
from ase.geometry import get_layers
from ase import Atoms

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

SCALE_FACTOR = int(args.scale)

K=1

iteration = int(args.iteration)

measure_frequency = int(args.measure_frequency)
log_interval = int(args.log_interval)

BASIC_NUMBER_CELLS = 5
NUMBER_OF_CELLS = BASIC_NUMBER_CELLS // K # Scaling the size!!! Basic number of atoms is 16.

PBC = (True, True, False)

sigma_for_gold = 4.08 * (3.6/2.33)**(-1)
SIGMA= K * sigma_for_gold # Scaling for sigma

ATOMIC_UNIT_MASS = 196.196 * (K**3) # Scaling for masses
EPSILON=0.4 * (K**3) # Scaling for masses
n_layers = NUMBER_OF_CELLS*2
THRESHOLD=5.

A = SIGMA * (3.6/2.33)
au = bulk("Au", "fcc", a=A, cubic=True)
cube = au.repeat((NUMBER_OF_CELLS, NUMBER_OF_CELLS, NUMBER_OF_CELLS))
cube.pbc = PBC

cube.calc = (
    LennardJones(epsilon=EPSILON, sigma=SIGMA)
)
layer_thikness = A/2

cube.set_masses(np.repeat([ATOMIC_UNIT_MASS], len(cube.get_masses()))) # Setting masses of the atoms


def create_masks(cube):

    layers, nlayers = get_layers(cube, (0, 0, 1), tolerance=layer_thikness-0.3)

    masks_layers = []

    for i in range(len(nlayers)):
        mask = np.where(layers == i)[0]

        masks_layers.append(mask)

    return masks_layers


MASKS_OF_LAYERS = create_masks(cube)
iteration_count = 0

# combmask = MASKS_OF_LAYERS[0]
# for mask in MASKS_OF_LAYERS[1:]:
#     combmask = np.concatenate([mask,combmask]) 

# # for k, mask in enumerate(MASKS_OF_LAYERS):

# write(f'init_cube.xyz', cube[combmask])

# view(cube)


# exit(0) 

class LastlayerLeft(Exception):

    def __init__(self, value, message="Only one layer left. Stopping."):
        self.value = value
        self.message = message
        super().__init__(self.message)

class CoarseGrane(Exception):

    def __init__(self, value, message="Value is too high"):
        self.value = value
        self.message = message
        super().__init__(self.message)

def compute_approximation_atoms(scale_factor, atoms_to_approximate):
    sigma_for_gold = 4.08 * (3.6/2.33)**(-1)
    SIGMA = scale_factor * sigma_for_gold # Scaling for sigma

    ATOMIC_UNIT_MASS = 196.196 * (scale_factor**3) # Scaling for masses
    EPSILON=0.4 * (scale_factor**3) # Scaling for masses

    A = SIGMA * (3.6/2.33)

    print(f"Для K={scale_factor} vs K=1:")
    print(f"Объём: {(A ** 3) * ((len(atoms_to_approximate)//(scale_factor**3)))} == {(4.08 ** 3)* len(atoms_to_approximate)}")
    print(f"Масса: {(len(atoms_to_approximate)//(scale_factor**3)) * ATOMIC_UNIT_MASS}  ==  {len(atoms_to_approximate) * 196.196}")
    print(f"Количество атомов (должно отличаться при k != 1): {(int((NUMBER_OF_CELLS)//(scale_factor)))**2 * 4} != {((BASIC_NUMBER_CELLS)**3)*4}")


    A = SIGMA * (3.6 / 2.33)
    au = bulk("Au", "fcc", a=A, cubic=True)
    plane = au.repeat((int((NUMBER_OF_CELLS)//(scale_factor)), int((NUMBER_OF_CELLS)//(scale_factor)), 1))

    plane.pbc = PBC

    plane.set_masses(np.repeat([ATOMIC_UNIT_MASS], len(plane.get_masses())))

    # mask_bottom = create_masks(plane)[0]
    # plane = plane[mask_bottom].copy()

    write(f'plane.xyz', plane)

    plane.calc = LennardJones(
        sigma=SIGMA,
        epsilon=EPSILON
    )

    return plane

def compute_temperatures(trajectory, cube, MASKS_OF_LAYERS, temperatures_by_layer, already_grained, n_layers):
    temps = []
    T_atom = np.zeros(len(cube))

    global iteration_count
    iteration_count += 1

    velocities = cube.get_velocities()
    masses = cube.get_masses()


    for mask in MASKS_OF_LAYERS:
        idxs = mask


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
    
    # shift on K + already grained to left. This is the gighesdt layer which is NOT goinig to be grained.
    # After this layer starts SCALE_FACTOR times layers to be checked and then grained  
    index_start_graining = n_layers-SCALE_FACTOR-already_grained

    if already_grained == 0:
        # Check highest SCALE_FACTOR layers
        temp_highest_groups = temps[index_start_graining:]
    else:
        # Check from index_start_graining layers which are not grained yet (there could be already grained atoms above them)
        temp_highest_groups = temps[index_start_graining:index_start_graining+SCALE_FACTOR]
    
    print(f'indexes: from {index_start_graining} to {index_start_graining + SCALE_FACTOR}, total size: {len(temp_highest_groups)}')

    if (len(temp_highest_groups) > 0) and np.all(np.array(temp_highest_groups) > THRESHOLD):
        print("Graining...")
        
        # collect indexes of atoms, which are not grained and not going to be grained at this stage
        combined_mask = MASKS_OF_LAYERS[0]
        for i in range(1, index_start_graining):
            combined_mask = np.concatenate([combined_mask, MASKS_OF_LAYERS[i]])

        # collect indexes of atoms, which are going to be grained at this stage
        mask_to_grain = MASKS_OF_LAYERS[index_start_graining]
        for i in range(index_start_graining, index_start_graining+SCALE_FACTOR):
            mask_to_grain = np.concatenate([mask_to_grain, MASKS_OF_LAYERS[i]])

        # collect indexes of atoms, which are grained ALREADY
        combined_mask_already_grained = MASKS_OF_LAYERS[0]
        for i in range(index_start_graining+SCALE_FACTOR, len(MASKS_OF_LAYERS)):
            combined_mask_already_grained = np.concatenate([combined_mask_already_grained, MASKS_OF_LAYERS[i]])

        
        print("combined_mask: ", len(combined_mask))
        print("mask_to_grain: ", len(mask_to_grain))
        print("combined_mask_already_grained: ", len(combined_mask_already_grained))

        grained_atoms = compute_approximation_atoms(SCALE_FACTOR, cube[mask_to_grain])

        from ase.build.tools import cut, stack

        old_grained_atoms = cube[combined_mask_already_grained].copy()

        if np.linalg.det(grained_atoms.cell) < 0:
            grained_atoms.set_cell(-grained_atoms.cell)
        if np.linalg.det(old_grained_atoms.cell) < 0:
            old_grained_atoms.set_cell(-old_grained_atoms.cell)
        
        old_grained_atoms.set_cell(cube.cell, scale_atoms=True)
        grained_atoms.set_cell(cube.cell, scale_atoms=True)

        grained_part = stack(grained_atoms, old_grained_atoms, distance=A * (SCALE_FACTOR**3), maxstrain=None) # либо сделать растяжение!

        cube = cube[combined_mask].copy() # slice last K layers

        cube_interface = stack(cube, grained_part, distance=A, maxstrain=None) # либо сделать растяжение!
        cube_interface.calc = (
            LennardJones(epsilon=EPSILON, sigma=SIGMA)
        )

        MASKS_OF_LAYERS = create_masks(cube_interface)
        combined_mask_already_grained = MASKS_OF_LAYERS[0]
        for i in range(index_start_graining, len(MASKS_OF_LAYERS)):
            combined_mask_already_grained = np.concatenate([MASKS_OF_LAYERS[i], combined_mask_already_grained])

        graine_status = np.zeros(len(cube_interface))

        graine_status[combined_mask_already_grained] = 1

        cube_interface.set_array("Grained", graine_status)
        raise CoarseGrane(cube_interface.copy())

trajectory = []
temperatures_by_layer = []


groups = {'bottom': MASKS_OF_LAYERS[0]}
temps = {'bottom': 300}

print(len(MASKS_OF_LAYERS))

# thermostat = mdmd.MultiGroupLangevinMD(cube,
#                                            groups=groups,
#                                            temps=temps,
#                                            timestep=2 * units.fs,
#                                            friction=0.01 / units.fs)

# thermostat.run(200)

# current_scale_factor = 1
already_grained = 0 
        
while True:
    thermostat = mdmd.MultiGroupLangevinMD(cube,
                                           groups=groups,
                                           temps=temps,
                                           timestep=2 * units.fs,
                                           friction=0.01 / units.fs)

    thermostat.attach(MDLogger(thermostat, cube, f'logs/md_{K}.log', header=True, stress=False, peratom=True, mode="a"), interval=log_interval)

    thermostat.attach(
        lambda: compute_temperatures(trajectory, cube, MASKS_OF_LAYERS, temperatures_by_layer, already_grained, n_layers),
        interval=measure_frequency,
    )

    try:
        thermostat.run(iteration - iteration_count)
    except LastlayerLeft:
        print("\033[0;31m" + f'Last layer left. Break after: {iteration_count*measure_frequency}.' + '\033[0m')
        break
    except CoarseGrane as e:
        cube = e.value
        
        cube.pbc = PBC
        cube.calc = (
            LennardJones(epsilon=EPSILON, sigma=SIGMA)
        )
        # n_layers = n_layers - 1


        MASKS_OF_LAYERS = create_masks(cube)
        groups = {
            'bottom': MASKS_OF_LAYERS[0],
        }
        already_grained += SCALE_FACTOR
        if already_grained == n_layers:
            print("\033[93m" + f'Scaled all atoms. Stop.' + '\033[0m')
            break
        print("\033[93m" + f' Highest {SCALE_FACTOR} layers are hot enough. Changing up to k = {SCALE_FACTOR} on interation: ' + str(iteration_count*measure_frequency) + f'. Already grained: {already_grained} layers.' + '\033[0m')
        continue
    except KeyboardInterrupt:
        break
    break
    

today = date.today()
curr_time = time.strftime("%H_%M")
write(f"records/cube_iteration_scale_{K}_{curr_time}_day_{today.day}_.extxyz", trajectory, format="extxyz")

for i in temperatures_by_layer:
    if len(i) < NUMBER_OF_CELLS*2:
        for k in range(NUMBER_OF_CELLS*2 - len(i)):
            i.append(0)

print(temperatures_by_layer)
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

ax.set_title("Gradient of the heat spreading")
ax.set_xlabel("Index of the layer")
ax.set_ylabel("Iteration index")
ax.set_yticks(range(len(tempetarures_with_measurment_val)))
ax.set_yticklabels(tempetarures_with_measurment_val.index)
ax.invert_yaxis()

plt.tight_layout() 
plt.savefig(f'records/temperature gradient_{K}_{curr_time}_day_{today.day}_.png')

plt.show()

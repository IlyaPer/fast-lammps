# fast-lammps extension

This tool allows to dramatically speed up your molecular dynamics simulations using dynamic coarse-graining algorithm. 
The example of the processes which were approximated with this extension^ heating aurum. The blocks where kinetic energy was lower than the threshold were approximated with larger particles (atoms).

![](build/simple_grain.gif)


## Getting started quickly
### How does it work?
This package speeds up ready molecular dynamics scripts and simulations, written on LAMMPS or ASE package. The only thing you need is to provide the initial conditions script in ``.in`` format to test-drive how fast your simalutions would become.

![](build/structure.jpg)


### Installation
You will need to ** run all scripts with this package in the virtual environment of lammps module**.

1. Clone the repository:
```
git clone git@github.com:IlyaPer/Granular-MD.git
cd Granular-MD
```

First scenario: independent one launch of the simulation. 
2. Instalation dependencies:
```
pip install -r requirements.txt
```

3. Simple run with ready made tools

Obligatory command-line arguments:  
- `-f <file>` — path to the input file of the simulation in the `.in` format.
- `-i <iterations>` — maximum number of iterations.
- `-m <step>` — step of coarse‑graining (default: 500; adaptive if possible?).
- `-k <size of cell to be grained>` — must be even. if 0 - no coarse graining would be provided, so the simulation would continue as it is. 0 is useful if you need other tools of the package (visualizing prometheus metrics with Grafana, for instance).

Additional command-line arguments:  
- `--prometheus_checkup` — with this flag enabled you will obtain CPU/GPU/Memory allocations data in any format, ready to be visualized with Grafana.
- `--experiment` — with this flag you will receive comparsion between CPU/GPU/Memory allocations data with dynamic coarse-graining and without it, available to be visualized in Grafana tool. This flag utomatically enables `--prometheus_checkup`.
- `--stress` — with this flag you will receive maximum possible memory/gpu/cpu consumption with synthetically made conditions (such as all regions to be grained simultaneously or retrieving extreme forces). If the simulation ended up successfully with this flag enabled - you would probably not receive any overflow errors from our side on your supercomputer run.
- `--reversed_graining` — This is an advanced tool which allows to setip several conditions to (1) approximate a group of atoms with one particle, as works with simple run (2) provide reverse-graining, if the approximated region is involved into an "interesting" process, where high precision is required. This allows to use package for simulating highly uncertain processes, where there is no assumptions on where the process will be interesting
 - `--analyze` — This is an advanced flag which allows to recieve information on how many resources is going to be required for running your simulation. It is running by it's own and doen't require any additonal flags except for `-f`. It provides information in a table format. This tool is needed if you run your simulation on a supercomputer and you are planning to count required CPU/GPU units.

Example of simple usage
```
python main.py -f 'heating_aurum.in' -k 2 -i 10000 -m 500
```  

Example of advanced usage with experiments  
```
python main.py -f 'heating_aurum.in' --experiment -i 10000 -m 500
```  

4. (additional) Integrating Grafana and Prometheus
If you need to speed up your calculations and predict peak consumption of your script without any "heavy tools" we provide built-in instruments of checking up the system requirements. 

## Advanced usage with your own mesh-refinement
Ready made mesh-refinement solutions are described in out documentation. Currently, only fcc structures are available to be grained. However, you can easily integrate your own solver for extracting regions to be approximated.

## Pipeline of the supercomputer experiment with fast-lammps

Package provides several useful tools for launching your already made simulation on supercomputer cluster with dynamic coarse graining.

1. Check up possible overflow issues with `--stress` flag
2. Run the flag experiment with `--experiment` flag.
3. Choose `--autodetect` to identify best way to  
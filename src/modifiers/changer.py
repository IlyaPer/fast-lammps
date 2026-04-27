APPROXIMATE = 1
GRANULATE = 2
import logging

class DynamicChanger:
    def __init__(self, extractor, lattice_constant, lattice_constant_cg, baby_mode=False):
        self.extractor = extractor
        self.baby_mode = baby_mode
        self.lattice_constant_cg = lattice_constant_cg
        self.lattice_constant = lattice_constant

    def accelerate(self,):
        """
        affirmative. execute acceleration.
        """
        self.extractor.extract_interesting_regions()
        cells = self.extractor
        
        for cell_to_granulate in self.extractor._get_cells_to_granulate():
            logging.info(f'Executing replacement')
            self._execute_lammps_replacement(cell_to_granulate, operation=GRANULATE)

        if not self.baby_mode:
            for cell_to_approximate in self.extractor._get_cells_to_approximate():
                self._execute_lammps_replacement(cell_to_approximate, operation=GRANULATE)


    def _execute_lammps_replacement(self, cell_to_granulate : tuple, operation : int):
        """
        Replace atoms with new one.
        """
        x_min, x_max, y_min, y_max, z_min, z_max, atom_ids  = cell_to_granulate
        velocities_region =  self.extractor.__get_velocities__()
        self.extractor.get_lammps_instance().command(f"region kill block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box")
        self.extractor.get_lammps_instance().command("group cell_atoms region kill")

        self.extractor.get_lammps_instance().command(f"lattice fcc {self.lattice_constant_cg-0.01}")
        if self.baby_mode:
            commands = [
                # "compute v_atoms cell_atoms property/atom vx vy vz",
                # "compute v_red cell_atoms reduce ave c_v_atoms[1] c_v_atoms[2] c_v_atoms[3]",
                # "variable n_kill equal count(cell_atoms)",
                # "run 0",
                # "variable vx_new equal c_v_red[1]",
                # "variable vy_new equal c_v_red[2]",
                # "variable vz_new equal c_v_red[3]",
                'delete_atoms group cell_atoms',
                'create_atoms 2 region kill',
                # "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
                # "uncompute v_atoms",
                # "uncompute v_red",
                # "uncompute pe_cell",
                # "variable vx_new delete",
                # "variable vy_new delete",
                # "variable vz_new delete",
                # "variable n_kill delete",
                "group cell_atoms delete",
                "region kill delete"
            ]
            for cmd in commands:
                self.extractor.get_lammps_instance().lammps_instance.command(cmd)

        # commands = [
        #     # f"group kill_group id {' '.join(map(str, value))}",
        #     "compute v_atoms cell_atoms property/atom vx vy vz",
        #     "compute v_red cell_atoms reduce ave c_v_atoms[1] c_v_atoms[2] c_v_atoms[3]",
        #     "variable n_kill equal count(cell_atoms)",
        #     "run 0",
        #     "variable vx_new equal c_v_red[1]",
        #     "variable vy_new equal c_v_red[2]",
        #     "variable vz_new equal c_v_red[3]",
        #     'delete_atoms group cell_atoms',
        #     'create_atoms 2 region kill',
        #     "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
        #     "uncompute v_atoms",
        #     "uncompute v_red",
        #     "uncompute pe_cell",
        #     "variable vx_new delete",
        #     "variable vy_new delete",
        #     "variable vz_new delete",
        #     "variable n_kill delete",
        #     "group cell_atoms delete",
        #     "region kill delete"
        # ]
        # for cmd in commands:
        #     self.extractor.get_lammps_instance().lammps_instance.command(cmd)
        

        # commands = [
        #     f"region kill block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box",
        #     "group cell_atoms region kill",
        #     # ... остальные команды ...
        #     "delete_atoms group cell_atoms",
        #     f"create_atoms 2 region kill", # пример
        #     "region kill delete",
        #     "group cell_atoms delete"
        # ]
        
        # for cmd in commands:
        #     lammps_instance.command(cmd)
            
        # logging.info(f"Modified cell at {key}")
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units


class MultiGroupLangevinMD(MolecularDynamics):
    """
    Langevin dynamics with per-group temperature control.

    This integrator allows splitting atoms into groups, each coupled to a
    separate heat bath at a specific temperature.

    Crucially, the friction and thermal noise are applied to velocities
    relative to the group's center-of-mass (COM) velocity. This prevents
    the thermostat from dampening the collective motion of a group.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object.
    groups : dict
        Dictionary mapping group names to lists of atom indices.
        e.g. {'left': [0, 1, ...], 'right': [10, 11, ...]}
        Atoms not listed in any group are not thermostatted (NVE).
    temps : dict
        Dictionary mapping group names to temperatures in Kelvin.
        e.g. {'left': 300.0, 'right': 100.0}
    timestep : float
        Time step in ASE time units (usually fs).
    friction : float
        Friction coefficient in inverse ASE time units.
    rng_seed : int, optional
        Seed for the random number generator.
    """

    def __init__(self, atoms, groups, temps, timestep, friction, rng_seed=None):
        super().__init__(atoms, timestep)
        self.groups = groups
        self.temps = temps
        self.friction = friction
        self.rng = np.random.default_rng(rng_seed)

        # Pre-calculate Langevin coefficients per group
        # c1 = exp(-gamma * dt)
        # c2 = sqrt(1 - c1^2) * sqrt(2 * T * kB / m) / sqrt(gamma) ?
        # Using ASE implementation logic:
        # sigma = sqrt(2 * T * friction * kB / m)
        # coeff1 = exp(-friction * dt)
        # coeff2 = sqrt(1 - coeff1^2) * sigma / sqrt(friction)

        # We store these coefficients for easy access during the step
        self.group_coeffs = {}
        masses = atoms.get_masses()

        for name, indices in groups.items():
            if name not in temps:
                raise ValueError(f"Temperature not defined for group '{name}'")

            T = temps[name]
            m_group = masses[indices]

            # Coefficients
            # c1 is the damping factor
            c1 = np.exp(-friction * timestep)

            # c2 is the noise amplitude
            # sigma^2 = 2*T*gamma*kB/m
            # We need the prefactor for the noise term in the integration step
            # Standard Langevin integration often uses: v(t+dt) = c1*v(t) + c2*noise
            # c2 = sqrt(k_B * T * (1 - c1^2) / m)
            # Note: ASE units check. units.kB is in eV/K. Masses in AMU.
            # We need consistent units. ASE internal velocities are sqrt(eV/AMU).

            c2 = np.sqrt(units.kB * T * (1 - c1**2) / m_group[:, None])

            self.group_coeffs[name] = {"c1": c1, "c2": c2, "indices": indices}

    def step(self):
        atoms = self.atoms

        # 1. Half-step velocity update (Velocity Verlet part 1)
        forces = atoms.get_forces()
        masses = atoms.get_masses()[:, None]
        velocities = atoms.get_velocities()

        # v = v + 0.5 * dt * F/m
        velocities += 0.5 * self.dt * forces / masses

        # 2. Langevin Step (Friction + Noise)
        # We apply this GROUP-WISE and RELATIVE to Group COM

        for name, params in self.group_coeffs.items():
            idxs = params["indices"]
            c1 = params["c1"]
            c2 = params["c2"]

            # Extract velocities for this group
            v_group = velocities[idxs]
            m_group = masses[idxs]

            # Calculate Group COM velocity
            # V_com = sum(m*v) / sum(m)
            total_mass = np.sum(m_group)
            v_com = np.sum(v_group * m_group, axis=0) / total_mass

            # Relative velocity (internal motion only)
            v_rel = v_group - v_com

            # Apply Langevin thermostat to relative velocity
            # Generate random noise (Gaussian with mean 0, var 1)
            noise = self.rng.standard_normal(size=v_rel.shape)

            v_rel_new = c1 * v_rel + c2 * noise

            # Add COM velocity back (conserving the group's collective momentum
            # effectively, though noise adds random momentum, the friction
            # won't drag the COM to zero).
            velocities[idxs] = v_rel_new + v_com

        atoms.set_velocities(velocities)

        # 3. Full position update
        # r = r + dt * v
        atoms.set_positions(atoms.get_positions(wrap=True) + self.dt * velocities)

        # 4. Re-calculate forces at new positions
        forces = atoms.get_forces()

        # 5. Final half-step velocity update (Velocity Verlet part 2)
        # v = v + 0.5 * dt * F/m
        velocities = atoms.get_velocities()  # get fresh ref
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)

    def all_group_temperatures(self):
        """Helper to return current calculated temperature of each group."""
        res = {}
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()

        for name, params in self.group_coeffs.items():
            idxs = params["indices"]
            v_group = velocities[idxs]
            m_group = masses[idxs][:, None]  # shape (N, 1)

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
                res[name] = current_T
            else:
                res[name] = 0.0
        return res
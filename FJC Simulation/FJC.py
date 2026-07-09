import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lammps import lammps

# Files are saved next to this script.
# os.chdir so LAMMPS receives bare filenames — avoids Windows spaces-in-path bug.
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
def path(f): return os.path.join(DIR, f)

# =============================================================================
# PARAMETERS
# =============================================================================

# --- Main detailed run (used for distribution plots A, B, C, D) ---
N_MAIN              = 100     # chain length for the main run  #Change to 200 and check if the standard deviation is decreasing
N_CHAINS_PER_BOX    = 50      # how many independent chains run simultaneously in one box 
BOND_LENGTH         = 1.0     # b: equilibrium bond length [LJ units]
BOND_STIFFNESS      = 5000.0  # K: spring constant in U=K(r-r0)² — very stiff → near-rigid
BOX_SIZE            = 500.0   # side length of cubic simulation box (chains ~10σ wide, never hit walls)
EQUIL_STEPS         = 100000  # steps discarded before data collection (let chain forget start)
PROD_STEPS_MAIN     = 300000  # steps used for analysis in the main run
DUMP_INTERVAL       = 500     # write a snapshot to the trajectory file every this many steps

# --- Scaling study (used for plots E, F: how Rg² grows with chain length) ---
# We simulate each chain length separately and measure <Rg²> and <R²>
SCALING_CHAIN_LENGTHS = [10, 20, 40, 60, 80, 100, 150, 200]

# Longer chains take more time to explore conformations (autocorrelation time ~ N²).
# We scale production steps with N as a practical compromise between accuracy and runtime.
def scaling_prod_steps(chain_length):
    return max(100000, int(3000 * chain_length))

# A LAMMPS dump frame always has exactly 9 header lines before the first atom line:
#   ITEM: TIMESTEP  /  value  /  ITEM: NUMBER OF ATOMS  /  value  /
#   ITEM: BOX BOUNDS pp pp pp  /  xlo xhi  /  ylo yhi  /  zlo zhi  /
#   ITEM: ATOMS id mol type x y z
DUMP_HEADER_LINES = 9

rng = np.random.default_rng(42)

# =============================================================================
# write_data: create LAMMPS input topology file
# =============================================================================
def write_data(chain_length, n_chains, filepath):
    """
    Writes a LAMMPS data file for n_chains FJC chains, each with chain_length monomers.

    Data file structure:
      Header  — total atom/bond counts, simulation box dimensions
      Masses  — mass of each atom type (only one type here)
      Atoms   — one line per atom: atom-ID  mol-ID  type  x  y  z
      Bonds   — one line per bond: bond-ID  type  atom1  atom2

    mol-ID (molecule ID) labels which chain each atom belongs to (1..n_chains).
    This is how we later separate atoms back into individual chains when reading the dump.
    """
    total_atoms = chain_length * n_chains
    total_bonds = (chain_length - 1) * n_chains

    # Spread chain starting positions evenly across the box on a 3D grid
    grid    = int(np.ceil(n_chains**(1/3))) + 1
    spacing = BOX_SIZE / grid
    grid_points  = [spacing * (k + 0.5) for k in range(grid)]
    start_positions = [(x, y, z)
                       for x in grid_points
                       for y in grid_points
                       for z in grid_points]
    rng.shuffle(start_positions)

    # Build each chain as a random walk: each step is a random direction on the unit sphere
    all_chain_coords = []
    for chain_idx in range(n_chains):
        coords    = np.zeros((chain_length, 3))
        coords[0] = np.array(start_positions[chain_idx])
        for bead in range(1, chain_length):
            # arccos(1-2u) samples polar angle θ uniformly on the sphere
            # (plain θ = π*u would cluster directions near the poles)
            theta        = np.arccos(1 - 2 * rng.random())
            phi          = 2 * np.pi * rng.random()
            step_vector  = BOND_LENGTH * np.array([
                               np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi),
                               np.cos(theta)])
            coords[bead] = coords[bead - 1] + step_vector
        all_chain_coords.append(coords)

    with open(filepath, 'w') as f:
        f.write(f"FJC  chain_length={chain_length}  n_chains={n_chains}\n\n")
        f.write(f"{total_atoms} atoms\n{total_bonds} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"0 {BOX_SIZE} xlo xhi\n")
        f.write(f"0 {BOX_SIZE} ylo yhi\n")
        f.write(f"0 {BOX_SIZE} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")
        f.write("Atoms\n\n")
        atom_id = 1
        for mol_id, coords in enumerate(all_chain_coords, start=1):
            for x, y, z in coords:
                f.write(f"{atom_id} {mol_id} 1 {x:.5f} {y:.5f} {z:.5f}\n")
                atom_id += 1
        f.write("\nBonds\n\n")
        bond_id = 1
        first_atom_of_chain = 1
        for _ in range(n_chains):
            for i in range(chain_length - 1):
                f.write(f"{bond_id} 1 "
                        f"{first_atom_of_chain + i} "
                        f"{first_atom_of_chain + i + 1}\n")
                bond_id += 1
            first_atom_of_chain += chain_length

# =============================================================================
# parse_dump: read trajectory → R² and Rg² for every chain in every frame
# =============================================================================
def parse_dump(filepath, chain_length, n_chains):
    """
    Reads every snapshot from a LAMMPS dump file.
    Groups atoms by mol-ID to reconstruct each chain's coordinates.
    Computes R² (end-to-end distance squared) and Rg² (radius of gyration squared)
    for every chain in every frame independently.

    Returns:
        R2_all_samples   — flat array of R²  values, length = n_frames × n_chains
        Rg2_all_samples  — flat array of Rg² values, length = n_frames × n_chains
        mean_rg_per_frame — mean Rg across all chains at each timestep (for time-series plot)
    """
    R2_all_samples   = []
    Rg2_all_samples  = []
    mean_rg_per_frame = []
    total_atoms = chain_length * n_chains

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            i += DUMP_HEADER_LINES      # skip 9 header lines, land on first atom data line

            # Read all atoms in this frame, grouped by their mol-ID
            chains_in_frame = {}        # mol_id → { atom_id → position }
            for _ in range(total_atoms):
                parts  = lines[i].split(); i += 1
                atom_id = int(parts[0])
                mol_id  = int(parts[1])
                position = np.array([float(parts[3]),
                                     float(parts[4]),
                                     float(parts[5])])
                chains_in_frame.setdefault(mol_id, {})[atom_id] = position

            # For each chain compute R² and Rg²
            rg_values_this_frame = []
            for mol_id in sorted(chains_in_frame):
                atom_dict = chains_in_frame[mol_id]
                # Sort by atom_id so bead order is 1→2→3→...→N
                coords = np.array([atom_dict[k] for k in sorted(atom_dict)])

                # End-to-end vector: from first bead to last bead
                end_to_end = coords[-1] - coords[0]
                R2_all_samples.append(end_to_end @ end_to_end)

                # Radius of gyration squared: mean squared distance of beads from COM
                centre_of_mass = coords.mean(axis=0)
                rg2 = np.mean(np.sum((coords - centre_of_mass)**2, axis=1))
                Rg2_all_samples.append(rg2)
                rg_values_this_frame.append(np.sqrt(rg2))

            mean_rg_per_frame.append(np.mean(rg_values_this_frame))
        else:
            i += 1

    return (np.array(R2_all_samples),
            np.array(Rg2_all_samples),
            np.array(mean_rg_per_frame))

# =============================================================================
# read_single_chain_frame: get one chain's coords from one frame (for 3D plot)
# =============================================================================
def read_single_chain_frame(filepath, frame_index, chain_length, n_chains, mol_id=1):
    """Returns the (chain_length, 3) coordinate array for one chain in one frame."""
    total_atoms = chain_length * n_chains
    with open(filepath) as f:
        lines = f.readlines()
    frames_seen = -1
    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            frames_seen += 1
            if frames_seen == frame_index:
                i += DUMP_HEADER_LINES
                chains_in_frame = {}
                for _ in range(total_atoms):
                    parts = lines[i].split(); i += 1
                    chains_in_frame.setdefault(int(parts[1]), {})[int(parts[0])] = [
                        float(parts[3]), float(parts[4]), float(parts[5])]
                atom_dict = chains_in_frame[mol_id]
                return np.array([atom_dict[k] for k in sorted(atom_dict)])
        i += 1
    return None

# =============================================================================
# run_lammps: run equilibration + production for one set of chains
# =============================================================================
def run_lammps(chain_length, n_chains, data_file, dump_file,
               equil_steps, prod_steps, silent=False):
    """
    Runs a LAMMPS simulation via the Python API.
    All analysis is done afterwards in Python by reading the dump file,
    so this function only handles the dynamics.

    Key LAMMPS commands used:
      units lj          — dimensionless units: length=, σ energy=ε, time=τ
      atom_style bond   — atoms store molecule-ID and bond connectivity
      boundary p p p    — periodic box in x,y,z (atoms wrap around walls)
      bond_style harmonic + bond_coeff
                        — bond energy U(r) = K(r - r0)² 
                          K=5000 >> kT=1 so bond length barely fluctuates
      (no pair_style)   — zero non-bonded forces = ideal/phantom chain
      fix langevin      — random thermal kicks + friction = implicit solvent at kT=1
      fix nve           — integrates Newton's equations (moves atoms each timestep)
      dump              — writes atom positions every DUMP_INTERVAL steps
      undump before reset_timestep
                        — LAMMPS forbids resetting the clock while a dump is open
    """
    bare_data_file = os.path.basename(data_file)
    bare_dump_file = os.path.basename(dump_file)
    lammps_args    = ["-screen", "none", "-log", "none"] if silent else []
    lmp            = lammps(cmdargs=lammps_args)

    for cmd in [
        "units lj",
        "atom_style bond",
        "boundary p p p",
        f"read_data {bare_data_file}",
        "bond_style harmonic",
        f"bond_coeff 1 {BOND_STIFFNESS} {BOND_LENGTH}",
        "minimize 1.0e-4 1.0e-6 500 5000",    # relax any initial bond strain
        "reset_timestep 0",
        f"dump d1 all custom {DUMP_INTERVAL} {bare_dump_file} id mol type x y z",
        "fix thermostat all langevin 1.0 1.0 1.0 12345",
        "fix integrator all nve",
        "timestep 0.005",
        "thermo 10000" if not silent else "thermo 99999999",
        "thermo_style custom step temp ebond",
        f"run {equil_steps}",          # equilibration — not saved, chain forgets its start
        "undump d1",                   # must close dump before reset_timestep
        "reset_timestep 0",
        f"dump d1 all custom {DUMP_INTERVAL} {bare_dump_file} id mol type x y z",
        f"run {prod_steps}",           # production — saved and analysed
    ]:
        lmp.command(cmd)
    lmp.close()

# =============================================================================
# PART 1: MAIN DETAILED RUN  (N_MAIN=100 monomers, N_CHAINS_PER_BOX=20 chains)
# =============================================================================
print(f"=== Part 1: Main run — chain_length={N_MAIN}, n_chains={N_CHAINS_PER_BOX} ===")
write_data(N_MAIN, N_CHAINS_PER_BOX, path("main_run.data"))
run_lammps(N_MAIN, N_CHAINS_PER_BOX,
           data_file   = path("main_run.data"),
           dump_file   = path("main_run.dump"),
           equil_steps = EQUIL_STEPS,
           prod_steps  = PROD_STEPS_MAIN,
           silent      = False)
print("    Done.")

# =============================================================================
# PART 2: PARSE MAIN TRAJECTORY + VERIFY THEORY
# =============================================================================
print("\n=== Part 2: Parsing main trajectory and verifying theory ===")
R2_main, Rg2_main, rg_timeseries = parse_dump(
    path("main_run.dump"), N_MAIN, N_CHAINS_PER_BOX)

n_bonds_main     = N_MAIN - 1
mean_R2_main     = R2_main.mean()
mean_Rg2_main    = Rg2_main.mean()
R2_theory_main   = n_bonds_main * BOND_LENGTH**2        # <R²>  = (N-1)b²
Rg2_theory_main  = n_bonds_main * BOND_LENGTH**2 / 6   # <Rg²> = (N-1)b²/6
total_samples    = len(R2_main)
total_frames     = len(rg_timeseries)

print(f"    {total_frames} frames × {N_CHAINS_PER_BOX} chains = {total_samples} samples")
print(f"\n{'Quantity':<30} {'Simulation':>12} {'Theory':>12} {'Error':>8}")
print("-" * 66)
for label, sim_val, theory_val in [
    ("<R²>",        mean_R2_main,              R2_theory_main),
    ("<Rg²>",       mean_Rg2_main,             Rg2_theory_main),
    ("<R²>/<Rg²>",  mean_R2_main/mean_Rg2_main, 6.0),
    ("R_rms",       np.sqrt(mean_R2_main),     np.sqrt(R2_theory_main)),
    ("Rg_rms",      np.sqrt(mean_Rg2_main),    np.sqrt(Rg2_theory_main)),
]:
    error_pct = 100 * abs(sim_val - theory_val) / theory_val
    print(f"{label:<30} {sim_val:>12.4f} {theory_val:>12.4f} {error_pct:>7.2f}%")

# =============================================================================
# PART 3: SCALING STUDY — simulate each chain length, collect <Rg²> and <R²>
# =============================================================================
print("\n=== Part 3: Scaling study ===")
scaling_mean_R2  = {}   # chain_length → mean <R²>
scaling_mean_Rg2 = {}   # chain_length → mean <Rg²>

for chain_len in SCALING_CHAIN_LENGTHS:
    prod_steps = scaling_prod_steps(chain_len)
    print(f"    chain_length={chain_len:3d}  ({prod_steps} prod steps) ...",
          end=" ", flush=True)
    write_data(chain_len, N_CHAINS_PER_BOX, path(f"scaling_N{chain_len}.data"))
    run_lammps(chain_len, N_CHAINS_PER_BOX,
               data_file   = path(f"scaling_N{chain_len}.data"),
               dump_file   = path(f"scaling_N{chain_len}.dump"),
               equil_steps = 50000,
               prod_steps  = prod_steps,
               silent      = True)
    R2_scaling, Rg2_scaling, _ = parse_dump(
        path(f"scaling_N{chain_len}.dump"), chain_len, N_CHAINS_PER_BOX)
    scaling_mean_R2[chain_len]  = R2_scaling.mean()
    scaling_mean_Rg2[chain_len] = Rg2_scaling.mean()
    theory_rg2 = (chain_len - 1) * BOND_LENGTH**2 / 6
    error = 100 * abs(Rg2_scaling.mean() - theory_rg2) / theory_rg2
    print(f"<Rg²>={Rg2_scaling.mean():.3f}  theory={theory_rg2:.3f}  error={error:.1f}%")

# =============================================================================
# PART 4: PLOTS
# =============================================================================
print("\n=== Part 4: Plotting ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Freely Jointed Chain — N={N_MAIN}, {N_CHAINS_PER_BOX} chains, b={BOND_LENGTH}, kT=1.0",
    fontsize=14)

# A: Rg time series — mean Rg across all chains, plotted per frame
ax = axes[0, 0]
timesteps = np.arange(total_frames) * DUMP_INTERVAL
ax.plot(timesteps, rg_timeseries, lw=0.8, color='steelblue', alpha=0.85,
        label=r'$\langle R_g(t)\rangle$ over chains')
ax.axhline(np.sqrt(Rg2_theory_main), color='red', lw=2, ls='--',
           label=fr'Theory $\sqrt{{\langle R_g^2\rangle}}='
                 fr'{np.sqrt(Rg2_theory_main):.3f}$')
ax.axhline(rg_timeseries.mean(), color='navy', lw=1.5, ls=':',
           label=fr'Sim mean $= {rg_timeseries.mean():.3f}$')
ax.set_xlabel('Timestep'); ax.set_ylabel(r'$\langle R_g\rangle$ [$\sigma$]')
ax.set_title(r'$R_g$ Time Series — mean over chains per frame')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# B: Distribution of R²
ax = axes[0, 1]
ax.hist(R2_main, bins=50, density=True, color='darkorange', alpha=0.7)
ax.axvline(mean_R2_main,   color='darkorange', lw=2, ls='--',
           label=fr'Sim $\langle R^2\rangle = {mean_R2_main:.2f}$')
ax.axvline(R2_theory_main, color='red',        lw=2, ls='-',
           label=fr'Theory $(N-1)b^2 = {R2_theory_main:.2f}$')
ax.set_xlabel(r'$R^2$ [$\sigma^2$]'); ax.set_ylabel('Probability density')
ax.set_title(r'Distribution of $R^2$  ($n=$' + f'{total_samples})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# C: Distribution of Rg²
ax = axes[0, 2]
ax.hist(Rg2_main, bins=50, density=True, color='mediumseagreen', alpha=0.7)
ax.axvline(mean_Rg2_main,   color='mediumseagreen', lw=2, ls='--',
           label=fr'Sim $\langle R_g^2\rangle = {mean_Rg2_main:.2f}$')
ax.axvline(Rg2_theory_main, color='red',             lw=2, ls='-',
           label=fr'Theory $(N-1)b^2/6 = {Rg2_theory_main:.2f}$')
ax.set_xlabel(r'$R_g^2$ [$\sigma^2$]'); ax.set_ylabel('Probability density')
ax.set_title(r'Distribution of $R_g^2$  ($n=$' + f'{total_samples})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# D: Gaussian P(R) check — does P(R) match the theoretical Gaussian distribution?
ax = axes[1, 0]
R_magnitudes = np.sqrt(R2_main)
hist_bins    = np.linspace(0, R_magnitudes.max() * 1.05, 50)
ax.hist(R_magnitudes, bins=hist_bins, density=True,
        color='mediumpurple', alpha=0.7,
        label=f'Simulation ({total_samples} samples)')
R_line    = np.linspace(0.01, hist_bins[-1], 300)
variance  = n_bonds_main * BOND_LENGTH**2 / 3.0   # per-component variance = Nb²/3
P_gauss   = (4 * np.pi * R_line**2
             * (1 / (2 * np.pi * variance))**1.5
             * np.exp(-R_line**2 / (2 * variance)))
ax.plot(R_line, P_gauss, 'r-', lw=2.5, label=r'Gaussian theory $P(R)$')
ax.set_xlabel(r'$R$ [$\sigma$]'); ax.set_ylabel('P(R)')
ax.set_title('Gaussian Distribution Check')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# E: Scaling plot — does <Rg²> grow linearly with chain length as Nb²/6?
ax = axes[1, 1]
chain_lengths_arr = np.array(SCALING_CHAIN_LENGTHS)
Rg2_sim_arr       = np.array([scaling_mean_Rg2[n] for n in SCALING_CHAIN_LENGTHS])
N_smooth          = np.linspace(chain_lengths_arr.min(), chain_lengths_arr.max(), 200)
ax.plot(chain_lengths_arr, Rg2_sim_arr,           'o', color='darkorange', ms=8,
        label=r'Sim $\langle R_g^2\rangle$')
ax.plot(N_smooth,          N_smooth*BOND_LENGTH**2/6, '-', color='red', lw=2,
        label=r'Theory $Nb^2/6$')
ax.set_xlabel('Chain length N'); ax.set_ylabel(r'$\langle R_g^2\rangle$ [$\sigma^2$]')
ax.set_title(r'Scaling: $\langle R_g^2\rangle = Nb^2/6$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# F: Universal ratio — <R²>/<Rg²> should equal 6 for all chain lengths
ax = axes[1, 2]
R2_sim_arr  = np.array([scaling_mean_R2[n] for n in SCALING_CHAIN_LENGTHS])
ratio_arr   = R2_sim_arr / Rg2_sim_arr
ax.plot(chain_lengths_arr, ratio_arr, 'o', color='steelblue', ms=8,
        label='Sim ratio')
ax.axhline(6.0, color='red', lw=2, ls='--', label='Theory = 6')
ax.set_xlabel('Chain length N')
ax.set_ylabel(r'$\langle R^2\rangle / \langle R_g^2\rangle$')
ax.set_title(r'Universal ratio $\langle R^2\rangle / \langle R_g^2\rangle = 6$')
ax.set_ylim(0, 10); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path("fjc_analysis.png"), dpi=150, bbox_inches='tight')

# =============================================================================
# PART 5: 3D CONFORMATIONS — show 3 different chains from the last frame
# =============================================================================
last_frame_index = total_frames - 1
fig2 = plt.figure(figsize=(15, 5))
fig2.suptitle(
    f"3D Conformations — 3 independent chains at last frame (chain_length={N_MAIN})",
    fontsize=13)
chain_colors = ['steelblue', 'darkorange', 'mediumseagreen']
for plot_idx, (mol_id, color) in enumerate(zip([1, 2, 3], chain_colors)):
    coords = read_single_chain_frame(
        path("main_run.dump"), last_frame_index, N_MAIN, N_CHAINS_PER_BOX, mol_id=mol_id)
    if coords is not None:
        ax = fig2.add_subplot(1, 3, plot_idx + 1, projection='3d')
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color=color, lw=1.5, alpha=0.8)
        ax.scatter(*coords[0],  color='green', s=80, label='First bead')
        ax.scatter(*coords[-1], color='red',   s=80, label='Last bead')
        ax.set_xlabel(r'X [$\sigma$]'); ax.set_ylabel(r'Y [$\sigma$]')
        ax.set_zlabel(r'Z [$\sigma$]')
        ax.set_title(f'Chain {mol_id}', fontsize=10)
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(path("fjc_conformations_3d.png"), dpi=150, bbox_inches='tight')

print("\n>>> All done. Files saved to:", DIR)
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lammps import lammps

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
def path(f): return os.path.join(DIR, f)
import sys
sys.stdout.reconfigure(encoding='utf-8')
# =============================================================================
# PARAMETERS
# =============================================================================
# FENE bond — Kremer-Grest standard
FENE_K     = 30.0    # 
FENE_R0    = 1.5     # max extension [σ]; force → ∞ as r → R0
FENE_EPS   = 1.0     # LJ ε inside FENE [ε]
FENE_SIGMA = 1.0     # LJ σ inside FENE [σ]

# WCA pair — good solvent
WCA_EPS    = 1.0
WCA_SIGMA  = 1.0
WCA_CUTOFF = 2.0**(1.0/6.0)   # = 1.12246σ; LJ minimum; shift makes U(r_cut)=0 exactly

BOND_INIT  = 0.9     # [σ]

def box_size_for(chain_length):
    return max(50.0, 10.0 * chain_length**0.6)

# Main detailed run
N_MAIN          = 50      # chain length
N_CHAINS        = 1    # simultaneous chains in one box
EQUIL_STEPS     = 300000  # steps discarded (need longer for real chain: τ_Rouse ~ N²)
PROD_STEPS_MAIN = 600000  # production steps
DUMP_INTERVAL   = 1000    # snapshot every N steps

SCALING_LENGTHS = [10, 20, 30, 50, 75, 100]

def scaling_prod_steps(N):
    return max(300000, int(8000 * N))

DUMP_HEADER_LINES = 9

rng = np.random.default_rng(42)   

# =============================================================================
# write_data — build LAMMPS topology + initial coordinates
# =============================================================================
def write_data(chain_length, n_chains, filepath):
    
    total_atoms = chain_length * n_chains
    total_bonds = (chain_length - 1) * n_chains

    current_box = box_size_for(chain_length)
    grid    = int(np.ceil(n_chains**(1.0/3.0))) + 1
    spacing = current_box / grid
    g_pts   = [spacing * (k + 0.5) for k in range(grid)]
    starts  = [(x, y, z) for x in g_pts for y in g_pts for z in g_pts]
    rng.shuffle(starts)

    all_coords = []
    exclusion_radius = 0.9  # The WCA safety limit for the SAW

    for c in range(n_chains):
        coords    = np.zeros((chain_length, 3))
        coords[0] = np.array(starts[c % len(starts)])
        
        for bead in range(1, chain_length):
            placed = False
            retries = 0
            
            while not placed and retries < 10000:
                theta       = np.arccos(1.0 - 2.0 * rng.random())
                phi         = 2.0 * np.pi * rng.random()
                step        = BOND_INIT * np.array([np.sin(theta) * np.cos(phi),
                                                    np.sin(theta) * np.sin(phi),
                                                    np.cos(theta)])
                candidate_pos = coords[bead - 1] + step
                
                if bead == 1:
                    coords[bead] = candidate_pos
                    placed = True
                else:
                    # Intra-chain overlaps
                    dists_intra = np.linalg.norm(coords[:bead-1] - candidate_pos, axis=1)
                    valid = np.all(dists_intra >= exclusion_radius)
                    
                    # Inter-chain overlaps (Global SAW)
                    if valid and len(all_coords) > 0:
                        for prev_chain in all_coords:
                            dists_inter = np.linalg.norm(prev_chain - candidate_pos, axis=1)
                            if not np.all(dists_inter >= exclusion_radius):
                                valid = False
                                break
                    
                    if valid:
                        coords[bead] = candidate_pos
                        placed = True
                
                retries += 1
            
            if not placed:
                print(f"Warning: SAW trapped at bead {bead}.")
                coords[bead] = candidate_pos
                
        all_coords.append(coords)

    with open(filepath, 'w') as f:
        f.write(f"KG chain  N={chain_length}  n_chains={n_chains}\n\n")
        f.write(f"{total_atoms} atoms\n")
        f.write(f"{total_bonds} bonds\n\n")
        f.write("1 atom types\n")
        f.write("1 bond types\n\n")
        f.write(f"0.0 {current_box:.1f} xlo xhi\n")
        f.write(f"0.0 {current_box:.1f} ylo yhi\n")
        f.write(f"0.0 {current_box:.1f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")

        f.write("Atoms\n\n")
        aid = 1
        for mol_id, coords in enumerate(all_coords, start=1):
            for x, y, z in coords:
                f.write(f"{aid} {mol_id} 1 {x:.5f} {y:.5f} {z:.5f}\n")
                aid += 1

        f.write("\nBonds\n\n")
        bid   = 1
        first = 1
        for _ in range(n_chains):
            for i in range(chain_length - 1):
                f.write(f"{bid} 1 {first + i} {first + i + 1}\n")
                bid += 1
            first += chain_length


# =============================================================================
# parse_dump 
# =============================================================================
def parse_dump(filepath, chain_length, n_chains):
    R2_all, Rg2_all, rg_per_frame = [], [], []
    total_atoms = chain_length * n_chains

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            i += DUMP_HEADER_LINES

            chains_in_frame = {}
            for _ in range(total_atoms):
                p   = lines[i].split(); i += 1
                aid = int(p[0]); mol = int(p[1])
                pos = np.array([float(p[3]), float(p[4]), float(p[5])])
                chains_in_frame.setdefault(mol, {})[aid] = pos

            frame_rg = []
            for mol in sorted(chains_in_frame):
                atoms  = chains_in_frame[mol]
                coords = np.array([atoms[k] for k in sorted(atoms)])

                Rvec = coords[-1] - coords[0]
                R2_all.append(Rvec @ Rvec)

                cm  = coords.mean(axis=0)
                rg2 = np.mean(np.sum((coords - cm)**2, axis=1))
                Rg2_all.append(rg2)
                frame_rg.append(np.sqrt(rg2))

            rg_per_frame.append(np.mean(frame_rg))
        else:
            i += 1

    return np.array(R2_all), np.array(Rg2_all), np.array(rg_per_frame)


def read_single_chain_frame(filepath, frame_idx, chain_length, n_chains, mol_id=1):
    total_atoms = chain_length * n_chains
    with open(filepath) as f:
        lines = f.readlines()
    count = -1; i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            count += 1
            if count == frame_idx:
                i += DUMP_HEADER_LINES
                mol_pos = {}
                for _ in range(total_atoms):
                    p = lines[i].split(); i += 1
                    mol_pos.setdefault(int(p[1]), {})[int(p[0])] = [
                        float(p[3]), float(p[4]), float(p[5])]
                atoms = mol_pos[mol_id]
                return np.array([atoms[k] for k in sorted(atoms)])
        i += 1
    return None


# =============================================================================
# run_lammps 
# =============================================================================
def run_lammps(chain_length, n_chains, data_file, dump_file,
               equil_steps, n_prod_steps, silent=False):
   
    df   = os.path.basename(data_file)
    dp   = os.path.basename(dump_file)
    args = ["-screen", "none", "-log", "none"] if silent else []
    lmp  = lammps(cmdargs=args)

    for cmd in [
        "units lj",
        "atom_style molecular",
        "boundary p p p",
        "special_bonds lj 0 1 1",
        f"read_data {df}",

        "bond_style fene",
        f"bond_coeff 1 {FENE_K} {FENE_R0} {FENE_EPS} {FENE_SIGMA}",

        # ── Jump directly to WCA potential (No Minimizer) ─────────────────
        f"pair_style lj/cut {WCA_CUTOFF:.10f}",
        f"pair_coeff 1 1 {WCA_EPS} {WCA_SIGMA} {WCA_CUTOFF:.10f}",
        "pair_modify shift yes",
        "reset_timestep 0",

        # ── Full NVT equilibration ──────────────────────────────
        "velocity all create 1.0 48279 dist gaussian",
        "fix thermostat all langevin 1.0 1.0 1.0 54321",
        "fix integrator all nve",
        "timestep 0.001",       
        "thermo 10000" if not silent else "thermo 99999999",
        "thermo_style custom step temp epair ebond", 
        f"run {equil_steps}",    
        "unfix thermostat",
        "unfix integrator",
        "reset_timestep 0",

        # ── Production ─────────────────────────────
        f"dump d1 all custom {DUMP_INTERVAL} {dp} id mol type xu yu zu",
        "fix thermostat all langevin 1.0 1.0 1.0 99999",
        "fix integrator all nve",
        f"run {n_prod_steps}",
    ]:
        lmp.command(cmd)
    lmp.close()


# =============================================================================
# PART 1: MAIN DETAILED RUN  (N=50, 10 chains)
# =============================================================================
print(f"=== Part 1: Main run — N={N_MAIN}, {N_CHAINS} chains, good solvent ===")
print(f"    Model: FENE bond (K={FENE_K}, R0={FENE_R0}) + WCA pair (eps={WCA_EPS}, "
      f"r_cut={WCA_CUTOFF:.4f} sigma)")
print(f"    No angle potential — standard flexible Kremer-Grest model")
write_data(N_MAIN, N_CHAINS, path("kg_main.data"))
run_lammps(N_MAIN, N_CHAINS,
           data_file    = path("kg_main.data"),
           dump_file    = path("kg_main.dump"),
           equil_steps  = EQUIL_STEPS,
           n_prod_steps = PROD_STEPS_MAIN,
           silent       = False)
print("    LAMMPS done.")


# =============================================================================
# PART 2: PARSE + STATISTICS
# =============================================================================
print("\n=== Part 2: Parsing trajectory and computing statistics ===")
R2_main, Rg2_main, rg_ts = parse_dump(path("kg_main.dump"), N_MAIN, N_CHAINS)

n_samples     = len(R2_main)
n_frames      = len(rg_ts)
mean_R2_main  = R2_main.mean()
mean_Rg2_main = Rg2_main.mean()

ideal_R2  = (N_MAIN - 1) * 1.0**2
ideal_Rg2 = (N_MAIN - 1) * 1.0**2 / 6.0
swelling  = mean_Rg2_main / ideal_Rg2   

print(f"    {n_frames} frames × {N_CHAINS} chains = {n_samples} samples")
print(f"\n    {'Quantity':<20} {'Simulation':>12} {'Ideal FJC':>12} {'Ratio':>8}")
print(f"    {'─'*56}")
print(f"    {'<R²>':<20} {mean_R2_main:>12.3f} {ideal_R2:>12.3f} {mean_R2_main/ideal_R2:>8.3f}")
print(f"    {'<Rg²>':<20} {mean_Rg2_main:>12.3f} {ideal_Rg2:>12.3f} {swelling:>8.3f}  ← α²")
print(f"    {'<R²>/<Rg²>':<20} {mean_R2_main/mean_Rg2_main:>12.3f} {'6.000':>12} {'(expect ~6)':>8}")
print(f"\n    Swelling α² = {swelling:.3f}")
if 1.1 < swelling < 10:
    print(f"    Chain is swollen vs ideal FJC — good solvent confirmed ✓")
elif swelling <= 1.1:
    print(f"    WARNING: α² ≈ 1 — chain behaves like ideal FJC, check equilibration")
else:
    print(f"    WARNING: α² too large — check for bugs or insufficient equilibration")


# =============================================================================
# PART 3: SCALING STUDY
# =============================================================================
print("\n=== Part 3: Scaling study ===")
sc_R2      = {}
sc_Rg2     = {}
sc_R2_err  = {}
sc_Rg2_err = {}

for N in SCALING_LENGTHS:
    ps = scaling_prod_steps(N)
    print(f"    N={N:3d}  ({ps} prod steps) ...", end=" ", flush=True)
    write_data(N, N_CHAINS, path(f"kg_scale_N{N}.data"))
    run_lammps(N, N_CHAINS,
               data_file    = path(f"kg_scale_N{N}.data"),
               dump_file    = path(f"kg_scale_N{N}.dump"),
               equil_steps  = max(100000, int(5000 * N)),
               n_prod_steps = ps,
               silent       = True)
    
    R2s, Rg2s, _ = parse_dump(path(f"kg_scale_N{N}.dump"), N, N_CHAINS)
    
    # Calculate the ensemble mean
    sc_R2[N]  = R2s.mean()
    sc_Rg2[N] = Rg2s.mean()
    
    # Calculate the Standard Error of the Mean (SEM)
    n_samples = len(R2s)
    sc_R2_err[N]  = R2s.std(ddof=1) / np.sqrt(n_samples)
    sc_Rg2_err[N] = Rg2s.std(ddof=1) / np.sqrt(n_samples)
    
    ideal = (N - 1) / 6.0
    print(f"<Rg²>={sc_Rg2[N]:.3f} ± {sc_Rg2_err[N]:.3f} ideal={ideal:.3f} swelling={sc_Rg2[N]/ideal:.2f}x")


# =============================================================================
# PART 4: PLOTS
# =============================================================================
print("\n=== Part 4: Plotting ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Real Polymer — Good Solvent (FENE + WCA, Kremer-Grest)   "
    f"N={N_MAIN}, {N_CHAINS} chains, kT=1.0",
    fontsize=13)

# ── A: Rg time series ────────────────────────────────────────────────────────
ax = axes[0, 0]
ts_steps = np.arange(n_frames) * DUMP_INTERVAL
ax.plot(ts_steps, rg_ts, lw=0.8, color='steelblue', alpha=0.85,
        label=r'$\langle R_g(t)\rangle$ — mean over chains')
ax.axhline(np.sqrt(mean_Rg2_main), color='red', lw=2, ls='--',
           label=fr'Sim $\sqrt{{\langle R_g^2\rangle}} = {np.sqrt(mean_Rg2_main):.3f}\sigma$')
ax.axhline(np.sqrt(ideal_Rg2), color='gray', lw=1.5, ls=':',
           label=fr'Ideal FJC $= {np.sqrt(ideal_Rg2):.3f}\sigma$')
ax.set_xlabel('Timestep'); ax.set_ylabel(r'$\langle R_g\rangle$ [$\sigma$]')
ax.set_title(r'$R_g$ Time Series — red > gray confirms swelling')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── B: Distribution of R² ────────────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(R2_main, bins=50, density=True, color='darkorange', alpha=0.7,
        label=f'Real chain ({n_samples} samples)')
ax.axvline(mean_R2_main, color='darkorange', lw=2, ls='--',
           label=fr'Sim $\langle R^2\rangle = {mean_R2_main:.2f}$')
ax.axvline(ideal_R2, color='gray', lw=2, ls=':',
           label=fr'Ideal $(N-1)b^2 = {ideal_R2:.2f}$')
ax.set_xlabel(r'$R^2$ [$\sigma^2$]'); ax.set_ylabel('Probability density')
ax.set_title(r'$R^2$ distribution — shift right = excluded volume swelling')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── C: Distribution of Rg² ───────────────────────────────────────────────────
ax = axes[0, 2]
ax.hist(Rg2_main, bins=50, density=True, color='mediumseagreen', alpha=0.7,
        label=f'Real chain ({n_samples} samples)')
ax.axvline(mean_Rg2_main, color='mediumseagreen', lw=2, ls='--',
           label=fr'Sim $\langle R_g^2\rangle = {mean_Rg2_main:.2f}$')
ax.axvline(ideal_Rg2, color='gray', lw=2, ls=':',
           label=fr'Ideal $(N-1)b^2/6 = {ideal_Rg2:.2f}$')
ax.set_xlabel(r'$R_g^2$ [$\sigma^2$]'); ax.set_ylabel('Probability density')
ax.set_title(r'$R_g^2$ distribution')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── D: P(R) — real chain vs ideal Gaussian ────────────────────────────────────
ax = axes[1, 0]
R_vals = np.sqrt(R2_main)
bins   = np.linspace(0, R_vals.max() * 1.05, 50)
ax.hist(R_vals, bins=bins, density=True, color='mediumpurple', alpha=0.7,
        label='Real chain P(R)')
R_line     = np.linspace(0.01, bins[-1], 300)
sig2_ideal = ideal_R2 / 3.0   
P_gauss    = (4 * np.pi * R_line**2
              * (1.0 / (2.0 * np.pi * sig2_ideal))**1.5
              * np.exp(-R_line**2 / (2.0 * sig2_ideal)))
ax.plot(R_line, P_gauss, 'r--', lw=2.5,
        label=r'Ideal Gaussian $P(R)$ (phantom chain)')
ax.set_xlabel(r'$R$ [$\sigma$]'); ax.set_ylabel('P(R)')
ax.set_title('P(R): real chain (purple) shifted right of ideal Gaussian')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── E: Log-log <Rg²> vs N — extract Flory exponent ν ─────────────────────────
ax = axes[1, 1]
Ns_arr      = np.array(SCALING_LENGTHS)
Rg2_arr     = np.array([sc_Rg2[N] for N in SCALING_LENGTHS])
Rg2_err_arr = np.array([sc_Rg2_err[N] for N in SCALING_LENGTHS])

mask      = Ns_arr >= 20
log_N     = np.log(Ns_arr[mask])
log_Rg2   = np.log(Rg2_arr[mask])
slope, intercept = np.polyfit(log_N, log_Rg2, 1)
nu_measured = slope / 2.0

N_smooth     = np.linspace(Ns_arr.min(), Ns_arr.max(), 200)
Rg2_fit      = np.exp(intercept) * N_smooth**slope
Rg2_ideal_ln = (N_smooth - 1) / 6.0

# Using errorbar instead of loglog for the data points
ax.errorbar(Ns_arr, Rg2_arr, yerr=Rg2_err_arr, fmt='o', color='darkorange', ms=9, 
            capsize=4, zorder=5, label=r'Sim $\langle R_g^2\rangle$ (real chain, KG model)')
ax.set_xscale('log')
ax.set_yscale('log')

ax.loglog(N_smooth, Rg2_fit, '-', color='red', lw=2.5,
          label=fr'Power-law fit: $\propto N^{{{slope:.3f}}}$, $\nu={nu_measured:.3f}$')
ax.loglog(N_smooth, Rg2_ideal_ln, '--', color='gray', lw=1.5,
          label=r'Ideal FJC: $(N-1)b^2/6$  ($\nu=0.500$)')
ax.set_xlabel('Chain length N'); ax.set_ylabel(r'$\langle R_g^2\rangle$ [$\sigma^2$]')
ax.set_title(fr'Log-log scaling: $\nu_\mathrm{{fit}}={nu_measured:.3f}$ '
             fr'(theory: $\nu=0.588$, ideal: $\nu=0.500$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

print(f"\n    Measured Flory exponent ν = {nu_measured:.4f}  (N≥20 fit)")
print(f"    Theory (3D good solvent)  ν = 0.5880")
print(f"    Ideal FJC (phantom chain) ν = 0.5000")

# ── F: Swelling factor α² vs N ───────────────────────────────────────────────
ax = axes[1, 2]
ideal_arr        = (Ns_arr - 1) / 6.0
swelling_arr     = Rg2_arr / ideal_arr
# Standard error propagates linearly through division by a constant
swelling_err_arr = Rg2_err_arr / ideal_arr

ax.errorbar(Ns_arr, swelling_arr, yerr=swelling_err_arr, fmt='o-', color='steelblue', 
            ms=8, lw=2, capsize=4, 
            label=r'$\alpha^2(N) = \langle R_g^2\rangle_\mathrm{real} / \langle R_g^2\rangle_\mathrm{ideal}$')

ax.axhline(1.0, color='gray', lw=2, ls='--', label='Ideal chain: α²=1')
ax.fill_between(Ns_arr, 1.0, swelling_arr, alpha=0.15, color='steelblue')
ax.set_xlabel('Chain length N'); ax.set_ylabel(r'Swelling factor $\alpha^2$')
ax.set_title(r'$\alpha^2 > 1$ and growing with $N$ — excluded volume effect')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path("kg_analysis.png"), dpi=150, bbox_inches='tight')

# =============================================================================
# PART 5: 3D CONFORMATIONS — random chains at random frames
# =============================================================================
viz_rng     = np.random.default_rng()

# Dynamically determine how many chains to plot (max 3)
num_to_plot = min(3, N_CHAINS)

rand_mols   = viz_rng.choice(np.arange(1, N_CHAINS + 1), size=num_to_plot, replace=False)
rand_frames = viz_rng.choice(np.arange(n_frames), size=num_to_plot, replace=False)

fig2 = plt.figure(figsize=(15, 5))
fig2.suptitle(
    f"3D Conformations — Kremer-Grest chain, N={N_MAIN}, good solvent  "
    f"(more irregular/extended than ideal FJC)",
    fontsize=12)
colors = ['steelblue', 'darkorange', 'mediumseagreen']

for idx, (mol, frame, color) in enumerate(zip(rand_mols, rand_frames, colors)):
    coords = read_single_chain_frame(
        path("kg_main.dump"), int(frame), N_MAIN, N_CHAINS, mol_id=int(mol))
    if coords is not None:
        # Dynamically set the number of columns based on num_to_plot
        ax = fig2.add_subplot(1, num_to_plot, idx + 1, projection='3d')
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color=color, lw=1.5, alpha=0.8)
        ax.scatter(*coords[0],  color='green', s=80, label='First bead')
        ax.scatter(*coords[-1], color='red',   s=80, label='Last bead')
        ax.set_xlabel(r'X [$\sigma$]'); ax.set_ylabel(r'Y [$\sigma$]')
        ax.set_zlabel(r'Z [$\sigma$]')
        ax.set_title(f'Chain {mol}, frame {frame}', fontsize=10)
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(path("kg_3d.png"), dpi=150, bbox_inches='tight')

print("\n>>> All done. Files saved to:", DIR)
plt.show()

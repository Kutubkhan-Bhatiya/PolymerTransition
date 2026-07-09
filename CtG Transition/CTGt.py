import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lammps import lammps

# All files saved next to this script.
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
def path(f): return os.path.join(DIR, f)

# =============================================================================
# PARAMETERS
# =============================================================================

# FENE bond 
FENE_K     = 30.0
FENE_R0    = 1.5
FENE_EPS   = 1.0
FENE_SIGMA = 1.0

# Full LJ pair — INCLUDES attractive well
LJ_EPS     = 1.0
LJ_SIGMA   = 1.0
LJ_CUTOFF  = 2.5

# Initial bond length — must be << FENE_R0 = 1.5σ to avoid explosion
BOND_INIT  = 0.9

# Chain length for the temperature sweep
N_MAIN     = 50

# Chains per box 
N_CHAINS   = 5

# Temperature sweep
T_VALUES   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# Three representative temperatures for scaling study
T_GOOD    = 5.0
T_THETA   = 3.0
T_POOR    = 1.0

# Scaling chain lengths
SCALING_N  = [10, 20, 30, 40, 50]

def equil_steps(T, N):
    base_steps = 150 * (N ** 2.18)
    t_penalty = max(1.0, 3.0 / T)
    return int(max(150000, (base_steps / 4.0) * t_penalty))

def prod_steps(T, N):
    base_steps = 150 * (N ** 2.18)
    t_penalty = max(1.0, 3.0 / T)
    return int(max(300000, base_steps * t_penalty))

DUMP_INTERVAL     = 1000
DUMP_HEADER_LINES = 9

rng = np.random.default_rng(42)

# =============================================================================
# write_data
# =============================================================================
def write_data(chain_length, n_chains, filepath):
    total_atoms = chain_length * n_chains
    total_bonds = (chain_length - 1) * n_chains

    current_box_size = max(40.0, chain_length * 1.2)

    grid    = int(np.ceil(n_chains**(1.0/3.0))) + 1
    spacing = current_box_size / grid
    g_pts   = [spacing * (k + 0.5) for k in range(grid)]
    starts  = [(x, y, z) for x in g_pts for y in g_pts for z in g_pts]
    rng.shuffle(starts)

    all_coords = []
    for c in range(n_chains):
        coords    = np.zeros((chain_length, 3))
        coords[0] = np.array(starts[c])
        for bead in range(1, chain_length):
            theta        = np.arccos(1.0 - 2.0 * rng.random())
            phi          = 2.0 * np.pi * rng.random()
            step         = BOND_INIT * np.array([np.sin(theta)*np.cos(phi),
                                                 np.sin(theta)*np.sin(phi),
                                                 np.cos(theta)])
            coords[bead] = coords[bead - 1] + step
        all_coords.append(coords)

    with open(filepath, 'w') as f:
        f.write(f"KG coil-to-globule  N={chain_length}  n_chains={n_chains}\n\n")
        f.write(f"{total_atoms} atoms\n{total_bonds} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"0.0 {current_box_size:.1f} xlo xhi\n")
        f.write(f"0.0 {current_box_size:.1f} ylo yhi\n")
        f.write(f"0.0 {current_box_size:.1f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")
        f.write("Atoms\n\n")
        aid = 1
        for mol_id, coords in enumerate(all_coords, start=1):
            for x, y, z in coords:
                f.write(f"{aid} {mol_id} 1 {x:.5f} {y:.5f} {z:.5f}\n")
                aid += 1
        f.write("\nBonds\n\n")
        bid = 1; first = 1
        for _ in range(n_chains):
            for i in range(chain_length - 1):
                f.write(f"{bid} 1 {first+i} {first+i+1}\n")
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
                Rvec   = coords[-1] - coords[0]
                R2_all.append(Rvec @ Rvec)
                cm     = coords.mean(axis=0)
                rg2    = np.mean(np.sum((coords - cm)**2, axis=1))
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
               temperature, n_equil, n_prod, silent=False):

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

        f"pair_style soft {LJ_CUTOFF}",
        "pair_coeff 1 1 100.0",
        "minimize 1.0e-4 1.0e-6 10000 100000",

        f"pair_style lj/cut {LJ_CUTOFF}",
        f"pair_coeff 1 1 {LJ_EPS} {LJ_SIGMA} {LJ_CUTOFF}",
        "pair_modify shift yes",
        "minimize 1.0e-4 1.0e-6 10000 100000",
        "reset_timestep 0",

        f"velocity all create {temperature} 48279 dist gaussian",
        "fix gentle_nve all nve/limit 0.05",
        f"fix gentle_lang all langevin {temperature} {temperature} 1.0 12345",
        "timestep 0.005",
        "thermo 10000" if not silent else "thermo 99999999",
        "thermo_style custom step temp epair ebond",
        "run 50000",
        "unfix gentle_nve",
        "unfix gentle_lang",

        f"fix thermostat all langevin {temperature} {temperature} 1.0 54321",
        "fix integrator all nve",
        "timestep 0.005",
        f"run {n_equil}",
        "unfix thermostat",
        "unfix integrator",
        "reset_timestep 0",

        f"dump d1 all custom {DUMP_INTERVAL} {dp} id mol type xu yu zu",
        f"fix thermostat all langevin {temperature} {temperature} 1.0 99999",
        "fix integrator all nve",
        f"run {n_prod}",
    ]:
        lmp.command(cmd)
    lmp.close()


# =============================================================================
# PART 1: TEMPERATURE SWEEP
# =============================================================================
print(f"=== Part 1: Temperature sweep  N={N_MAIN}, {N_CHAINS} chains ===")
print(f"    T values: {T_VALUES}")
print()

#results_T stores a 4-tuple: (mean_R2, mean_Rg2, std_Rg2, rg_timeseries)
results_T = {}

for T in T_VALUES:
    # n_eq = equil_steps(T, N_MAIN)
    # n_pr = prod_steps(T, N_MAIN)
    # print(f"    T={T:.1f}  equil={n_eq}  prod={n_pr} ...", end=" ", flush=True)
    # write_data(N_MAIN, N_CHAINS, path(f"ctg_T{T:.1f}.data"))
    # run_lammps(N_MAIN, N_CHAINS,
    #            data_file   = path(f"ctg_T{T:.1f}.data"),
    #            dump_file   = path(f"ctg_T{T:.1f}.dump"),
    #            temperature = T,
    #            n_equil     = n_eq,
    #            n_prod      = n_pr,
    #            silent      = True)
    R2s, Rg2s, rg_ts = parse_dump(path(f"ctg_T{T:.1f}.dump"), N_MAIN, N_CHAINS)
    results_T[T] = (R2s.mean(), Rg2s.mean(), Rg2s.std(), rg_ts)
    print(f"<Rg²>={Rg2s.mean():.3f}  <R²>={R2s.mean():.3f}")


# =============================================================================
# PART 2: N-SCALING AT THREE TEMPERATURES
# =============================================================================
print(f"\n=== Part 2: Scaling study at T_good={T_GOOD}, T_theta={T_THETA}, T_poor={T_POOR} ===")

scaling_results = {T: {} for T in [T_GOOD, T_THETA, T_POOR]}

for T in [T_GOOD, T_THETA, T_POOR]:
    print(f"\n  T = {T:.1f}:")
    for N in SCALING_N:
        # n_eq = equil_steps(T, N)
        # n_pr = prod_steps(T, N)
        # print(f"    N={N:3d}  equil={n_eq}  prod={n_pr} ...", end=" ", flush=True)
        # write_data(N, N_CHAINS, path(f"ctg_scale_T{T:.1f}_N{N}.data"))
        # run_lammps(N, N_CHAINS,
        #            data_file   = path(f"ctg_scale_T{T:.1f}_N{N}.data"),
        #            dump_file   = path(f"ctg_scale_T{T:.1f}_N{N}.dump"),
        #            temperature = T,
        #            n_equil     = n_eq,
        #            n_prod      = n_pr,
        #            silent      = True)
        R2s, Rg2s, _ = parse_dump(
            path(f"ctg_scale_T{T:.1f}_N{N}.dump"), N, N_CHAINS)
        scaling_results[T][N] = (R2s.mean(), Rg2s.mean())
        print(f"<Rg²>={Rg2s.mean():.3f}")


# =============================================================================
# PART 3: FLORY EXPONENT FITS
# =============================================================================
print(f"\n=== Part 3: Flory exponent fits ===")
nu_measured = {}

for T in [T_GOOD, T_THETA, T_POOR]:
    Ns   = np.array(SCALING_N)
    Rg2s = np.array([scaling_results[T][N][1] for N in SCALING_N])
    slope, intercept = np.polyfit(np.log(Ns), np.log(Rg2s), 1)
    nu_measured[T] = slope / 2.0
    label = {T_GOOD: "good solvent", T_THETA: "theta point", T_POOR: "poor solvent"}[T]
    print(f"  T={T:.1f} ({label}): 2ν={slope:.3f}  →  ν={nu_measured[T]:.3f}")

print(f"\n  Expected:")
print(f"  Good solvent : ν = 0.588")
print(f"  Theta point  : ν = 0.500")
print(f"  Poor solvent : ν = 0.333")

# Also estimate T_theta from Part 1 data:
# Because the Kremer-Grest model possesses inherent local stiffness 
# (the beads are wider than the equilibrium bond length), it deviates 
# from a phantom FJC. We must apply the Flory Characteristic Ratio (C_inf).
C_inf = 1.75
ideal_Rg2 = C_inf * (N_MAIN - 1) / 6.0

T_arr_full  = np.array(T_VALUES)
Rg2_arr_full = np.array([results_T[T][1] for T in T_VALUES])
# Find crossing by linear interpolation between adjacent points
T_theta_measured = None
for idx in range(len(T_arr_full) - 1):
    if (Rg2_arr_full[idx] - ideal_Rg2) * (Rg2_arr_full[idx+1] - ideal_Rg2) < 0:
        # Linear interpolation
        frac = (ideal_Rg2 - Rg2_arr_full[idx]) / (Rg2_arr_full[idx+1] - Rg2_arr_full[idx])
        T_theta_measured = T_arr_full[idx] + frac * (T_arr_full[idx+1] - T_arr_full[idx])
        break
if T_theta_measured:
    print(f"\n  T_theta (measured from <Rg²> crossing ideal): {T_theta_measured:.2f}")
else:
    print(f"\n  T_theta: no crossing found in T range — check T_VALUES range")


# =============================================================================
# PART 4: PLOTS (PROFESSIONAL PUBLICATION STYLE - FIXED LIMITS)
# =============================================================================
print("\n=== Part 4: Plotting ===")

T_arr   = np.array(T_VALUES)
R2_arr  = np.array([results_T[T][0] for T in T_VALUES])
Rg2_arr = np.array([results_T[T][1] for T in T_VALUES])
Rg2_std = np.array([results_T[T][2] for T in T_VALUES])

# --- Apply Professional Matplotlib Styling ---
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2.0,
    'lines.markersize': 8
})

fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
fig.suptitle(
    f"Coil-to-Globule Transition — FENE + Full LJ ($r_c=2.5\\sigma$) | N={N_MAIN}, {N_CHAINS} chains",
    fontsize=16, fontweight='bold', y=0.98)

c_poor   = '#1f77b4' 
c_theta  = '#2ca02c' 
c_good   = '#d62728' 
c_orange = '#ff7f0e' 

T_crit = T_theta_measured if T_theta_measured else 3.0

# ── A: <Rg²> vs T ────────────────────────────────────────────────────────────
ax = axes[0, 0]
# Calculate the absolute max of the error bar to prevent cutoff
max_error_y = np.max(Rg2_arr + Rg2_std/np.sqrt(N_CHAINS))
y_limit_A = max_error_y * 1.35 # Add 35% empty headroom for the legend

ax.errorbar(T_arr, Rg2_arr, yerr=Rg2_std/np.sqrt(N_CHAINS),
            fmt='o-', color=c_poor, ms=9, lw=2, capsize=5, capthick=1.5,
            label=r'Sim $\langle R_g^2\rangle$', zorder=4)
ax.axhline(ideal_Rg2, color='gray', lw=1.5, ls='--', zorder=1,
           label=fr'Ideal Ref. $C_\infty(N-1)b^2/6={ideal_Rg2:.2f}$')
ax.axvline(T_crit, color='#7f7f7f', lw=1.5, ls=':', zorder=2,
           label=fr'$T_\theta={T_crit:.2f}$')

ax.fill_betweenx([0, y_limit_A], 0, T_crit,
                 alpha=0.08, color=c_poor, label='Poor solvent', zorder=0)
ax.fill_betweenx([0, y_limit_A], T_crit, T_arr.max()+0.5,
                 alpha=0.08, color=c_orange, label='Good solvent', zorder=0)

ax.set_xlabel('Temperature $T$ [$\epsilon/k_B$]')
ax.set_ylabel(r'$\langle R_g^2\rangle$ [$\sigma^2$]')
ax.set_xlim(T_arr.min()-0.2, T_arr.max()+0.2)
ax.set_ylim(0, y_limit_A) # Enforce the new limit
ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgrey')
ax.grid(True, alpha=0.3, ls='--')

# ── B: α²(T) ─────────────────────────────────────────────────────────────────
ax = axes[0, 1]
alpha2 = Rg2_arr / ideal_Rg2
y_limit_B = alpha2.max() * 1.35 # 35% headroom

ax.plot(T_arr, alpha2, 'o-', color=c_orange, ms=9, lw=2.5, zorder=4)
ax.axhline(1.0, color='gray', lw=1.5, ls='--', zorder=2, label=r'Ideal Ref: $\alpha^2=1$')
ax.axvline(T_crit, color='#7f7f7f', lw=1.5, ls=':', zorder=2)

ax.fill_between(T_arr, alpha2, 1.0, where=alpha2 < 1.0,
                alpha=0.15, color=c_poor, label=r'Collapsed ($\alpha^2<1$)', zorder=1)
ax.fill_between(T_arr, alpha2, 1.0, where=alpha2 > 1.0,
                alpha=0.15, color=c_orange, label=r'Swollen ($\alpha^2>1$)', zorder=1)

ax.set_xlabel('Temperature $T$ [$\epsilon/k_B$]')
ax.set_ylabel(r'Swelling Factor $\alpha^2$')
ax.set_ylim(0, y_limit_B)
ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgrey')
ax.grid(True, alpha=0.3, ls='--')

# ── C: P(R) at three temperatures ────────────────────────────────────────────
ax = axes[0, 2]
colors_T = {1.0: c_poor, 3.0: c_theta, 5.0: c_good}
labels_T = {1.0: 'Globule (T=1.0)', 3.0: 'Theta (T=3.0)', 5.0: 'Coil (T=5.0)'}

R_line  = np.linspace(0.01, 30, 300)
sig2    = ideal_Rg2 * 2
P_gauss = (4*np.pi*R_line**2 * (1/(2*np.pi*sig2))**1.5 * np.exp(-R_line**2/(2*sig2)))

ax.plot(R_line, P_gauss, color='black', ls='--', lw=1.8, label='Ideal Gaussian', zorder=5)

for T_plot in [1.0, 3.0, 5.0]:
    R2s, _, _ = parse_dump(path(f"ctg_T{T_plot:.1f}.dump"), N_MAIN, N_CHAINS)
    ax.hist(np.sqrt(R2s), bins=35, density=True, alpha=0.6,
            color=colors_T[T_plot], edgecolor='black', linewidth=0.6, label=labels_T[T_plot], zorder=3)

ax.set_xlabel(r'End-to-End Distance $R$ [$\sigma$]')
ax.set_ylabel('Probability Density $P(R)$')
ax.set_ylim(0, 0.40) # Hardcoded empty space at top for legend
ax.legend(loc='upper right', framealpha=0.95, edgecolor='lightgrey')
ax.grid(True, alpha=0.3, ls='--')

# ── D: Log-log <Rg²> vs N at 3 temperatures ──────────────────────────────────
ax = axes[1, 0]
Ns_arr      = np.array(SCALING_N)
theory_nu   = {T_GOOD: 0.588, T_THETA: 0.500, T_POOR: 0.333}
plot_colors = {T_GOOD: c_good, T_THETA: c_theta, T_POOR: c_poor}
N_smooth    = np.linspace(Ns_arr.min(), Ns_arr.max(), 200)

for T_plot in [T_GOOD, T_THETA, T_POOR]:
    Rg2s = np.array([scaling_results[T_plot][N][1] for N in SCALING_N])
    slope, intercept = np.polyfit(np.log(Ns_arr), np.log(Rg2s), 1)
    nu   = slope / 2.0
    Rg2_fit = np.exp(intercept) * N_smooth**slope
    
    label_str = fr'T={T_plot:.1f}: $\nu={nu:.3f}$ (Th: {theory_nu[T_plot]:.3f})'
    ax.loglog(Ns_arr, Rg2s, 'o', color=plot_colors[T_plot], ms=9, zorder=4)
    ax.loglog(N_smooth, Rg2_fit, '-', color=plot_colors[T_plot], lw=2.2, label=label_str, zorder=3)

ax.set_xlabel('Chain Length $N$')
ax.set_ylabel(r'$\langle R_g^2\rangle$ [$\sigma^2$]')
# Legend in top left is naturally clear of lines going bottom-left to top-right
ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgrey') 
ax.grid(True, alpha=0.3, which='both', ls='--')

# ── E: Rg time series ────────────────────────────────────────────────────────
ax = axes[1, 1]
max_ts_value = 0
for T_plot, col in [(5.0, c_good), (3.0, c_theta), (1.0, c_poor)]:
    _, _, _, rg_ts = results_T[T_plot] 
    if rg_ts.max() > max_ts_value:
        max_ts_value = rg_ts.max() # Find the highest spike across all 3 arrays
    ts = np.arange(len(rg_ts)) * DUMP_INTERVAL
    ax.plot(ts, rg_ts, lw=1.0, color=col, alpha=0.85,
            label=f'T={T_plot:.1f} ($\mu={rg_ts.mean():.1f}\sigma$)')

ax.set_xlabel('Timestep')
ax.set_ylabel(r'Instantaneous $R_g(t)$ [$\sigma$]')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_ylim(1.5, max_ts_value * 1.35) # Clear the upper right entirely for the legend
ax.legend(loc='upper right', framealpha=0.95, edgecolor='lightgrey')
ax.grid(True, alpha=0.3, ls='--')

# ── F: ν(T) ──────────────────────────────────────────────────────────────────
ax = axes[1, 2]
T_fit  = np.array([T_POOR, T_THETA, T_GOOD])
nu_fit = np.array([nu_measured[T] for T in T_fit])

ax.plot(T_fit, nu_fit, 'o', color='rebeccapurple', ms=12, zorder=5,
        markeredgecolor='black', markeredgewidth=1, label=r'Measured Exponent $\nu$')

ax.axhline(0.588, color=c_good,  lw=1.8, ls='--', zorder=2, label='Good Solvent (0.588)')
ax.axhline(0.500, color=c_theta, lw=1.8, ls='--', zorder=2, label='Theta Point (0.500)')
ax.axhline(0.333, color=c_poor,  lw=1.8, ls='--', zorder=2, label='Globule (0.333)')

ax.axvline(T_crit, color='#7f7f7f', lw=1.5, ls=':', zorder=1)

ax.set_xlabel('Temperature $T$ [$\epsilon/k_B$]')
ax.set_ylabel(r'Scaling Exponent $\nu$')
ax.set_ylim(0.20, 0.85) # Creates massive empty space in the top left
ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgrey')
ax.grid(True, alpha=0.3, ls='--')

# --- Final Layout Adjustments ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(path("coil_to_globule_professional.png"), dpi=300, bbox_inches='tight', facecolor='white')
print("    Saved coil_to_globule_professional.png")

# =============================================================================
# PART 5: 3D CONFORMATIONS (FIXED TITLE OVERLAP)
# =============================================================================
from matplotlib.ticker import MaxNLocator

fig2, ax_list = plt.subplots(1, 3, figsize=(16, 5.5),
                              subplot_kw={"projection": "3d"})

# Removed the y=0.98 parameter to allow dynamic placement
fig2.suptitle(
    "Chain Conformations at Different Temperatures (N=50)\n"
    "Fixed Spatial Scale Demonstrating Relative Coil-to-Globule Collapse",
    fontsize=14, fontweight='bold')

viz_rng = np.random.default_rng()
FIXED_SPAN = 12.0

for idx, (ax_3d, T_plot, color) in enumerate(zip(ax_list,
                                 [T_POOR, T_THETA, T_GOOD],
                                 ['#1f77b4', '#2ca02c', '#ff7f0e'])):
    
    _, mean_Rg2_here, _, rg_ts_here = results_T[T_plot]
    n_fr  = len(rg_ts_here)
    frame = int(viz_rng.integers(n_fr // 2, n_fr))
    mol   = int(viz_rng.integers(1, N_CHAINS + 1))
    coords = read_single_chain_frame(
        path(f"ctg_T{T_plot:.1f}.dump"), frame, N_MAIN, N_CHAINS, mol_id=mol)
    
    if coords is not None:
        ax_3d.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                   color='black', lw=1.2, alpha=0.4, zorder=1)
        
        ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      color=color, s=70, alpha=0.95, edgecolor='black', linewidth=0.6, zorder=2)
        
        ax_3d.scatter(*coords[0],  color='#2ca02c', s=120, edgecolor='black', linewidth=1, zorder=5, label='First bead')
        ax_3d.scatter(*coords[-1], color='#d62728', s=120, edgecolor='black', linewidth=1, zorder=5, label='Last bead')
        
        com = coords.mean(axis=0)
        ax_3d.set_xlim(com[0] - FIXED_SPAN, com[0] + FIXED_SPAN)
        ax_3d.set_ylim(com[1] - FIXED_SPAN, com[1] + FIXED_SPAN)
        ax_3d.set_zlim(com[2] - FIXED_SPAN, com[2] + FIXED_SPAN)
        ax_3d.set_box_aspect([1, 1, 1]) 
        
        ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        ax_3d.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1), "linewidth": 0.8})
        ax_3d.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1), "linewidth": 0.8})
        ax_3d.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1), "linewidth": 0.8})
        
        ax_3d.xaxis.set_major_locator(MaxNLocator(4))
        ax_3d.yaxis.set_major_locator(MaxNLocator(4))
        ax_3d.zaxis.set_major_locator(MaxNLocator(4))
        
        ax_3d.view_init(elev=25, azim=45)
        
        label_map = {T_POOR: "Globule", T_THETA: "Theta State", T_GOOD: "Swollen Coil"}
        ax_3d.set_title(
            f'{label_map[T_plot]} (T = {T_plot:.1f})\n'
            fr'$\langle R_g \rangle = {np.sqrt(mean_Rg2_here):.2f}\sigma$',
            fontsize=12, pad=10)
        
        ax_3d.set_xlabel(r'X [$\sigma$]', labelpad=5)
        ax_3d.set_ylabel(r'Y [$\sigma$]', labelpad=5)
        ax_3d.set_zlabel(r'Z [$\sigma$]', labelpad=5)
        
        if idx == 1:
            ax_3d.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='grey')

plt.subplots_adjust(wspace=0.05)

# This forces the layout to reserve the top 10% of the image for the main title, preventing overlap
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig(path("coil_to_globule_3d_professional.png"), dpi=300, bbox_inches='tight', facecolor='white')
print("    Saved coil_to_globule_3d_professional.png")
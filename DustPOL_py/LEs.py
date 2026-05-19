import numpy as np
import matplotlib.pyplot as plt

## SMALL GRAIN CAN NOT HAVE THE PERFECT INTERNAL ALIGNMENT

# Constants and simulation parameters
N_steps = 100000  # number of time steps
dt = 1e-4         # time step
T_gas = 20       # gas temperature [K]
T_dust = 20       # dust temperature [K]
delta_m = 0.01    # magnetic damping factor
tau_gas_eff = 1.0 # normalized gas damping time
tau_ed_eff = 10.0 # normalized electric dipole damping time
tau_m = 100.0     # normalized magnetic alignment time

# Initialize arrays for J' components and beta angle
Jx, Jy, Jz = np.zeros(N_steps), np.zeros(N_steps), np.ones(N_steps)  # start with Jz = 1
beta = np.zeros(N_steps)

# Redefine constants relevant for Bii calculation
I_parallel = 1.0          # normalized moment of inertia along symmetry axis
k_B = 1.38e-16            # Boltzmann constant in erg/K
tau_H_parallel = 1.0      # normalized H damping time
Gtot_parallel = 1.0       # total excitation coefficient (assumed unity for simplicity)
Gtot_perp = 1.0
# Compute Bii using the formula: Bii = 2 * I_parallel * k_B * T_gas / tau_H_parallel * Gtot_parallel
# This gives actual (not dimensionless) Bii, we normalize later if needed

Bxx_body = 2 * I_parallel * k_B * T_gas / tau_H_parallel * Gtot_perp  # erg/s
Byy_body = Bxx_body
# Byy_body = 2 * I_parallel * k_B * T_gas / tau_H_parallel * Gtot_parallel  # erg/s
Bzz_body = 2 * I_parallel * k_B * T_gas / tau_H_parallel * Gtot_parallel  # erg/s

B_par  = Bzz_body
B_perp = Bxx_body #= Byy_body

# Langevin dynamics loop
for t in range(1, N_steps):
    Jx_t, Jy_t, Jz_t = Jx[t-1], Jy[t-1], Jz[t-1]

    # Drift coefficients
    Ax = -Jx_t / tau_gas_eff - (2/3) * Jx_t**3 / tau_ed_eff - Jx_t / tau_m
    Ay = -Jy_t / tau_gas_eff - (2/3) * Jy_t**3 / tau_ed_eff - Jy_t / tau_m
    Az = -Jz_t / tau_gas_eff - (2/3) * Jz_t**3 / tau_ed_eff  # no magnetic alignment along z

    # Diffusion coefficients
    # Directional cosines
    cos_beta = Jz[t-1] / J_mag
    sin_beta = np.sqrt(1 - cos_beta**2)
    beta[t] = np.arccos(np.clip(cos_beta, -1, 1))
    eta[t] = np.arctan2(Jy[t-1], Jx[t-1])

    cos_theta = 1.0 ##perfect internal alignment: theta = 0.0degree <<-- wrong assumption
    sin_theta = 0.0 ##perfect internal alignment: theta = 0.0degree <<-- wrong assumption

    cos_eta = np.cos(eta[t])
    sin_eta = np.sin(eta[t])
    cos2_beta = cos_beta**2
    sin2_beta = sin_beta**2
    cos2_eta = cos_eta**2
    sin2_eta = sin_eta**2

    # Compute Bzz, Bxx, Byy from angular forms (Eqs C3–C5)
    Bzz = (B_par * (0.5 * sin_theta**2 * sin2_beta + cos_theta**2 * cos2_beta) +
           B_perp * (0.5 * (1 + cos_theta**2) * sin2_beta + sin_theta**2 * cos2_beta))

    Bxx = (B_par * (0.5 * sin_theta**2 * (cos2_eta + sin2_eta * cos2_beta) +
                    cos_theta**2 * sin2_eta * sin2_beta) +
           B_perp * (0.5 * (1 + cos_theta**2) * (cos2_eta + sin2_eta * cos2_beta) +
                     sin_theta**2 * sin2_eta * sin2_beta))

    Byy = (B_par * (0.5 * sin_theta**2 * (sin2_eta + cos2_eta * cos2_beta) +
                    cos_theta**2 * cos2_eta * sin2_beta) +
           B_perp * (0.5 * (1 + cos_theta**2) * (sin2_eta + cos2_eta * cos2_beta) +
                     sin_theta**2 * cos2_eta * sin2_beta))

    # Bxx_array[t] = Bxx
    # Byy_array[t] = Byy
    # Bzz_array[t] = Bzz

    # Normalize Bii to make it consistent with dimensionless form used in Langevin equations
    # Normalization factor: 2 * I_parallel * k_B * T_gas / tau_H_parallel
    Bxx_normalized = Bxx * tau_H_parallel / (2 * I_parallel * k_B * T_gas) + T_dust / T_gas * delta_m
    Byy_normalized = Byy * tau_H_parallel / (2 * I_parallel * k_B * T_gas) + T_dust / T_gas * delta_m
    Bzz_normalized = Bzz * tau_H_parallel / (2 * I_parallel * k_B * T_gas)

    # Update using Euler-Maruyama
    dqx, dqy, dqz = np.random.normal(0, np.sqrt(dt), 3)
    Jx[t] = Jx_t + Ax * dt + np.sqrt(Bxx_normalized) * dqx
    Jy[t] = Jy_t + Ay * dt + np.sqrt(Byy_normalized) * dqy
    Jz[t] = Jz_t + Az * dt + np.sqrt(Bzz_normalized) * dqz

    # Compute angle beta (angle between J and B-field, which is along z)
    J_mag = np.sqrt(Jx[t]**2 + Jy[t]**2 + Jz[t]**2)
    beta[t] = np.arccos(Jz[t] / J_mag)

# Convert beta to degrees for plotting
beta_deg = np.degrees(beta)

# Plot results
plt.figure(figsize=(10, 4))
plt.hist(beta_deg, bins=100, density=True, color='skyblue', edgecolor='k')
plt.xlabel('Angle β (degrees)')
plt.ylabel('Probability Density')
plt.title('Distribution of Angle β Between Angular Momentum and Magnetic Field')
plt.grid(True)
plt.tight_layout()
plt.show()

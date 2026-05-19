import numpy as np
import matplotlib.pyplot as plt
from DustPOL_py import DustPOL, constants
from scipy.interpolate import interp1d

Rstar_CM = 3.1e13
AU = constants.au
RSUN = constants.Rsun

# precompute common conversions once
Rstar_rsun = Rstar_CM / RSUN  
r0_au      = 5.20 * Rstar_CM / AU
rin_au     = 8.7  * Rstar_CM / AU
rout_au    = 2e4  * Rstar_CM / AU

# --- Example 9: AGB star with an extinction curve ---
exe = DustPOL('input_template.dustpol',ratd=True, amax=0.25e-4,dust_type='sil',alpha=0.3333)

IK_params = {
    'Mloss': 4.5e-6, # mass loss rate in Msun/yr
    'Tstar':2100, # stellar temperature in K
    'Rstar':Rstar_rsun, # stellar radius in Rsun
    'vwind': 24.0,  # wind velocity in km/s
    'T0': 707, # gas temperature at r0 in K -- define gas temperature (Tram et al. 2020)
    'r0': r0_au, # reference radius in au -- define gas temperature (Tram et al. 2020)
    'alpha_gas': 0.79, # power-law index for gas temperature profile (Tram et al. 2020)
    'rin': rin_au, # inner radius of the envelop in au (where dust starts to form)
    'rout': rout_au, # outer radius of the envelope in au
    'points':200,
    'sampling_type': 'log_space' # sampling type for the radial grid (linear or log)
}

r_los = 0*AU # line of sight distance from the star (0 means directly towards the star)
wum, ext_curve = exe.isoAGB_los_extinction(
                                r_los=r_los, 
                                AGB_params=IK_params, 
                                partial_alignment=True
                                )
fig = plt.figure(1)
ax = fig.gca()
ax.loglog(wum, ext_curve, '-', label=f'line-of-sight: {r_los/AU:.0f} au from center')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Extinction $(A_{\\lambda}/A_{\\rm V})$')
ax.set_title('Extinction Curve for IK Tau Star')
ax.legend(fontsize=20)

fext = interp1d(wum, ext_curve, bounds_error=False, fill_value='extrapolate')
AJ = fext(1.25) # extinction at J band (1.25 micron)
AH = fext(1.65) # extinction at H band (1.65 micron)
AK = fext(2.2)  # extinction at K band (2.2 micron)
print(f"Extinction Ratio: H/J bands : {AH/AJ:.2f} mag")
print(f"Extinction Ratio: K/J bands : {AK/AJ:.2f} mag")
print(f"Extinction Ratio: K/H bands : {AK/AH:.2f} mag")


plt.show()

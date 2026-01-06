import numpy as np
import matplotlib.pyplot as plt
from DustPOL_py import DustPOL
from scipy.interpolate import interp1d

## PARAMETERS CAN BE SET VIA CODE WHEN INITIALIZING THE DustPOL CLASS -- THIS OPTION IS USEFUL FOR FITTING PURPOSES
## To see which parameters can be set, and their names, please check using
## In the IPYTHON console:
## >>> from DustPOL_py import DustPOL
## >>> help(DustPOL)

## Example of how to set advanced parameters for the DustPOL model via code
## for starlight polarization and extinction curve calculations using RAT + DG alignments for Astrodust + PAH grains 
## Astrodust follows Hensley & Draine (2023) size distribution
## PAHs follows Hensley et al. (2023) size distribution
params = {
                "dust_type":"astro+pah",
                "f_max":1.0,
                "f_min":"DG",
                "Bfield":5.0e-6,
                "U" : 1.0,
                "ngas" : 1e3,
                "mean_lam":1.2e-4,
                "gamma": 1.0,
                "GSD_law":"HD23", 
            }
exe = DustPOL('input_template.dustpol', **params)

w,pext=exe.cal_pol_abs(NH=0.0,verbose=True,save_output=True)

_,Aext=exe.extinction_curve(verbose=True)
fA_model=interp1d(w,Aext,bounds_error=False, fill_value="extrapolate")
ext_model = Aext/fA_model(0.55) #A_lambda/A_V
fig,ax=plt.subplots(figsize=(10,6))
ax.loglog(w,pext,label='Polarization')
ax1=ax.twinx()
ax1.loglog(w,ext_model,'r--',label='Extinction Curve')
ax.set_xlabel('Wavelength $(\\rm \\mu m)$')
ax.set_ylabel('Polarization per H $(\\rm p_{ext}/N_H)$')
ax1.set_ylabel('Extinction Curve $(\\rm A_{\\lambda}/N_H)$')
ax.legend(loc='upper left', fontsize=20)
ax1.legend(loc='upper right', fontsize=20)
plt.show()
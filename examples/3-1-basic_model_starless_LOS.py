import numpy as np
import matplotlib.pyplot as plt
from DustPOL_py import DustPOL, constants, tools

## SET THE INPUT PARAMETERS FOR THE MODEL
## 1- The main input parameters can be set directly at the input file '***.dustpol file'

## calling the DustPOL class
exe = DustPOL('input_template_starless.dustpol')
amax=exe.amax*1e4

## 2- Computing the degree of polarization along a certain LOSsc

## Choose the locations of LOSs (distance from the center)
los_range = np.array([0.0,0.1,0.3])*constants.pc

## computing degree of dust polarization along these LOSs
for los in los_range:
    exe.isoCloud_los(
                    los,
                    progress=True,
                    save_output=True,
                    filename_output=f"pol_r0={los/constants.pc:.2f}pc_amax={amax:.2f}"
                    )

## Plotting the results
output_abs = [
    f"{exe.output_dir}/pol_r0={los/constants.pc:.2f}pc_amax={amax:.2f}_abs.dat"
    for los in los_range
]
output_emi = [
    f"{exe.output_dir}/pol_r0={los/constants.pc:.2f}pc_amax={amax:.2f}_emi.dat"
    for los in los_range
]

fig_abs,ax_abs=plt.subplots(figsize=(9,3))
fig_emi,ax_emi=plt.subplots(figsize=(9,3))
tools.analysis.plot_pl(output_abs,color='k',ax=ax_abs)
tools.analysis.plot_pl(output_emi,color='k',ax=ax_emi)
                 
plt.show()

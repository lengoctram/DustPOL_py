import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.weight'] = 'normal'
from DustPOL_py import DustPOL, constants

## SET THE INPUT PARAMETERS FOR THE MODEL
## 1- The main input parameters can be set directly at the input file '***.dustpol file'

exe    = DustPOL('input_template_protostar.dustpol')

## If we don't want to save the output
w,pext_per_Ngas = exe.cal_pol_abs(save_output=False)
fig,ax=plt.subplots()
ax.loglog(w,pext_per_Ngas)
ax.set_xlabel('wavelength $(\\rm \\mu m)$')
ax.set_ylabel('$p_{\\rm ext}/N_{\\rm H}$')

## If we want to have the output on the disk use 'save_output=True' and give "filename_output"
## The output file will be stored at the output_dir directory in the input file
## In this case, the extinction curve (A_lambda/Ngas) for the randomely oriented grains is also written
w,pext_per_Ngas = exe.cal_pol_abs(save_output=True, filename_output='pabs')

plt.show()

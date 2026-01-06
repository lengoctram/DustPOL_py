import matplotlib.pyplot as plt
import os
from DustPOL_py import DustPOL, tools

#### NOTE: IT MUST BE RUN WITH "IF __NAME__ == '__MAIN__':" 
####       TO AVOID PROBLEMS WITH MULTI-PROCESSING ON SOME SYSTEMS (E.G., SPAWN METHOD ON MACOS)

## SET THE INPUT PARAMETERS FOR THE MODEL
## 1- The main input parameters can be set directly at the input file '***.dustpol file'

def model_exec():
    ## calling the DustPOL class
    exe = DustPOL('input_template_starless.dustpol')
    amax=exe.amax*1e4

    ## 2- Computing the degree of polarization on POS for entire cloud (a function of Av)
    filename_output=f'p_amax={amax:.2f}_pos' ## NOTE THE OUTPUT NAME is just the name without _abs.dat or _emi.dat
    # if os.path.exists(filename_output+'_abs.dat') is False: ## to avoid overwriting if the file already exists
    exe.isoCloud_pos(filename_output=filename_output)

    ## 3- Plotting the results of p_vs_wavelength
    _,ax_abs=plt.subplots(figsize=(9,3))
    tools.analysis.plot_pl(
                    f'{exe.output_dir}/{filename_output}_abs.dat',
                    av_range=[1,5,10,20],
                    color='k',ax=ax_abs
                    )

    _,ax_emi=plt.subplots(figsize=(9,3))
    tools.analysis.plot_pl(
                    f'{exe.output_dir}/{filename_output}_emi.dat',
                    av_range=[1,5,10,20],
                    color='k',ax=ax_emi
                    )
    
    ## 4- Plotting the results of p_vs_Av
    
    ## 4-1 pext/Av_vs_Av at 0.65micron for absorption polarization
    _,ax_abs = plt.subplots(figsize=(9,3))
    tools.analysis.plot_pav(
                    f'{exe.output_dir}/{filename_output}_abs.dat',
                    wavelength=0.65,
                    color='k',ax=ax_abs
                    )

    ## 4-2 pem_vs_Av at 850micron for emission polarization    
    _,ax_emi=plt.subplots(figsize=(9,3))
    tools.analysis.plot_pav(
                    f'{exe.output_dir}/{filename_output}_emi.dat',
                    wavelength=850,
                    color='k',ax=ax_emi
                    )

    ## 5- Show the results of p-I for emission polarization
    _,ax_pi=plt.subplots(figsize=(6,6))
    tools.analysis.plot_pI(
                    f'{exe.output_dir}/{filename_output}_emi.dat',
                    wavelength=850,
                    color='k',ax=ax_pi
                    )

if __name__ == "__main__":
    model_exec()
plt.show()

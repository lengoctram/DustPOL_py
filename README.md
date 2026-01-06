---
title: "DustPOL-py: a numerical modeling for linear dust polarization"
---

## DustPOL-py - numerical modelling - v1.8
DustPOL-py computes multi-wavelength polarization of starlight absorption and thermal dust emission based on:
- Radiative Torque alignment (RAT-A)
- Magnetically enhanced RAT (MRAT)
- Radiative Torque Disruption (RAT-D)

Features
- Polarization spectra for diffuse ISM, molecular clouds, dense cores, protostars (POS and LOS).
- High-performance computation (ProcessPoolExecutor or joblib).
- Flexible parameter overrides via code for fitting workflows.
- Silicate, graphite, Astrodust and PAHs (and their combinations) compositions
- Multiple size distribution: MRN, WD01, HD23
- The routine will save the output files (wavelength and degree of polarization) for further analysis. 
- Built-in analysis and plotting routines.

Manuals and GUI (a bit outdated)
- Docs: https://lengoctram.github.io/DustPOL-website/
- Web GUI: https://dustpol-py.streamlit.app

## Installation
** It is recommended to use a virtual environment to 
prevent conflicts with existing Python packages. **
For silicon chip:
  conda create -n DustPOL_py
  conda activate DustPOL_py
  conda config --env --set subdir osx-arm64
  conda install python=3.12, numpy, matplotlib, ...

For Intel chip
  conda create -n DustPOL_py
  conda activate DustPOL_py
  conda config --env --set subdir osx-64
  conda install python=3.12, numpy, matplotlib, ...

1- Download the source files from here

2- Go to the directory

3- From the terminal, type
  In principle
      make install

  Otherwise, try
      pip install .
  or
      pip install -e .

## Authors
```Le Ngoc Tram```, Hyeseung Lee, and Thiem Hoang

## Contributors
Pham N. Diep, Nguyen B. Ngoc, Bao Truong, Ngan Lê

## History:
2025   : Tram incorporated the modelling for a starless and protostar
2025   : Tram incorporated the DG alignment
2025   : Tram modified the main routines for overriding the input parameters (useful for performing fitting)

2025   : Tram optimised the model with cached memories 

2025   : Tram added the PAHs composition

2024   : Tram added the modulation for starless core and embedded high-performance-computation techniques

2024   : Tram re-structured the DustPOL-py infractructure to python class object (modulation)

2024   : Tram implemented a two-phase model: cold and warm dust layers along the LOS

2023   : Tram optimized and improved the code to work with maximum grain size lower than the disruption size

2022   : Thiem implemented MRAT in align.py to account for iron inclusions

2020   : Tram improved Hyeseung's code

2019   : Hyeseung modified the Dustpol Code from Thiem, adding RATD (maximum grain size is higher than the disruption size)

## Dependencies

1- Python 3

2- Numpy

3- Matplotlib

4- Scipy

5- Astropy

6- Joblib for parallelization (installation: https://joblib.readthedocs.io/en/latest/installing.html)

7- Concurrency for parallelization

8- Pands

## Bugs
Please reach out to us at <nle@strw.leidenuniv.nl> or <nle@mpifr-bonn.mpg.de>

## macOS multiprocessing notes
- On macOS (Ventura/Sonoma), Python uses “spawn”. Interactive IPython/Jupyter sessions may fail with ProcessPoolExecutor due to __main__.__spec__=None.
- Solutions:
  - Run scripts as modules (python examples/4-2-basic_model_protostar_POS.py) under a main guard.
  - In interactive environments, the library can fallback to joblib backend='loky'.

## More information and citations

1- Tram et al. (2025) <https://www.aanda.org/articles/aa/pdf/2025/11/aa53917-25.pdf>

2- Tram et al. (2024) <https://www.aanda.org/articles/aa/pdf/2024/09/aa50127-24.pdf>

3- Tram et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T>

4- Lee et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L>






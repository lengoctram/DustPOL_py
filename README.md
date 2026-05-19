# DustPOL_py: a numerical modeling for linear dust polarization (version v1.8)

## Multi-wavelength polarization Modeling
DustPOL_py computes multi-wavelength polarization of starlight absorption and thermal dust emission based on:
- Radiative Torque alignment (RAT-A)
- Magnetically enhanced RAT (MRAT)
- Radiative Torque Disruption (RAT-D)
- Paramagnetic relaxation alignment (DG)

## Features
- Extinction curve through the envelop of an evolved star (AGB or RSG).
- Polarization spectra for diffuse ISM, molecular clouds, dense cores, protostars (POS and LOS).
- Built-in analysis and plotting routines.
- High-performance computation (ProcessPoolExecutor or joblib), with fallbacks.
- Flexible parameter overrides via code for fitting workflows.
- Multiple grain compositions: silicate, graphite, PAH and astrodust, and their combinations.
- Multiple grain size distributions: MRN, WD01, HD23

### Manuals and GUI (not yet updated)
- Docs: https://lengoctram.github.io/DustPOL-website/
- Web GUI: https://dustpol-py.streamlit.app

### Examples 
- Python scripts are provided in the examples/ folder

## Installation
** It is recommended to use a virtual environment to prevent conflicts with existing Python packages. **

0- Silicon chip
  ```
  conda create -n DustPOL_py
  conda activate DustPOL_py
  conda config --env --set subdir osx-arm64
  conda install python=3.12, numpy, matplotlib, ...
  ```

0- Intel chip
  ```
  conda create -n DustPOL_py
  conda activate DustPOL_py
  conda config --env --set subdir osx-64
  conda install python=3.12, numpy, matplotlib, ...
  ```

1- Download the source files
  ```
  Clone: git clone https://github.com/lengoctram/DustPOL_py.git
  ```

2- Go to the directory
  ```
  cd DustPOL_py-main
  ```

3- From the terminal, type

    recommended:  
    
        make install
        
        
    or
        
        pip install .
        
        
    or 
        
        pip install -e .
        
## Authors
**<ins> Le Ngoc Tram** </ins>, Hyeseung Lee, and Thiem Hoang

## Contributors
Pham N. Diep, Nguyen B. Ngoc, Bao Truong, Ngan Lê

## Dependencies

1- Python 3

2- Numpy

3- Matplotlib

4- Scipy

5- Astropy

6- Joblib for parallelization (installation: https://joblib.readthedocs.io/en/latest/installing.html)

7- Concurrency for parallelization

8- Pandas

## Bugs
Please reach out to us at <nle@strw.leidenuniv.nl>

macOS multiprocessing notes

- On macOS (Ventura/Sonoma), Python uses “spawn”. Interactive IPython/Jupyter sessions may fail with ProcessPoolExecutor due to __main__.__spec__=None.

- Solutions:
  
  - Run scripts as modules (python examples/4-2-basic_model_protostar_POS.py) under a main guard.

  - In interactive environments, the library can fallback to joblib backend='loky'.

## More information and citations

1- Tram et al. (2025) <https://www.aanda.org/articles/aa/pdf/2025/11/aa53917-25.pdf>

2- Tram et al. (2024) <https://www.aanda.org/articles/aa/pdf/2024/09/aa50127-24.pdf>

3- Tram et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T>

4- Lee et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L>

## Special thanks
L. Tram wishes to express his gratitude to Prof. Karl M. Menten and Dr. Yannick Giraud-Heraud, who have sadly passed away, for their support and encouragement.

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from DustPOL_py import DustPOL

def dust_model(row):
    parameterDictionary = {
        "U":row.U,
        "ngas":row.ngas,
        "amax":row.amax,
        "f_min":row.fmin,
        # "outputFile":row.outputFile,
        "Tgas":100,
        "align_func":'L20',
        "pstiff":1.0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exe = DustPOL('input_template.dustpol',**parameterDictionary)
        exe.cal_pol_abs(verbose=True, save_output=True, filename_output=row.outputFile)


def make_grids(U_range, ngas_range, amax_range, fmin_range):
    U, ngas, amax, fmin = np.meshgrid(U_range, ngas_range, amax_range, fmin_range, indexing='ij')

    # Flatten and stack the parameter combinations
    parameterSpace = np.stack([U.ravel(), ngas.ravel(), amax.ravel(), fmin.ravel()], axis=1)

    # Create a DataFrame
    model_table = pd.DataFrame(parameterSpace, columns=["U", "ngas", "amax", "fmin"])


    model_table["outputFile"] = model_table.apply(
        lambda row: f"U={row.U:.2f}_ngas={row.ngas:.1e}_amax={row.amax*1e4:.2f}_fmin={row.fmin:.3f}", axis=1
        )

    model_table.head(-1).apply(dust_model, axis=1)


if __name__=='__main__':
    U_range    = np.arange(1,10+1,1)
    ngas_range = np.array([0.1,1.0, 10.0, 50., 100.0])
    amax_range = np.logspace(np.log10(0.1),np.log10(0.5),20)*1e-4#np.arange(0.1,0.5+0.05,0.05)*1e-4
    fmin_range = np.array([0.0,0.05])#np.array([0.0,0.05,0.1,0.15,0.2])

    make_grids(U_range, ngas_range, amax_range,fmin_range)

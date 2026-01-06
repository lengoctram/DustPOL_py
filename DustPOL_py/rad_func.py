from numpy import *
from .read import *
import scipy.integrate as integrate
import os
import warnings
from scipy import interpolate
#from common import path

#suppress warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
H   = 6.625e-27
C   = 3e10
K   = 1.38e-16
amu = 1.6605402e-24 #atomic mass unit
yr  = 365.*24.*60.*60.
# -------------------------------------------------------------------------
# Dust :: w (wavelength), a (grain size), T_dust (dust temperature), alpha (axial ratio)
# -------------------------------------------------------------------------
#alpha = 0.3333
# min/maximum grain size
#amin = arange[0]#1. *1.e-7 # [cm]
#amax = arange[-1]#1. *1.e-4 # [cm]
# w
def wave(path):
    filename = path+"data/LAMBDA.DAT"
    q = genfromtxt(filename,skip_header = 4, dtype=['float'],names=['wave'], usecols= (0))
    w = q['wave'] *1e-4       # in cm
    return w
    
# a
# def a_dust(path,UINDEX):
#     q = readDC(path+"U={:.2f}/SDIST.RES".format(UINDEX),7,1,2,70,2)
#     a_gra = q[0,:,0]
#     a_sil = q[0,:,1]    
#     return a_gra, a_sil

def get_DustEM(path, UINDEX):
    """This routine to read the data from the temperature distribution
        computed by DustEM model
        inputs:
            - path: path-to-data-folder
            - U   : radiation strength
        outputs:
            - a   : grain sizes used in DustEM
            - na  : nsize (na = len(a))
            - nT  : number of grids of Temperature
            - Teq0: temperature at equilibrium for pah0
            - Teq1: temperature at equilibrium for pah1
    """        
    file_path = path + f"TEMP_U={UINDEX:.1f}.RES"
    if not os.path.exists(file_path):
        file_path = path + f"TEMP_U={UINDEX:.0f}.RES"
        if not os.path.exists(file_path):
            file_path = path + f"TEMP_U={UINDEX:.2f}.RES"
    data_lines = [] 
    with open(file_path, "r") as file:
        for line in file:
            # Check if the line starts with '#'
            if line.startswith("#"):
                # Remove the '#' and strip whitespace
                cleaned_line = line[1:].strip()
                data_lines.append(cleaned_line.split())
    file.close()
    data = np.array(data_lines[6:])
    a=data[:,1]; Teq=data[:,2]; nT=eval(data[0,6])
    
    a=np.array(a,dtype=np.float64)
    a=a[:int(len(a)/2)]
    na = len(a)

    Teq=np.array(Teq,dtype=np.float64)
    Teq_pah0=Teq[:na]
    Teq_pah1=Teq[na:]

    return a, na, nT, Teq_pah0, Teq_pah1

# T_dust
def T_dust(path,UINDEX):
    a_init, na, nT, Teq_gra, Teq_sil = get_DustEM(path, UINDEX)

    T_gra = zeros([na,nT]);
    T_sil = zeros([na,nT]);
    
    dP_dlnT_gra = zeros([na, nT])
    dP_dlnT_sil = zeros([na, nT])
    
    q = genfromtxt(path+"TEMP_U={:.2f}.RES".format(UINDEX),skip_header = 8,dtype=['float','float','float','float','float'],names=['T','dP_dlnT','C','U', 'dP/dU'],usecols= (0,1,2,3,4))
    ##print('T=',q['T'])
    for i in range(na):
        for j in range(nT):
            ij = i*nT +j
            ik = na*nT + ij
            T_gra[i,j] = q['T'][ij]
            T_sil[i,j] = q['T'][ik]
            dP_dlnT_gra[i,j] = q['dP_dlnT'][ij]
            dP_dlnT_sil[i,j] = q['dP_dlnT'][ik]
            if T_gra[i,j] <=2.7:
                T_gra[i,j] = 2.7
            if T_sil[i,j] <= 2.7:
                T_sil[i,j] = 2.7

    return [a_init, Teq_gra, Teq_sil, T_gra, T_sil, dP_dlnT_gra, dP_dlnT_sil]

# T_dust
def T_dust_pah(path,UINDEX):
    """This routine to read the data from the temperature distribution
        computed by DustEM model
        inputs:
            - path: path-to-data-folder
            - U   : radiation strength
        outputs:
            - Teq0  : temperature at equilibrium for pah0
            - Teq1  : temperature at equilibrium for pah1
            - T_pah0: temperature range for pah0
            - T_pah1: temperature range for pah1
            - dP_dlnT_pah0: distribution for T_pah0
            - dP_dlnT_pah0: distribution for T_pah1
    """        

    a_pah, na, nT, Teq_pah0, Teq_pah1 = get_DustEM(path, UINDEX)

    T_pah0 = zeros([na,nT]);
    T_pah1 = zeros([na,nT]);
    
    dP_dlnT_pah0 = zeros([na, nT])
    dP_dlnT_pah1 = zeros([na, nT])
    # try:
    #     q = genfromtxt(path+f"TEMP_U={UINDEX:.1f}.RES".format(UINDEX),skip_header = 7,dtype=['float','float','float','float','float'],names=['T','dP_dlnT','C','U', 'dP/dU'],usecols= (0,1,2,3,4))
    # except:
    #     q = genfromtxt(path+f"TEMP_U={UINDEX:.0f}.RES".format(UINDEX),skip_header = 7,dtype=['float','float','float','float','float'],names=['T','dP_dlnT','C','U', 'dP/dU'],usecols= (0,1,2,3,4))
    for fmt in ["{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}"]:
        fname = path + f"TEMP_U={fmt}.RES".format(UINDEX)
        try:
            q = np.genfromtxt(
                fname,
                skip_header=7,
                dtype=['float','float','float','float','float'],
                names=['T','dP_dlnT','C','U', 'dP/dU'],
                usecols=(0,1,2,3,4)
            )
            break
        except Exception as e:
            q = None
    if q is None:
        raise FileNotFoundError(f"Could not find TEMP_U file for UINDEX={UINDEX}! Please check the files!")

    for i in range(na):
        data_block_pah0 = q[i*nT:(i+1)*nT]
        data_block_pah0_arr2d = np.vstack([tuple(row) for row in data_block_pah0])
        T_pah0[i,:]=data_block_pah0_arr2d[:,0]
        dP_dlnT_pah0[i,:]=data_block_pah0_arr2d[:,1]
 
        data_block_pah1 = q[(i+na)*nT:(i+na+1)*nT]
        data_block_pah1_arr2d = np.vstack([tuple(row) for row in data_block_pah1])

        T_pah1[i,:]=data_block_pah1_arr2d[:,0]
        dP_dlnT_pah1[i,:]=data_block_pah1_arr2d[:,1]

        # T_pah0[T_pah0<2.7] = 2.7
        # T_pah1[T_pah1<2.7] = 2.7
    return [a_pah, Teq_pah0, Teq_pah1, T_pah0, T_pah1, dP_dlnT_pah0, dP_dlnT_pah1]

# -------------------------------------------------------------------------
# PLANCK FUNCTION
# -------------------------------------------------------------------------
def planck_1(w,na,T,dP_dlnT):
    nw = len(w)
    nT = len(T[0,:])
    B = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    for i in range(na):
        B_T= zeros([nT, nw])
        for j in range(nT):
            for k in range(nw):
                B_T[j,k] = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,j])-1)

        for m in range(nw):
            B[i,m] = integrate.trapezoid(dP_dlnT[i]*B_T[:,m]/T[i],T[i])

        #printProgressBar(i+1, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    return B

def planck_integrated_Tdust(w_new,a_new,w,a,T,dP_dlnT):
    nw = len(w); na = len(a)
    nT = len(T[0,:])
    B = zeros([na, nw])

    for i in range(na):
        func_dPdT = dP_dlnT[i]
        func_T    = T[i]
        for k in range(nw):
            #for j in range(nT):
            #    B_T[j,k] = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,j])-1)
            B_T      = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,:])-1)
            B[i,k] = integrate.trapezoid(func_dPdT*B_T/func_T, func_T) ##2darray

    ##interpolate to w_new and a_new
    f_B = interpolate.RectBivariateSpline(a, w, B, kx = 5, ky = 5 ) ##allowing extra-polation
    
    # Meshgrid
    A_new, W_new = np.meshgrid(a_new, w_new, indexing='ij')
    B = f_B(a_new,w_new)

    # Mask for out-of-bounds `a` values
    a_min, a_max = a.min(), a.max()
    mask = (A_new < a_min) | (A_new > a_max)

    # Set out-of-bounds values to zero
    B[mask] = 0.0

    return B

def planck_equi(w,na,T):
    """Planck function is computed from the equipartition (Drain 2011)
        T = T0*(a/1e-5)^{-1/15} U^{1/6}
    """
    nw = len(w)
    B = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    for i in range(na):
        # for k in range(nw):
        #     B[i,k]      = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i])-1)

        # for k in range(nw):
        x = (H * C) / (w * K * T[i])               # dimensionless
        pref = (2 * H * C**2) / (w**5)

        Bi = np.empty_like(w)
        with np.errstate(over='ignore', divide='ignore', under='ignore', invalid='ignore'): 
            large = x > 700
            mid = ~large
            Bi[large] = pref[large] * np.exp(-x[large])        # 1/(exp(x)-1) ~ exp(-x)
            Bi[mid] = pref[mid] / (np.exp(x[mid]) - 1)
        B[i,:]      = Bi#= 2*H*C**2/(w)**5 /(exp(H*C/w/K/T[i])-1)
    return B

def planck_function(wavelength, T):
    """
    Compute the blackbody radiation intensity using the Planck function in SI units.
    :param wavelength: Wavelength in meters.
    :param T: Temperature in Kelvin.
    :return: Blackbody radiation intensity (energy density per unit wavelength).
    """
    # Avoid division by zero for wavelength or temperature
    wavelength = np.clip(wavelength, 1e-20, None)  # Prevent zero wavelengths
    if T<1e-6: T=1e-6 # Prevent zero or near-zero temperatures

    # Compute the exponential term safely
    exponent = H * C / (wavelength * K * T)
    safe_exponent = np.clip(exponent, None, 700)  # Clamp exponent to avoid overflow

    # Planck function
    B_lambda = (2 * H * C**2) / (wavelength**5) * (1. / (np.exp(safe_exponent) - 1.))
    return B_lambda 

import numpy as np

def radiation_intensity_uv(wavelength):
    """
    Compute UV radiation intensity for an array of wavelengths in cm.

    Parameters:
        wavelength (array-like): Wavelengths in cm.

    Returns:
        np.ndarray: Radiation intensities for each wavelength.
    """
    wavelength = np.asarray(wavelength)
    wl_um = wavelength * 1e4  # convert from cm to microns

    intensity_uv = np.zeros_like(wavelength)

    # Region 1: 0.134 μm < wl <= 0.246 μm
    mask1 = (wl_um > 0.134) & (wl_um < 0.246)
    intensity_uv[mask1] = 2.373e-14 * wl_um[mask1] ** (-0.6678)

    # Region 2: 0.110 μm < wl <= 0.134 μm
    mask2 = (wl_um > 0.110) & (wl_um < 0.134)
    intensity_uv[mask2] = 6.825e-13 * wl_um[mask2]

    # Region 3: 0.0912 μm < wl <= 0.110 μm
    mask3 = (wl_um > 0.0912) & (wl_um < 0.110)
    intensity_uv[mask3] = 1.287e-9 * wl_um[mask3] ** 4.4172

    return intensity_uv

def radiation_intensity(wavelength, x=1):
    """
    Compute the energy density u_lambda at Av=0
    :param wavelength: Wavelength in centimeters.
    :param x: Scaling factor for the intensity of the radiation field.
    :return: Energy density u_lambda (energy per unit wavelength).
    :UNIT of erg cm-3 cm-1
    """
    # Weighting factors and temperatures
    W = np.array([1e-14, 1.65e-13, 4e-13])  # Weighting factors (dimensionless)
    T = np.array([7500, 4000, 3000])     # Blackbody temperatures (Kelvin)

    # Ultraviolet component contribution (u_UV, assumed constant)
    u_UV = radiation_intensity_uv(wavelength)#/wavelength

    # Compute weighted blackbody contributions
    # blackbody_components = sum(W[i] * (4 * np.pi / C) * planck_function(wavelength, T[i]) for i in range(len(W))) ##unit of Jm-3 m-1 (Jm-3 lam-1)
    # blackbody_components = np.sum( 
        # (W[i] * (4 * np.pi / C) * planck_function(wavelength, T[i]) for i in range(len(W))), axis=0)

    B = np.array([planck_function(wavelength, Ti) for Ti in T])  # shape (n_T, n_wl)
    blackbody_components = (4 * np.pi / C) * np.sum(W[:, None] * B, axis=0)

    # Combine all contributions, scaling by factor x
    u_rad_lam = x * (u_UV + blackbody_components) + (4*np.pi/C) * planck_function(wavelength,2.73)

    if len(wavelength)%2 == 0:
        u_rad = integrate.trapezoid(u_rad_lam,x=wavelength)
        lambA = integrate.trapezoid(u_rad_lam*wavelength,x=wavelength)/integrate.trapezoid(u_rad_lam,x=wavelength)
    else:
        u_rad = integrate.simpson(u_rad_lam,x=wavelength)
        lambA = integrate.simpson(u_rad_lam*wavelength,x=wavelength)/integrate.simpson(u_rad_lam,x=wavelength)
    return u_rad_lam,u_rad,lambA

def uISRF_mathis83(dpc):
    C   = 3e10
    q      = readD('./data/Mathis83/GMC_'+str(dpc)+'_ISRF.dat',2,2)
    lamb   = q[0,:] * 1.e-4 #[cm]
    uISRF  = q[1,:] / lamb / C #u_rad

    lmax   = where(abs(lamb-20.e-4) == min(abs(lamb-20.e-4)))
    lrange = lamb[0:lmax[0][0]]
    urange = uISRF[0:lmax[0][0]]
    u_ISRF = trapezoid(urange,x=lrange)
    return u_ISRF

# compute radiation strength from given radiation field
def LambU_mathis83(Av,uISRF,dpc):
    C   = 3e10
    Wmax = 20.e-4 # select wavelengths less than 20.e-4 cm
    
    q = readD('./data/Mathis83/GMC_'+str(dpc)+'_Av'+str(Av)+'.dat',2,2)
    lamb = q[0,:] * 1.e-4 #[cm]
    urad = q[1,:] / lamb / C #u_rad
    
    # averaged wavelength in a range of 0.1 - 20 microns
    lmax = where(abs(lamb-Wmax) == min(abs(lamb-Wmax)))
    lrange = lamb[0:lmax[0][0]]
    urange = urad[0:lmax[0][0]]
    
    u_rad = trapezoid(urange,x=lrange)
    lambA = trapezoid(urange*lrange,x=lrange)/u_rad
    print('u_rad=',u_rad,'u_ISRF=',uISRF)
    # radiation factor
    U = u_rad / uISRF
    
    return U, lambA

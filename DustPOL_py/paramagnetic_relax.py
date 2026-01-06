import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.integrate import quad
from . import constants
# import constants
K   = constants.K
amu = constants.amu #atomic mass unit
def Gamma_par(a_b_ratio):
    e_ecc = (1-a_b_ratio**2)**(0.5) #eccentricity
    ge_ecc = 1./(2*e_ecc) * np.log((1+e_ecc)/(1-e_ecc))
    
    G = 3 + 4*(1-e_ecc**2)*ge_ecc - e_ecc**(-2) * (1 - (1-e_ecc**2)**(2) * ge_ecc)
    return 3./16 * G

def K_omega_sil(a,rho,Tgas,Tdust,omega_factor=1.0,fp=1./7, n23=1.0):
    tau_2 = 2.9e-12/(fp*n23)#2.9e-11
    I_par = 8*np.pi/15 * rho * a**5
    omega2_th = 2*K*Tgas/I_par
    omega = omega_factor * omega2_th**(0.5)
    
    # term_1 = (Tdust/15)**(-1)
    # term_2 = 1./(1 + (omega*tau_2/2)**2)**2
    # return 1.2e-13 * term_1 * term_2
    chi_0 = 0.03 * n23 * fp * (20./Tdust)
    numerator = chi_0 * tau_2
    denominator = (1. + (omega * tau_2 / 2)**2)**2
    return numerator / denominator

def K_omega_sil_spm(a, rho, Tgas, Tdust, Ncl, phi_sp, omega_factor=1.0, p=5.5):
    I_par = 8*np.pi/15 * rho * a**5
    omega2_th = 2*K*Tgas/I_par
    omega = omega_factor * omega2_th**(0.5)
    
    tau_sp = 1e-9 * np.exp(0.011 * Ncl/Tdust)
    chi_0  = 0.026 * Ncl * phi_sp * (p/5.5)**(2) * (20./Tdust)
    numerator = chi_0 * tau_sp
    denominator = (1 + (omega * tau_sp / 2)**2)**2    
    return numerator / denominator

def K_omega_resonant(a, Tdust, B, n23=1.0, fp=1./7, spm=False, Ncl=1., phi_sp= 1./7):
    """
    Placeholder for resonant relaxation calculation.
    This function should be implemented based on the specific model for resonant relaxation.
    """
    m=6; p=5.5
    if spm:
        chi_0 = 0.026 * Ncl * phi_sp * (p/5.5)**(2) * (20./Tdust)
        tau_2 = 1e-9 * np.exp(0.011 * Ncl/Tdust)  # Adjusted for resonant relaxation
    else:
        chi_0 = 0.03 * n23 * fp * (20./Tdust)
        tau_2 = 2.9e-12/(fp*n23)  # Adjusted for resonant relaxation
    numerator = chi_0 * tau_2

    Tl = 63 * (1e-7/a) #K
    tau_1_inf = 1e-6  # Infinite relaxation time for resonant relaxation
    tau_1 = tau_1_inf * (77/Tdust)**(m+1) * (Tdust/Tl)**m * np.exp(Tdust/Tl) * scipy.special.gamma(m+1) * scipy.special.zeta(m)
    denominator_second_term = 8 * (tau_1/1e6) * (tau_2/2e-9) * (B/5e-6)**2
    demominator = 1 + denominator_second_term 
 
    return numerator/demominator 
       
def delta_mag(a,a_b_ratio,rho,B,nH,Tgas,Tdust,omega_factor=1.0, spm=False, Ncl=0, phi_sp=0):
    mH = 1.00784 * amu

    if spm:
        K_omega_DG = K_omega_sil_spm(a,rho,Tgas,Tdust,Ncl,phi_sp,omega_factor=omega_factor)
    else:
        K_omega_DG = K_omega_sil(a,rho,Tgas,Tdust,omega_factor=omega_factor)
        
    K_omega_res = K_omega_resonant(a,Tdust,B,spm=spm,Ncl=Ncl,phi_sp=phi_sp)
    
    # numerator = np.sqrt(np.pi) * K_omega * B**2 #tau_DG
    # demonator = 1.2*nH*np.sqrt(2*K*mH)*np.sqrt(Tgas) * a * Gamma_par(a_b_ratio) #tau_gas
    # return numerator/demonator
    V_grain = 4./3 * np.pi * a**3
    tau_gas = 3./(4*np.sqrt(np.pi)) * 1./ (1.2*nH*np.sqrt(2*K*mH)*np.sqrt(Tgas) * a**4 * Gamma_par(a_b_ratio))
    tau_DG  = 1./(K_omega_DG * B**2 * V_grain)
    tau_res = 1./(K_omega_res * B**2 * V_grain)
    tau_param  = np.minimum(tau_DG, tau_res)#tau_DG
    return tau_gas/tau_param

def q_x(x):
    """
    Vectorized version of q_x for array or scalar x.
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float64)

    pos = x > 0
    neg = x < 0
    zero=x == 0
    result[pos] = 1./x[pos] * ( ((1.+x[pos])/x[pos])**0.5 * np.arcsinh(np.sqrt(x[pos])) - 1 )
    result[neg] = 1./x[neg] * ( (-(1.+x[neg])/x[neg])**0.5 * np.arcsin(np.sqrt(-x[neg])) - 1 )
    result[zero] = 1./3  # Handle the case where x is zero
    return -1./3 + result

def compute_Q_J(x):
    """
    Vectorized version of compute_Q_J for array or scalar x.
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float64)
    nonzero = x != 0
    result[nonzero] = 3./2 * np.abs(q_x(x[nonzero]))
    # result[x == 0] remains 0
    return result if result.shape != () else result.item()

def integrand(t, xi):
	return t**2 * np.exp(t**2 - xi**2)
	# return t**2 * np.exp(t**2)

def integrand_tot(t,xi):
	return np.exp(t**2 - xi**2)
	# return np.exp(t**2)

# def compute_Q_X(a_b_ratio,Tgas,Tdust):
# 	T_ratio = Tgas/Tdust
# 	h = 2. / (1. + a_b_ratio**2)
# 	J_factor = np.sqrt((1+0.5*a_b_ratio**2)*(1.+1./T_ratio)) 
# 	xi = 1./np.sqrt(2) * J_factor* np.sqrt(T_ratio) * np.sqrt(h - 1) 

# 	# Perform numerical integration
# 	integral, _ = quad(integrand, 0, xi, args=(xi))
# 	integral_tot, _ = quad(integrand_tot, 0, xi, args=(xi))
	
# 	# Compute Q_x from Eq. 38
# 	Q_x = 3./(2.*xi*xi) * integral/integral_tot - 1./2

# 	return Q_x

def compute_Q_X(a_b_ratio, Tgas, Tdust, alpha_DG):
    """
    Vectorized version for arrays of Tgas and Tdust.
    Returns an array of Q_x with the same shape as Tgas/Tdust.
    """
    Tgas = np.asarray(Tgas)
    Tdust = np.asarray(Tdust)
    T_ratio = Tgas / Tdust
    h = 2. / (1. + a_b_ratio**2)
    # The factor of 0.5 is included in the case of gas and IR damping are accounted for
    # J_factor = 0.5 * np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))
    # J_factor = 0.1 * np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))
    J_factor = alpha_DG * np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))
        
    xi = 1. / np.sqrt(2) * J_factor * np.sqrt(T_ratio) * np.sqrt(h - 1)

    # Prepare output array
    Q_x = np.zeros_like(xi, dtype=np.float64)

    # Vectorized numerical integration using np.nditer for efficiency
    it = np.nditer(xi, flags=['multi_index'])
    while not it.finished:
        xi_val = float(it[0])
        integral, _ = quad(integrand, 0, xi_val, args=(xi_val))
        integral_tot, _ = quad(integrand_tot, 0, xi_val, args=(xi_val))
        Q_x[it.multi_index] = 3. / (2. * xi_val * xi_val) * integral / integral_tot - 1. / 2
        it.iternext()

    return Q_x

# def compute_Q_X_v2(a_b_ratio,J2_factor=1):
# 	h = 2. / (1. + a_b_ratio**2)
# 	def fTE(theta):
# 		return np.exp(-J2_factor*(1. + (h-1)*np.sin(theta)*np.sin(theta)))
# 	def integrand(theta):
# 		return np.cos(theta)*np.cos(theta) * fTE(theta) * np.sin(theta)
# 	def integrand_tot(theta):
# 		return fTE(theta) * np.sin(theta)

# 	# Compute Q_x from Eq. 38
# 	Q_x = 3./2 * (quad(integrand,0,np.pi)[0]/quad(integrand_tot,0,np.pi)[0] -1./3)
# 	return Q_x

def compute_R(a,a_b_ratio,rho,B,nH,Tgas,Tdust,alpha_DG,spm=False,Ncl=1.0,phi_sp=1./7):

    T_ratio = Tgas / Tdust
    
    # The factor of 0.5 is included in the case of gas and IR damping are accounted for
    # J_factor = 0.5*np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))
    # J_factor = 0.1*np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))
    J_factor = alpha_DG * np.sqrt((1 + 0.5 * a_b_ratio**2) * (1. + 1. / T_ratio))

    delta = delta_mag(a,a_b_ratio,rho,B,nH,Tgas,Tdust,omega_factor=J_factor,spm=spm,Ncl=Ncl,phi_sp=phi_sp)

    x = delta/(1.+delta) * (Tdust-Tgas)/Tgas
        
    return compute_Q_J(x) * compute_Q_X(a_b_ratio,Tgas,Tdust,alpha_DG)

if __name__ == "__main__":
    # Example usage
    s=1.4;a_b_ratio=1./s
    U = 1.0; nH=1e3 #cm-3
    B=50e-6 #G
    # Tgas=50
    omega_factor=1.0
    
    a_range = np.logspace(np.log10(4e-4),np.log10(0.1),100) * 1.e-4 #cm
    # R_values=[]
    # R_values_TE=[];R_values_spm=[]
    # QX_values=[]
    # QJ_values=[]
    # for a in a_range:    
    Tdust =16.4 * pow(U,1./6) * pow(a_range/1e-5,-1./15)
    Tgas = 3 * Tdust

    delta = delta_mag(a_range,a_b_ratio,3.0,B,nH,Tgas,Tdust,omega_factor=omega_factor)

    x = delta/(1.+delta) * (Tdust-Tgas)/Tgas
    QX_values = compute_Q_X(a_b_ratio,Tgas,Tdust)
    QJ_values = compute_Q_J(x)
    R_values  = compute_R(a_range,a_b_ratio,3.0,B,nH,Tgas,Tgas,Tdust,omega_factor=omega_factor)
    R_values_spm = compute_R(a_range,a_b_ratio,3.0,B,nH,Tgas,Tgas,Tdust,omega_factor=omega_factor,spm=True,Ncl=1.,phi_sp=1./7)
    plt.loglog(a_range*1e4,R_values,'k-',label=rf'R=Q$_{{\sf J}}$Q$_{{\sf X}}$')
    plt.loglog(a_range*1e4,R_values_spm,'r-',label=rf'R=Q$_{{\sf J}}$Q$_{{\sf X}}$')
    plt.loglog(a_range*1e4,QJ_values,'k--',label=rf'Q$_{{\sf J}}$')
    plt.loglog(a_range*1e4,QX_values,'k.',label=rf'Q$_{{\sf X}}$')
    plt.legend()
    plt.ylim([1e-4,1.0])
    plt.show()
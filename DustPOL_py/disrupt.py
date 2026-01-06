from .plt_setup import *
from .read import *
# import align
# import importlib
# import common; importlib.reload(common)
# from common import gamma,rho,Smax,u_ISRF,parallel
# from common import *
from astropy import log
# ------------------------------------------------------
# [DESCRIPTION] disrupted grain size
# 1. wRAT : angular velocity of grain rotation by radiation
#    input:
#	U = radiation field strength
#	a = grain size
#	lambA = mean wavelength of radiation field
#    output:
#	w_rat = angular velocity of grain rotation
#
# 2. a_disrupt : disrupted grain size
#    input : 
#	UINDEX = U = radiation field strength
#	Smax = tensile strength
#    output : 
#	amax = disrupted grain size#		
# ------------------------------------------------------
def wRAT(U,u_ISRF,a,n_gas,T_gas,lambA,gamma,rho):
    na = len(a)
    w_rat = np.zeros((len(a)))
    
    for ia in range(na):
        # --------------- Qgam ---------------
        if a[ia] <= lambA/1.8:
            Qgam = 2 * (lambA/a[ia])**(-2.7)
        else:
            Qgam = 0.4

        # --------------- Torq ---------------
        torq_rat = a[ia]**2 *gamma*(U*u_ISRF) * lambA * Qgam / 2

        # --------------- Wrat ---------------
        #FIR = 9.1e-6 * U**(2/3.) * (30/n_gas) * sqrt(100/T_gas) / a[ia]
        FIR = 4.0e-6 * U**(2/3.) * (30./n_gas) * sqrt(100./T_gas) / a[ia]
        t_gas = 8.74e9 * a[ia] * (rho/3.) * (30./n_gas) * sqrt(100./T_gas)
        t_damp = t_gas / (1+FIR)
        I = 8.*pi*rho*a[ia]**5 / 15.
        w_rat[ia] = torq_rat * t_damp / I *(365*24*60*60) # [Hz] : years -> seconds
    
    return w_rat

class radiative_disruption():
    def __init__(self, parent):
        self.U = parent.U
        self.u_ISRF=parent.u_ISRF
        self.amax=parent.amax
        self.ngas=parent.ngas
        self.Tgas=parent.Tgas
        self.mean_lam=parent.mean_lam
        self.gamma=parent.gamma
        self.Smax=parent.Smax
        self.rho = parent.rho
        self.verbose=parent.verbose

    def a_disrupt(self,a_size):
        wd = 2./a_size * sqrt(self.Smax/self.rho)
        wr = wRAT(self.U,self.u_ISRF,a_size,self.ngas,self.Tgas,self.mean_lam,self.gamma,self.rho)
        ratio = wr/wd
        try:
            idd  = max(where(ratio<=1)[0])
            acrit=a_size[idd+1]
            if (self.verbose):
                log.info('   *** Checking disruption: \033[1;36m occured \033[0m')
                # log.info('   *** a_disr = %.2f(um)'%(acrit*1e4))
                if (acrit>self.amax):
                    log.info('   *** a_disr = %.2f(um) > amax= %.2f'%(acrit*1e4,self.amax*1e4))
                else:
                    log.info('   *** a_disr = %.2f(um)'%(acrit*1e4))
        except:
            lmax=max(where(a_size<=self.amax+0.1*self.amax)[0])
            acrit = a_size[lmax]
            if (self.verbose):
                log.info('   *** Checking disruption: \033[1;36m no \033[0m')

        return acrit
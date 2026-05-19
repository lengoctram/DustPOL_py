import numpy as np
import scipy.integrate as integrate
from astropy import log
from joblib import Parallel, delayed#, Memory
from scipy.interpolate import interp1d
from scipy.special import beta as beta_fn, betainc
from . import align, constants
from  .decorators import auto_refresh

class isoAGB_profile(object):
    def __init__(self):
        self.Tdmax=1500#10**(3.5)
        # self.Rv=4.0
        # nw = 129
        self.wave1 = constants.H*constants.C*1e4/(13.6*constants.eV) * 1e-4 #cm
        self.wave2 = 20.0 * 1e-4 #cm
        # self.wave = np.logspace(np.log10(wave1),np.log10(wave2),nw)#np.exp(np.log(wave2/wave1)*np.arange(nw)/(nw-1) + np.log(wave1))
        # self.lamcgs = self.wave*1e-4

    @auto_refresh
    def isoAGB_model(self,parent):
        self.n0_gas    = parent.ngas
        self.w         = parent.w
        self.points    = parent.points
        self.Mloss     = parent.Mloss
        self.vexp      = parent.vwind
        self.Tstar     = parent.Tstar
        self.Rstar     = parent.Rstar
        self.T0        = parent.T0
        self.r0        = parent.r0
        self.alpha_gas = parent.alpha_gas
        self.rin       = parent.rin
        self.rout      = parent.rout
        sampling_type  = parent.sampling_type

        if self.Mloss is None or self.Mloss <= 0.0:
            raise ValueError("Mloss must be provided and > 0")
        try:
            self.Mloss = float(self.Mloss)
        except:
            raise ValueError("Mloss must be a number (in Msun/yr)")


        if self.points is None:
            raise ValueError("points must be provided")
        try:
            self.points = int(self.points)
        except:
            raise ValueError("points (number of sampling points) must be an integer")

        if sampling_type=='lin_space':
            self.rr = np.linspace(self.rin,self.rout,self.points)
        elif sampling_type=='log_space':
            self.rr = np.logspace(np.log10(self.rin),np.log10(self.rout),self.points)

        self.x = self.y = self.z = np.concatenate((-self.rr[::-1],np.array([0]),self.rr[::1]))
        self.X,  self.Y = np.meshgrid(self.x, self.y); self.Z=self.Y
        
        # compute a grid of the gas density profile
        self.nH = np.array([self.ngas_AGB(self.Mloss,self.vexp)(ri) for ri in self.rr])
        self.fnH = interp1d(self.rr,self.nH,axis=0,fill_value='extrapolate')
            
        return [self.x,self.y,self.z],self.rr#self.rr.max()

    @auto_refresh
    def ngas_AGB(self,Mloss,vexp):
        return lambda r: 1e7*(Mloss/(1e-5)) * (10.0*1e5/vexp) * pow(1e15,2) * pow(r,-2)

    @auto_refresh
    def Tgas_AGB(self,r):
        return self.T0 * pow(r/self.r0,-self.alpha_gas)

    @auto_refresh
    def Tdust_AGB(self,r):
        return self.Tstar * pow(self.Rstar/2/r,2./5)
    
    @auto_refresh
    def Ngas_surface2cell(self,r, Mloss, vexp):
        # integrate n(r) from infinity to r
        return 1e7 * (Mloss/(1e-5)) * (10.0*1e5/vexp) * pow(1e15,2) * pow(r,-1)

    @auto_refresh
    def Ngas_center2cell(self,r, rin, Mloss, vexp):
        # integrate n(r) from rin to r
        return 1e7 * (Mloss/(1e-5)) * (10.0*1e5/vexp) * pow(1e15,2) * (1./rin-1./r)
    
    @auto_refresh
    def optical_depth_local(self,parent,r,f_align=False):
        """Compute the optical depth for a local physical condition, integrated over the grain size distribution.

        Args:
            parent (_type_): global parameters
            f_align (bool, optional): optical depth for non-polarized grains (f_align=False) or polarized grains (f_align=True).

        Returns:
            optical depth per unit radial length integrated over the grain size distribution
        """
        a = parent.a ## to make sure that a is updated automatically when parent.a is updated
        dtau = np.zeros_like(a, dtype=float)
        for dusttype in parent.dust_type.split("+"):
            if dusttype.lower()=='astro' or dusttype.lower()=='astro+pah':
                raise ValueError(f"dust_type '{dusttype}' is not supported in AGB's envelope. Use 'sil','car' or 'sil+car' instead.")

            Qext = getattr(parent, f'Qext_{dusttype}', None)
            dn_da= getattr(parent, f'dn_da_{dusttype}', None)
            if Qext is None or dn_da is None:
                raise ValueError(f"Missing Qext or dn_da for dust type '{dusttype}' on parent. Expected attributes: Qext_{dusttype}, dn_da_{dusttype}")
            
            if (f_align):
                ali_cl = align.alignment_class(parent)
                fa     = ali_cl.f_ali()
                Qpol = getattr(parent, f'Qpol_{dusttype}', None)
                dtau = dtau + (Qext + fa * Qpol*(2./3 - np.sin(parent.B_angle)*np.sin(parent.B_angle))) * np.pi * a**2 * dn_da * self.fnH(r)
            else:
                dtau = dtau + Qext * np.pi * a**2 * dn_da * self.fnH(r)

        if len(a)%2 == 0:
            tau_local = integrate.trapezoid(dtau,a)
        else:
            tau_local = integrate.simpson(dtau,a)
        return tau_local

    @auto_refresh
    def radiation_center2cell(self,r,tau=0):
        """
            This function to calculate the radiation strength at distance r from the center
            input:
                - Tstar: temperature of the protostar/star (in unit of K)
                - r: distance from the center to the cell
            output:
                - U: radiation strength at distance r
        """
        #intrinsic radiation from the AGB star
        ZZ		  = constants.H*constants.C/(constants.K*self.Tstar)
        BBstar	  = (2*constants.H*constants.C**2./self.w**5.)*(1./(np.exp(ZZ/self.w)-1.))    
        ulambda_0 = np.pi/constants.C * BBstar * (self.Rstar/r)**2 
        ulambda   = ulambda_0 * np.exp(-tau)
        
        mask = (self.w>=self.wave1) & (self.w<=self.wave2)
        new_wave = self.w[mask]
        ulambda = ulambda[mask]

        if len(new_wave)%2 == 0:
            U = integrate.trapezoid(ulambda, new_wave)/constants.uISRF
            wave_mean = integrate.trapezoid(ulambda*new_wave, new_wave)/integrate.trapezoid(ulambda, new_wave)
        else:
            U = integrate.simpson(ulambda, new_wave)/constants.uISRF
            wave_mean = integrate.simpson(ulambda*new_wave, new_wave)/integrate.simpson(ulambda, new_wave)
        return U,wave_mean
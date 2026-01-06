import numpy as np
import scipy.integrate as integrate
from astropy import log
from joblib import Parallel, delayed#, Memory
from scipy.interpolate import interp1d
from scipy.special import beta as beta_fn, betainc
from . import extcurves, align, constants
from  .decorators import auto_refresh
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

class isoProtostar_profile(object):
    def __init__(self):
        self.Tdmax=1500#10**(3.5)
        # self.Rv=4.0
        nw = 129
        wave1 = constants.H*constants.C*1e4/(13.6*constants.eV)#0.091 #um
        wave2 = 20.0 #um
        self.wave = np.logspace(np.log10(wave1),np.log10(wave2),nw)#np.exp(np.log(wave2/wave1)*np.arange(nw)/(nw-1) + np.log(wave1))
        self.lamcgs = self.wave*1e-4
        # Initialize placeholders for radial extinction profiles
        self._tauV_profile = None
        self._tauB_profile = None
        self._tauV_interp = None
        self._tauB_interp = None

    @auto_refresh
    def isoProtostar_model(self,parent):
        # self.a=parent.a
        self.n0_gas  = parent.ngas
        self.rout    = parent.rout
        self.nsample = parent.nsample
        self.rflat   = parent.rflat
        self.p       = parent.p
        self.Lstar   = parent.Lstar
        self.Tstar   = parent.Tstar
        sampling_type = parent.sampling_type
        
        if self.Lstar is None or self.Lstar <= 0.0:
            raise ValueError("Lstar must be provided and > 0")
        try:
            self.Lstar = float(self.Lstar)
        except:
            raise ValueError("Lstar must be a number (in erg/s)")

        if self.p is None:
            raise ValueError("Density profile index p must be provided")
        try:
            self.p = float(self.p)
        except:
            raise ValueError("Density profile index p must be a number")

        if self.rout is None or self.rout <= 0.0:
            raise ValueError("rout must be provided and > 0")
        try:
            self.rout = float(self.rout)
        except:
            raise ValueError("rout must be a number (in cm)")

        if self.rflat is None or self.rflat < 0.0:
            raise ValueError("rflat must be provided and >= 0")
        try:
            self.rflat = float(self.rflat)
        except:
            raise ValueError("rflat must be a number (in cm)")
        if self.nsample is None:
            raise ValueError("nsample must be provided")
        try:
            self.nsample = int(self.nsample)
        except:
            raise ValueError("nsample (number of sampling points) must be an integer")

        self.rsub    = self.get_rsub(self.Lstar)        
        if sampling_type=='lin_space':
            self.rr = np.linspace(self.rsub,self.rout,self.nsample)#np.linspace(0,self.rout/5e3,self.nsample)
        elif sampling_type=='log_space':
            self.rr = np.logspace(np.log10(self.rsub*0.001),np.log10(self.rout),self.nsample)

        self.x = self.y = self.z = np.concatenate((-self.rr[::-1],np.array([0]),self.rr[::1]))
        self.X,  self.Y = np.meshgrid(self.x, self.y); self.Z=self.Y
        
        self.nH = np.array([self.ngas_protostar(self.n0_gas,self.rflat,self.p)(ri) for ri in self.rr])
        
        # --- Radius-dependent tau_per_H profiles (V & B) ---
        lamV = 0.55e-4
        lamB = 0.44e-4
        # Compute only if not cached or grain state changed (parent can toggle flag parent._grain_state_dirty)
        needs_recompute = (self._tauV_profile is None or getattr(parent, '_grain_state_dirty', False))
        if needs_recompute:
            self._tauV_profile = self._tau_per_N_lambda_profile(parent, lamV)
            self._tauB_profile = self._tau_per_N_lambda_profile(parent, lamB)
            # Build interpolators (allow extrapolation with end values)
            self._tauV_interp = interp1d(self.rr, self._tauV_profile, kind='linear',
                                         bounds_error=False, fill_value=(self._tauV_profile[0], self._tauV_profile[-1]))
            self._tauB_interp = interp1d(self.rr, self._tauB_profile, kind='linear',
                                         bounds_error=False, fill_value=(self._tauB_profile[0], self._tauB_profile[-1]))
            if hasattr(parent, '_grain_state_dirty'):
                parent._grain_state_dirty = False
        # Backward-compatible attribute names used elsewhere
        self.tauV_per_H = self._tauV_interp  # callable now
        self.tauB_per_H = self._tauB_interp
        
        self.Av_map          = np.zeros((len(self.x),len(self.y)))
        self.Av_map_los      = np.zeros((len(self.x),len(self.y)))
        self.Av_map_2calcule = np.zeros((len(self.x),len(self.z)))
        self.align_map       = np.zeros((len(self.x),len(self.z)))
        self.Tdust_map       = np.zeros((len(self.x),len(self.z)))
        self.wavelength_map  = np.zeros((len(self.x),len(self.z)))        
        return [self.x,self.y,self.z],self.rr#self.rr.max()

    @auto_refresh
    def Av_los_by_dust(self,parent,r0):
        """
            This function to calculate the Av along the line-of-sight at location on POS:r0
            This function to returns the observed Av.
            input:
                - the location on the POS: r0
                    e.g., on the POS (oxy): r0^2 = x^2 + y^2
                    e.g., on the observer plane (oxz): r0 = x
            output:
                - Av at r0
        """
        msk = np.where(self.rr>r0)[0]
        rnew=self.rr[msk]
        nnew=self.nH[msk]
        s = np.sqrt(rnew*rnew - r0*r0) ##conversion variable from 'r' to 's'
        nn0=self.ngas_protostar(self.n0_gas,self.rflat,self.p)(np.array([r0]))
        if isinstance(nn0,float):
            nn0=np.array([nn0])
        n = np.concatenate((nnew[::-1],nn0,nnew[::1]))
        s = np.concatenate((-s[::-1],np.array([0]),s[::1]))
        ds = s[1:]-s[:-1] #[len(n),] array

        w=parent.w
        a=parent.a
        dust_type=parent.dust_type

        if dust_type.lower()=='astro':
            log.info('max(a)=%.3e'%(a.max()*1e4))
            Qext_astro = parent.Qext_astro
            dn_da_astro= parent.dn_da_astro

            fQext_astro   = interp1d(w,Qext_astro,axis=0)
            Qext_astro_V  = fQext_astro(0.55e-4)

            #optical depth of astrodust
            fastro_ext = Qext_astro_V * np.pi *a*a * dn_da_astro
            dtau_astro = integrate.simpson(fastro_ext, a) * n
            tau = integrate.simpson(dtau_astro, s)
            return 1.086*tau
            
        elif dust_type.lower()=='astro+pah':
            Qext_astro = parent.Qext_astro
            Qext_pah   = parent.Qext_pah
            dn_da_astro= parent.dn_da_astro
            dn_da_pah  = parent.dn_da_pah

            fQext_astro = interp1d(w,Qext_astro,axis=0)
            fQext_pah   = interp1d(w,Qext_pah,axis=0)
            Qext_astro_V= fQext_astro(0.55e-4)
            Qext_pah_V  = fQext_pah(0.55e-4)

            fastro_ext  = Qext_astro_V * np.pi *a*a * dn_da_astro
            dtau_astro  = integrate.simpson(fastro_ext, a) * n

            fpah_ext    = Qext_pah_V * np.pi *a*a * dn_da_pah
            dtau_pah    = integrate.simpson(fpah_ext, a) * n

            dtau = dtau_astro + dtau_pah
            tau  = integrate.simpson(dtau, s)
            return 1.086*tau

        else:
            Qext_sil=parent.Qext_sil
            Qext_amCBE=parent.Qext_amCBE
            dust_type = parent.dust_type
            dn_da_sil =parent.dn_da_sil
            dn_da_gra =parent.dn_da_gra
            
            # log.info('max(a)=%.3f (um)'%(a.max()*1e4))
            fQext_sil   = interp1d(w,Qext_sil,axis=0)
            fQext_car   = interp1d(w,Qext_amCBE,axis=0)
            Qext_sil_V  = fQext_sil(0.55e-4)
            Qext_car_V  = fQext_car(0.55e-4)

            # msk = np.where(self.rr>r0)[0]
            # rnew=self.rr[msk]
            # nnew=self.nH[msk]
            # s = np.sqrt(rnew*rnew - r0*r0)
            # nn0=self.ngas_starless(self.n0_gas,self.rflat)(np.array([r0]))
            # if isinstance(nn0,float):
            #     nn0=np.array([nn0])
            # n = np.concatenate((nnew[::-1],nn0,nnew[::1]))
            # s = np.concatenate((-s[::-1],np.array([0]),s[::1]))
            # ds = s[1:]-s[:-1] #[len(n),] array
            #optical depth of silicate
            fsil_ext= Qext_sil_V * np.pi *a*a * dn_da_sil
            dtau_sil = integrate.simpson(fsil_ext, a) * n
            #optical depth of carbon
            fgra_ext= Qext_car_V * np.pi *a*a * dn_da_gra
            dtau_gra = integrate.simpson(fgra_ext, a) * n
            dtau = dtau_sil+dtau_gra

            # f = interp1d(s,dtau,axis=0)
            # tau = romberg(f,s[0],s[-1])[0]
            tau = integrate.simpson(dtau, s)
            # print('Av=',1.086*tau)
            return 1.086*tau
        
    @auto_refresh
    def Av_los_by_gas(self, r0):
        
        msk = np.where(self.rr>r0)[0]
        rnew=self.rr[msk]
        nnew=self.nH[msk]
        s = np.sqrt(rnew*rnew - r0*r0) ##conversion variable from 'r' to 's'
        nn0=self.ngas_protostar(self.n0_gas,self.rflat,self.p)(np.array([r0]))
        if isinstance(nn0,float):
            nn0=np.array([nn0])
        n = np.concatenate((nnew[::-1],nn0,nnew[::1]))
        s = np.concatenate((-s[::-1],np.array([0]),s[::1]))
        
        # Column density Ngas
        if len(s)%2 == 0:
            Ngas = integrate.trapezoid(n, s)
        else:
            Ngas = integrate.simpson(n, s)
        
        # # Radius along each segment for tau(r)
        # r_path = np.sqrt(r0**2 + s**2)
        # tauV_path = self.tauV_per_H(r_path)
        # tauB_path = self.tauB_per_H(r_path)
        
        # if len(s)%2 == 0:
        #     I_V = integrate.trapezoid(n * tauV_path, s)
        #     I_B = integrate.trapezoid(n * tauB_path, s)
        # else:
        #     I_V = integrate.simpson(n * tauV_path, s)
        #     I_B = integrate.simpson(n * tauB_path, s)
        
        # eps = 1e-30
        # Rv_los = float(I_V / max(I_B - I_V, eps))
        Rv_los = 4.0
        return Ngas, (Ngas / 5.8e21) * Rv_los
        
    # @auto_refresh
    # def get_Rv_local(self,parent):
    #     w     = parent.w
    #     a     = parent.a
    #     dn_da = getattr(parent, f'dn_da_{dusttype}', None)

    #     dtau = np.zeros_like(a)
    #     for dusttype in parent.dust_type.split("+"):
            
    #         Qext = getattr(parent, f'Qext_{dusttype}', None) 
    #         dn_da= getattr(parent, f'dn_da_{dusttype}', None)

    #         dtau = dtau + Qext * np.pi * a**2 * dn_da

    #     if len(a)%2 == 0:
    #         tau_per_Ngas = integrate.trapezoid(dtau,a)
    #     else:
    #         tau_per_Ngas = integrate.simpson(dtau,a)

    #     Alamb_per_NH = 1.086*tau_per_Ngas #mag cm^2
    #     fextinction=interp1d(w,Alamb_per_NH,bounds_error=False, fill_value="extrapolate")
    #     Rv_r = fextinction(0.55e-4)/(fextinction(0.44e-4)-fextinction (0.55e-4)) #RV= A_V/(A_B-A_V)
    #     return Rv_r
    
    # @auto_refresh
    def _tau_per_N_lambda(self, parent, lam):
        """
        τ_λ per H nucleus at radius r:
          τ_λ/H(r) = ∫ π a^2 Q_ext(a, λ; r) [dn/da](a; r) da
        weight(a, r) can be used to modulate Q_ext or dn/da radially (e.g., alignment).
        """
        # Deprecated scalar version kept for compatibility; now delegates to profile first element.
        prof = self._tau_per_N_lambda_profile(parent, lam)
        return float(prof[0])

    @auto_refresh
    def _tau_per_N_lambda_profile(self, parent, lam):
        """
        Compute τ_λ per H nucleus as a function of radius over self.rr.
        Allows for optional radial variation of size distributions via user-supplied
        callable attributes on parent named dn_da_<dusttype>_r_func(r, a) -> array(len(a)).
        Returns array tau_λ/H(r_i) with shape (nr,).
        """
        a = parent.a
        rgrid = self.rr
        tau_r = np.zeros_like(rgrid, dtype=float)
        # Prebuild Qext interpolators per dust type (λ only) once
        q_interp = {}
        for dusttype in parent.dust_type.split("+"):
            Qext = getattr(parent, f'Qext_{dusttype}', None)
            if Qext is None:
                continue
            q_interp[dusttype] = interp1d(parent.w, Qext, axis=0, bounds_error=False, fill_value="extrapolate")
        # Loop radii
        even_a = (len(a) % 2 == 0)
        for i, r in enumerate(rgrid):
            dtau_a = 0.0
            for dusttype in parent.dust_type.split("+"):
                if dusttype not in q_interp:
                    continue
                Qlam = q_interp[dusttype](lam)  # (na,)
                # Radial size distribution hook
                radial_func_name = f'dn_da_{dusttype}_r_func'
                if hasattr(parent, radial_func_name):
                    dn_da_r = getattr(parent, radial_func_name)(r, a)
                else:
                    dn_da_r = getattr(parent, f'dn_da_{dusttype}', None)
                if dn_da_r is None:
                    continue
                dtau_a = dtau_a + (Qlam * np.pi * a**2 * dn_da_r)
            if even_a:
                tau_r[i] = integrate.trapezoid(dtau_a, a)
            else:
                tau_r[i] = integrate.simpson(dtau_a, a)
        return tau_r

    # @auto_refresh
    # def get_map_Av(self,parent):
    #     def func_para(i,j):
    #         # print(self.X[i,j],self.Y[i,j])
    #         r0 = np.sqrt(self.X[i,j]*self.X[i,j]+self.Y[i,j]*self.Y[i,j])
    #         return self.Av_func(parent,r0)
    #         # Av_map[i,j]=self.Av_func(r0)
    #         # return Av_map

    #     out=Parallel(n_jobs=-1,verbose=1, prefer='threads')(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.y)))
    #     i=0
    #     for xi in range(len(self.x)):
    #         for yi in range(len(self.y)):
    #             Av_val = out[i]
    #             self.Av_map[xi,yi]=Av_val
    #             i+=1
    #     self.Av_map[self.Av_map==0.0] = np.nan
    #     return self.Av_map

    #     #self.Av_map[self.Av_map==0.0] = np.nan
    #     # return self.Av_map

    @auto_refresh
    def get_map_Av_surface2cell(self,parent):
        self.isoProtostar_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            return self.Av_surface2cell(self.n0_gas, self.rflat, self.p)(r)#self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)

        out=Parallel(n_jobs=-1,verbose=1)(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.z)))
        i=0
        for xi in range(len(self.x)):
            for zi in range(len(self.z)):
                Av_val = out[i]
                self.Av_map_2calcule[xi,zi]=Av_val
                i+=1
        self.Av_map_2calcule[self.Av_map_2calcule==0.0] = np.nan
        # pc=constants.pc.cgs.value
        # fig,ax=plt.subplots(figsize=(9,9))
        # im = plt.imshow(self.Av_map_2calcule,interpolation='bilinear',origin='lower',cmap='magma',extent=[self.x[0]/pc,self.x[-1]/pc,self.z[0]/pc,self.z[-1]/pc])
        # cbar=plt.colorbar(im,ax=ax,format='%.2f',shrink=0.8)

        return self.Av_map_2calcule

    @auto_refresh
    def get_map_Av_los(self,parent):
        self.isoProtostar_model(parent)
        def func_para(i,j):
            r0 = np.sqrt(self.X[i,j]*self.X[i,j]+self.Y[i,j]*self.Y[i,j])
            return self.Av_los_by_gas(r0)[1]

        out=Parallel(n_jobs=-1,verbose=1)(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.y)))
        i=0
        for xi in range(len(self.x)):
            for yi in range(len(self.y)):
                Av_val = out[i]
                self.Av_map_los[xi,yi]=Av_val
                i+=1
        self.Av_map_los[self.Av_map_los==0.0] = np.nan
        # pc=constants.pc.cgs.value
        # fig,ax=plt.subplots(figsize=(9,9))
        # im = plt.imshow(self.Av_map_2calcule,interpolation='bilinear',origin='lower',cmap='magma',extent=[self.x[0]/pc,self.x[-1]/pc,self.z[0]/pc,self.z[-1]/pc])
        # cbar=plt.colorbar(im,ax=ax,format='%.2f',shrink=0.8)

        return self.Av_map_los

    @auto_refresh
    def get_map_align(self,parent):
        # self.U0=parent.U
        self.u_ISRF=parent.u_ISRF
        self.rho=parent.rho
        self.amin=parent.amin
        self.amax=parent.amax
        # self.gamma=parent.gamma
        self.T0_gas=parent.Tgas
        # self.mean_lam0=parent.mean_lam
        self.RATalign=parent.RATalign
        self.f_min=parent.f_min
        self.f_max=parent.f_max
        self.a=parent.a
        self.na=parent.na
        self.alpha=parent.alpha
        self.Bfield = parent.Bfield
        self.Ncl = parent.Ncl
        self.phi_sp = parent.phi_sp
        self.fp = parent.fp
        self.align_func = parent.align_func
        self.pstiff = parent.pstiff
        self.verbose= False
        
        ## special parameters (just the names) for radiation routine
        self.U_ISRF=parent.U
        self.mean_lam_ISRF=parent.mean_lam
        
        self.isoProtostar_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                # Av = self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)
                # Av = self.Av_surface2cell(self.n0_gas, self.rflat, self.p)(r)
                
                # self.U = self.U_starless(self.U0,Av)
                # self.Tgas=self.Tgas_starless(self.U0,Av,self.T0_gas)
                # self.mean_lam=self.lamda_starless(self.mean_lam0,Av)
                self.mean_lam, self.U, self.gamma = self.radiation(self,r) ##<<--- updated mean_lam, U, gamma at r
                self.Tgas = self.Tdust_protostar(self.U) #Assume Tgas=Tdust
                self.ngas=self.ngas_protostar(self.n0_gas,self.rflat,self.p)(r)
                ali_cl=align.alignment_class(self)
                return ali_cl.Aligned_Size_v2()

        out=Parallel(n_jobs=-1,verbose=1,prefer='threads')(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.z)))
        i=0
        for xi in range(len(self.x)):
            for zi in range(len(self.z)):
                align_val = out[i]
                self.align_map[xi,zi]=align_val
                i+=1
        # self.align_map[self.Av_map==0.0] = np.nan
        return self.align_map

    @auto_refresh
    def get_test(self,parent):
        self.isoProtostar_model(parent)
        rsub = self.get_rsub(self.Lstar)
        r = np.logspace(np.log10(rsub*0.001),np.log10(self.rout),40)
        Tshell = self.get_Tshell(self.Tstar)
           
        for ri in r:
            Avs = self.Av_centre2cell(self.n0_gas, self.rflat, self.p)(ri)

            _,U = self.radiation_star(ri,rsub,self.Tstar)
            
            _,Ushell = self.radiation_shell(ri,rsub,Tshell)
            Utot = U + Ushell
            Tdust = 16.4*pow(Utot,1./6)
            print('ri/pc=',ri/constants.pc,'Avs=',Avs, f'U={U:.3e}', f'Ushell={Ushell:.3e}', f'Tdust={Tdust:.2f}')

    @auto_refresh
    def get_map_Tdust(self,parent):
        self.U0=parent.U
        # self.T0_gas=parent.Tgas
        self.mean_lam0=parent.mean_lam0

        self.isoProtostar_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                Tdust = self.Tdust_protostar(parent,r)
                # print('tset=', self.Tdust_protostar(self.Lstar,self.Tstar,self.U0,self.mean_lam0,0.009163803996714385*constants.pc))
                return Tdust

        out=Parallel(n_jobs=-1,verbose=1,prefer='threads')(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.z)))
        i=0
        for xi in range(len(self.x)):
            for zi in range(len(self.z)):
                Td_val = out[i]
                self.Tdust_map[xi,zi]=Td_val
                i+=1
        return self.Tdust_map

    @auto_refresh
    def get_map_mean_wavelength(self,parent):
        self.U0=parent.U
        self.T0_gas=parent.Tgas
        self.mean_lam0=parent.mean_lam

        self.isoProtostar_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                # Av=self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)
                Av = self.Av_surface2cell(self.n0_gas, self.rflat, self.p)(r)
                return self.lamda_starless(self.mean_lam0,Av)

        out=Parallel(n_jobs=-1,verbose=1,prefer='threads')(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.z)))
        i=0
        for xi in range(len(self.x)):
            for zi in range(len(self.z)):
                wavelength_val = out[i]
                self.wavelength_map[xi,zi]=wavelength_val*1e4 #um
                i+=1
        return self.wavelength_map

    @auto_refresh
    def ngas_protostar(self,n0,Rflat,p):
        ##Any emprical formulation can be used
        ##here: Hoang et al. 2021 is adopted
        ##Note: r is the radial distance from center to the envelope
        # return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-2.0))
        # return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-3./2))
        # return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-p))
        return lambda r: n0/(1.0+pow(r/Rflat,2)**(p/2))

    # @auto_refresh
    # def Av_2calcule(self,n0,Rflat,p): Plummer sphere
    #     """
    #         This function to calculate the Av from the cloud's surface to the center
    #         This function to returns the Av for radiation attenuation.
    #         *** Note: this Av is differ from the observed Av.
    #         input:
    #             - n0: peaked gas density
    #             - rflat: below which ngas=n0=const.
    #             - Rv: total-to-selective extinction ratio.
    #                   If NA, Rv=4.0 
    #     """
    #     # def Av_inward_ana(r,p,n0,Rflat):
    #     #p=2.0#3./2
    #     # r_ratio = r/Rflat
    #     Av_c = 10.3*(n0/1e8)*(Rflat/(10.*constants.au))*(self.Rv/4.0)
    #     return lambda r: np.where(r<=Rflat, Av_c*(p/(p-1)-r/Rflat), Av_c/(p-1)*pow(r/Rflat,1-p))
    
    @auto_refresh
    def plummer_column_inf_to_r(self,r, n0, Rflat, p):
        """
        N(r) = ∫_{s=r}^{∞} n0 / [1 + (s/Rflat)^2]^{p/2} ds
        Valid for p > 1.
        """
        # x = np.asarray(r, dtype=float) / Rflat
        x = r/Rflat
        a = 0.5
        b = 0.5*(p - 1.0)
        # Regularized incomplete beta argument z = x^2 / (1 + x^2)
        z = (x*x) / (1.0 + x*x)
        J = 0.5 * beta_fn(a, b) * (1.0 - betainc(a, b, z))
        return n0 * Rflat * J

    @auto_refresh
    def plummer_column_0_to_r(self,r, n0, Rflat, p):
        """
        N(r) = ∫_{s=r}^{∞} n0 / [1 + (s/Rflat)^2]^{p/2} ds
        Valid for p > 1.
        """
        # x = np.asarray(r, dtype=float) / Rflat
        x = r/Rflat
        a = 0.5
        b = 0.5*(p - 1.0)
        # Regularized incomplete beta argument z = x^2 / (1 + x^2)
        z = (x*x) / (1.0 + x*x)
        J = 0.5 * beta_fn(a, b) * betainc(a, b, z)
        return n0 * Rflat * J


    @auto_refresh
    def Rv_surface2cell(self, parent, n0, Rflat, p, n_r=256):
        n_r = 256
        def _Rv(r):
            r = float(r)
            rmax = float(self.rout)
            if np.round(r/constants.pc,5) >= np.round(rmax/constants.pc,5):
                return 0.0#np.nan  # undefined at the surface (zero path length)
            r_start = max(r*(1.0+1e-6), r + 1e-12)
            R = np.logspace(np.log10(r_start), np.log10(rmax), int(n_r))  # increasing r -> rout

            # Density and local τ per H along the path
            nR = self.ngas_protostar(n0, Rflat, p)(R)
            tauV = self.tauV_per_H(R)
            tauB = self.tauB_per_H(R)

            # Integrate from r to rout
            I_V = integrate.trapezoid(nR * tauV, R)
            I_B = integrate.trapezoid(nR * tauB, R)

            # 1.086 cancels in the ratio
            eps = 1e-30
            return float(I_V / max(I_B - I_V, eps))
        return _Rv
    
    @auto_refresh
    def Av_surface2cell(self, n0, Rflat, p):
        """
        Exact A_V from infinity to radius r for Plummer-like n(r):
        n(r) = n0 / [1 + (r/Rflat)^2]^{p/2}
        Uses N_H -> A_V conversion: A_V = (N_H / 5.8e21) * Rv
        Returns a callable A_V(r).
        """
                            
        def _Av(r):

            Ngas = self.plummer_column_inf_to_r(r, n0, Rflat, p) 
            # Rv   = self.Rv_surface2cell(parent, n0, Rflat, p)(r)
            Rv=4.0
            return (Ngas / 5.8e21) * Rv
        return _Av

    # @auto_refresh
    # def fun_Av_s(self,r,rsub): #plummer sphere
    #     if r>=rsub:
    #         # print('rsub/pc=',rsub/constants.pc, 'n0_gas=',self.n0_gas,'rflat/au=',self.rflat/constants.au,'p=',self.p)
    #         rr=np.logspace(np.log10(rsub),np.log10(r),100)
    #         Ngas = self.n0_gas * integrate.simpson(1./(1.+pow(rr/self.rflat,self.p)),rr)
    #         Avs = Ngas/5.8e21 * self.Rv
    #     else:
    #         rr=np.logspace(np.log10(r),np.log10(rsub),100)
    #         Avs=0.0
    #     return Avs

    @auto_refresh
    def Rv_centre2cell(self, n0, Rflat, p, n_r=256):
        """
        Return a callable Rv(r): total-to-selective extinction from the center to radius r.
        r is scalar. Uses:
          A_V(r) = 1.086 ∫_0^r n(s) τ_V/H(s) ds
          A_B(r) = 1.086 ∫_0^r n(s) τ_B/H(s) ds
          R_V(r) = A_V / (A_B - A_V)
        """

        # choose a small inner cutoff to avoid singularities at r=0
        rin  = max(self.rsub*1e-3, 1e-12*self.rout)

        def _Rv(r):
            r = float(r)
            if r <= rin:
                return 0.0#np.nan  # zero path length -> undefined
            R = np.logspace(np.log10(rin), np.log10(r), int(n_r))  # increasing 0->r

            # Density and local τ per H along the path
            nR = self.ngas_protostar(n0, Rflat, p)(R)
            tauV = self.tauV_per_H(R)
            tauB = self.tauB_per_H(R)

            # Integrate from 0 to r
            I_V = integrate.trapezoid(nR * tauV, R)
            I_B = integrate.trapezoid(nR * tauB, R)

            eps = 1e-30
            return float(I_V / max(I_B - I_V, eps))

        return _Rv
    
    @auto_refresh
    def Av_centre2cell(self, n0, Rflat, p):
        """
        Exact A_V from radius r to infinity for Plummer-like n(r):
        n(r) = n0 / [1 + (r/Rflat)^2]^{p/2}
        Uses N_H -> A_V conversion: A_V = (N_H / 5.8e21) * Rv
        Returns a callable A_V(r).
        """

        def _Av(r):
            N_r  = self.plummer_column_0_to_r(r, n0, Rflat, p)
            # Rv_r = self.Rv_centre2cell(parent, n0, Rflat, p)(r)
            Rv_r=4.0
            return (N_r / 5.8e21) * Rv_r, Rv_r
        return _Av
    
    @auto_refresh
    def Tdust_protostar(self,Urad):
        """
            This function to calculate the dust temperature at distance r from the protostar/star
            input:
                - Lstar: luminosity of the protostar/star (in unit of Lsun)
                - Tstar: temperature of the protostar/star (in unit of K)
                - r: distance from the center to the cell
            output:
                - Tdust: dust temperature at distance r
        """
        Tdust = pow(Urad,1./6.)*16.4 #K
        return Tdust
        
    @auto_refresh
    def radiation_surface2cell(self,parent,r):
        #For the case of starless core, we parameterize the radiation strength as a function of Av
        #as in Hoang et al. 2021 (Equation 34)
        # log.info('[U_starless]U0=%.2f'%U0)
        # Av = self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)


        U_ISRF = parent.U_ISRF
        mean_lambda_ISR = parent.mean_lam_ISRF
        Av = self.Av_surface2cell(self.n0_gas, self.rflat, self.p)(r)
        wave_mean = mean_lambda_ISR*(1+0.27*Av**0.76)
        Urad = U_ISRF/(1+0.42*pow(Av,1.22))
        return wave_mean, Urad
    
    @auto_refresh
    def get_Tshell(self,Tstar):
        """
            This function to calculate the temperature of the dust shell at rsub
            input:
                - Tstar: temperature of the protostar/star (in unit of K)
            output:
                - Tshell: temperature of the dust shell (in unit of K)
        """
        Tshell = self.Rstar_rsub(Tstar)
        Tshell = Tshell**0.5 * Tstar
        return Tshell

    @auto_refresh
    def get_rsub(self,Lstar):
        """
            This function to calculate the sublimation radius of the dust shell
            input:
                - Lstar: luminosity of the protostar/star (in unit of Lsun)
            output:
                - rsub: sublimation radius of the dust shell (in unit of cm)
        """
        rsub = 155.3*(Lstar/1e6/constants.Lsun)**(0.5) * (self.Tdmax/1500.)**(-5.6/2) * constants.au
        return rsub
    
    @auto_refresh
    def radiation(self,parent,r):
        """
            This function to calculate the radiation strength at distance r
            input:
                - Lstar: luminosity of the protostar/star (in unit of Lsun)
                - Tstar: temperature of the protostar/star (in unit of K)
                - U0: radiation strength at the cloud's surface
                - mean_lambda0: mean wavelength of the radiation field at the cloud's surface (in unit of micron)
                - r: distance from the center to the cell
            output:
                - U: radiation strength at distance r
        """
        Lstar = parent.Lstar
        Tstar = parent.Tstar
            
        rsub = self.get_rsub(Lstar)
        Tshell = self.get_Tshell(Tstar)
           
        uwave_red_star, U_star = self.radiation_star(r,rsub,Tstar)

        if r<rsub:
            U_shell=0.0
            uwave_red_shell=0.0
        else:
            uwave_red_shell,U_shell = self.radiation_shell(r,rsub,Tshell)
        wave_mean_ISRF_att, U_ISRF_att = self.radiation_surface2cell(parent,r)
        
        if U_star+U_shell<U_ISRF_att:
            Utot=U_ISRF_att
            wave_mean=wave_mean_ISRF_att
            gamma=0.3 # anisotropic radiation field
            return wave_mean, Utot,gamma
        else:
            Utot = U_star + U_shell + U_ISRF_att
            gamma=0.8
            # print('r/pc =', r/constants.pc, 'Ustar=', U_star, 'Ushell=', U_shell, 'Utot=', Utot)
            uwave_red = uwave_red_star + uwave_red_shell
            wave_mean = integrate.simpson(uwave_red*self.lamcgs, self.lamcgs)/integrate.simpson(uwave_red, self.lamcgs)
            # if Utot<U_ISRF:
            #     Utot=U_ISRF
            #     wave_mean=wave_mean_ISRF
        
            # Tdust = pow(Utot,1./6.)*16.4 #K
            return wave_mean, Utot, gamma
    
    @auto_refresh
    def Rstar_rsub(self,Tstar):
        f1 = np.sqrt(1e6 * constants.Lsun)
        f2 = 155.3*constants.au * pow(self.Tdmax/1500.,-5.6/2) * np.sqrt(4*np.pi*constants.sigma_SB)*pow(Tstar,2)
        return f1/f2

    @auto_refresh
    def radiation_shell(self,r,rsub,Tshell):
        TEM		= Tshell #temperature of protostars/stars
        W		= 1.0
        ZZ		= constants.H*constants.C/(constants.K*TEM)

        AV,_   = self.Av_centre2cell(self.n0_gas, self.rflat, self.p)(r)
        RV     = 4.0
        #Spectral energy density of a star
        I3stars	= W*(2*constants.H*constants.C**2./self.lamcgs**5.)*(1./(np.exp(ZZ/self.lamcgs)-1.))

        uwave_star = (4*np.pi*self.lamcgs/constants.C)*(I3stars)/self.lamcgs
        
        [A_lambda_AV, _] = extcurves.extcurve_obs(self.wave, RV)

        # AV = AV.reshape(len(AV),1)
        # A_lambda_AV=A_lambda_AV.reshape(1,len(A_lambda_AV))
        tau_wave = AV*A_lambda_AV/1.086
        uwave_red   = uwave_star*np.exp(-tau_wave)
        urad_red	= integrate.simpson(uwave_red,self.lamcgs)
        Urad  = urad_red/constants.uISRF * pow(r/rsub,-2)

        return uwave_red,Urad

    @auto_refresh    
    def radiation_star(self,r,rsub,Tstar):
        TEM		= Tstar #temperature of protostars/stars
        W		= 1.0
        ZZ		= constants.H*constants.C/(constants.K*TEM)

        AV,_   = self.Av_centre2cell(self.n0_gas, self.rflat, self.p)(r)
        RV=4.0
        #Spectral energy density of a star
        I3stars	= W*(2*constants.H*constants.C**2./self.lamcgs**5.)*(1./(np.exp(ZZ/self.lamcgs)-1.))

        uwave_star = (4*np.pi*self.lamcgs/constants.C)*(I3stars)/self.lamcgs

        [A_lambda_AV, _] = extcurves.extcurve_obs(self.wave, RV)

        # AV = AV.reshape(len(AV),1)
        # A_lambda_AV=A_lambda_AV.reshape(1,len(A_lambda_AV))
        tau_wave    = AV*A_lambda_AV/1.086
        uwave_red   = uwave_star*np.exp(-tau_wave)
        urad_red    = integrate.simpson(uwave_red,self.lamcgs)    
        Rstar       = self.Rstar_rsub(Tstar)*rsub
        Urad        = urad_red/constants.uISRF * pow(r/Rstar,-2)
        return uwave_red,Urad

    @auto_refresh
    def func_Tshell(self,Tstar):
        # return pow(4,1./4) * np.sqrt(Rstar_rsub(Tstar)) * Tstar
        return np.sqrt(self.Rstar_rsub(Tstar)) * Tstar

    # def func_Tshell_from_U(self,r,rsub,Ustar):
    #     f = interp1d(r,Ustar)
    #     Ustar_rsub=f(rsub)
    #     a1=16.4*pow(Ustar_rsub,1./6)
    #     a2=19.5*pow(Ustar_rsub,1./5.6)
    #     return (0.625*pow(a1,4) + 0.375*pow(a2,4))**(1./4)

# if __name__=='__main__':
# 	# r = np.logspace(np.log10(rmin),np.log10(rmax),40)
# 	# r = np.linspace(rmin,rmax,50)
# 	r = np.logspace(np.log10(rsub*0.001),np.log10(rmax),40) 
# 	# r = np.logspace(0.0,np.log10(rmax),40) 

# 	# r = np.logspace(np.log10(rsub),np.log10(rmax),40) 
# 	densities    = ngas(r)#(r)
# 	temperatures_star,U_star,wave_star = radiation_star(r,Tstar)
# 	temperatures_shell=np.zeros(len(r))
# 	U_shell=np.zeros(len(r))
# 	wave_shell=np.zeros(len(r))
# 	for i,ri in enumerate(r):
# 		if ri<1.*rsub:
# 			temperatures_shell[i]=0.0
# 			U_shell[i]=0.0
# 		else:
# 			# temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],Tdmax) #this matches better to NM2004 but don't understand whey Tshell=Tsub!
# 			temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],func_Tshell_from_U(r,U_star)) ##This method is identical to the next method
# 			# temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],func_Tshell(Lstar))             ##This method is identical to the previous method

# 	# temperatures_num_shell,Uana_shell = radiation_ana_shell(r,Tdmax)
# 	# temperatures_num_star,Uana_star = radiation_ana_star(r,Tstar)

# 	Utot = U_shell+U_star
# 	Td1 = 19.5*pow(Utot, 1./5.6)#16.4*pow(U,1./6)
# 	Td2 = 16.4*pow(Utot,1./6)
# 	# Td_cal = (0.5*(pow(Td1,4) + pow(Td2,4)))**(1./4)#0.5*(Td1+Td2)
# 	Td_cal = (0.375*pow(Td1,4) + 0.625*pow(Td2,4))**(1./4)#0.5*(Td1+Td2)

# 	# Utot_ana = Uana_shell+Uana_star ##analytical
# 	# Td3 = 19.5*pow(Utot_ana, 1./5.6)#16.4*pow(U,1./6)
# 	# Td4 = 16.4*pow(Utot_ana,1./6)
# 	# Td_ana = 0.5*(Td3+Td4)

# 	fig,ax=plt.subplots(figsize=(8,8))
# 	ax.loglog(r/pc,densities,'k-',ls=ls[keys[0]],lw=2.5,label='Volume density')
# 	ax.loglog([],[],'k',ls=ls[keys[1]],label='Temperature (this work)')
# 	# ax.loglog([],[],'k',ls=ls[keys[2]],label='Temperature (analytical, $n_{\\rm gas}\sim r^{-2.4}$)')

# 	# ax.set_title('low-mass protostar embedded',pad=20)
# 	ax.set_xlabel('$\\sf Distance\\,(pc)$')
# 	ax.set_ylabel('$\\sf Density\\,(cm^{-3})$')
# 	ax.legend(loc='lower left', fontsize=20)
# 	text = sci_notation(Lstar/Lsun,sig_fig=1)
# 	ax.text(0.03,0.3,'$\\sf L_{\\ast}='+str(text)+'L_{\\odot}$', transform=ax.transAxes)

# 	secax = ax.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
# 	secax.set_xscale('log')
# 	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

# 	ax.tick_params(axis='x', which='both', top=False, labeltop=False)
# 	secax.tick_params(axis='x', which='both', top=True)

# 	ax1=ax.twinx()
# 	# ax1.loglog(r/pc,temperatures,'g.')	
# 	ax1.loglog(r/pc,Td_cal,'k',ls=ls[keys[1]],lw=2.5)
# 	# ax1.loglog(r/pc,temperatures_shell,'r',ls=ls[keys[1]])
# 	# ax1.loglog(r/pc,temperatures_star,'b',ls=ls[keys[1]])
# 	# ax1.loglog(r/pc,Td_ana,'k',ls=ls[keys[2]])
# 	# ax1.loglog(r/pc,temperatures_wrong,'r')
# 	# ax1.loglog(r/pc,temperatures_shell,'k',ls=ls[keys[2]])

# 	# ax1.loglog(r/pc,temperatures_shell,'b-')
# 	# ax1.loglog(r/pc,temperatures_star,'b--')
# 	# ax1.loglog(r/pc,temperatures_num_shell,'b--')
# 	# ax1.loglog(r/pc,temperatures_num_star,'b--')
# 	# ax1.loglog(r/au,Tdust_Hoang(r,Lstar/20),'k:')
# 	ax1.set_ylabel('$\\sf Temperature\\,(K)$')
# 	if params=='Sabatini':
# 		ax1.set_ylim(1,3e3)
# 	else:
# 		ax1.set_ylim([10,1000])

# 	fig2,ax2=plt.subplots(figsize=(8,8))
# 	# ax2.loglog(r/pc,Td_cal,'k-',lw=3,label='Total')
# 	# r_nodust = np.logspace(np.log10(rsub*0.001),np.log10(rsub),40) 
# 	#_,Ustar_nodust = radiation_star(r_nodust,Tstar)#
# 	r_nodust=r
# 	Ustar_nodust=2*Lstar/(4*np.pi*r*r * c *uISRF)
# 	ax2.loglog(r/pc,U_star,'k',ls=ls[keys[0]],lw=2.5,label='U(T$_{\\ast}$)')
# 	ax2.loglog(r_nodust/pc,Ustar_nodust,'gray',ls=ls[keys[1]],lw=2.5,label='U(T$_{\\ast}$, nodust)')
# 	ax2.loglog(r/pc,U_shell,'k',ls=ls[keys[2]],lw=2.5,label='U(T$_{\\sf shell}$)')
# 	# ax2.axvline(x=rsub/pc,color='gray')
# 	ax2.legend(loc='lower left',fontsize=20)
# 	ax2.set_xlim([1*au/pc,1])
# 	# ax2.set_ylim([10,1000])
# 	ax2.set_xlabel('$\\sf Distance\\,(pc)$')
# 	ax2.set_ylabel('Radiation (U)')
# 	text = sci_notation(Lstar/Lsun,sig_fig=1)
# 	ax2.text(0.65,0.9,'$\\sf L_{\\ast}='+str(text)+'L_{\\odot}$', transform=ax2.transAxes)

# 	secax = ax2.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
# 	secax.set_xscale('log')
# 	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

# 	ax2.tick_params(axis='x', which='both', top=False, labeltop=False)
# 	secax.tick_params(axis='x', which='both', top=True)

# 	fig3,ax3=plt.subplots(figsize=(8,8))
# 	ax3.loglog(r/pc,wave_star*1e4,'k',ls=ls[keys[0]],label='Stellar radiation')
# 	ax3.loglog(r/pc,wave_shell*1e4,'k',ls=ls[keys[1]],label='Dust shell')
# 	ax3.set_xlabel('$\\sf Distance\\,(pc)$')
# 	ax3.set_ylabel('Mean wavelength, $\\sf \\bar{\\lambda} \\,(\\mu m)$')
# 	ax3.legend(fontsize=20)
# 	ax3.set_xlim([1*au/pc,1])
# 	L_text = sci_notation(Lstar/Lsun,sig_fig=1)
# 	ax3.text(0.6,0.2,'$\\sf L_{\\ast}='+str(L_text)+'L_{\\odot}$', transform=ax3.transAxes)
# 	n_text = sci_notation(nin,sig_fig=1)
# 	ax3.text(0.6,0.12,'$\\sf n_{0}='+str(n_text)+'\, cm^{-3}$', transform=ax3.transAxes)

# 	secax = ax3.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
# 	secax.set_xscale('log')
# 	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

# 	ax3.tick_params(axis='x', which='both', top=False, labeltop=False)
# 	secax.tick_params(axis='x', which='both', top=True)

# 	# fig,ax=plt.subplots(figsize=(8,8))
# 	# ax.loglog(r/au,mean_lam*1e4)
# 	# fig=plt.figure(figsize=(20,6))
# 	# gs  = gridspec.GridSpec(1,2, figure=fig, width_ratios=[5,3],top=0.95,bottom=0.05,left=0.1,right=0.95)
# 	# axs = gs.subplots()
# 	# ax1,ax2=axs

# 	# # files_st = glob.glob('../grid_folder_10Lsun/starts/*.dat')
# 	# files_st = glob.glob('../grid_folder/starts/*.dat')
# 	# files_st.sort(key=os.path.getmtime)

# 	# # files = glob.glob('../grid_folder_10Lsun/*_0.dat')
# 	# files = glob.glob('../grid_folder/*_0.dat')
# 	# files.sort(key=os.path.getmtime)
# 	# for i in range(1,len(files_st),2):
# 	# 	df = uclchem.analysis.read_output_file(files[i])
# 	# 	ax1.semilogy(df['Time']/1e6,df['Density'],'k-')
# 	# 	ax2.plot(df['Time'],df['gasTemp'],'k-')

# 	# 	df = uclchem.analysis.read_output_file(files_st[i])
# 	# 	ax1.semilogy(df['Time']/1e6-df['Time'].max()/1e6,df['Density'],'k-')
# 	# 	ax2.plot(df['Time']-df['Time'].max(),df['gasTemp'],'k-')
# 	# ax1.set(yscale='symlog',ylim=(30,5e8))
# 	# # ax1.set_ylim([0,7e8])
# 	# ax2.set_xscale('symlog')
# 	# ax2.set_xlim([-1.5,1e5])
# 	# ax2.set_ylim([-20,280])

# 	# ax1.set_xlabel('$\\sf Time\\,(Myr)$')
# 	# ax1.set_ylabel('$\\sf Density\\,(cm^{-3})$')
# 	# ax1.text(0.15,0.8,'$\\sf L_{\\ast}=10L_{\\odot}$', transform=ax1.transAxes)
# 	# ax1.text(0.15,0.9,'low-mass protostar embedded', transform=ax1.transAxes)
# 	# ax2.set_xlabel('$\\sf Time\\,(yr)$')
# 	# ax2.set_ylabel('$\\sf Temperature\\,(K)$')

plt.show()
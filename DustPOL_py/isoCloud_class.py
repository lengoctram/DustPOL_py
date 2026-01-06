import numpy as np
import scipy.integrate as integrate
from astropy import log
from joblib import Parallel, delayed#, Memory
from scipy.interpolate import interp1d
from . import align, size_distribution, constants, decorators
from  .decorators import auto_refresh

# import matplotlib.pyplot as plt

class isoCloud_profile(object):
    # def __init__(self,parent):
    #     self.a=parent.a
    #     self.na=parent.na
    #     self.w=parent.w
    #     self.nsample=parent.nsample
    #     self.Qext_sil=parent.Qext_sil
    #     self.Qext_amCBE=parent.Qext_amCBE
    #     self.GSD_law=parent.GSD_law
    #     self.power_index=parent.power_index
    #     self.dust_to_gas_ratio=parent.dust_to_gas_ratio
    #     self.n0_gas=parent.ngas
    #     self.rflat=parent.rflat
    #     self.rout=parent.rout
    #     self.U0  = parent.U
    #     self.u_ISRF=parent.u_ISRF
    #     # self.rho=parent.rho
    #     # self.amin=parent.amin
    #     # self.amax=parent.amax
    #     # self.gamma=parent.gamma
    #     # self.RATalign=parent.RATalign
    #     # self.f_min=parent.f_min
    #     # self.f_max=parent.f_max


    @auto_refresh
    def isoCloud_model(self,parent):
        # self.a=parent.a
        self.n0_gas  = parent.ngas
        self.rout    = parent.rout
        self.nsample = parent.nsample
        self.rflat   = parent.rflat
        self.p       = parent.p
        sampling_type = parent.sampling_type
        
        if self.rout is None:
            raise ValueError("rout (outer radius) must be specified")
        try:
            self.rout = float(self.rout)
        except:
            raise ValueError("rout (outer radius) must be a number (in cm)")
        
        if self.rflat is None:
            raise ValueError("rflat (flat radius) must be specified")
        try:
            self.rflat = float(self.rflat) 
        except:
            raise ValueError("rflat (flat radius) must be a number (in cm)")
        
        if self.nsample is None:
            raise ValueError("nsample (number of sampling points) must be specified")
        try:
            self.nsample = int(self.nsample)
        except:
            raise ValueError("nsample (number of sampling points) must be an integer")
        
        if self.rout is None:
            raise ValueError("rout (outer radius) must be specified")
        try:
            self.rout = float(self.rout)
        except:
            raise ValueError("rout (outer radius) must be a number (in cm)")
        
        # log.info('max(a)=%.3f (um)'%(self.a.max()*1e4))
        # log.info('n0_gas=%.3e'%(self.n0_gas))
        #self.rr = np.linspace(0,self.rout/2e3,self.nsample)
        if sampling_type=='lin_space':
            self.rr = np.linspace(0,self.rout,self.nsample)#np.linspace(0,self.rout/5e3,self.nsample)
        elif sampling_type=='log_space':
            self.rr = np.logspace(np.log10(1e-5),np.log10(self.rout),self.nsample)
        # self.rr = np.logspace(np.log10(1e-5),np.log10(self.rout),self.nsample)
        
        # self.rr = list(np.linspace(0,self.rout/4.1e3,int(self.nsample/2)))+list(np.linspace(self.rout/4.e3,self.rout/2.e3,int(self.nsample/2))) #5e3
        # self.rr=np.array(self.rr)
        
        # rr = np.linspace(0,rout/1e4,nsample)
        self.x = self.y = self.z = np.concatenate((-self.rr[::-1],np.array([0]),self.rr[::1]))
        self.X,  self.Y = np.meshgrid(self.x, self.y); self.Z=self.Y
        
        self.nH = np.array([self.ngas_starless(self.n0_gas,self.rflat,self.p)(ri) for ri in self.rr])

        self.Av_map          = np.zeros((len(self.x),len(self.y)))
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
        nn0=self.ngas_starless(self.n0_gas,self.rflat,self.p)(np.array([r0]))
        if isinstance(nn0,float):
            nn0=np.array([nn0])
        n = np.concatenate((nnew[::-1],nn0,nnew[::1]))
        s = np.concatenate((-s[::-1],np.array([0]),s[::1]))
        ds = s[1:]-s[:-1] #[len(n),] array

        w=parent.w
        a=parent.a
        dust_type=parent.dust_type

        if dust_type.lower()=='astro':
            # log.info('max(a)=%.3e'%(a.max()*1e4))
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
    def get_dtau(self,parent,n):
        dust_type=parent.dust_type
        a=parent.a
        if dust_type.lower()=='astro':
            Qext_astro=parent.Qext_astro
            dn_da_astro=parent.dn_da_astro

            fastro_ext= Qext_astro * np.pi *a*a * dn_da_astro * n
            dtau_astro = integrate.simpson(fastro_ext, a)
            return dtau_astro

        elif dust_type.lower()=='astro+pah':
            Qext_astro=parent.Qext_astro
            dn_da_astro=parent.dn_da_astro

            Qext_pah=parent.Qext_pah
            dn_da_pah=parent.dn_da_pah

            fastro_ext= Qext_astro * np.pi *a*a * dn_da_astro * n
            dtau_astro = integrate.simpson(fastro_ext, a)

            fpah_ext= Qext_pah * np.pi *a*a * dn_da_pah * n
            dtau_pah = integrate.simpson(fpah_ext, a)
            return dtau_astro + dtau_pah

        else:
            Qext_sil=parent.Qext_sil
            Qext_amCBE=parent.Qext_amCBE
            dn_da_sil=parent.dn_da_sil
            dn_da_gra=parent.dn_da_gra

            #optical depth of silicate
            fsil_ext= Qext_sil * np.pi *a*a * dn_da_sil * n
            dtau_sil = integrate.simpson(fsil_ext, a)
            #optical depth of carbon
            fgra_ext= Qext_amCBE * np.pi *a*a * dn_da_gra * n
            dtau_gra = integrate.simpson(fgra_ext, a)
            return dtau_sil+dtau_gra #I think, (dtau_sil+dtau_gra)*ds

    @auto_refresh
    def get_map_Av_los(self,parent):
        def func_para(i,j):
            # print(self.X[i,j],self.Y[i,j])
            r0 = np.sqrt(self.X[i,j]*self.X[i,j]+self.Y[i,j]*self.Y[i,j])
            return self.Av_los_by_dust(parent,r0)
            # Av_map[i,j]=self.Av_func(r0)
            # return Av_map

        out=Parallel(n_jobs=-1,verbose=1, prefer='threads')(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.y)))
        i=0
        for xi in range(len(self.x)):
            for yi in range(len(self.y)):
                Av_val = out[i]
                self.Av_map[xi,yi]=Av_val
                i+=1
        self.Av_map[self.Av_map==0.0] = np.nan
        return self.Av_map

        #self.Av_map[self.Av_map==0.0] = np.nan
        # return self.Av_map

    @auto_refresh
    def get_map_Av_2calcule(self,parent):
        self.isoCloud_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            return self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)

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

    # def get_align(self):
    #     def func_para(i,j):
    #         r0=self.x[i]
    #         Av = self.Av_func(np.abs(r0))
    #         self.U = self.U_starless(self.U0,Av)
    #         self.Tgas=self.Tgas_starless(self.U0,Av,16.4)
    #         self.mean_lam=self.lamda_starless(1.3e-4,Av)
    #         self.ngas=self.ngas_starless(self.n0_gas,self.rflat)(np.sqrt(self.z[j]*self.z[j]+r0*r0))
    #         ali_cl=align.alignment_class(self)
    #         return ali_cl.Aligned_Size_v2()

    #     out=Parallel(n_jobs=-1,verbose=1)(delayed(func_para)(i, j) for i in range(len(self.x)) for j in range(len(self.z)))
    #     i=0
    #     for xi in range(len(self.x)):
    #         for zi in range(len(self.z)):
    #             align_val = out[i]
    #             self.align_map[xi,zi]=align_val
    #             i+=1
    #     # self.align_map[self.Av_map==0.0] = np.nan
    #     return self.align_map

    @auto_refresh
    def get_map_align(self,parent):
        self.U0=parent.U
        self.u_ISRF=parent.u_ISRF
        self.rho=parent.rho
        self.amin=parent.amin
        self.amax=parent.amax
        self.gamma=parent.gamma
        self.T0_gas=parent.Tgas
        self.mean_lam0=parent.mean_lam
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
        
        self.isoCloud_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                Av = self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)
                self.U = self.U_starless(self.U0,Av)
                self.Tgas=self.Tgas_starless(self.U0,Av,self.T0_gas)
                self.mean_lam=self.lamda_starless(self.mean_lam0,Av)
                self.ngas=self.ngas_starless(self.n0_gas,self.rflat,self.p)(r)
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
    def get_map_Tdust(self,parent):
        self.U0=parent.U
        self.T0_gas=parent.Tgas

        self.isoCloud_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                Av=self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)
                return self.Tdust_starless(self.U0,Av,self.T0_gas,0.1e-4)

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

        self.isoCloud_model(parent)
        def func_para(i,j):
            r = np.sqrt(self.X[i,j]*self.X[i,j]+self.Z[i,j]*self.Z[i,j])
            if r>=self.rr.max():
                return np.nan
            else:
                Av=self.Av_2calcule(self.n0_gas,self.rflat,self.p)(r)
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
    def Av_2calcule(self,n0,Rflat,p,Rv=4.0):
        """
            This function to calculate the Av from the cloud's surface to the core
            This function to returns the Av for radiation attenuation.
            *** Note: this Av is differ from the observed Av.
            input:
                - n0: peaked gas density
                - rflat: below which ngas=n0=const.
                - Rv: total-to-selective extinction ratio.
                      If NA, Rv=4.0 
        """
        # def Av_inward_ana(r,p,n0,Rflat):
        #p=2.0#3./2
        # r_ratio = r/Rflat
        Av_c = 10.3*(n0/1e8)*(Rflat/(10.*constants.au))*(Rv/4.0)
        return lambda r: np.where(r<=Rflat, Av_c*(p/(p-1)-r/Rflat), Av_c/(p-1)*pow(r/Rflat,1-p))

    @auto_refresh
    def ngas_starless(self,n0,Rflat,p):
        ##Any emprical formulation can be used
        ##here: Hoang et al. 2021 is adopted
        ##Note: r is the radial distance from center to the envelope
        # return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-2.0))
        # return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-3./2))
        return lambda r: np.where(r<=Rflat, n0, n0 * (np.maximum(r,1e-30) / Rflat) ** (-p))

    @auto_refresh
    def U_starless(self,U0,Av):
        #For the case of starless core, we parameterize the radiation strength as a function of Av
        #as in Hoang et al. 2021 (Equation 34)
        # log.info('[U_starless]U0=%.2f'%U0)
        return U0/(1+0.42*pow(Av,1.22))

    @auto_refresh
    def lamda_starless(self,mean_lamda0,Av):
        ##See Equation 35 in Hoang et al. 2021
        # mean_lamda0=1.3e-4 #cm
        return mean_lamda0*(1+0.27*Av**0.76)

    @auto_refresh
    def Tgas_starless(self,U0,Av,Td0):
        T0=Td0*U0**(1./6) #K
        return T0/(1+0.42*pow(Av,1.22))**(1./6)

    @auto_refresh
    def Tdust_starless(self,U0,Av,Td0,a):
        T0=(a/1e-5)**(-1./15) * Td0*U0**(1./6) #K
        return T0/(1+0.42*pow(Av,1.22))**(1./6)

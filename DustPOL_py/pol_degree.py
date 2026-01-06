import numpy as np
import os,re
from . import rad_func, align, disrupt, qq#, starless_class, size_distribution
from astropy import log, constants
# import pol_degree, radiation
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.colors import LogNorm
from  .decorators import auto_refresh
from .check_vars import check_variable_name,check_variable_combination

class pol_degree(object):
    # def __init__(self,
    #                 U,mean_lam,gamma, ##radiation
    #                 Tgas,Tdust,ngas, ##physical conditions
    #                 ratd,Smax, 	##rotational disruption
    #                 amin,amax,power_index,dust_type,dust_to_gas_ratio,GSD_law, ##grain-size
    #                 RATalign,f_min,f_max,alpha, 		##alignment physics
    #                 B_angle			##B-field inclination
    #             ):
    def __init__(self,parent):
        self.U = parent.U
        self.gamma=parent.gamma
        self.mean_lam=parent.mean_lam
        self.Tgas=parent.Tgas
        self.Tdust=parent.Tdust
        self.ngas=parent.ngas
        self.ratd=parent.ratd
        self.Smax=parent.Smax
        self.amin=parent.amin
        self.amax=parent.amax
        self.a   =parent.a
        self.na  =parent.na
        self.power_index=parent.power_index
        self.dust_type=parent.dust_type
        self.dust_to_gas_ratio=parent.dust_to_gas_ratio
        self.GSD_law=parent.GSD_law
        self.RATalign=parent.RATalign
        self.Bfield=parent.Bfield
        self.Ncl=parent.Ncl
        self.phi_sp=parent.phi_sp
        self.fp=parent.fp
        self.f_min=parent.f_min
        self.f_max=parent.f_max
        self.alpha=parent.alpha
        self.B_angle=parent.B_angle
        self.Urange_tempdist=[]
        self.u_ISRF=parent.u_ISRF
        self.rho=parent.rho
        self.w  = parent.w
        self.align_func = parent.align_func
        self.pstiff=parent.pstiff
        self.verbose=parent.verbose
        self.alpha_DG=parent.alpha_DG
        self.temp_ratio=parent.temp_ratio
        
        ## get the 2layer dust model
        self.model_layer = parent.model_layer
        self.fheat       = parent.fheat
        self.fscale      = parent.fscale
        self.fscale_car  = parent.fscale_car

        ##get the alignment size and alignment function
        ali_cl  = align.alignment_class(self)
        # a_ali   = ali_cl.Aligned_Size_v2()
        self.fa = ali_cl.f_ali()

        if self.fscale is not None:
            self.U    = getattr(parent, f'U_add', None) # Update the U value for the second layer
            self.a    = getattr(parent, f'a_add', None) # Update the grain size for the second layer
            ali_cl    = align.alignment_class(self)
            self.fa_add    = ali_cl.f_ali()
            
            self.U = parent.U # Reset the U value for the first layer
            self.a = parent.a # Reset the grain size for the first layer 
               

    @auto_refresh
    def tau_xy(self,parent,Ngas):
        """
        Calculate the optical depth along the x-axis and y-axis of the grain.
        """

        tau_x_total = 0.0
        tau_y_total = 0.0
        for dusttype in parent.dust_type.split("+"):
            dn_da = getattr(parent, f'dn_da_{dusttype}', None)
            Qext  = getattr(parent, f'Qext_{dusttype}', None)
            Qpol  = getattr(parent, f'Qpol_{dusttype}', None)
            
            # polarized tau for each dusttype
            _,dP_abs_dict,_ = self._pol_degree_absorption_thin_(parent)
            tau_pol_ = dP_abs_dict[dusttype]/self.ngas/100 * Ngas

            # extinction per gas number density for each dusttype
            dtau_ = (Qext + self.fa * Qpol*(2./3 - np.sin(self.B_angle)*np.sin(self.B_angle))) * np.pi * self.a**2 * dn_da
            if len(self.a)%2 == 0:
                tau_ = integrate.trapezoid(dtau_,self.a) * Ngas
            else:
                tau_ = integrate.simpson(dtau_,self.a) * Ngas

            tau_x_ = tau_-tau_pol_
            tau_y_ = tau_+tau_pol_
            
            tau_x_total += tau_x_
            tau_y_total += tau_y_
        return tau_x_total, tau_y_total

    @auto_refresh
    def _pol_degree_absorption_thin_(self,parent): #optically thin assumption
        dP_abs_dict = {}
        # dP_abs_a_return={}
        for dusttype in parent.dust_type.split("+"):
            dn_da = getattr(parent, f'dn_da_{dusttype}', None)
            Qpol_abs = getattr(parent, f'Qpol_abs_{dusttype}', None)
            # print("dusttype=",dusttype,"B_angle=",self.B_angle,'Qpol_abs=',Qpol_abs,'dn_da=',dn_da,'fa=',self.fa)
            dP_abs_a = dn_da * np.sin(self.B_angle)**2 * Qpol_abs * np.pi* self.a**2 * self.fa
            dP_abs_a[dP_abs_a<0]=0.0 #remove negative values
            # dP_abs_a_return[dusttype] = dP_abs_a

            if len(self.a)%2 ==0:
                dP_abs_dict[dusttype]  = integrate.trapezoid(dP_abs_a, self.a) * self.ngas*100#* exp(-tau)
            else:
                dP_abs_dict[dusttype]  = integrate.simpson(dP_abs_a, self.a) * self.ngas*100#* exp(-tau)

        # total degree of polarization over all dusttype
        dP_abs = np.sum(list(dP_abs_dict.values()), axis=0)
        return self.w,dP_abs_dict,dP_abs#,dP_abs_a_return

    # @auto_refresh
    # def _pol_degree_absorption_general_(self,parent,Ngas):
    #     w,dP_abs_dict,_ = self._pol_degree_absorption_thin_(parent)
    #     P_abs_dict={}
    #     for dusttype in parent.dust_type.split("+"):
    #         P_abs_dict[dusttype] = np.tanh(dP_abs_dict[dusttype]/self.ngas/100 * Ngas) * 100
        
    #     # total degree of polarization over all dusttype
    #     P_abs = np.sum(list(P_abs_dict.values()), axis=0)
    #     return w,P_abs
    
    @auto_refresh
    def _pol_degree_emission_thin(self,parent):
        Iem_dict={}
        Ipol_dict={}
        for dusttype in parent.dust_type.split("+"):
            dn_da = getattr(parent, f'dn_da_{dusttype}', None)
            Qabs  = getattr(parent, f'Qabs_{dusttype}', None)
            Qpol  = getattr(parent, f'Qpol_{dusttype}', None)
            BB    = getattr(parent, f'BB_{dusttype}', None)
            f_mass= getattr(parent, f'f_mass_{dusttype}', None)

            # if (Ngas is None): # optically thin assumption
            Iem   = f_mass * self.fIem(self.a,self.fa,dn_da,Qabs,Qpol,BB) * self.w
            Ipol  = f_mass * self.fIpol(self.a,self.fa,dn_da,Qpol,BB) * self.w
            
            
            # If the dust model is one layer, then Iem_add and Ipol_add are zero        
            if self.fscale is None: #one layer dust model
                Iem_add = 0.0
                Ipol_add= 0.0
                        
            # If the dust model is two layers, then calculate the additional layer
            else: #two layer dust model
                a_add     = getattr(parent, f'a_add', None)
                dn_da_add = getattr(parent, f'dn_da_add_{dusttype}', None)
                Qabs_add  = getattr(parent, f'Qabs_add_{dusttype}', None)
                Qpol_add  = getattr(parent, f'Qpol_add_{dusttype}', None)
                BB_add    = getattr(parent, f'BB_add_{dusttype}', None)
                    
                Iem_add = f_mass * self.fscale * self.fIem(a_add,self.fa_add,dn_da_add,Qabs_add,Qpol_add,BB_add) * self.w
                Ipol_add= f_mass * self.fscale * self.fIpol(a_add,self.fa_add,dn_da_add,Qpol_add,BB_add) * self.w

            # Sum the two layers
            Iem  += Iem_add
            Ipol += Ipol_add

            Iem[Iem<0]=0.
            Ipol[Ipol<0]=0.
            Iem_dict[dusttype] = Iem
            Ipol_dict[dusttype]= Ipol
            # Iem_dict[dusttype+'_add'] = Iem_add
            # Ipol_dict[dusttype+'_add']= Ipol_add
            
        Iem_tot = np.sum(list(Iem_dict.values()), axis=0)
        Ipol_tot= np.sum(list(Ipol_dict.values()), axis=0) 
        # Pem = np.where(Iem_tot !=0, Ipol_tot/Iem_tot * 100, 0)
        den = np.nan_to_num(Iem_tot, nan=0.0, posinf=0.0, neginf=0.0)
        num = np.nan_to_num(Ipol_tot, nan=0.0, posinf=0.0, neginf=0.0)
        mask = den != 0.0
        Pem = np.zeros_like(den)
        Pem[mask] = (num[mask] / den[mask]) * 100.0
        return self.w,[Iem_dict,Ipol_dict],[Iem_tot,Ipol_tot],Pem

    @auto_refresh
    def _pol_degree_emission_general(self,parent,Ngas):
        Ipol_dict = {}
        Iem_dict  = {}

        BB = {}
        tau_x_total = 0.0
        tau_y_total = 0.0
        for dusttype in parent.dust_type.split("+"):
            dn_da        = getattr(parent, f'dn_da_{dusttype}', None)
            Qext  = getattr(parent, f'Qext_{dusttype}', None)
            Qpol  = getattr(parent, f'Qpol_{dusttype}', None)
            
            # Qabs_x       = getattr(parent, f'Qabs_x_{dusttype}', None) 
            # Qabs_y       = getattr(parent, f'Qabs_y_{dusttype}', None) 
            BB[dusttype] = getattr(parent, f'BB_{dusttype}', None)[0,:] # BB is a 2D array, we take the first row for the wavelength

            # polarized tau for each dusttype
            _,dP_abs_dict,_ = self._pol_degree_absorption_thin_(parent)
            tau_pol_ = dP_abs_dict[dusttype]/self.ngas/100 * Ngas

            # extinction per gas number density for each dusttype
            dtau_ = (Qext + self.fa * Qpol*(2./3 - np.sin(self.B_angle)*np.sin(self.B_angle))) * np.pi * self.a**2 * dn_da
            if len(self.a)%2 == 0:
                tau_ = integrate.trapezoid(dtau_,self.a) * Ngas
            else:
                tau_ = integrate.simpson(dtau_,self.a) * Ngas

            tau_x_ = tau_-tau_pol_
            tau_y_ = tau_+tau_pol_

            Ipol_dict[dusttype] = (np.exp(-tau_x_) - np.exp(-tau_y_))/2.0
            Ipol_dict[dusttype] = Ipol_dict[dusttype] * BB[dusttype].T
            
            Iem_dict[dusttype] = (2.0 - (np.exp(-tau_x_) + np.exp(-tau_y_)))/2.0
            Iem_dict[dusttype] = Iem_dict[dusttype] * BB[dusttype].T
            
            tau_x_total += tau_x_
            tau_y_total += tau_y_
            
        # total emission intensities                      
        Iem_tot  = (2.0 - (np.exp(-tau_x_total) + np.exp(-tau_y_total)))/2.0
        Ipol_tot = (np.exp(-tau_x_total) - np.exp(-tau_y_total))/2.0
        Iem_tot  = Iem_tot  * np.sum(list(BB.values()), axis=0)
        Ipol_tot = Ipol_tot * np.sum(list(BB.values()), axis=0)

        # total degree of polarization
        Pem = np.where(Iem_tot !=0, Ipol_tot/Iem_tot * 100, 0)
        return self.w,[Iem_dict,Ipol_dict],[Iem_tot,Ipol_tot],Pem
    
    # ==================== Total emission intensities ====================
    def fIem(self,a,fa,dn_da,Qabs,Qpol,BB):
        arr1=self.ngas * dn_da * (Qabs+Qpol*fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* a**2 
        arr2=BB.T * np.array([np.exp(-0.0)]).T
        dI_em=np.multiply(arr1,arr2)
        if len(a)%2 == 0:
            return integrate.trapezoid(dI_em, a)
        else:
            return integrate.simpson(dI_em, a)
    # ==================== Polarized emission intensities ====================
    def fIpol(self,a,fa,dn_da,Qpol,BB):
        arr2=BB.T * np.array([np.exp(-0.0)]).T
        arr3=self.ngas * dn_da * Qpol * np.pi* a**2 * fa *np.sin(self.B_angle)*np.sin(self.B_angle)
        dI_pol=np.multiply(arr3,arr2)
        if len(a)%2 == 0:
            return integrate.trapezoid(dI_pol, a)
        else:
            return integrate.simpson(dI_pol, a)
            
    def fIem_sil(self,tau):
        # -------- Silicate grain --------
        ##Total intensities
        arr1=self.ngas*self.dn_da_sil* (self.Qabs_sil+self.Qpol_sil*self.fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* self.a**2 
        arr2=self.B_sil.T * np.array([np.exp(-tau)]).T
        dI_em_sil=np.multiply(arr1,arr2)
        if len(self.a)%2 is 0:
            self.Iem_sil = integrate.trapezoid(dI_em_sil, self.a)
        else:
            self.Iem_sil = integrate.simpson(dI_em_sil, self.a)
        return

    def fIpol_sil(self,tau):
        ##Polarized emission intensities
        arr2=self.B_sil.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_sil* self.Qpol_sil* np.pi* self.a**2 * self.fa/2 *np.sin(self.B_angle)*np.sin(self.B_angle)
        dI_pol_sil=np.multiply(arr3,arr2)
        if len(self.a)%2 is 0:
            self.Ipol_sil = integrate.trapezoid(dI_pol_sil, self.a)
        else:
            self.Ipol_sil = integrate.simpson(dI_pol_sil, self.a)
        return

        # ##Polarized absorption intensities
        # dI_pol_abs_sil = dn_da_sil* Qpol_abs_sil* pi* a**2 * fa
        # Ipol_abs_sil   = integrate.simpson(dI_pol_abs_sil, a) #* exp(-tau)

    def fIem_car(self,tau):
        # -------- Carbonaceous grain --------
        ##Total intensities
        arr1=self.ngas*self.dn_da_gra* self.Qabs_amCBE* np.pi* self.a**2
        arr2=self.B_gra.T * np.array([np.exp(-tau)]).T
        dI_em_amCBE=np.multiply(arr1,arr2)
        if len(self.a)%2 is 0:
            self.Iem_amCBE  = integrate.trapezoid(dI_em_amCBE, self.a)    
        else:
            self.Iem_amCBE  = integrate.simpson(dI_em_amCBE, self.a)
        return

    def fIpol_car(self,tau):
        ##Polarized emission intensities
        arr2=self.B_gra.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_gra* self.Qpol_amCBE* np.pi* self.a**2 * self.fa/2
        dI_pol_amCBE=np.multiply(arr3,arr2)
        if len(self.a)%2 is 0:
            self.Ipol_amCBE = integrate.trapezoid(dI_pol_amCBE, self.a)
        else:
            self.Ipol_amCBE = integrate.simpson(dI_pol_amCBE, self.a)
        return
        # ##Polarized absorption intensities
        # dI_pol_abs_amCBE = dn_da_gra* Qpol_abs_amCBE* pi* a**2 * fa
        # Ipol_abs_amCBE   = integrate.simpson(dI_pol_abs_amCBE, a) #* exp(-tau)

    def fIem_astro(self,tau):
        # -------- Silicate grain --------
        ##Total intensities
        # arr1=self.ngas*self.dn_da_astro* (self.Qabs_astro+self.Qpol_astro*self.fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* self.a**2 
        arr1=self.ngas*self.dn_da_astro* self.Qabs_astro* np.pi* self.a**2
        arr2=self.B_astro.T * np.array([np.exp(-tau)]).T
        dI_em_astro=np.multiply(arr1,arr2)
        if len(self.a)%2 is 0:
            self.Iem_astro = integrate.trapezoid(dI_em_astro, self.a)    
        else:
            self.Iem_astro = integrate.simpson(dI_em_astro, self.a)
        return

    def fIem_pah(self,tau):
        # -------- PAH grain --------
        ##Total intensities
        # arr1=self.ngas*self.dn_da_astro* (self.Qabs_astro+self.Qpol_astro*self.fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* self.a**2 
        arr1=self.ngas*self.dn_da_pah* self.Qabs_pah* np.pi* self.a**2
        arr2=self.B_pah.T * np.array([np.exp(-tau)]).T
        dI_em_pah=np.multiply(arr1,arr2)
        if len(self.a)%2 is 0:
            self.Iem_pah = integrate.trapezoid(dI_em_pah, self.a)    
        else:
            self.Iem_pah = integrate.simpson(dI_em_pah, self.a)
        return

    def fIpol_astro(self,tau):
        ##Polarized emission intensities
        arr2=self.B_astro.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_astro* self.Qpol_astro* np.pi* self.a**2 * self.fa *np.sin(self.B_angle)*np.sin(self.B_angle)
        dI_pol_astro=np.multiply(arr3,arr2)
        if len(self.a)%2 is 0:
            self.Ipol_astro = integrate.trapezoid(dI_pol_astro, self.a)
        else:
            self.Ipol_astro = integrate.simpson(dI_pol_astro, self.a)
        return

    # # ------- dust grain size -------
    # def grain_size_distribution(self):
    #     # #a = rad_func.a_dust(UINDEX)[1]
    #     # a = rad_func.a_dust(10.0)[1] ##good for prolate shape
    #     # # a = Data_sil[0,0,:]*1e-4; a=a[where(a<=amax)[0]]
    #     # if self.amax>max(a):
    #     #     log.error('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))
    #     #     raise IOError('Value of amax!')
    #     # na = len(a)
    #     # print('[input]na=',na)
    #     # #

    #     # # ------- update grain size -------
    #     # self.lmin = min(np.where(a>=self.amin)[0])
        
    #     # if (self.ratd == 'on'):
    #     #     disruption_ = disrupt.radiative_disruption(self) 
    #     #     a_disr = disruption_.a_disrupt(a)
    #     #     if a_disr>self.amax:
    #     #         self.lmax=abs(np.log10(a)-np.log10(self.amax)).argmin()
    #     #     else:
    #     #         self.lmax = abs(np.log10(a)-np.log10(a_disr)).argmin()
    #     #     # lmax = abs(log10(a)-log10(a_disr)).argmin()
    #     # if (self.ratd == 'off'):
    #     #     self.lmax = max(np.where(a<=self.amax+0.1*self.amax)[0])
    #     # a = a[self.lmin:self.lmax+1]
    #     # na = len(a)
    #     log.info('na=%d'%self.na)
    #     if self.dust_type=='astro' or self.dust_type=='Astro':
    #         dn_da_astro = size_distribution.dnda_astro(self.a)
    #         return dn_da_astro
    #     else:
    #         dn_da_gra = size_distribution.dnda(6,'carbon',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
    #         dn_da_sil = size_distribution.dnda(6,'silicate',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
    #         return dn_da_sil, dn_da_gra


    # def get_coefficients_files(self):
    #     #BELOW DOESN'T WORK FOR OBLATE SHAPE WITH S=2
    #     if float(self.alpha)==0.3333:
    #         hdr_lines = 4
    #         skip_lines=4
    #         len_a_sil=70
    #         len_a_car=100
    #         len_w=800
    #         num_cols=8
    #     elif float(self.alpha)==2.0:
    #         hdr_lines=4
    #         skip_lines=4
    #         len_a_sil=160
    #         len_a_car=160
    #         len_w=104
    #         num_cols=8
    #     else:
    #         log.error('Values of alpha is not regconized! [\033[1;5;7;91m failed \033[0m]')
    #     self.Data_sil = rad_func.readDC('./data/Q_aSil2001_'+str(self.alpha)+'_p20B.DAT',hdr_lines,skip_lines,len_a_sil,len_w,num_cols)
    #     self.Data_mCBE = rad_func.readDC('./data/Q_amCBE_'+str(self.alpha)+'.DAT',hdr_lines,skip_lines,len_a_car,len_w,num_cols)
    #     return

# class 
#     def starless_core(self):
#         fig,ax=plt.subplots()
#         phys_ = starless_class.starless_profile(self)
#         # Av_   = phys_.get_Av()
#         # align_=phys_.get_align()
#         # im = plt.imshow(align_.T*1e4,interpolation='bilinear',origin='lower',cmap='magma',norm=LogNorm())
#         # cbar=plt.colorbar(im,ax=ax,format='%.2f',shrink=0.8)

#         # plt.imshow(align_*1e4,interpolation='bilinear',origin='lower',cmap='magma',norm=LogNorm())

        
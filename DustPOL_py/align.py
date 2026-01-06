from .plt_setup import *
from .read import *
from . import paramagnetic_relax

# import importlib
# import common; importlib.reload(common)
from sys import exit
from astropy import log

# from common import u_ISRF, rho, amin, gamma, RATalign, \
                   # K, H, C, K, amu, yr \
# ------------------------------------------------------
# [DESCRIPTION] aligned grain size and alignment function
# input:
#	U = radiation field strength
# output:
#	1. a_ali = aligned grain size at a radiation field strength
#	2. fali = alignment function at a radiation field strength
# ------------------------------------------------------
"""
Sukhbinder
5 April 2017
Based on:
"""
yr  = 365.*24.*60.*60.
H   = 6.625e-27
C   = 3e10
K   = 1.38e-16

def _rect_inter_inner(x1,x2):
   n1=x1.shape[0]-1
   n2=x2.shape[0]-1
   X1=c_[x1[:-1],x1[1:]]
   X2=c_[x2[:-1],x2[1:]]
   S1=tile(X1.min(axis=1),(n2,1)).T
   S2=tile(X2.max(axis=1),(n1,1))
   S3=tile(X1.max(axis=1),(n2,1)).T
   S4=tile(X2.min(axis=1),(n1,1))
   return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
   S1,S2,S3,S4=_rect_inter_inner(x1,x2)
   S5,S6,S7,S8=_rect_inter_inner(y1,y2)

   C1=np.less_equal(S1,S2)
   C2=np.greater_equal(S3,S4)
   C3=np.less_equal(S5,S6)
   C4=np.greater_equal(S7,S8)

   ii,jj=np.nonzero(C1 & C2 & C3 & C4)
   return ii,jj

def intersection(x1,y1,x2,y2):
   """
   INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
   usage:
      x,y=intersection(x1,y1,x2,y2)
   Example:
   a, b = 1, 2
   phi = np.linspace(3, 10, 100)
   x1 = a*phi - b*np.sin(phi)
   y1 = a - b*np.cos(phi)
   x2=phi
   y2=np.sin(phi)+2
   x,y=intersection(x1,y1,x2,y2)
   plt.plot(x1,y1,c='r')
   plt.plot(x2,y2,c='g')
   plt.plot(x,y,'*k')
   plt.show()
   """
   ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
   n=len(ii)
   
   dxy1=np.diff(c_[x1,y1],axis=0)
   dxy2=np.diff(c_[x2,y2],axis=0)

   T=np.zeros((4,n))
   AA=np.zeros((4,4,n))
   AA[0:2,2,:]=-1
   AA[2:4,3,:]=-1
   AA[0::2,0,:]=dxy1[ii,:].T
   AA[1::2,1,:]=dxy2[jj,:].T

   BB=np.zeros((4,n))
   BB[0,:]=-x1[ii].ravel()
   BB[1,:]=-x2[jj].ravel()
   BB[2,:]=-y1[ii].ravel()
   BB[3,:]=-y2[jj].ravel()

   for i in range(n):
      try:
         T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
      except:
         T[:,i]=np.NaN

   in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

   xy0=T[2:,in_range]
   xy0=xy0.T
   return xy0[:,0],xy0[:,1]

# ================================ a_ali function
def Qgam(a,lamb):
    if a <= lamb/1.8:
        return 2 * (lamb/a)**(-2.7)
    else:
        return 0.4

def Qgam_v2(lamb):
    return lambda a: where(a <= lamb/1.8, 2 * (lamb/a)**(-2.7), 0.4)

#
def torq(a,U,u_ISRF,gamma,lamb):
    # torq_rat = gamma*a**2 * (U*u_ISRF) * lamb * (Qgam(a,lamb)/2.)
    torq_rat = gamma*a**2 * (U*u_ISRF) * lamb * (Qgam_v2(lamb)(a)/2.)
    return torq_rat
#

# total damping time
def tdamp(a, U, nH, T_gas, rho):
    
    a_7 = a/1e-7
    FIR    = (60.8/a_7)*pow(U,2/3.)*(20./nH)*(100./T_gas)**0.5
    
    t_gas = 8.74e4 * (a/1e-5) * (rho/3) * (30/nH) * sqrt(100./T_gas)*yr # in sec
    t_damp = t_gas/(1. + FIR)
    
    return t_damp
#

def wrat(a,U,u_ISRF,gamma,lamb,nH,T_gas,rho):
    
    #FIR = 9.1e-6 * U**(2/3.) * (30/nH) * sqrt(100/T_gas)/a

    a_7 = a/1e-7
    FIR    = (60.8/a_7)*pow(U,2/3.)*(20./nH)*(100./T_gas)**0.5

    t_gas = 8.74e4 * (a/1e-5) * (rho/3) * (30/nH) * sqrt(100./T_gas)*yr # in sec
    t_damp = t_gas/(1 + FIR)
    Ia = 8*pi*rho*a**5./15.
    w_rat = torq(a,U,u_ISRF,gamma,lamb) * t_damp/Ia # [Hz] : years -> seconds
    w_th  = sqrt(K*T_gas/Ia)
    return w_rat,w_th
    
def Aligned_Size(U, amax, n_gas, T_gas, lambA):
    a = logspace(log10(amin), log10(amax), 1500) # [cm]: recreate an arbitrary smooth grain-size array
    na = len(a)
    ratio=[]
    for j in range(na):
        #F_IR = (0.91/(a[j]*1e5)) * U**(2/3) * (30./n_gas) *(100./T_gas)**0.5
        #OMEGA_RAT = 200* (30./n_gas) *(100/T_gas) *U *2.4*1e-3/1e-2*(a[j]/1e-5)**3.2/(1+F_IR)
        #F_IR = (0.4/(a[j]*1.e5)) * U**(2./3) * (30./n_gas) *(100./T_gas)**0.5
        #OMEGA_RAT = 4174. *pow(rho/3,0.5) * gamma * (lambA*1e4/0.5)**(-1.7) * (a[j]/1.e-5)**(3.2) *U * (10./n_gas) * (1./(1+F_IR)) * (100./T_gas)
        #print('U=',U)
        #print('check=',abs(OMEGA_RAT))
        #print('ngas=',n_gas)
        WWrat, WWth = wrat(a[j],U, gamma,lambA,n_gas, T_gas)
        ratio.append(WWrat/WWth)
        #print('ratio=',ratio)
        #a_ali=intersection(a,ratio,a,3*ones(na))[0]
        #if abs(ratio - 3.0) < 1e-2:
        #    a_ali = a[j]
        #else:
        #    a_ali = max(a)
    if max(ratio)<3.0:
        log.warning('*** \033[1;5;33m St<3, set a_align == a_max \033[0m')
    try:
        a_ali=intersection(a,array(ratio),a,3*ones(na))[0][0]
    except:
        a_ali=np.nan#max(a)
    return a_ali

class alignment_class():
    def __init__(self,parent):
        self.u_ISRF = parent.u_ISRF 
        self.rho=parent.rho
        # self.amin = parent.amin
        # self.amax = parent.amax
        self.gamma= parent.gamma
        self.RATalign=parent.RATalign
        self.Tgas = parent.Tgas
        # self.Tdust = parent.Tdust
        self.ngas = parent.ngas
        self.mean_lam=parent.mean_lam
        self.U=parent.U
        self.rho=parent.rho
        self.alpha = parent.alpha
        self.f_min = getattr(parent, "f_min", None) #self.f_min = parent.f_min
        if self.f_min is None:
            self.alpha_DG = parent.alpha_DG
            self.temp_ratio = parent.temp_ratio
        self.f_max = parent.f_max
        self.a = parent.a
        self.Bfield = parent.Bfield
        self.Ncl = parent.Ncl
        self.phi_sp = parent.phi_sp
        self.fp = parent.fp
        self.align_func = parent.align_func
        self.pstiff = parent.pstiff
        self.verbose = getattr(parent, "verbose", False)#parent.verbose

    def Aligned_Size_v2(self):
        # a_new = logspace(log10(self.amin), log10(5.e-4), 1500) # [cm]: recreate an arbitrary smooth grain-size array
        a_new = logspace(log10(3.e-8), log10(5.e-4), 10000) # [cm]: recreate an arbitrary smooth grain-size array
        na = len(a_new)
        WWrat, WWth = wrat(a_new,self.U, self.u_ISRF, self.gamma,self.mean_lam,self.ngas, self.Tgas,self.rho)
        ratio=WWrat/WWth
        try:
            idd = where(ratio<=3)[0]
            a_ali=max(a_new[idd])
        except:
            a_ali=np.nan
        # log.info('[Aligned_Size_v2]a_ali=%.3f'%(a_ali*1e4))
        return a_ali

        # ==============ALIGNMENT FUNCTION
    def f_ali(self):#U, a, a_ali, nH, Tgas, f_max, f_min=0.0):#f_min = 1e-3):
        #log.info('Maximum grain alignment efficiency: f_max=%.2f'%(f_max))
        #If alignment physics is RAT only
        a_ali=self.Aligned_Size_v2()
        if a_ali>self.a.max():
            fali=zeros(len(self.a))
            if (self.verbose):
                log.info(' ---> U=%.3f, ngas=%.3e (cm-3), amin=%.3e (um), \033[1;5;7;35m amax=%.3f (um) <= a_ali=%.3f (um) \033[0m, f_max=%.3f, Tgas=%.3f (K), mean_lam=%.3f (um) \033[0m \t\t'%(self.U,self.ngas,self.a.min()*1e4, self.a.max()*1e4,a_ali*1e4,self.f_max,self.Tgas,self.mean_lam*1e4))
            return fali
            
        if(self.RATalign == 'rat'):

            if self.f_min is None:
                #For the moment, we assume that Trot = Tgas (thermal rotation of grain)
                # Tdust = 16.4#self.Tdust * pow(self.a/1e-5,-1./15)
                #Bfield =5.e-6#Bfield_nH(self.ngas) * 1e-6 # convert to G  
                if self.alpha > 1.0:
                    f_min = paramagnetic_relax.compute_R(self.a,1./self.alpha,self.rho,self.Bfield,self.ngas,self.Tgas,self.Tgas/self.temp_ratio,self.alpha_DG,spm=True,Ncl=self.Ncl,phi_sp=self.phi_sp)
                else:
                    f_min = paramagnetic_relax.compute_R(self.a,self.alpha,self.rho,self.Bfield,self.ngas,self.Tgas,self.Tgas/self.temp_ratio,self.alpha_DG,spm=True,Ncl=self.Ncl,phi_sp=self.phi_sp)
                # f_min[self.a >= a_ali] = 0.0

            else:
                f_min = self.f_min
                    
            if self.align_func.lower() == 'l20':
                fali = f_min + (self.f_max-f_min)*(1.-np.exp(-(0.5*self.a/a_ali)**3))
            elif self.align_func.lower() == 'g18':
                fali = f_min + 0.5*(self.f_max-f_min) * (1.0+np.tanh(np.log(self.a/a_ali)/self.pstiff))
            elif self.align_func.lower() == 'h15':
                fali = f_min + (self.f_max-f_min)*(1.-np.exp(-(self.a/a_ali)**3))
            else:
                raise ValueError('Alignment function would be: H15 or L20 or G18!')
        
            if (self.verbose):
                if self.align_func.lower()=='g18':
                    log.info("Alignment function: %s with pstiff: %.2f"%(self.align_func,self.pstiff))
                else:
                    log.info("Alignment function: %s"%(self.align_func))                    
              
                log.info('\033[1;3;35m ---> U=%.3f, ngas=%.3e (cm-3), amin=%.3e (um), amax=%.3f (um), a_ali=%.3f (um), f_max=%.3f, Tgas=%.3f (K), mean_lam=%.3f (um) \033[0m \t\t'%(self.U, self.ngas,self.a.min()*1e4, self.a.max()*1e4,a_ali*1e4,self.f_max,self.Tgas,self.mean_lam*1e4))
                if (self.f_min is None):
                    log.info('      \033[1;3;33m [NOTE] Calculating f_min by DG theory (paramagnetic relaxation) with Ncl=%.0e, Bfield=%.0f (umG) \033[0m' % (self.Ncl, self.Bfield*1e6))            
                else:
                    log.info('      \033[1;3;33m [NOTE] f_min is given: f_min=%.3e \033[0m' % self.f_min)
                     
        else:
        #If alignment physics is MRAT (RAT + magnetic relaxation)
            fmag = np.zeros((len(self.a)))
            fali = np.zeros((len(self.a)))
            for ia in range(len(self.a)):
                #fmag[ia] = f_highJ(delta_mag(self.U,self.a[ia],self.ngas,self.Tgas,0,0,0,0),self.f_max) ##phi_sp, fp, Bfield, Ncl are zeros NOW !!! which is wrong --> need to pass the proper values
                fmag[ia] = f_highJ(delta_mag(self.U,self.a[ia],self.ngas,self.Tgas,self.phi_sp,self.fp,self.Bfield,self.Ncl),self.f_max) ##phi_sp, fp, Bfield, Ncl are zeros NOW !!! which is wrong --> need to pass the proper values
                if self.align_func.lower() == 'l20':
                    fali[ia] = self.f_min + (fmag[ia]-self.f_min) * (1-np.exp(-(0.5*self.a[ia]/a_ali)**3))
                elif self.align_func.lower() == 'g18':
                    fali[ia] = self.f_min + 0.5*(fmag[ia]-self.f_min) * (1.0+np.tanh(np.log(self.a[ia]/a_ali)/self.pstiff))
                elif self.align_func.lower() == 'h15':
                    fali[ia] = self.f_min + (fmag[ia]-self.f_min) * (1-np.exp(-(self.a[ia]/a_ali)**3))                   
                else:
                    raise ValueError('Alignment function would be: H15 or L20 or G18!') 
            #endfor
        #endif
            if (self.verbose):
                if self.align_func.lower()=='g18':
                    log.info("Alignment function: %s with pstiff: %.2f"%(self.align_func,self.pstiff))
                else:
                    log.info("Alignment function: %s"%(self.align_func))
                if a_ali >= self.a.max():
                    log.info(' ---> U=%.3f, ngas=%.3e (cm-3), amin=%.3e (um), \033[1;5;7;35m amax=%.3f (um) <= a_ali=%.3f (um) \033[0m, f_highJ=%.3f, Tgas=%.3f (K), mean_lam=%.3f (um) \033[0m \t\t'%(self.U,self.ngas,self.a.min()*1e4, self.a.max()*1e4,a_ali*1e4,fmag[-1],self.Tgas,self.mean_lam*1e4))
                else:
                    log.info('\033[1;3;35m ---> U=%.3f, ngas=%.3e (cm-3), amin=%.3e (um), amax=%.3f (um), a_ali=%.3f (um), f_highJ=%.3f, Tgas=%.3f (K), mean_lam=%.3f (um) \033[0m \t\t'%(self.U,self.ngas,self.a.min()*1e4, self.a.max()*1e4,a_ali*1e4,fmag[-1],self.Tgas,self.mean_lam*1e4))

        #print ('max_fali=', fali.max(), fmag.max())
        
        return fali
#


#Dec 27, 2022: Thiem added the MRAT feature
# Beginning of MRAT implementation
#Magnetic fields from Crutcher+ for nH>300 cm-3 derived from measurements
def Bfield_nH(nH):
    # parallel component of Bfield along the line of sight
    B_par = np.where(nH<=300, 10, 10*(nH/300)**0.65)#muG
    # Statistical correction for total Bfield
    # return np.sqrt(B_par**2 / (1-np.pi**2/16))
    return B_par * 2 #muG

#Modeling f_highJ by MRAT theory
def f_highJ(delta_mag, fhiJ_RAT):
    if (delta_mag >= 10):
        f_hiJ = 1
    elif(delta_mag > 1):
        f_hiJ = 0.5
    else:
        f_hiJ = fhiJ_RAT
    #
    return f_hiJ
#

#Magnetic relaxation strength, delta_mag
def delta_mag(U, acm, nH, Tgas, phi_sp, fp, Bfield, Ncl):
    # from common import phi_sp, fp, Bfield, Ncl
    # muG = 1e-6 #microGauss, Bfield input given in muG
    Td  = 16.4*pow(U, 1./6) #assumed equilibrium temperature for large grains
	
    #for paramagnetic material
    delta_mag_pm = 18.97*pow(acm/1e-5,-1)*pow(nH/1e4,-1)*(fp/0.1)*pow(Bfield/1e-3,2)*pow(Tgas/10.,-1./2)*(10./Td) #Eq 6 in HL16

    #For superparamagnetic material (grains with embedded iron clusters). See Hoang+2022
    delta_mag_spm = 56*pow(acm/1e-5,-1)*pow(nH/1e4,-1)*(Ncl*phi_sp/0.01)*pow(Bfield*1e6/1e3,2)*pow(Tgas/10.,-1./2)*(10./Td)

    # delta_mag_spm = 0.56*pow(acm/1e-5,-1)*pow(nH/1e4,-1)*(Ncl*phi_sp/0.01)*pow(Bfield/1e-3,2)*pow(Tgas/10.,-1./2)*(10./Td)
    #If all iron in PM: delta_mag_pm(fp=0.1)
    #If all iron in SPM:delta_mag_spm (phi_sp=0.3).
    # SPM with Ncl=1 yields delta_mag_spm(phi_sp=0.3, Ncl=1) = delta_mag_pm(fp=0.1)
	
    return delta_mag_spm
#
#End of Thiem's MRAT new feature

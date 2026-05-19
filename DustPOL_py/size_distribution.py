from .plt_setup import *
from .read import *
# from common import *
# import importlib
# import common; importlib.reload(common)
# from common import mgas,dust_to_gas_ratio
from scipy.special import erf

# ------------------------------------------------------
# [DESCRIPTION] GRAIN SIZE DISTRIBUTION dn/da
# input:
#	INDEX = 0 - 14 at Rv
#	dusttype = 'carbon' for carbonaceous or PAHs, 'silicate' for silicate
#	aef = effective grain size
#	sizedist = 'MRN', 'WD01', 'DL07'
# output:
#	1/ngas * dnda
# ------------------------------------------------------

# mgas =2*1.3*1.6605402e-24 #atomic mass unit #90%H + 10%He
rho_gas =1.4*1.6605402e-24 #atomic mass unit #90%H + 10%He
def AMRN_dust_gas_ratio(a0, a1, rho_dust, ratio=0.01, beta=-3.5):
    f1= ratio*rho_gas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*rho_dust*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*rho_dust*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2

def AMRN_sil(a0, a1, rho_s,rho_c, ratio=0.01, beta=-3.5):
    f1= ratio*rho_gas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2 * 1.12/(1.12*rho_s+rho_c)

def AMRN_car(a0, a1, rho_s, rho_c, ratio=0.01, beta=-3.5):
    f1= ratio*rho_gas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2 * 1./(1.12*rho_s+rho_c)

def dnda_hd23(acm,GSD_params=None):

    BAd,a0Ad,sigmaAd,A0,A1,A2,A3,A4,A5=GSD_params
    A_array=np.array([A1,A2,A3,A4,A5])

    component1 = 1./(2*sigmaAd**2)*(np.log(acm/a0Ad))**2
    dnda = BAd/acm*np.exp(-component1) 
        
    component2 = np.zeros((5,len(acm)))
    for i in range(1,6):
        component2[i-1]=(np.log(acm/1.e-8))**(i)
    component2 = np.sum(A_array[:,None]*component2,axis=0)

    return dnda + A0/acm * exp(component2)

def dnda_pah(a,B1,B2):
    a01=4.*1e-8 ##AA to cm
    a02=30.*1e-8##AA to cm
    sigma=0.40
    component1 = 1./(2*sigma**2) * (np.log(a/a01))**2
    component2 = 1./(2*sigma**2) * (np.log(a/a02))**2
    return B1/a * np.exp(-component1) + B2/a*np.exp(-component2)

def dnda_WD01(acm,dusttype,GSD_params=None,ed=False): #alpha,beta,at,ac,Cs
    alpha,beta,at,ac,Cs,a01,a02,sigma,B1,B2 = GSD_params

    dnda=(Cs/acm)*(acm/at)**alpha
    if (beta >= 0.):
        dnda = dnda*(1.+beta*acm/at)
    else:
        dnda = dnda/(1.-beta*acm/at)
    # dnda = where(beta>=0,  dnda*(1.+beta*acm/at), dnda/(1.-beta*acm/at))

    if (ed):
        # if (a > at):
        dnda[acm>at] = dnda[acm>at]*np.exp(((at-acm[acm>at])/ac)**3)
        # else: 
        # dnda = dnda
        #endif
    #endif
    if dusttype=='sil':
        return dnda
    if dusttype=='car':
        dnda_log=(B1/acm)*np.exp(-0.5*(np.log(acm/a01)/sigma)**2)+ (B2/acm)*np.exp(-0.5*(np.log(acm/a02)/sigma)**2)
        return dnda + dnda_log

# def dnda_WD01_carbon(acm,GSD_params=None,ed=False):
#     alpha,beta,at,ac,Cs,a01,a02,sigma,B1,B2 = GSD_params
#     dnda_log=(B1/acm)*np.exp(-0.5*(np.log(acm/a01)/sigma)**2)+ (B2/acm)*np.exp(-0.5*(np.log(acm/a02)/sigma)**2)

#     dnda_nonlog=(Cs/acm)*(acm/at)**alpha
#     if (beta >= 0.):
#         dnda_nonlog = dnda_nonlog*(1.+beta*acm/at)
#     else:
#         dnda_nonlog = dnda_nonlog/(1.-beta*acm/at)
#     # dnda = where(beta>=0,  dnda*(1.+beta*acm/at), dnda/(1.-beta*acm/at))

#     if (ed):
#         dnda_nonlog[acm>at] = dnda_nonlog[acm>at]*np.exp(((at-acm[acm>at])/ac)**3)
#     #endif
#     dnda = dnda_log + dnda_nonlog
#     return dnda
    
# def dnda_pah(a, sigma, fC = 0.05, C_H = 4e-4):
#     # AMRN= 6.9e-26
#     # dna = AMRN*a**(-3.5) # power law term
#     rho_PAH = 2.2
#     mC = 12.01*1.6605402e-24
#     a0 = np.array([4.0,30])*1e-8
    
#     dnda_log= np.exp(-0.5*(np.log(a/a0)/sigma)**2)/a
#     dnda_MRN= AMRN*a**(-3.5)
    
#     NC_log  = scp.integrate.simps(dnda_log*rho_PAH*(4*np.pi*a**3./3.), a)/mC #total C atoms in log-mod per H
    
#     NC      = fC*C_H # fC = fraction of C abundance in log-normal distribution
#     B       = (NC)/NC_log
    
#     #using Eq 9 in Hensley & Draine (2017)
    
#     xC  = 3.*sigma/pow(2.,0.5)+ np.log(a0/np.min(a))/(sigma*np.sqrt(2))
#     supl   = mC*(C_H*fC)/(1+ erf(xC))
#     B_HD17 = 3./pow(2*np.pi,1.5)*exp(-4.5*sigma**2)/(a0**3*rho_PAH*sigma)*supl
    
#     #print 'B=', B, B_HD17
#     dn_da = dnda_log*B_HD17 #+ dnda_MRN
    
#     return dn_da

def dnda(dusttype,aef,power_index,dust_to_gas_ratio=0.01):
    
    na = len(aef)
    dnda_gr    = np.zeros(na)
    for i in range(na):
        a = aef[i] ##cm

        #---MRN distribution----------------
        if dusttype=='sil':
            #print ('[dnda]: We are using silicate grains')
            AMRN=AMRN_sil(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
            dnda    = AMRN* pow(a,power_index)

        elif dusttype=='car':
            #print ('[dnda]: We are using carbon grains')
            AMRN=AMRN_car(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
            dnda    = AMRN* pow(a,power_index)

        elif dusttype=='astro':
            # a0,a1,rho_dust,ratio,beta=GSD_params
            #Normalization constant for a fixed 
            C = AMRN_dust_gas_ratio(aef.min(),aef.max(),2.7,ratio=dust_to_gas_ratio,beta=power_index)
            dnda = C*pow(a,power_index) #* np.exp(-acm/a1)

        else:
            dnda = np.nan

        dnda_gr[i]  = dnda

    return dnda_gr

def dnda_ref(INDEX,dusttype,aef,sizedist,power_index,dust_to_gas_ratio=0.01):
    # if (sizedist==GSD_law):
    #     if(sizedist == 'MRN'):
    #         #log.info('   *** [re-check] MRN with the power of %f [\033[1;7;34m ok \033[0m]'%(power_index))
    #     elif (sizedist=='WD01') or (sizedist=='DL07'):
    #         #log.info('   *** [re-check] %s grain-size distribution [\033[1;7;34m ok \033[0m]'%(sizedist))
    #     else:
    #         log.error('   \033[1;91m Grain-size distribution is not valid [MRN/WD01/DL07] \033[0m')
    #         raise IOError('Grain size distribution')
    # else:
    #     #log.info('   *** [re-check] %s grain-size distribution [\033[1;5;7;91m failed \033[0m]'%(sizedist))
    #     log.error('       \033[1;91m Grain-size distribution is not valid [use MRN/WD01/DL07] \033[0m')
    #     raise IOError('Grain size distribution')
    #if dusttype=='silicate':
    #    log.info('   *** [re-check] We are using silicate grains')
    #elif dusttype=='carbon':
    #    log.info('   *** [re-check] We are using carbon grains')
    #else:
    #    log.error('%s is not defined!'%(sizedist))
    
    na = len(aef)
    dnda_gr    = np.zeros(na)
    for i in range(na):
        a = aef[i]

        #---MRN distribution----------------
        # if(sizedist == 'MRN'):
            
        #log.info('We are using the MRN-like distribution with the power of %f '%(power_index))
        #AMRN=AMRN_dust_gas_ratio(aef[0], aef[-1], rhoo, ratio=dust_to_gas_ratio, beta=MRN_power)
        #dnda    = AMRN* pow(a,MRN_power)
        if dusttype=='silicate':
            #print ('[dnda]: We are using silicate grains')
            AMRN=AMRN_sil(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
        if dusttype=='carbon':
            #print ('[dnda]: We are using carbon grains')
            AMRN=AMRN_car(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
        dnda_MRN    = AMRN* pow(a,power_index)

        # else:
        #log.info('We are using %s grain-size distribution'%(sizedist))
        data = readD('./data/values.dat',2,10)
        if (dusttype == 'carbon'):
            ALPHA = data[0,INDEX]
            BETA  = data[1,INDEX]
            A_T   = data[2,INDEX]*1.E-4
            A_C   = data[3,INDEX]*1.E-4
            C_j   = data[4,INDEX]
            C_MRN = 10**(-25.13)
            rhoo  = 2.2
        if (dusttype == 'silicate'):
            ALPHA = data[5,INDEX]
            BETA  = data[6,INDEX]
            A_T   = data[7,INDEX]*1.E-4
            A_C   = 1.E-5
            C_j   = data[8,INDEX]
            C_MRN = 10**(-25.11)
            rhoo  = 3.0
        #BC5       = data[9,INDEX]
        #if(sizedist == 'DL07'):
        #    BC5   = 0.92*BC5
        B1        = 2.0496E-7
        B2        = 9.6005E-11
    
        
        dnda=(C_j/a)*(a/A_T)**ALPHA #the factor C_j/a in ==. 4
        if (BETA >= 0.):
            dnda = dnda*(1.+BETA*a/A_T)
        else:
            dnda = dnda/(1.-BETA*a/A_T)
        #endif

        if (a > A_T):
            dnda = dnda*np.exp(((A_T-a)/A_C)**3)  ##Silicate grain
        else: 
            dnda = dnda
        #endif

        if (dusttype == 'carbon'):
            dnda_nonlog    = dnda
            
            if(sizedist == 'WD01'):
                a01 = 3.5E-8
                a02 = 3.E-7
                SIG1= 0.4
                SIG2= 0.4
                BC5 = data[9,INDEX]
                dndaVSG=(B1/a)*np.exp(-0.5*(np.log(a/a01)/SIG1)**2)+ (B2/a)*np.exp(-0.5*(np.log(a/a02)/SIG2)**2)
                if (dndaVSG >=  0.0001*dnda):
                    dnda    = dnda_nonlog+BC5*dndaVSG
                       
            elif(sizedist == 'DL07'):
                a01    = 4.E-8
                a02    = 2.E-7
                SIG1= 0.4
                SIG2= 0.55
                BC5   = 0.92*BC5

                a0        = np.array([4.0,20.0]) #angstrom
                a0        = a0*(1e-8)    #cm
                a_min    = 3.556e-8    #cm
                sigma    = np.array([0.4, 0.55])

                a_m        = a0*np.exp(3.*sigma**2.)
                mC        = 12.*(1.67e-24)
                rho_C    = 2.2

                bJ        = np.array([BC5*0.75,BC5*0.25])
                bJ        = bJ*(1e-5)

                xj        = np.log(a_m/a_min)/(sigma*np.sqrt(2))
                n0_j    = (3./(2.*np.pi)**1.5)*(np.exp(4.5*sigma**2.)/(1+erf(xj)))*(mC/(rho_C*a_m**3.*sigma))*bJ
                        
                supl    = (np.log(a/a0))**2./(2.*sigma**2.)

                #if(dusttype == 'carbon'):
                dnda = dnda_nonlog + ((n0_j/a)*np.exp(-supl)).sum()
        dnda_gr[i]  = dnda + dnda_MRN

    return dnda_gr

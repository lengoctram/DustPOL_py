"""There was a problem for a factor of 4 different in calculating the radiation energy, using Planck function or specific luminosity.
For the current version, Planck functions are mostly used with the temperature as the input parameter.
However, this factor uncertain might be embeded somewhere.

This factor is due to:
	- BB function accounts for all direction radiation
	- Luminosity accounts for the radial direction outwards from the central source.
		--> isotropic radiation: urad(BB) = 4urad(L)
		--> otherwise: urad(BB) = 4urad(L)
	
	For Lbol < 1e5: surrounding medium is optically thin (free escape): urad(BB) = 4urad(L)
	For Lbol > 1e5: surrounding medium is optically thick (radiation is absorbed and re-emitted many times --> isotropized): urad(BB) = urad(L)
"""
import numpy as np # type: ignore
import scipy
import scipy as scp # type: ignore
import matplotlib.pyplot as plt # type: ignore
from astropy import constants # type: ignore
from collections import OrderedDict
ls                  = OrderedDict(
                                 [
                                  ('solid',               (0, ())),
                                  ('dashed',              (0, (5, 5))),
                                  # ('solid',               (0, ())),
                                  ('dashdotted',          (0, (5, 4, 1, 6))),
                                  ('dotted',              (0, (1, 5))),

                                  ('loosely dashed',      (0, (5, 15))),
                                  ('densely dashed',      (0, (5, 1))),
                                  ('loosely dotted',      (0, (1, 10))),
                                  ('densely dotted',      (0, (1, 1))), 
                                  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

                                  ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                                  ('densely dashdotted',  (0, (3, 1, 1, 1))),

                                  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
keys = list(ls.keys())

au = constants.au.cgs.value
pc = constants.pc.cgs.value
c  = constants.c.cgs.value
h  = constants.h.cgs.value
kB = constants.k_B.cgs.value
eV = 1.602e-12
uISRF=8.64e-13
Lsun = constants.L_sun.cgs.value
Rsun = constants.R_sun.cgs.value
sigma_SB = constants.sigma_sb.cgs.value


rin=0.1*pc#0.05*pc #0.005pc(for L<1e2) #0.03*pc#(for L=1e3) #0.05*pc (elsewhere)
nin=3e6#1.5e6
Lstar=1e5*Lsun
Tdmax=1500#10**(3.5)

if Lstar==6e6*Lsun:
	Tstar=5e4
elif Lstar==1.e6*Lsun or Lstar==6.3e5*Lsun:
	Tstar=4.5e4#2.5e4#3.e4#1.5e4
elif Lstar==4.7e5*Lsun or Lstar==8.7e5*Lsun:
	Tstar=4.5e4
elif Lstar==1.e5*Lsun:
	Tstar=3.e4
elif Lstar==1e4*Lsun:
	Tstar=2.5e4
elif Lstar==2.5e3*Lsun:
	Tstar=2.2e4
elif Lstar==1e3*Lsun:
	Tstar=1.8e4
elif Lstar==1e2*Lsun:
	Tstar=1.e4
elif Lstar==1e1*Lsun:
	Tstar=8000.
elif Lstar==Lsun:
	Tstar=6000.
elif Lstar==0.1*Lsun:
	Tstar=1.0
else:
	raise ValueError('Lstar is not correct!')

# Rstar = 11*Rsun##np.sqrt(Lstar/(4*np.pi*sigma_SB*pow(Tstar,4)))#11*R_sun#5.3*R_sun
#35*Rsun
Rv=4.0
p=2.4
rsub = 155.3*(Lstar/1e6/Lsun)**(0.5) * (Tdmax/1500.)**(-5.6/2) * au
print('rsub/pc=',rsub/pc)
# rsub = 108.7*(Lstar/1e6/Lsun)**(0.5) * (Tdmax/1500.)**(-5.6/2) * au
# rsub = (Tdmax/Tstar)**(-2) * 11.*Rsun
rmin=1*au; rmax=2*0.5*pc # nin = 1e6

rotdesorb=False

# [wave1, wave2] = [h*c*1e4/(13.6*eV), h*c*1e4/(6.*eV)]#20.] #lower and upper cutoff of wavelengths for RATs
[wave1, wave2] = [h*c*1e4/(13.6*eV), 20.] #lower and upper cutoff of wavelengths for RATs


def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    return a + " \\times 10^" + str(b)

def pc_to_au(pc_value):
	return pc_value*pc/au

def au_to_pc(au_value):
	return au_value*au/pc

## define the gas volume density
def ngas(r):
	##Any emprical formulation can be used
	##here: Hoang et al. 2021 is adopted
	##Note: r is the radial distance from center to the envelope
	
	#return lambda r: np.where(r<=rin, nin, nin*(r/rin)**(-p))
	return nin * pow(rin,p)/(pow(rin,p)+pow(r,p))
	# return lambda r: np.where(r<=Rflat, n0, n0*(r/Rflat)**(-3./2))

# def func_Av_s(rsub):
# 	# Av_c = 10.3 * (nin/1.e8) * (rin/10./au) * (Rv/4.0)
# 	NH2=(rin-rsub)*nin + nin*rin/(p-1)*(1-pow(r/rin,1-p))
# 	return lambda r: np.where(r <= rin, nin*(r-rsub)*Rv/(5.8e21), NH2/5.8e21 * Rv)

def fun_Av_s_num(r,Rv=4.0):
	Avs=np.zeros(len(r))
	Avc=10.3*(nin/1e8)*(rin/10/au)*(Rv/4.0)
	for i in range(len(r)):	
		if r[i]<=rin:
			Avs[i]=Avc * (r[i]/rin)
		else:
			Avs[i]= Avc*(1.+1./(p-1)*(1.-(r[i]/rin)**(1.-p)) )
	return Avs

def fun_Av_s(r,Rv=4.0):
	Avs=np.zeros(len(r))
	for i in range(len(r)):
		if r[i]>=rsub:
			rr=np.logspace(np.log10(rsub),np.log10(r[i]),100)
			Ngas = nin * scp.integrate.simpson(1./(1.+pow(rr/rin,p)),rr)
			Avs[i] = Ngas/5.8e21 * Rv
		else:
			rr=np.logspace(np.log10(r[i]),np.log10(rsub),100)
			Avs[i]=0.0
		# Ngas = nin * scp.integrate.simps(1./(1.+pow(rr/rin,p)),rr)
		# Avs[i] = Ngas/5.8e21 * Rv

		# print('Ngas=',Ngas,'Av=',Avs)
	return Avs

def radiation_ana_shell(r,Tshell):
	if Tshell==647.:
		c1=0.02330625; c2 = 1.10216326 
	elif Tshell==1000.:
		c1=0.044; c2=1.134
	elif Tshell==1200.:
		c1=0.05898573;c2=1.15000396
	elif Tshell==1500.:
		c1=0.08452511; c2=1.17010132
	elif Tshell==1819.:
		c1=0.116; c2=1.187
	elif Tshell==2000.:
		c1=0.13; c2= 1.21
	elif Tshell==2300.:
		c1=0.17584011; c2=1.20705394
	elif Tshell==3000.:
		c1=0.253; c2=1.248
	elif Tshell==5000.:
		c1=0.587; c2=1.355
	elif Tshell==7000.:
		c1=0.991; c2 = 1.467
	elif Tshell==10000.:
		c1=1.594; c2=1.688
	elif Tshell==15000.:
		c1=2.669; c2=1.953
	elif Tshell==30000.:
		c1=4.94; c2=2.48
	elif Tshell == 50000.:
		c1=6.86; c2=2.63
	else:
		raise IOError('Tshell could not be found!')
	# Av_s = func_Av_s(rsub)(r)
	Av_s = fun_Av_s(r)#fun_Av_s_num(r)#
	# Uin = Lstar/(4*np.pi*c*uISRF) #Lstar/(4*np.pi*rin*rin * c*uISRF)
	Urad = Lstar/(4*np.pi*r*r*c*uISRF) *1./(1.+c1*pow(Av_s,c2)) #* pow(rsub,-2.0)#* pow(r,-2.0)
	Td1 = 19.5*pow(Urad, 1./5.6)#16.4*pow(U,1./6)
	Td2 = 16.4*pow(Urad,1./6)
	Td = 0.5*(Td1+Td2)
	return Td,Urad

def radiation_ana_star(r,Tstar):
	if Tstar==1000.:
		c1=0.044; c2=1.134
	elif Tstar==1500.:
		c1=0.08452511; c2=1.17010132
	elif Tstar==2000.:
		c1=0.13; c2= 1.21
	elif Tstar==2300.:
		c1=0.17584011; c2=1.20705394
	elif Tstar==3000.:
		c1=0.253; c2=1.248
	elif Tstar==5000.:
		c1=0.587; c2=1.355
	elif Tstar==7000.:
		c1=0.991; c2 = 1.467
	elif Tstar==10000.:
		c1=1.594; c2=1.688
	elif (Tstar>=15000.) and (Tstar<=20000.):
		c1=2.669; c2=1.953
	elif Tstar==30000.:
		c1=4.94; c2=2.48
	elif Tstar==45000.:
		c1=6.60; c2=2.66
	elif Tstar<=50000.:
		c1=6.86; c2=2.63
	else:
		raise IOError('Tstar could not be found!')

	Av_s = fun_Av_s(r)#fun_Av_s_num(r)#
	# Ustar=Lstar/(4*np.pi*rsub*rsub*c*uISRF)*1./(1.+c1*pow(Av_s,c2)) * pow(r/rsub,-2)
	Ustar=Lstar/(4*np.pi*r*r*c*uISRF)*1./(1.+c1*pow(Av_s,c2))
	Td3 = 19.5*pow(Ustar, 1./5.6)#16.4*pow(U,1./6)
	Td4 = 16.4*pow(Ustar,1./6)
	Tds = 0.5*(Td3+Td4)
	return Tds,Ustar

	# Td = 16.4*(Urad)**(1./6) #* (1+c1*pow(Av_s,c2))**(-p/2)
	# # # Td = 16.4*(Uin)**(1./6) *(r/rin)**(-1./3) *(1+c1*pow(Av_s,c2))**(-1/6)
	# Td_cal = np.where(Td<=Tdmax,Td,Tdmax)
	# Urad_max = (Tdmax/16.4)**(6)
	# return Td, np.where(Td<=Tdmax,Urad,Urad_max)
	# return Td,Urad

def radiation_shell(r,Tshell):
	nw  = 129
	TEM		= Tshell #temperature of protostars/stars
	W		= 1.0
	ZZ		= h*c/(kB*TEM)

	# Total wavelength range above Lyman limit at wave1=0.091um (13.6ev) due to H ionization
	#wave = exp(log(20./0.091)*arange(nw)/(nw-1) + log(0.091))
	wave = np.logspace(np.log10(wave1),np.log10(wave2),nw)#np.exp(np.log(wave2/wave1)*np.arange(nw)/(nw-1) + np.log(wave1))
	lamcgs = wave*1e-4

	# Testing with GMC at D = 5 kpc
	NAV= len(r)#50
	AV = fun_Av_s(r)#fun_Av_s_num(r)#
	#reddening radfield of a stellar spectrum inside GMC
	# uwave_star = np.zeros(nw)
	# uwave_red = np.zeros((NAV, nw))
	# urad_red = np.zeros(NAV)

	#Spectral energy density of a star
	wavelen	= wave
	wavelen1= wavelen*(1e-4) # in cgs
	I3stars	= W*(2*h*c**2./wavelen1**5.)*(1./(np.exp(ZZ/wavelen1)-1.))
	# print('test=',scp.integrate.simps(I3stars,wavelen1)/1e10)
	# print('I3star=',I3stars)
	uwave_star = (4*np.pi*wavelen1/c)*(I3stars)/wavelen1
	# uwave_star[i,:] = (pi*wavelen1/c)*(I3stars)/wavelen1
	#
	# Compute radfield in a GMC using the reddening law: ulam= u0*exp(-tau)
	import extcurves
	[A_lambda_AV, _] = extcurves.extcurve_obs(wave, Rv)
	# for iv in range(0, NAV):
	# tau_wave = AV[iv]*A_lambda_AV/1.086

	AV = AV.reshape(len(AV),1)
	A_lambda_AV=A_lambda_AV.reshape(1,len(A_lambda_AV))
	tau_wave = AV*A_lambda_AV/1.086
	# print('AV=',AV)
	# print('tau_wave=',tau_wave)
	# for istar in range(0, nstar):
	uwave_red   = uwave_star*np.exp(-tau_wave)
	# print('uwave_red=',uwave_red)
	urad_red	= scp.integrate.simpson(uwave_red,lamcgs)
    
	mean_wave = scp.integrate.simpson(uwave_red*lamcgs,lamcgs)/urad_red
	# print('urad_red=',urad_red)
	#
	U  = urad_red/uISRF*pow(r/rsub,-2)
	# Td = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	# Td = 16.4*pow(U,1./6)
	Td1 = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	Td2 = 16.4*pow(U,1./6)
	# Td = (0.5*(pow(Td1,4) + pow(Td2,4)))**(1./4)#0.5*(Td1+Td2)
	Td = (0.375*pow(Td1,4) + 0.625*pow(Td2,4))**(1./4)
	return Td,U,mean_wave

# def Rstar_rsub(Tstar):
# 	f1 = np.sqrt(1e6 * Lsun)
# 	f2 = 1.25*pow(Tstar,2)*au
# 	return f1/f2

def Rstar_rsub(Tstar):
	f1 = np.sqrt(1e6 * Lsun)
	f2 = 155.3*au * pow(Tdmax/1500.,-5.6/2) * np.sqrt(4*np.pi*sigma_SB)*pow(Tstar,2)
	return f1/f2

def radiation_star(r,Tstar):
	nw  = 129
	TEM		= Tstar #temperature of protostars/stars
	W		= 1.0
	ZZ		= h*c/(kB*TEM)

	# Total wavelength range above Lyman limit at wave1=0.091um (13.6ev) due to H ionization
	#wave = exp(log(20./0.091)*arange(nw)/(nw-1) + log(0.091))
	wave = np.logspace(np.log10(wave1),np.log10(wave2),nw)#np.exp(np.log(wave2/wave1)*np.arange(nw)/(nw-1) + np.log(wave1))
	lamcgs = wave*1e-4

	# Testing with GMC at D = 5 kpc
	NAV= len(r)#50
	AV = fun_Av_s(r)#fun_Av_s_num(r)#
	print('Avs=',AV)
	#reddening radfield of a stellar spectrum inside GMC
	# uwave_star = np.zeros(nw)
	# uwave_red = np.zeros((NAV, nw))
	# urad_red = np.zeros(NAV)

	#Spectral energy density of a star
	wavelen	= wave
	wavelen1= wavelen*(1e-4) # in cgs
	I3stars	= W*(2*h*c**2./wavelen1**5.)*(1/(np.exp(ZZ/wavelen1)-1))
	# print('test=',scp.integrate.simps(I3stars,wavelen1)/1e10)
	# print('I3star=',I3stars)
	uwave_star = (4*np.pi*wavelen1/c)*(I3stars)/wavelen1
	# uwave_star[i,:] = (pi*wavelen1/c)*(I3stars)/wavelen1
	#
	# Compute radfield in a GMC using the reddening law: ulam= u0*exp(-tau)
	import extcurves
	[A_lambda_AV, A_lambda_NH] = extcurves.extcurve_obs(wave, Rv)
	# for iv in range(0, NAV):
	# tau_wave = AV[iv]*A_lambda_AV/1.086

	AV = AV.reshape(len(AV),1)
	A_lambda_AV=A_lambda_AV.reshape(1,len(A_lambda_AV))
	tau_wave = AV*A_lambda_AV/1.086
	# print('AV=',AV)
	# print('tau_wave=',tau_wave)
	# for istar in range(0, nstar):
	uwave_red   = uwave_star*np.exp(-tau_wave)
	# print('uwave_red=',uwave_red)
	urad_red	= scp.integrate.simpson(uwave_red,lamcgs)
    
	mean_wave = scp.integrate.simpson(uwave_red*lamcgs,lamcgs)/urad_red
	# print('urad_red=',urad_red)
	#
	Rstar=Rstar_rsub(Tstar)*rsub
	# Rstar=Lstar/(4*np.pi*sigma_SB*pow(Tstar,4)) #Rstar_rsub(Tstar)*rsub
	# Rstar=np.sqrt(Rstar)
	# Rstar=11.*Rsun
	U  = urad_red/uISRF*pow(r/Rstar,-2)
	# Td = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	# Td = 16.4*pow(U,1./6)
	Td1 = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	Td2 = 16.4*pow(U,1./6)
	# Td = (0.5*(pow(Td1,4) + pow(Td2,4)))**(1./4)#0.5*(Td1+Td2)
	Td = (0.375*pow(Td1,4) + 0.625*pow(Td2,4))**(1./4)
	return Td,U,mean_wave

def radiation_wrong(r):
	nw  = 129
	TEM		= Tstar #temperature of protostars/stars
	W		= 1.0
	ZZ		= h*c/(kB*TEM)
	
	# rsub = 155.3*(Lstar/1e6/Lsun)**(0.5) * (2500/1500.)**(-5.6/2) * au

	# Total wavelength range above Lyman limit at wave1=0.091um (13.6ev) due to H ionization
	#wave = exp(log(20./0.091)*arange(nw)/(nw-1) + log(0.091))
	wave = np.logspace(np.log10(wave1),np.log10(wave2),nw)#np.exp(np.log(wave2/wave1)*np.arange(nw)/(nw-1) + np.log(wave1))
	lamcgs = wave*1e-4

	# Testing with GMC at D = 5 kpc
	NAV= len(r)#50
	AV = fun_Av_s_num(r)#fun_Av_s(r)
	#reddening radfield of a stellar spectrum inside GMC
	# uwave_star = np.zeros(nw)
	# uwave_red = np.zeros((NAV, nw))
	# urad_red = np.zeros(NAV)

	#Spectral energy density of a star
	wavelen	= wave
	wavelen1= wavelen*(1e-4) # in cgs
	I3stars	= W*(2*h*c**2./wavelen1**5.)*(1/(np.exp(ZZ/wavelen1)-1))
	# print('test=',scp.integrate.simps(I3stars,wavelen1)/1e10)
	# print('I3star=',I3stars)
	uwave_star = (4.*np.pi*wavelen1/c)*(I3stars)/wavelen1
	# uwave_star[i,:] = (pi*wavelen1/c)*(I3stars)/wavelen1
	#
	# Compute radfield in a GMC using the reddening law: ulam= u0*exp(-tau)
	import extcurves
	[A_lambda_AV, A_lambda_NH] = extcurves.extcurve_obs(wave, Rv)
	# for iv in range(0, NAV):
	# tau_wave = AV[iv]*A_lambda_AV/1.086

	AV = AV.reshape(len(AV),1)

	A_lambda_AV=A_lambda_AV.reshape(1,len(A_lambda_AV))
	tau_wave = AV*A_lambda_AV/1.086
	# print('AV=',AV)
	# print('tau_wave=',tau_wave)
	# for istar in range(0, nstar):
	uwave_red   = uwave_star*np.exp(-tau_wave)
	# print('uwave_red=',uwave_red)
	urad_red	= scp.integrate.simpson(uwave_red,lamcgs)
    
	mean_wave = scp.integrate.simpson(uwave_red*lamcgs,lamcgs)/urad_red
	# print('urad_red=',urad_red)
	#
	U  = urad_red/uISRF*pow(r/rsub,-2)
	# Td = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	# Td = 16.4*pow(U,1./6)
	Td1 = 19.5*pow(U, 1./5.6)#16.4*pow(U,1./6)
	Td2 = 16.4*pow(U,1./6)
	Td = 0.5*(Td1+Td2)
	return Td,U#,mean_wave

def mean_lam(r):
    if Tstar==3000.:
        c3=0.206; c4=0.675
    elif Tstar==5000.:
        c3=0.410; c4=0.647
    elif Tstar==7000.:
        c3=0.595; c4=0.647
    elif Tstar==10000.:
        c3=0.942; c4=0.626
    else:
        raise IOError('Tstar could not be found!')
    Av_s = fun_Av_s_num(r)#fun_Av_s()(r)
    mean_lam_star = 0.54/Tstar ##cm
    return mean_lam_star * (1. + c3*pow(Av_s,c4))

# def Tdust(r):
# 	Td = 16.4*(radiation(r))**(1./6)
# 	return Td
	
## range of radius 'r'
# rmin=func_rsub(Tdmax) #1*au#0.01*rin 
# rmax=1e3*rin #--low-mass
# rmax=100.0*pc#1e2*rin # nin = 1e6
# rmax=3e2*rin #nin = 1e7

# def Tdust_Hoang(r,Luv):
# 	return 1800*(pow(Luv/5e12/Lsun,0.5)*pow(1*pc/r,2))**(1./5.6)

def func_Tshell(Tstar):
	# return pow(4,1./4) * np.sqrt(Rstar_rsub(Tstar)) * Tstar
	return np.sqrt(Rstar_rsub(Tstar)) * Tstar

def func_Tshell_from_U(r,Ustar):
	f = scipy.interpolate.interp1d(r,Ustar)
	Ustar_rsub=f(rsub)
	a1=16.4*pow(Ustar_rsub,1./6)
	a2=19.5*pow(Ustar_rsub,1./5.6)
	return (0.625*pow(a1,4) + 0.375*pow(a2,4))**(1./4)

if __name__=='__main__':
	# r = np.logspace(np.log10(rmin),np.log10(rmax),40)
	# r = np.linspace(rmin,rmax,50)
	r = np.logspace(np.log10(rsub*0.001),np.log10(rmax),40) 
	# r = np.logspace(0.0,np.log10(rmax),40) 

	# r = np.logspace(np.log10(rsub),np.log10(rmax),40) 
	densities    = ngas(r)#(r)
	temperatures_wrong,U_wrong = radiation_wrong(r)
	temperatures_star,U_star,wave_star = radiation_star(r,Tstar)
	temperatures_shell=np.zeros(len(r))
	U_shell=np.zeros(len(r))
	wave_shell=np.zeros(len(r))
	for i,ri in enumerate(r):
		if ri<1.*rsub:
			temperatures_shell[i]=0.0
			U_shell[i]=0.0
		else:
			# temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],Tdmax) #this matches better to NM2004 but don't understand whey Tshell=Tsub!
			# temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],func_Tshell_from_U(r,U_star)) ##This method is identical to the next method
			temperatures_shell[i],U_shell[i],wave_shell[i] = radiation_shell([ri],func_Tshell(Tstar))             ##This method is identical to the previous method

	# temperatures_num_shell,Uana_shell = radiation_ana_shell(r,Tdmax)
	# temperatures_num_star,Uana_star = radiation_ana_star(r,Tstar)

	Utot = U_shell+U_star
	# Td1 = 19.5*pow(Utot, 1./5.6)#16.4*pow(U,1./6)
	Td2 = 16.4*pow(Utot,1./6)
	# Td_cal = (0.5*(pow(Td1,4) + pow(Td2,4)))**(1./4)#0.5*(Td1+Td2)
	# Td_cal = (0.375*pow(Td1,4) + 0.625*pow(Td2,4))**(1./4)#0.5*(Td1+Td2)
	Td_cal = Td2
	for i,ri in enumerate(r):
		print('r/pc=',ri/pc, f'Ustar={U_star[i]:.3e}', f'Ushell={U_shell[i]:.3e}', f'Tshell={func_Tshell(Tstar):.3f}', f'Tdust={Td_cal[i]:.2f}')


	# Utot_ana = Uana_shell+Uana_star ##analytical
	# Td3 = 19.5*pow(Utot_ana, 1./5.6)#16.4*pow(U,1./6)
	# Td4 = 16.4*pow(Utot_ana,1./6)
	# Td_ana = 0.5*(Td3+Td4)

	fig,ax=plt.subplots(figsize=(8,8))
	ax.loglog(r/pc,densities,'k-',ls=ls[keys[0]],lw=2.5,label='Volume density')
	ax.loglog([],[],'k',ls=ls[keys[1]],label='Temperature (this work)')
	# ax.loglog([],[],'k',ls=ls[keys[2]],label='Temperature (analytical, $n_{\\rm gas}\sim r^{-2.4}$)')

	# ax.set_title('low-mass protostar embedded',pad=20)
	ax.set_xlabel('$\\sf Distance\\,(pc)$')
	ax.set_ylabel('$\\sf Density\\,(cm^{-3})$')
	ax.legend(loc='lower left', fontsize=20)
	text = sci_notation(Lstar/Lsun,sig_fig=1)
	ax.text(0.03,0.3,'$\\sf L_{\\ast}='+str(text)+'L_{\\odot}$', transform=ax.transAxes)

	secax = ax.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
	secax.set_xscale('log')
	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

	ax.tick_params(axis='x', which='both', top=False, labeltop=False)
	secax.tick_params(axis='x', which='both', top=True)

	ax1=ax.twinx()
	# ax1.loglog(r/pc,temperatures,'g.')	
	ax1.loglog(r/pc,Td_cal,'k',ls=ls[keys[1]],lw=2.5)
	# ax1.loglog(r/pc,temperatures_shell,'r',ls=ls[keys[1]])
	# ax1.loglog(r/pc,temperatures_star,'b',ls=ls[keys[1]])
	# ax1.loglog(r/pc,Td_ana,'k',ls=ls[keys[2]])
	# ax1.loglog(r/pc,temperatures_wrong,'r')
	# ax1.loglog(r/pc,temperatures_shell,'k',ls=ls[keys[2]])

	# ax1.loglog(r/pc,temperatures_shell,'b-')
	# ax1.loglog(r/pc,temperatures_star,'b--')
	# ax1.loglog(r/pc,temperatures_num_shell,'b--')
	# ax1.loglog(r/pc,temperatures_num_star,'b--')
	# ax1.loglog(r/au,Tdust_Hoang(r,Lstar/20),'k:')
	ax1.set_ylabel('$\\sf Temperature\\,(K)$')

	fig2,ax2=plt.subplots(figsize=(8,8))
	# ax2.loglog(r/pc,Td_cal,'k-',lw=3,label='Total')
	# r_nodust = np.logspace(np.log10(rsub*0.001),np.log10(rsub),40) 
	#_,Ustar_nodust = radiation_star(r_nodust,Tstar)#
	r_nodust=r
	Ustar_nodust=2*Lstar/(4*np.pi*r*r * c *uISRF)
	ax2.loglog(r/pc,U_star,'k',ls=ls[keys[0]],lw=2.5,label='U(T$_{\\ast}$)')
	ax2.loglog(r_nodust/pc,Ustar_nodust,'gray',ls=ls[keys[1]],lw=2.5,label='U(T$_{\\ast}$, nodust)')
	ax2.loglog(r/pc,U_shell,'k',ls=ls[keys[2]],lw=2.5,label='U(T$_{\\sf shell}$)')
	# ax2.axvline(x=rsub/pc,color='gray')
	ax2.legend(loc='lower left',fontsize=20)
	ax2.set_xlim([1*au/pc,1])
	# ax2.set_ylim([10,1000])
	ax2.set_xlabel('$\\sf Distance\\,(pc)$')
	ax2.set_ylabel('Radiation (U)')
	text = sci_notation(Lstar/Lsun,sig_fig=1)
	ax2.text(0.65,0.9,'$\\sf L_{\\ast}='+str(text)+'L_{\\odot}$', transform=ax2.transAxes)

	secax = ax2.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
	secax.set_xscale('log')
	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

	ax2.tick_params(axis='x', which='both', top=False, labeltop=False)
	secax.tick_params(axis='x', which='both', top=True)

	fig3,ax3=plt.subplots(figsize=(8,8))
	ax3.loglog(r/pc,wave_star*1e4,'k',ls=ls[keys[0]],label='Stellar radiation')
	ax3.loglog(r/pc,wave_shell*1e4,'k',ls=ls[keys[1]],label='Dust shell')
	ax3.set_xlabel('$\\sf Distance\\,(pc)$')
	ax3.set_ylabel('Mean wavelength, $\\sf \\bar{\\lambda} \\,(\\mu m)$')
	ax3.legend(fontsize=20)
	ax3.set_xlim([1*au/pc,1])
	L_text = sci_notation(Lstar/Lsun,sig_fig=1)
	ax3.text(0.6,0.2,'$\\sf L_{\\ast}='+str(L_text)+'L_{\\odot}$', transform=ax3.transAxes)
	n_text = sci_notation(nin,sig_fig=1)
	ax3.text(0.6,0.12,'$\\sf n_{0}='+str(n_text)+'\, cm^{-3}$', transform=ax3.transAxes)

	secax = ax3.secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
	secax.set_xscale('log')
	secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)

	ax3.tick_params(axis='x', which='both', top=False, labeltop=False)
	secax.tick_params(axis='x', which='both', top=True)

	# fig,ax=plt.subplots(figsize=(8,8))
	# ax.loglog(r/au,mean_lam*1e4)
	# fig=plt.figure(figsize=(20,6))
	# gs  = gridspec.GridSpec(1,2, figure=fig, width_ratios=[5,3],top=0.95,bottom=0.05,left=0.1,right=0.95)
	# axs = gs.subplots()
	# ax1,ax2=axs

	# # files_st = glob.glob('../grid_folder_10Lsun/starts/*.dat')
	# files_st = glob.glob('../grid_folder/starts/*.dat')
	# files_st.sort(key=os.path.getmtime)

	# # files = glob.glob('../grid_folder_10Lsun/*_0.dat')
	# files = glob.glob('../grid_folder/*_0.dat')
	# files.sort(key=os.path.getmtime)
	# for i in range(1,len(files_st),2):
	# 	df = uclchem.analysis.read_output_file(files[i])
	# 	ax1.semilogy(df['Time']/1e6,df['Density'],'k-')
	# 	ax2.plot(df['Time'],df['gasTemp'],'k-')

	# 	df = uclchem.analysis.read_output_file(files_st[i])
	# 	ax1.semilogy(df['Time']/1e6-df['Time'].max()/1e6,df['Density'],'k-')
	# 	ax2.plot(df['Time']-df['Time'].max(),df['gasTemp'],'k-')
	# ax1.set(yscale='symlog',ylim=(30,5e8))
	# # ax1.set_ylim([0,7e8])
	# ax2.set_xscale('symlog')
	# ax2.set_xlim([-1.5,1e5])
	# ax2.set_ylim([-20,280])

	# ax1.set_xlabel('$\\sf Time\\,(Myr)$')
	# ax1.set_ylabel('$\\sf Density\\,(cm^{-3})$')
	# ax1.text(0.15,0.8,'$\\sf L_{\\ast}=10L_{\\odot}$', transform=ax1.transAxes)
	# ax1.text(0.15,0.9,'low-mass protostar embedded', transform=ax1.transAxes)
	# ax2.set_xlabel('$\\sf Time\\,(yr)$')
	# ax2.set_ylabel('$\\sf Temperature\\,(K)$')

plt.show()
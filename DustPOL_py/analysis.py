import numpy as np
import matplotlib.pyplot as plt
import pwlf, os
import importlib
from scipy.interpolate import interp1d
from scipy import integrate
from scipy import interpolate
from sympy import Symbol
from sympy.utilities import lambdify
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from astropy.io import ascii
from collections import OrderedDict
from . import decorators 
from  .decorators import auto_refresh

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

@auto_refresh
def PiecewiseLineFit(xdata,ydata,nlines=2,force_break=False, x0=None, get_func=False, sigma=None):
    x = Symbol('x')
    def get_symbolic_eqn(pwlf_, segment_number):
        if pwlf_.degree < 1:
            raise ValueError('Degree must be at least 1')
        if segment_number < 1 or segment_number > pwlf_.n_segments:
            raise ValueError('segment_number not possible')
        # assemble degree = 1 first
        get_beta=[];get_se=[]
        for line in range(segment_number):
            if line == 0:
                my_eqn = pwlf_.beta[0] + (pwlf_.beta[1])*(x-pwlf_.fit_breaks[0])
                get_beta.append(pwlf_.beta[0])
                get_beta.append(pwlf_.beta[1])
            else:
                my_eqn += (pwlf_.beta[line+1])*(x-pwlf_.fit_breaks[line])
                get_beta.append(pwlf_.beta[line+1])
        # assemble all other degrees
        #if pwlf_.degree > 1:
        #    for k in range(2, pwlf_.degree + 1):
        #        for line in range(segment_number):
        #            beta_index = pwlf_.n_segments*(k-1) + line + 1
        #            my_eqn += (pwlf_.beta[beta_index])*(x-pwlf_.fit_breaks[line])**k
        return my_eqn.simplify(), get_beta
 
    if (force_break):
        xx=np.array([min(xdata),x0,max(xdata)])
        my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
        res2 = my_pwlf_2.fit_with_breaks(xx)
    else:
        try:
            if sigma==None: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
            else: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata, weights=sigma)
        except:
            if sigma.any()==None: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
            else: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata, weights=sigma)
        res2  = my_pwlf_2.fitfast(nlines,pop=50)
    eqn_list = []
    f_list = []
    beta_list=[]
    if (get_func):
        print ("--------------Fitting information----------------")
    for i in range(my_pwlf_2.n_segments):
        eqns, betas = get_symbolic_eqn(my_pwlf_2, i + 1)
        eqn_list.append(eqns)
        if (get_func):
            print('Fitting function: ', i + 1)
            print(eqn_list[-1])
        f_list.append(lambdify(x, eqn_list[-1]))
    # if (fitto=='temp'):
    	# slopee=betas[2]+betas[1]
    # print('Decreased slope = ', slopee)
    # elif (fitto=='NH'):
        # slopee=betas[1]
        # print('Increased slope = ', slopee)

    slopee=np.array([betas[i] for i in range(1,nlines+1)]).sum()
    beta_list.append(betas[1])
    beta_list.append(betas[2]+betas[1])
    xHat = np.linspace(min(xdata), max(xdata), num=10000)
    yHat = my_pwlf_2.predict(xHat)

    # print('slopes=',my_pwlf_2.slopes,my_pwlf_2.n_segments)
    p = my_pwlf_2.p_values(method='non-linear', step_size=1e-4)
    se = my_pwlf_2.se  # standard errors
    parameters = np.concatenate((my_pwlf_2.beta,
                                 my_pwlf_2.fit_breaks[1:-1]))


    values = np.zeros((parameters.size, 5), dtype=np.object_)
    values[:, 1] = np.around(parameters, decimals=3)
    values[:, 2] = np.around(se, decimals=3)
    values[:, 3] = np.around(parameters / se, decimals=3)
    values[:, 4] = np.around(p, decimals=3)
    # if (get_func):
    #     header = ['Parameter type', 'Parameter value', 'Standard error', 't',
    #               'P > np.abs(t) (p-value)']
    #     print(*header, sep=' | ')
    #     for i, row in enumerate(values):
    #         if i < my_pwlf_2.beta.size:
    #             row[0] = 'Beta'
    #             print(*row, sep=' | ')
    #         else:
    #             row[0] = 'Breakpoint'
    #             print(*row, sep=' | ')

    if (force_break):
        x_break=x0
        e_break=0
    else:
        x_break=res2
        e_break=se[-1]

    return xHat, yHat, x_break, e_break, slopee, res2

def read_headers(filename):
	with open(filename, "r") as file:
		# Loop through each line in the file
		for line in file:
			# Check if the line contains "dust_composition"
			if "dust_composition" in line:
				# Split the line by "=" and strip extra spaces
				key, value = line.strip().split("=")
				if key.strip() == "dust_composition":
					dust_composition = value.strip()  # Get the value
			elif "amax" in line:
				key, value=line.strip().split("=")
				if key.strip() == "amax":
					amax=value.strip("(um)")
	params={}
	params['dust_type'] = dust_composition
	params['amax']      = float(amax)
	return params

@auto_refresh
def get_Av(filename):
	##get Av_array
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	Av_str = lines[7].split(',')
	Av_ls=[]
	for i in range(len(Av_str)):
		try:
			Av_ls.append(eval(Av_str[i]))
		except:
			redo=Av_str[i].split()
			for j in range(len(redo)):
				try:
					Av_ls.append(eval(redo[j]))
				except:
					continue
	Av_=np.array(Av_ls)
	Av_=np.array(list(dict.fromkeys(Av_))) ##de-duplicated values
	if len(Av_)==1:
		return float(Av_)
	else:
		return Av_

@auto_refresh
def plot_pl_single_files(filenames,ax,**plot_kwargs):

	# if isinstance(filenames,list) or isinstance(filenames,np.array):
	if isinstance(filenames,str):
		filenames = [filenames]

	# try:
	# 	amax=np.ones(len(filenames))*params['amax']
	# except:
	# 	amax=np.ones(len(filenames))*np.nan
	for i,filename in enumerate(filenames):
		params=read_headers(filename)
		if params['dust_type'] == 'astro':
			dust_type='astrodust'
		elif params['dust_type']=='astro+pah':
			dust_type='astrodust+PAHs'
		elif params['dust_type']=='sil':
			dust_type='silicate'
		elif params['dust_type']=='sil+car':
			dust_type='silicate+carbon'
		else:
			raise IOError('dust_type is not regconized!')

		Av_=get_Av(filename)

		##get column's name
		f=open(filename,'r')
		lines=f.readlines()
		f.close()
		names = lines[7].split()

		##get the data
		# data=ascii.read(filename,header_start=7, include_names=names)
		data_=np.loadtxt(filename,skiprows=10)

		##wavelength
		#w=data['w'].data
		w=data_[:,0]
		if dust_type in ['astrodust','astrodust+PAHs','silicate']:
			p=data_[:,1]
		else:
			p=data_[:,2]

		p[p<0]=np.nan
		if 'abs' in filename:
			ax.semilogx(w,p/Av_,ls=ls[keys[i]],**plot_kwargs,label='$\\rm A_{V}=%.2f\\,mag.$'%Av_)
		elif 'emi' in filename:
			ax.semilogx(w,p,ls=ls[keys[i]],**plot_kwargs,label='$\\rm A_{V}=%.2f\\,mag.$'%Av_)
		else:
			raise IOError('')
	ax.set_xlabel('$\\rm Wavelength\\, (\\mu m)$')
	if 'abs' in filenames[0]:	
		ax.set_ylabel('$\\rm p_{ext}/A_{V}\\,(\\%/mag.)$')
		ax.text(0.8, 0.9, '$\\rm aligned\\, grain: %s$'%(dust_type), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		try:
			if isinstance(params['amax'],list) or isinstance(params['amax'],np.ndarray):
				ax.text(0.8, 0.8, '$\\rm a_{max}=%s\\, \\mu m$'%(list(params['amax'])), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
			else:
				ax.text(0.8, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%(params['amax']), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		except:
			ax.text(0,0,[])
		ax.legend(loc='upper right',bbox_to_anchor=(0.96,0.7),frameon=False)
		ax.set_xlim([0.03,1e3])
	elif 'emi' in filenames[0]:
		ax.set_ylabel('$\\rm p_{em}\\,(\\%)$')
		ax.text(0.2, 0.9, '$\\rm aligned\\, grain: %s$'%(dust_type), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		try:
			if isinstance(params['amax'],list) or isinstance(params['amax'],np.ndarray):
				ax.text(0.2, 0.8, '$\\rm a_{max}=%s\\, \\mu m$'%(list(params['amax'])), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
			else:
				ax.text(0.2, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%(params['amax']), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		except:
			ax.text(0,0,[])
		ax.legend(loc='lower left', bbox_to_anchor=(0.05,0.2),frameon=False)
		ax.set_xlim([10,3000])
	else:
		raise IOError('str(abs) or str(emi) must be in the filename!')	

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())

@auto_refresh
def plot_pl(filenames,av_range=None,ax=None,**plot_kwargs):
	""" This routine plots the output file of DustPOL-py
		------------------------------------------------
		inputs:
			-filenames     : (str,list,array or tuple) name of datafile(s)
			-dustpol_params: the parameters used to generate the above datafiles
			-av_range      : the list of Av values for different LOS to plotting 
							 only used for the datafile returned from isoCloud_pos [with same input parameters]
			-matplotlib parameters
		outputs:
		    -matplotlib plots
	"""
	if ax is None:
		fig,ax=plt.subplots(figsize=(10,6))

	if (av_range is None) or (len(list(av_range))==1):
		plot_pl_single_files(filenames,ax,**plot_kwargs)
		return

	if isinstance(filenames,list) or isinstance(filenames,tuple) or isinstance(filenames,np.ndarray):
		raise IOError('Only single filename is accepted with your call!')

	if isinstance(filenames,list) or isinstance(filenames,np.ndarray):
		filenames=filenames[0]

	params=read_headers(filenames)
	if params['dust_type'] == 'astro':
		dust_type='astrodust'
	elif params['dust_type']=='astro+pah':
		dust_type='astrodust+PAHs'
	elif params['dust_type']=='sil':
		dust_type='silicate'
	elif params['dust_type']=='sil+car':
		dust_type='silicate+carbon'
	else:
		raise IOError('dust_type is not regconized!')

	Av_=get_Av(filenames)

	##get column's name
	f=open(filenames,'r')
	lines=f.readlines()
	f.close()
	names = lines[7].split()

	##get the data
	# data=ascii.read(filename,header_start=7, include_names=names)
	data_=np.loadtxt(filenames,skiprows=10)

	##wavelength
	#w=data['w'].data
	w=data_[:,0]

	for i,iav in enumerate(av_range):
		if iav>Av_.max() or iav<Av_.min():
			raise IOError('Value of of input Av=%.2f is outof the boundary!'%iav)
			continue

		##Interpolate the computed data over Av_ and w:
		##Follow: https://stackoverflow.com/questions/39332053/using-scipy-interpolate-interpn-to-interpolate-a-n-dimensional-array
		## ---> get the value of p% at a costumized Av_ 
		interp_x=iav
		interp_y=w
		interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
		interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((len(w), 2))

		if 'abs' in filenames:
			sort_indices = np.argsort(Av_)
			Av_sort   = Av_[sort_indices]
			data_sort = data_[:,1:][:,sort_indices]

			interp_arr = interpolate.interpn((Av_sort,w), data_sort.T, interp_points)
			##check for negative values
			interp_arr[interp_arr<0]=np.nan
			ax.semilogx(w,interp_arr/iav,ls=ls[keys[i]],**plot_kwargs,label='$\\rm A_{V}=%.0f$'%iav)
		elif 'emi' in filenames:
			sort_indices = np.argsort(Av_)
			Av_sort   = Av_[sort_indices]
			data_sort = data_[:,1:][:,::2][:,sort_indices]

			interp_arr = interpolate.interpn((Av_sort,w), data_sort.T, interp_points)
			ax.semilogx(w,interp_arr,ls=ls[keys[i]],**plot_kwargs,label='$\\rm A_{V}=%.0f$'%iav)

	# if float(alpha<1):
	#     plt.text(0.8, 0.73, '$\\sf prolate\\, grain: s=1/3$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# else:
	#     plt.text(0.8, 0.73, '$\\sf oblate\\, shape: s=2$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# plt.text(0.8, 0.7, '$\\sf f_{max}=%.1f$'%(f_max), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.set_xlabel('$\\rm Wavelength\\, (\\mu m)$')
	if 'abs' in filenames:
		ax.text(0.8, 0.9, '$\\rm aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		try:
			ax.text(0.8, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%params['amax'], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		except:
			ax.text(0,0,[])
		ax.set_ylabel('$\\rm p_{ext}/A_{V}\\,(\\%/mag.)$')
		ax.legend(loc='upper right',bbox_to_anchor=(0.93,0.8),frameon=False)
	elif 'emi' in filenames:
		ax.text(0.2, 0.9, '$\\rm aligned\\, grain: %s$'%dust_type.lower(), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		try:
			ax.text(0.2, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%params['amax'], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		except:
			ax.text(0,0,[])
		ax.set_ylabel('$\\rm p_{em}\\,(\\%)$')
		ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.1),frameon=False)

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	if 'abs' in filenames:
		ax.set_xlim([0.03,1e3])
	elif 'emi' in filenames:
		ax.set_xlim([10,3000])
	# plt.savefig('Plambda_abs_amax=%.2f.pdf'%(amax))
	plt.show()

@auto_refresh
def plot_pav(filenames,wavelength=850,ax=None,show_break=False,get_info=False,**plot_kwargs):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))
	# save={}
	# filesave='models.dat'
	lam = wavelength

	# if isinstance(amax_range,float) or isinstance(amax_range,int):
	# 	amax_range=[amax_range]

	if isinstance(filenames,str):
		filenames=[filenames]

	# if not np.equal(len(amax_range),len(filenames)):
	# 	raise IOError('length of amax_range and of filenames must be the same!')

	for i,filename in enumerate(filenames):
		filename=filenames[i]
		params=read_headers(filename)
		amax=params["amax"]
		if params['dust_type'] == 'astro':
			dust_type='astrodust'
		elif params['dust_type']=='astro+pah':
			dust_type='astrodust+PAHs'
		elif params['dust_type']=='sil':
			dust_type='silicate'
		elif params['dust_type']=='sil+car':
			dust_type='silicate+carbon'
		else:
			raise IOError('dust_type is not regconized!')

		data=np.loadtxt(filename,skiprows=10)
		Av_=get_Av(filename); new_Av_=np.linspace(Av_.min(),Av_.max(),100)
		w  =data[:,0]
		x,y=np.shape(data)
		data_=[];I_=[]
		if 'abs' in filename:
			for j in range(1,y):
				data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

		elif 'emi' in filename:
			for j in range(1,y,2):
				data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

			# for j in range(1,y):
			# 	data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))
			# I_=data_[1::2]
			# data_=data_[::2]

		data_=np.array(data_)#; print(len(Av_),len(data_))
		f_ip = interpolate.interp1d(Av_,data_,kind='cubic')
		new_data_=f_ip(new_Av_)#*np.sin(30.*np.pi/180)*np.sin(30.*np.pi/180)

		if 'abs' in filename:
			#plt.loglog(Av_,data_/Av_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))
			ax.loglog(new_Av_,new_data_/new_Av_,ls=ls[keys[i]],**plot_kwargs,label='$\\rm a_{max}=%.2f\\, \\mu m$'%(amax))
			x=np.log10(new_Av_)
			# x=np.log10(Av_num[idd])
			y=np.log10(new_data_/new_Av_)
			f=interp1d(new_Av_,new_data_/new_Av_)
			# save['Av(amax=%.2f)'%(amax)]=new_Av_
			# save['p/Av(amax=%.2f)'%(amax)]=new_data_/new_Av_
			# output(args,filesave,save).file_save()
			if (show_break):
				if (get_info):
					get_func=True
				else:
					get_func=False
				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=2,get_func=get_func)
				if (slope<=-0.85):
					x0=10**(xbreak[1])
					y0=f(x0)
					ax.plot(x0,y0,'o',**plot_kwargs)
					ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)

		elif 'emi' in filename:
			sort_indices = np.argsort(Av_)
			Av_sort   = Av_[sort_indices]
			data_sort = data_[sort_indices]

			ax.loglog(Av_sort,data_sort,ls=ls[keys[i]],**plot_kwargs,label='$\\rm a_{max}=%.2f\\, \\mu m$'%(amax))
			# plt.loglog(new_Av_,new_data_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))

			x=np.log10(Av_sort)#[Av_>=1.5])
			y=np.log10(data_sort)#[Av_>=1.5])
			f=interp1d(Av_sort,data_sort)

			if (show_break):
				if (get_info):
					get_func=True
				else:
					get_func=False
				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=3,get_func=get_func)
				if (slope<=-0.85):
					x0=10**(xbreak[2])
					y0=f(x0)
					ax.plot(x0,y0,'o',**plot_kwargs)
					ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)
					# ax.plot(10**(xHAT),10**(yHAT),'r-')

	if 'abs' in filenames[0]:
		ax.text(0.7, 0.9, '$\\rm \\lambda= %.2f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.83, '$\\rm aligned\\, grain: %s$'%dust_type.lower(), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		# ax.set_xticks([10.0],['10'])
		ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),frameon=False)
		ax.set_xlabel('$\\rm A_{V}\\, (mag.)$')
		ax.set_ylabel('$\\rm p_{ext}/A_{V}\\,(\\%/mag.)$')
		# ax.set_xlim([1,20])
		ax.set_ylim([np.min(new_data_/new_Av_)*0.8,np.max(new_data_/new_Av_)*2.0])
		# ax.set_xticks([10.0])
		# ax.set_xticklabels(['10'])
		# ax.set_xticks([1.00])
		# ax.set_xticklabels(['1.0'])

	elif 'emi' in filenames[0]:
		ax.text(0.7, 0.9, '$\\rm \\lambda= %.0f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.83, '$\\rm aligned\\, grain: %s$'%dust_type.lower(), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		# ax.set_xticks([10.0])
		# ax.set_xticklabels(['10'])
		ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),frameon=False)
		ax.set_xlabel('$\\rm A_{V}\\, (mag.)$')
		ax.set_ylabel('$\\rm p_{em}\\,(\\%)$')
		# ax.set_xlim([5,60])
		ax.set_ylim([np.min(data_)*0.8,np.max(data_)*2.0])
	# ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# ax.get_xaxis().set_minor_formatter(ScalarFormatter())
	# ax.get_yaxis().set_major_formatter(ScalarFormatter())
	# ax.get_yaxis().set_minor_formatter(ScalarFormatter())

	plt.show()
	# return new_Av_,new_data_/new_Av_
##set exponential format for figure's axes
# current_values = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:,.1e}'.format(x) for x in current_values])

@auto_refresh
def plot_pI(filenames,wavelength,ax=None,show_break=False,get_info=False,**plot_kwargs):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))

	if isinstance(filenames,str):
		filenames=[filenames]

	for i,filename in enumerate(filenames):
		params=read_headers(filename)
		amax=params["amax"]
		if params['dust_type'] == 'astro':
			dust_type='astrodust'
		elif params['dust_type']=='astro+pah':
			dust_type='astrodust+PAHs'
		elif params['dust_type']=='sil':
			dust_type='silicate'
		elif params['dust_type']=='sil+car':
			dust_type='silicate+carbon'
		else:
			raise IOError('dust_type is not regconized!')

		data=np.loadtxt(filename,skiprows=10)
		w  =data[:,0]
		x,y=np.shape(data)
		data_=[];I_=[]
		
		for j in range(1,y,2):
			data_.append(interp1d(w*1e-4,data[:,j],axis=0)(wavelength*1e-4))#(0.65e-4))
		for j in range(2,y,2):
			I_.append(interp1d(w*1e-4,data[:,j],axis=0)(wavelength*1e-4))#(0.65e-4))

		data_=np.array(data_); I_=np.array(I_)
		new_I_=np.linspace(I_.min(),I_.max(),100)
		f_ip = interp1d(I_,data_,kind='cubic')
		new_data_=f_ip(new_I_)

		sort_indices = np.argsort(I_)
		I_sort   = I_[sort_indices]
		data_sort = data_[sort_indices]

		ax.loglog(I_sort/I_sort.max(),data_sort,ls=ls[keys[i]],**plot_kwargs,label='$\\rm a_{max}=%.1f\\, \\mu m$'%(amax))
		# plt.loglog(new_I_/new_I_.max(),new_data_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))

		x=np.log10(I_sort/I_sort.max())#[Av_>=1.5])
		y=np.log10(data_sort)#[Av_>=1.5])
		f=interp1d(I_sort/I_sort.max(),data_sort)

		if (show_break):
			if (get_info):
				get_func=True
			else:
				get_func=False			
			xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=2,get_func=get_func)
			if (slope<=-0.85):
				x0=10**(xbreak[1])
				y0=f(x0)
				ax.plot(x0,y0,'o',**plot_kwargs)
				ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)
				# ax.plot(10**(xHAT),10**(yHAT),'r-')

	# I_loss=np.linspace(0.2,1,20)
	# p_loss=pow(I_loss,-1)
	# plt.plot(I_loss[p_loss<=10],1.5*p_loss[p_loss<=10],color='gray')
	# fp_loss = interp1d(I_loss,p_loss)
	# plt.text(0.35, fp_loss(0.35)+2.0, '$\\sf I^{-1}$', color='gray', horizontalalignment='center', verticalalignment='center',rotation=-55)

	ax.text(0.7, 0.9, '$\\rm \\lambda= %.0f\\, \\mu m$'%(wavelength), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.text(0.7, 0.83, '$\\rm aligned\\, grain: %s$'%dust_type.lower(), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	#plt.xticks([10.0],['10'])
	ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),frameon=False)
	ax.set_xlabel('$\\rm I/max(I)$')
	ax.set_ylabel('$\\rm p_{em}\\,(\\%)$')
	#plt.xlim([5,60])
	ax.set_ylim([np.min(data_sort)*0.8,np.max(data_sort)*2.0])

	# ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_xaxis().set_minor_formatter(ScalarFormatter())
	# ax.get_yaxis().set_major_formatter(ScalarFormatter())
	# ax.get_yaxis().set_minor_formatter(ScalarFormatter())

##set exponential format for figure's axes
# current_values = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:,.1e}'.format(x) for x in current_values])

@auto_refresh
def plot_lamav(filenames,ax=None,**plot_kwargs):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))

	# if isinstance(amax_range,float) or isinstance(amax_range,int):
	# 	amax_range=[amax_range]

	if isinstance(filenames,str):
		filenames=[filenames]

	# if not np.equal(len(amax_range),len(filenames)):
	# 	raise IOError('length of amax_range and of filenames must be the same!')

	for j,filename in enumerate(filenames):

		# if dust_type.lower()=='astro':
		# 	dust_type='astrodust'
		# 	filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'
		# else:
		# 	if dust_type=='sil':
		# 		filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'
		# 	elif dust_type=='mix':
		# 		filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'

		Av_=get_Av(filename)

		# ##get column's name
		# f=open(filename,'r')
		# lines=f.readlines()
		# f.close()
		# names = lines[7].split()
		params=read_headers(filename)
		if params['dust_type'] == 'astro':
			dust_type='astrodust'
		elif params['dust_type']=='astro+pah':
			dust_type='astrodust+PAHs'
		elif params['dust_type']=='sil':
			dust_type='silicate'
		elif params['dust_type']=='sil+car':
			dust_type='silicate+carbon'
		else:
			raise IOError('dust_type is not regconized!')

		##get the data
		# data=ascii.read(filename,header_start=7, include_names=names)
		data_=np.loadtxt(filename,skiprows=10)

		##wavelength
		#w=data['w'].data
		w=data_[:,0]

		sort_indices = np.argsort(Av_)
		Av_sort   = Av_[sort_indices]
		data_sort = data_[:,1:][:,sort_indices]

		av_range=Av_sort

		lam_max=np.zeros(len(av_range))
		for i,iav in enumerate(av_range):
			if iav>Av_sort.max() or iav<Av_sort.min():
				raise IOError('Value of of input Av=%.2f is outof the boundary!'%iav)
				continue

			##Interpolate the computed data over Av_ and w:
			##Follow: https://stackoverflow.com/questions/39332053/using-scipy-interpolate-interpn-to-interpolate-a-n-dimensional-array
			## ---> get the value of p% at a costumized Av_ 
			interp_x=iav
			interp_y=w
			interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
			interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((len(w), 2))

			interp_arr = interpolate.interpn((Av_sort,w), data_sort.T, interp_points)

			##Finding lambda max
			idl = np.argmax(interp_arr)
			lam_max[i]=w[idl]

		ax.plot(av_range,lam_max,ls=ls[keys[j]],**plot_kwargs,label='$\\rm a_{\\rm max}=%.2f\\,\\mu m$'%params["amax"])

	ax.set_xlabel('$\\rm A_{V}\\, (mag.)$')
	ax.text(0.27, 0.9, '$\\rm aligned\\, grain: %s$'%dust_type.lower(), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# ax.text(0.25, 0.85, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.set_ylabel('$\\rm \\lambda_{max}\\,(\\mu m)$')
	ax.legend(loc='lower right',bbox_to_anchor=(0.93,0.4),frameon=False)

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# if pol=='abs':
	# 	ax.set_xlim([0.03,1e3])
	# elif pol=='emi':
	# 	ax.set_xlim([10,3000])
	# # plt.savefig('Plambda_abs_amax=%.2f.pdf'%(amax))
	plt.show()
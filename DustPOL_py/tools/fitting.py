import numpy as np 
import sys
import lmfit
from astropy import log
from lmfit import Parameters, minimize, fit_report
from scipy.interpolate import interp1d

def import_dustpol():
	if 'DustPOL_py' not in sys.modules:
		import DustPOL_py
	else:
		DustPOL_py = sys.modules['DustPOL_py']
	return DustPOL_py.DustPOL

class DustPOLfit:
	def __init__(self, x_data, y_data, yerr_data, input_file, params_dict, pol='abs'):
		self.model=import_dustpol() ##get DustPOL_py imported
		self.x_data=x_data
		self.y_data=y_data 
		self.yerr_data=yerr_data
		self.input_file=input_file
		self.params_dict=params_dict
		self.params_names = self.params_dict.keys()
		# self.models = self.params_dict['model']
		self.pol=pol

	def compute_dustpol(self,**kwargs):
		log.info('[inside compute_dustpol]: ',kwargs)
		if self.pol.lower() == 'abs':
			w,pext = self.model(self.input_file, **kwargs).cal_pol_abs(verbose=True)
			if w is None or pext is None or len(w)==0 or len(pext)==0: 
				return np.full_like(self.x_data,np.nan)

			# model returns in wavelength in micron and normalized p
			return w,pext/pext.max()

		elif self.pol.lower() == 'em':
			w,pem = self.model(self.input_file, **kwargs).cal_pol_em(verbose=True)
			if w is None or pem is None or len(w)==0 or len(pem)==0: 
				return np.full_like(self.x_data,np.nan)

			# model returns in wavelength in micron and normalized p
			return w,pem/pem.max()
		else:
			log.error('pol type not recognized. Choose either "abs" or "em" ')
			return np.full_like(self.x_data,np.nan)

	def func_model(self,**kwargs):
		x_model,y_model=self.compute_dustpol(**kwargs)
		f_model = interp1d(x_model,y_model, bounds_error=False, fill_value="extrapolate")
		return f_model(self.x_data)

	def residual(self,params):
		y_model = self.func_model(**params.valuesdict())
		resids = y_model - self.y_data
		weighted = np.sqrt(resids**2/self.yerr_data**2)
		return weighted

	def lmfit_fitting(self, **kwargs):
		keys = self.params_dict.keys()
		params = lmfit.Parameters()
		for param_name in self.params_names:
			##access the lower-and upper-limits
			try:
				lower_lim = self.params_dict[param_name]['min']
				upper_lim = self.params_dict[param_name]['max']
				params.add(param_name, min=lower_lim,max=upper_lim)
			except:
				params.add(param_name)
		result = lmfit.minimize(self.residual, params ,**kwargs)
		print(lmfit.fit_report(result))
		return result
##import Built-in functions
import numpy as np
import time,copy,os
import scipy.integrate as integrate
# import atexit
from joblib import Memory
import warnings
import concurrent.futures
from joblib import Parallel, delayed
from astropy import log
from scipy.interpolate import interp1d

##import customized functions
from .decorators import auto_refresh, printProgressBar
from .check_vars import check_variable_combination
from . import rad_func
from . import disrupt
from . import qq
from . import radiation
from . import align 
from . import size_distribution
from . import pol_degree
from . import DustPOL_io
from . import constants
from . import isoCloud_class
from . import isoProtostar_class
from . import extcurves
from . import tools
from .tools import analysis, fitting

##Ignore the warnings
warnings.filterwarnings('ignore')

# Worker function for isoProtostar_los with multiprocessing to avoid pickling issues
def _isoProtostar_los_worker(args):
    input_file, kwargs, r0, progress, get_info = args
    exe = DustPOL(input_file, **kwargs)
    return exe.isoProtostar_los(r0, progress=progress, get_info=get_info)

def _isoCloud_los_worker(args):
    input_file, kwargs, r0, progress, get_info = args
    exe = DustPOL(input_file, **kwargs)
    return exe.isoCloud_los(r0, progress=progress, get_info=get_info)

class DustPOL:
    """This is the main routine of the DustPOL-py
        Inputs:
        -------
            These args are passed from the input file or override from the manual setup
                + U,mean_lam,gamma, ##radiation
                + Tgas,Tdust,ngas,  ##physical conditions
                + ratd,Smax,        ##rotational disruption
                + amin,amax,power_index,dust_type,dust_to_gas_ratio,GSD_law, ##grain-size
                + RATalign,f_min,f_max,alpha,   ##alignment physics
                + B_angle                       ##B-field inclination
        Outputs:
        --------
            Main [checked] sub-routines to call-out
                + extinction       : to compute the dust extinction curve
                + SED_dust         : to compute the dust continuum SED
                + cal_pol_abs      : to compute the degree of absorption polarisation
                + cal_pol_emi      : to compute the degree of emisison polarisation
                + isoCloud_los     : to compute the degree of absorption and emission polarisations
                                            of starless core at a given line of sight (r0 -- x-coorinate)
                + isoCloud_pos     : to compute the degree of absorption and emission polarisations in starless core
                + isoProtostar_los : to compute the degree of absorption and emission polarisations
                                            of protostar at a given line of sight (r0 -- x-coorinate)
                + isoProtostar_pos : to compute the degree of absorption and emission polarisations in protostar

        Examples:
        ---------
        from DustPOL_py import DustPOL
        exe = DustPOL(input_file)
        
        exe.extinction()   <<-- to compute the dust extinction curve
        exe.SED_dust()     <<-- to compute the dust continuum SED
        exe.cal_pol_abs()  <<-- to compute the degree of absorption polarisation 
        exe.cal_pol_emi()  <<-- to compute the degree of emisison polarisation

        exe.isoCloud_los() <<-- Starless core with the line of sight
        exe.isoCloud_pos() <<-- Starless core on position of sky
        exe.isoProtostar_los() <<-- Protostar with the line of sight
        exe.isoProtostar_pos() <<-- Protostar on position of sky
        
        "To override the input parameters" <<-- this is useful for the fitting
        e.g.: to replace the value of U=10
                    exe = DustPOL(input_file,U=10)
              to replace the value of U=10 and ngas=1e5
                    exe = DustPOL(input_file,U=10,ngas=1e5)

        The list of parameter names that can be overrided
        ---------------------------------------
        parameters names        |       Description
        ---------------------------------------
        - output_dir            |       directory to save output files [option]
        - ratd                  |       True/False to turn on/off RAT-D mechanism [option]
        - Lstar                 |       stellar luminosity [erg s-1] (for protostar model)
        - p                     |       density profile index (for isolated cloud/protostar model)
        - rin                   |       inner radius of the isolated cloud/protostar [cm]
        - rout                  |       outer radius of the isolated cloud/protostar [cm]
        - rflat                 |       flat radius of the isolated cloud/protostar [cm]
        - dust_type             |       dust model type: 'astro', 'astro+pah', 'sil', 'sil+car', 'sil+car+pah'
        - dust_to_gas_ratio     |       dust-to-gas mass ratio (default:0.01)
        - alpha                 |       grain axial ratio (default:1.4)
        - rho                   |       grain mass density [g cm-3] (default:3.0 g cm-3)
        - U                     |       radiation intensity (dimensionless): 1 is for aISRF, 10 is for 10xaISRF
        - gamma                 |       anisotropic degree of the radiation field
                                |         0 means isotropy; 1 means completely anisotropy
                                |         - For a diffuse ISM, gamma<=0.1
                                |         - For a cloud/core , gamma=0.3
                                |         - For a star-forming region, gamma=0.7-1.0
        - mean_lam              |       mean wavelength of the radiation field (centimeter)
                                |         - For aISRF: mean_lam = 1.2-1.3 micron
        - Tgas                  |       gas temperature [K]
        - Tdust                 |       dust temperature [K]
        - ngas                  |       gas number density [cm-3]
        - Smax                  |       maximum tensile strength [erg cm-3]
        - amin                  |       minimum grain size [cm]
        - amax                  |       maximum grain size [cm]
        - Bfield                |       magnetic field strength [Gauss]
        - Ncl                   |       number of iron cluster
        - phi_sp                |       volume filling factor of iron cluster
        - f_min                 |       alignment efficiency for grains with a<align 
                                |         - "DG" or None: for DG alignment
                                |         - float value between 0.0 and 1.0
        - f_max                 |       alignment efficiency for grains with a>align 
                                |         - Set to 1.0 for perfect alignment
        - B_angle               |       angle between the magnetic field and the line of sight [degree]
        - car_align             |       True/False to turn on/off the alignment of carbon grain
        - align_func            |       alignment function L20[default] or G18 or H15 
                                |         - If G18 is chosen, give the value of pstiff[default=1.0]  
        - fheat                 |       heating fraction of the dust in the 2-layer dust model [2layer dust model]
        - fscale                |       scaling factor for the intensities between 2-layer dust model [2layer dust model]
        - power_index           |       power index for grain size distribution [No-unit]
        - PAHs_data_type        |       DL07[default] or HD23
        - B1,B2                 |       parameters for the HD23 grain-size distribution law for PAHs
        - BAd, a0Ad,sigmaAd,A0,A1,A2,A3,A4,A5       |     parameters for the HD23 grain-size distribution law for Astrodust
    """
    # memory = Memory(location='./__pycache__', verbose=0) # Class-level cache to store already computed function
    # atexit.register(memory.clear, warn=False)  # Clear the cache at the end of the program
    # # memory.clear(warn=False) # Clear the cache at the start of the class
    
    @auto_refresh
    def __init__(self,input_params_file,**kwargs):
        ##Initialize global input parameters from the input file, 
        ##   with optional overrides from kwargs.
        
        PAHs_data_type_default = 'DL07'#'HD23'
        self.PAHs_data_type = kwargs.get("PAHs_data_type",PAHs_data_type_default)

        self.input_params_file = input_params_file    
        params = DustPOL_io.input_params(
                self.input_params_file,
                self.PAHs_data_type.lower(),
                overwrites=kwargs
                )

        # Assign atributes with kwargs
        self.output_dir       = kwargs.get("output_dir", params.output_dir)    
        self.ratd             = kwargs.get("ratd", params.ratd)               
        self.dust_type        = kwargs.get("dust_type", params.dust_type)      
        self.dust_to_gas_ratio= kwargs.get("dust_to_gas_ratio", params.dust_to_gas_ratio) #No-unit
        self.RATalign         = kwargs.get("RATalign", params.RATalign)        
        self.alpha            = kwargs.get("alpha", params.alpha)              #No-unit
        self.rho              = kwargs.get("rho", params.rho)                  #g cm-3
         
        self.U           = kwargs.get("U", params.U)                     # No-unit
        self.gamma       = kwargs.get("gamma", params.gamma)             # No-unit
        self.mean_lam    = kwargs.get("mean_lam", params.mean_lam)       # cm
        self.Tgas        = kwargs.get("Tgas", params.Tgas)               # K
        self.Tdust       = kwargs.get("Tdust", params.Tdust)             # K
        self.ngas        = kwargs.get("ngas", params.ngas)               # cm-3
        self.Smax        = kwargs.get("Smax", params.Smax)               # erg cm-3
        self.amin        = kwargs.get("amin", params.amin)               # cm
        self.amax        = kwargs.get("amax", params.amax)               # cm

        # Grain size distribution parameters
        self.GSD_law     = kwargs.get("GSD_law", params.GSD_law)          #[option]
        self.power_index = kwargs.get("power_index", params.power_index) #No-unit
        
        ### Astrodust size distribution parameter for GSD_law='HD23'        
        self.BAd         = kwargs.get("BAd",params.BAd) 
        self.a0Ad        = kwargs.get("a0Ad",params.a0Ad) 
        self.sigmaAd     = kwargs.get("sigmaAd",params.sigmaAd)
        self.A0 = kwargs.get("A0",params.A0) 
        self.A1 = kwargs.get("A1",params.A1)
        self.A2 = kwargs.get("A2",params.A2)
        self.A3 = kwargs.get("A3",params.A3)
        self.A4 = kwargs.get("A4",params.A4)
        self.A5 = kwargs.get("A5",params.A5)
        ### PAHs size distribution parameter for GSD_law='HD23'
        self.B1 = kwargs.get("B1",params.B1) 
        self.B2 = kwargs.get("B2",params.B2)
        
        ### Parameters for WD01 size distribution (sil==silicate, car==carbon)
        self.alpha_wd01_sil = kwargs.get("alpha_wd01_sil",params.alpha_wd01_sil) 
        self.beta_wd01_sil  = kwargs.get("beta_wd01_sil",params.beta_wd01_sil)
        self.at_wd01_sil    = kwargs.get("at_wd01_sil",params.at_wd01_sil)
        self.ac_wd01_sil    = kwargs.get("ac_wd01_sil",params.ac_wd01_sil)
        self.Cs_wd01_sil    = kwargs.get("Cs_wd01_sil",params.Cs_wd01_sil)
        self.a01_wd01_sil   = params.a01_wd01_sil
        self.a02_wd01_sil   = params.a02_wd01_sil
        self.sigma_wd01_sil = params.sigma_wd01_sil
        self.B1_wd01_sil    = params.B1_wd01_sil
        self.B2_wd01_sil    = params.B2_wd01_sil

        self.alpha_wd01_car = kwargs.get("alpha_wd01_car",params.alpha_wd01_car) 
        self.beta_wd01_car  = kwargs.get("beta_wd01_car",params.beta_wd01_car)
        self.at_wd01_car    = kwargs.get("at_wd01_car",params.at_wd01_car)
        self.ac_wd01_car    = kwargs.get("ac_wd01_car",params.ac_wd01_car)
        self.Cs_wd01_car    = kwargs.get("Cs_wd01_car",params.Cs_wd01_car)
        self.a01_wd01_car   = kwargs.get("a01_wd01_car",params.a01_wd01_car)
        self.a02_wd01_car   = kwargs.get("a02_wd01_car",params.a02_wd01_car)
        self.sigma_wd01_car = kwargs.get("sigma_wd01_car",params.sigma_wd01_car)
        self.B1_wd01_car    = kwargs.get("B1_wd01_car",params.B1_wd01_car)
        self.B2_wd01_car    = kwargs.get("B2_wd01_car",params.B2_wd01_car)

        # Radiation field parameters if Tdust is given
        if self.U is None:
            if float(self.Tdust) <= 0.0:
                raise ValueError('Using Tdust--> U? SORRY - U is not defined, please set Tdust in the input file/by the keyword [\033[1;5;7;91m failed \033[0m]')
            else:
                self.U = pow(self.Tdust/16.4, 6.0)
        else:
            if float(self.Tdust) > 0.0:
                log.warning('U is given --> Tdust is estimated from U (to avoid warning: set Tdust=0 in the input file)')
            self.Tdust = pow(self.U, 1.0/6.0) * 16.4  # Tdust in K, U is dimensionless
                
        # MRAT parameters
        self.Bfield = kwargs.get("Bfield",params.Bfield) # Gauss
        self.Ncl    = kwargs.get("Ncl",params.Ncl)       # number of iron cluster
        self.phi_sp = kwargs.get("phi_sp", params.phi_sp)# volume filling factor of iron cluster 
        self.fp     = kwargs.get("fp", params.fp)        # fraction of paramagnetic atoms
        
        # Alignment efficiencies
        self.f_min  = kwargs.get("f_min",  params.f_min)    #%
        self.f_max  = kwargs.get("f_max",  params.f_max)    #% 
        self.B_angle= kwargs.get("B_angle",params.B_angle)  #radiant
        # -- need a specific value of f_min for DG alignment 
        if str(self.f_min).lower() == 'dg':
            self.f_min = None

        # Parameters for 2layer-dust model
        self.model_layer = params.model_layer  # 2layer dust model
        self.fheat       = kwargs.get("fheat", params.fheat)  # heating fraction of the dust
        self.fscale     = kwargs.get("fscale", params.fscale)  # scaling factor for the dust temperature
        self.fscale_car  = params.fscale_car  # scaling factor for the carbon abundance
        
        # HPC setup
        self.parallel = params.parallel       #[option] parallelization calculation
        if (self.parallel):
            self.max_workers = params.max_workers    #[if parallel]: numbers of CPU cores
        self.verbose = kwargs.get("verbose", False)  #[option] to print out the log info
        self.Urange_tempdist=[]

        # Parameters for isolated cloud
        self.Lstar  = kwargs.get("Lstar", params.Lstar)       #cgs
        self.Tstar  = params.Tstar                            #K
        self.p      = kwargs.get("p", params.p)               #density profile index
        self.rflat  = kwargs.get("rflat", params.rflat)       #cm
        self.rout   = kwargs.get("rout", params.rout)         #cm
        self.nsample= kwargs.get("nsample", params.nsample)   #number of sampling points
        self.sampling_type = kwargs.get("sampling_type", 'lin_space') #linear/log sampling

        if self.sampling_type not in ['lin_space','log_space']:
            raise ValueError(f"Invalid sampling_type: {self.sampling_type} -- choose 'lin_space' or 'log_space'")
        
        # # ------- get constants -------
        self.pc    = constants.pc    #cm
        self.h     = constants.H
        self.c     = constants.C
        self.eV    = constants.eV
        self.u_ISRF= params.u_ISRF   #erg cm-3

        # ------- get path to directory -------
        self.path=params.path

        # ------- Declare the alignment for carbon ------------
        self.carbon_alignment = kwargs.get("car_align", params.car_align)

        # ------- Declare the alignment function ------------
        self.align_func = kwargs.get("align_func", params.align_func)
        self.pstiff     = kwargs.get("pstiff", params.pstiff)

        # ------- Declare the parameter for DG alignment ------------
        self.alpha_DG   = kwargs.get("alpha_DG", 1.0)    # the parameter for IR-damping for very small grains (alpha_DG=1.0 is for no IR-damping)
        self.temp_ratio  = kwargs.get("temp_ratio", 2.0) # the ratio of Tgas/Tdust for DG alignment

        # ------- Save the kwargs to an instance attribute ------------
        self.kwargs = kwargs

        # ------- get cross-section files -------
        self.Data_Qfiles_1 = params.Data_Qfiles_1
        self.Data_Qfiles_2 = params.Data_Qfiles_2
        self.Data_Qfiles_3 = params.Data_Qfiles_3
        self.Data_Qfiles_4 = params.Data_Qfiles_4

        # ------- get dusttypes -------
        self.dusttype_1 = params.dusttype_1
        self.dusttype_2 = params.dusttype_2
        self.dusttype_3 = params.dusttype_3
        self.dusttype_4 = params.dusttype_4
        
        # ------- get the mass fraction of dusttype -------
        for dusttype in [self.dusttype_1, self.dusttype_2, self.dusttype_3, self.dusttype_4]:
            if dusttype is None:
                continue
            f_mass = getattr(params, f'f_mass_{dusttype}', None)
            setattr(self, f'f_mass_{dusttype}', f_mass)
        
        # ------- Initialization wavelength, grain size from the file  ------- 
        self.get_grainsize_wavelength() ## -->> wavelength (self.w) and grain size (self.a)
        
        # ------- Update grain size (if RAT-D occurs)  ------- 
        self.update_grain_size(self.a,verbose=self.verbose)## -->> update self.a

        # ------- Initialization grain-size distribution -------
        self.grain_size_distribution()  # update self.a -->> update self.dnda

        # ------- Initialization Qext,Qabs,Qpol and Qpol_abs -------
        self.get_coefficients_data() ## self.a -->> self.Qdata

        # Take snapshots once so we can restore quickly between LOS steps
        if not hasattr(self, "_grain_snapshot"):
            self._snapshot_initial_grain_state()
        if not hasattr(self, "_Qcoeffs_init"):
            self._snapshot_initial_Qcoeffs()

        # # ------- copy some values -------
        self.U_init        = self.U
        self.mean_lam_init = self.mean_lam

    # --------- Snapshots for fast restore between points ----------
    def _snapshot_initial_grain_state(self):
        """Save the initial grain grids and dn/da arrays."""
        snap = {}
        snap["a"] = None if self.a is None else self.a.copy()
        snap["a_add"] = None if getattr(self, "a_add", None) is None else self.a_add.copy()
        for i in range(1, 5):
            dusttype = getattr(self, f'dusttype_{i}', None)
            if not dusttype:
                continue
            snap[f"dn_da_{dusttype}"] = copy.deepcopy(getattr(self, f"dn_da_{dusttype}", None))
            snap[f"dn_da_add_{dusttype}"] = copy.deepcopy(getattr(self, f"dn_da_add_{dusttype}", None))
        self._grain_snapshot = snap

    def _restore_initial_grain_state(self):
        """Restore a, a_add, and dn/da to initial state."""
        if not hasattr(self, "_grain_snapshot"):
            return
        snap = self._grain_snapshot
        if snap.get("a") is not None:
            self.a = snap["a"].copy()
            self.na = len(self.a)
        if snap.get("a_add") is not None:
            self.a_add = snap["a_add"].copy()
            self.na_add = len(self.a_add)
        for i in range(1, 5):
            dusttype = getattr(self, f'dusttype_{i}', None)
            if not dusttype:
                continue
            setattr(self, f"dn_da_{dusttype}", copy.deepcopy(snap.get(f"dn_da_{dusttype}", None)))
            setattr(self, f"dn_da_add_{dusttype}", copy.deepcopy(snap.get(f"dn_da_add_{dusttype}", None)))

    def _snapshot_initial_Qcoeffs(self):
        """Save the initial Q* arrays for each dust type."""
        names = ("Qext","Qabs","Qpol","Qpol_abs","Qabs_y","Qabs_x",
                    "Qext_add","Qabs_add","Qpol_add","Qpol_abs_add")
        pack_all = {}
        for i in range(1, 5):
            dusttype = getattr(self, f'dusttype_{i}', None)
            if not dusttype:
                continue
            pack = {}
            for nm in names:
                pack[nm] = copy.deepcopy(getattr(self, f"{nm}_{dusttype}", None))
            pack_all[dusttype] = pack
        self._Qcoeffs_init = pack_all

    def _restore_initial_Qcoeffs(self):
        """Restore Q* arrays (must match the initial a/w grid)."""
        if not hasattr(self, "_Qcoeffs_init"):
            return
        for dusttype, pack in self._Qcoeffs_init.items():
            for nm, arr in pack.items():
                setattr(self, f"{nm}_{dusttype}", copy.deepcopy(arr))

    @auto_refresh
    def get_grainsize_wavelength(self):
        # Prepare lists
        a_arrays = []
        w_arrays = []

        # Loop over Qfile attributes
        for i in range(1, 5):  # handles Data_Qfiles_1 to _4
            data = getattr(self, f"Data_Qfiles_{i}", None)

            if data is not None:
                w = data[1, :, 0] * 1e-4  # wavelength in cm
                a = data[0, 0, :] * 1e-4  # grain size in cm
            else:
                w = None
                a = None

            w_arrays.append(w)
            a_arrays.append(a)

        # Filter out None values before taking lengths
        lengths_a = [len(a) if a is not None else -1 for a in a_arrays]
        lengths_w = [len(w) if w is not None else -1 for w in w_arrays]

        # Get longest arrays
        max_index_a = np.argmax(lengths_a)
        max_index_w = np.argmax(lengths_w)

        self.a = a_arrays[max_index_a]
        self.w = w_arrays[max_index_w]
        return

    @auto_refresh
    def get_coefficients_data(self):

        for i in range(1, 5):
            data     = getattr(self, f'Data_Qfiles_{i}', None)
            dusttype = getattr(self, f'dusttype_{i}', None)

            if data is not None and dusttype is not None:
                [Qext, Qabs, Qpol, Qpol_abs], [Qabs_y, Qabs_x] = qq.Qext_grain(data, self.w, self.a, dusttype,self.alpha)
                if self.fscale is not None:
                    [Qext_add,Qabs_add,Qpol_add,Qpol_abs_add],_ = qq.Qext_grain(data, self.w, self.a_add, dusttype,self.alpha)
                else:
                    Qext_add = Qabs_add = Qpol_add = Qpol_abs_add = None
                if dusttype=='car':
                    if not self.carbon_alignment:       # if carbon grain is not aligned
                        Qpol     = np.zeros_like(Qext)  #    --> Qpol_car = 0
                        Qpol_abs = np.zeros_like(Qext)  #    --> Qpol_abs_car = 0
                        Qpol_add     = np.zeros_like(Qext_add) #    --> Qpol_car_add = 0
                        Qpol_abs_add = np.zeros_like(Qext_add) #    --> Qpol_abs_car_add = 0
                        
                setattr(self, f'Qext_{dusttype}', Qext)
                setattr(self, f'Qabs_{dusttype}', Qabs)
                setattr(self, f'Qpol_{dusttype}', Qpol)
                setattr(self, f'Qpol_abs_{dusttype}', Qpol_abs)
                setattr(self, f'Qabs_y_{dusttype}', Qabs_y)                
                setattr(self, f'Qabs_x_{dusttype}', Qabs_x)

                if self.fscale is not None:
                    setattr(self, f'Qext_add_{dusttype}', Qext_add)
                    setattr(self, f'Qabs_add_{dusttype}', Qabs_add)
                    setattr(self, f'Qpol_add_{dusttype}', Qpol_add)
                    setattr(self, f'Qpol_abs_add_{dusttype}', Qpol_abs_add)
                else:
                    setattr(self, f'Qext_add_{dusttype}', None)
                    setattr(self, f'Qabs_add_{dusttype}', None)
                    setattr(self, f'Qpol_add_{dusttype}', None)
                    setattr(self, f'Qpol_abs_add_{dusttype}', None)
                
        return
                
    @auto_refresh
    def update_radiation_aISRF(self,Av):
        self.get_mean_wavelength_aISRF(Av) ## -->> self.urad_ISRF_red and self.mean_lam
        self.U = self.urad_ISRF_red/self.u_ISRF
        log.info(f'   *** [\033[0;1;1;80m U(Av=0.0 mag) = {self.U_init:.2f} --> U(Av={Av:.1f} mag) = {self.U:.2f} \033[0m]')
        return
     
    @auto_refresh
    def get_mean_wavelength_aISRF(self,Av):
        nw = len(self.w)
        wave1,wave2=self.h * self.c * 1.e4/(13.6 * self.eV), 20. #0.91um to 20um
        wavelength = 1.e-4 * np.exp(np.log(wave2/wave1) * np.arange(nw)/(nw-1) + np.log(wave1))

        _,tau_rand = self.extinction_curve(verbose=False, f_align=False)
        fextinction=interp1d(self.w, tau_rand ,bounds_error=False, fill_value="extrapolate")
        tau_rand = fextinction(wavelength)
        # Rv = fextinction(0.55e-4)/(fextinction(0.44e-4)-fextinction (0.55e-4)) #RV= A_V/(A_B-A_V)
        A_lambda_Av = tau_rand/fextinction(0.55e-4)

        tau_wavelength = Av * A_lambda_Av/1.086
        uwave_ISRF,_,mean_lam_ISRF = rad_func.radiation_intensity(wavelength, x=self.U)
        uwave_ISRF_red = uwave_ISRF * np.exp(-tau_wavelength)

        self.urad_ISRF_red = integrate.simpson(uwave_ISRF_red,x=wavelength)
        self.mean_lam = integrate.simpson(uwave_ISRF_red * wavelength, x=wavelength)/integrate.simpson(uwave_ISRF_red,x=wavelength)
        log.info(f'   *** [\033[0;1;1;80m mean_lam_ISRF(Av=0.0 mag)={mean_lam_ISRF*1e4:.3f} um --> mean_lam(Av={Av:.1f} mag)={self.mean_lam*1e4:.3f} um \033[0m]')
        return 

    @auto_refresh
    def update_grain_size(self,a,verbose=True):
        self.verbose=verbose
        a_init = a.copy()  # Store the original grain size array
        
        if self.amax>max(a):
            raise ValueError('SORRY - your amax=%.5f, but it should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(self.amax*1e4, max(a)*1e4))

        self.lmin = np.searchsorted(a, self.amin)
        self.lmax = np.searchsorted(a, min(self.amax, disrupt.radiative_disruption(self).a_disrupt(a))) if self.ratd else np.searchsorted(a, self.amax + 0.1 * self.amax)

        self.a  = a[self.lmin:self.lmax]
        self.na = len(self.a)
        
        # [second dust layer] Update grain size for the additional grain size
        if self.fscale is not None:
            Tdust1  = pow(self.U,1/6) * 16.4    # Dust temperature for the first layer
            Tdust2  = Tdust1/self.fheat         # Dust temperature for the second layer
            self.U_add   = (Tdust2/16.4)**6     # Radiation intensity for the second layer
            self.U  = self.U_add                # Update global U value for second layer

            lmax_add    = np.searchsorted(a_init, min(self.amax, disrupt.radiative_disruption(self).a_disrupt(a_init))) if self.ratd else np.searchsorted(a_init, self.amax + 0.1 * self.amax)
            self.a_add  = a_init[self.lmin:lmax_add]
            self.na_add = len(self.a_add)
            
            self.Tdust_add = Tdust2 * np.ones_like(self.a_add) # Dust temperature for the second layer in an array
            self.U = self.U_init                               # Reset U to the initial value

        ## [Only one dust layer] physical parameters for the second layer dust model is None
        else:
            self.a_add     = None
            self.na_add    = None
            self.Tdust_add = None
            self.U_add     = None
        return

    @auto_refresh
    def dP_dT(self,dusttype):
        ##This function reads the dust temperature distribution, pre-calculated
        ##by DustEM code, and returns the grain size, temperature and dP/dlnT
        ## Note: dusttype should be 'pah', 'sil', or 'car' because the dP/dlnT for Astrodust has not been calculated yet
        if dusttype=='pah':
            U_retrieve = radiation.radiation_retrieve(self).retrieve(PAHs=True)
            if self.PAHs_data_type.lower()=='dl07':
                path_to_data = self.path+"data/PAHs/dp_dlnT/"
                qT = rad_func.T_dust_pah(path_to_data,U_retrieve )
                a_init = qT[0]
                T_init = qT[3]     ##2darray: [na_pah,nT_pah] 
                dP_dlnT_init=qT[5] ##2darray: [na_pah,nT_pah]
                return a_init, T_init, dP_dlnT_init##original from the DustEM
            else:
                return None

        elif dusttype=='sil':
            U_retrieve = radiation.radiation_retrieve(self).retrieve(PAHs=False)
            path_to_data = self.path+"data/sil_car/dp_dlnT/"
            qT = rad_func.T_dust(path_to_data, U_retrieve)
            a_init = qT[0]
            T_init = qT[4]
            dP_dlnT_init = qT[6]
            return a_init,T_init,dP_dlnT_init
        
        elif dusttype=='car':
            U_retrieve = radiation.radiation_retrieve(self).retrieve(PAHs=False)
            path_to_data = self.path+"data/sil_car/dp_dlnT/"
            qT = rad_func.T_dust(path_to_data, U_retrieve)
            a_init=qT[0]
            T_init = qT[3]
            dP_dlnT_init = qT[5]
            return a_init,T_init,dP_dlnT_init

    @auto_refresh
    def grain_size_distribution(self):
        """
        Compute the grain-size distribution for each dust type.
        Output: 1D array as a function of grain size.
        Note: the grain size distribution can be a sum over multiple laws [e.g. MRN + WD01 + ...] 
        """
        dusttype=self.dust_type.lower()
        gsd_law =self.GSD_law.lower()
        
        # GSD_params_MRN           = [self.a.min(), amax, self.rho, self.dust_to_gas_ratio, self.power_index]
        GSD_params_HD23     = [self.BAd, self.a0Ad, self.sigmaAd,
                               self.A0, self.A1, self.A2, self.A3, self.A4, self.A5]

        GSD_params_WD01_sil = [self.alpha_wd01_sil,self.beta_wd01_sil,self.at_wd01_sil,self.ac_wd01_sil,self.Cs_wd01_sil,\
                               self.a01_wd01_sil,self.a02_wd01_sil,self.sigma_wd01_sil,self.B1_wd01_sil,self.B2_wd01_sil]

        GSD_params_WD01_car = [self.alpha_wd01_car,self.beta_wd01_car,self.at_wd01_car,self.ac_wd01_car,self.Cs_wd01_car,\
                               self.a01_wd01_car,self.a02_wd01_car,self.sigma_wd01_car,self.B1_wd01_car,self.B2_wd01_car]

        ed_extension = '_ed' in gsd_law

        dn_da_hd23    = size_distribution.dnda_hd23(self.a, GSD_params=GSD_params_HD23)
        dn_da_pah     = size_distribution.dnda_pah(self.a, self.B1, self.B2)
        dn_da_add_hd23= size_distribution.dnda_hd23(self.a_add, GSD_params=GSD_params_HD23) if self.fscale is not None else None
        dn_da_add_pah = size_distribution.dnda_pah(self.a_add, self.B1, self.B2) if self.fscale is not None else None

        # Map gsd_law names to precomputed distributions
        precomputed_dn_da = {
            'hd23'    : dn_da_hd23,
        }                               #first dust layer
        precomputed_dn_da_add = {
            'hd23'    : dn_da_add_hd23,
        }                               #second dust layer    
        
        for i in range(1,5):
            dusttype = getattr(self, f'dusttype_{i}', None)
            
            if dusttype is None:
                continue

            if dusttype == 'pah':
                setattr(self, f'dn_da_{dusttype}', dn_da_pah)
                if self.fscale is not None:
                    setattr(self, f'dn_da_add_{dusttype}', dn_da_add_pah)
                continue

            dn_da_list = []
            dn_da_add_list = []
            for law in gsd_law.split("+"):
                if law == 'mrn':
                    dn_da_ = size_distribution.dnda(dusttype, self.a, self.power_index, self.dust_to_gas_ratio)
                    if self.fscale is not None:
                        dn_da_add_=size_distribution.dnda(dusttype, self.a_add, self.power_index, self.dust_to_gas_ratio)
                
                elif law=='wd01' or law=='wd01_ed':
                    if dusttype=='sil':
                        # dn_da_ = precomputed_dn_da[law]
                        dn_da_ =  size_distribution.dnda_WD01(self.a, dusttype, GSD_params=GSD_params_WD01_sil,ed=ed_extension)
                        if self.fscale is not None:
                            dn_da_add_ = size_distribution.dnda_WD01(self.a_add, dusttype, GSD_params=GSD_params_WD01_sil,ed=ed_extension)
                    elif dusttype=='car':
                        dn_da_ =  size_distribution.dnda_WD01(self.a, dusttype, GSD_params=GSD_params_WD01_car,ed=ed_extension)
                        if self.fscale is not None:
                            dn_da_add_ = size_distribution.dnda_WD01(self.a_add, dusttype, GSD_params=GSD_params_WD01_car,ed=ed_extension) 
                elif law in precomputed_dn_da:
                    dn_da_ = precomputed_dn_da[law]
                    if self.fscale is not None:
                        dn_da_add_ = precomputed_dn_da_add[law]
                else:
                    print(f"Warning: Unknown gsd_law '{law}' for dusttype_{i}")
                    continue          

                if dn_da_ is not None and not np.isnan(dn_da_).any():
                    dn_da_list.append(dn_da_)
                    if self.fscale is not None and dn_da_add_ is not None:
                        dn_da_add_list.append(dn_da_add_)
                else:
                    print(f"Warning: Invalid or NaN dn_da for gsd_law '{law}'")
            
            # Sum all valid dn_da arrays
            if dn_da_list:
                total_dn_da = np.sum(dn_da_list, axis=0)
                setattr(self, f'dn_da_{dusttype}', total_dn_da)

            if self.fscale is not None:
                if dn_da_add_list:
                    total_dn_da_add = np.sum(dn_da_add_list, axis=0)
                    setattr(self, f'dn_da_add_{dusttype}', total_dn_da_add)
                else:
                    setattr(self, f'dn_da_add_{dusttype}', None)                
        return

    @auto_refresh
    # @memory.cache 
    def get_Planck_function(self,Tdust,a_cal,a_init=None,dP_dlnT=None):
        ##This function calculates the Planck-function
        ##The output is a 2d-array: a function of wavelength and grain-size

        if dP_dlnT is None:
            B_  = rad_func.planck_equi(self.w,len(a_cal),Tdust) ##Tdust must be an array with 'na' element
        else:
            B_ = rad_func.planck_integrated_Tdust(self.w,a_cal,self.w,a_init,Tdust,dP_dlnT) ##2darray -- function of U and na
        return B_
    
    @auto_refresh
    def extinction_curve(self,verbose=False,f_align=True,save_output=False,filename_output=None):
        """
        This function return the extinction curve, normalized by Ngas
        Inputs:
        -------
            - verbose: if True, print the information about the dust size distribution
            - f_align: if True, calculate the extinction for the aligned dust grains
        Outputs:
        -------
            - w: wavelength in micron
            - A_per_Ngas: extinction per gas number density (A_lambda/Ngas)
        """
        if (verbose):
            self.sdist_info_to_print()
        dtau = np.zeros_like(self.a)
        for dusttype in self.dust_type.split("+"):
            
            Qext = getattr(self, f'Qext_{dusttype}', None) 
            dn_da= getattr(self, f'dn_da_{dusttype}', None)
            if (f_align):
                # self.U = self.U_init             ## -->> reset self.U
                # self.mean_lam=self.mean_lam_init ## --> reset self.mean_lambda
                # self.update_radiation_aISRF(Av)
                # self.verbose=verbose
                if not verbose:
                    self.verbose = False
                else:
                    self.verbose = True
                ali_cl = align.alignment_class(self)
                fa     = ali_cl.f_ali()
                Qpol = getattr(self, f'Qpol_{dusttype}', None)
                dtau = dtau + (Qext + fa * Qpol*(2./3 - np.sin(self.B_angle)*np.sin(self.B_angle))) * np.pi * self.a**2 * dn_da
            else:
                self.U = self.U_init             ## -->> reset self.U
                self.mean_lam=self.mean_lam_init ## --> reset self.mean_lambda
                dtau = dtau + Qext * np.pi * self.a**2 * dn_da

        if len(self.a)%2 == 0:
            tau_per_Ngas = integrate.trapezoid(dtau,self.a)
        else:
            tau_per_Ngas = integrate.simpson(dtau,self.a)

        Alamb_per_NH = 1.086*tau_per_Ngas
        # fextinction=interp1d(self.w,A_lambda_NH,bounds_error=False, fill_value="extrapolate")
        # Rv = fextinction(0.55e-4)/(fextinction(0.44e-4)-fextinction (0.55e-4)) #RV= A_V/(A_B-A_V)

        if (save_output):
            #Save the output
            data_save={}
            data_save['wavelength'] = self.w*1e4

            data_save['Alambda/Ngas (mag./cm-2)'] = Alamb_per_NH
            if filename_output is None:
                self.outputFile = 'extcurve.dat'
            else:
                self.outputFile = filename_output+'_extcurve.dat'
            DustPOL_io.output(self,data_save)
            
        return self.w*1e4, Alamb_per_NH

    @auto_refresh
    def SED_dust(self,Av=0.0,verbose=True):
        """This routine returns the SED in the unit of Intensity/NH
        Inputs:
        -------        
            - Av: extinction in mag
            - verbose: if True, print the information
        Outputs:
        -------
            - w: wavelength in micron
            - Iem_Ngas_dict: dictionary with the intensity for each dust type and total, normalized by Ngas
                e.g. {'silicate': I_sil/Ngas, 'carbon': I_car/Ngas, 'pah': I_pah/Ngas, 'astro': I_astro/Ngas, 'total': I_total/Ngas}
        """
        
        # get total intensity for each dust type
        Iem_dict = self.cal_pol_emi(Av=Av, verbose=verbose, intensity_output=True)[1]

        # declear the dictionary to store the intensity per Ngas
        Iem_Ngas_dict = {}
        for dusttype in self.dust_type.split('+'):
            Iem_Ngas_dict[dusttype] = Iem_dict[dusttype]/self.ngas
        
        # sum the intensity for all dust types
        dust_types_set = set(dt.strip() for dt in self.dust_type.lower().split('+'))
        if dust_types_set == {'sil', 'car'}: # mass fraction correction for silicate and carbon dust
            Iem_Ngas_dict['total'] = (
                0.625 * Iem_Ngas_dict.get('sil', 0) +
                0.375 * Iem_Ngas_dict.get('car', 0)
            )
        else:    
            Iem_Ngas_dict['total'] = np.sum(list(Iem_Ngas_dict.values()), axis=0)

        if (verbose):
            self.sdist_info_to_print()

        return self.w*1e4, Iem_Ngas_dict

    @auto_refresh
    def tau_and_I_xy(self, Ngas):
        """
        This function calculates the intensity of the polarized light along the major axis of the dust grains
        Returns:
        -------
            - Ipar: intensity of the emission polarized along the y-axis of the dust grains
        """
        # Calculate the intensity of emission polarized light along the major axis of the dust grains
        BB={}
        for dusttype in self.dust_type.split("+"):
            Tdust = 16.4* self.U**(1./6) * np.ones(self.na)#* (self.a/1.e-5)**(-1./15)#
            BB[dusttype] = self.get_Planck_function(Tdust,self.a)[0,:]

        tau_x,tau_y = pol_degree.pol_degree(self).tau_xy(self,Ngas)
        Ix = (1-np.exp(-tau_x)) * np.sum(list(BB.values()), axis=0)
        Iy = (1-np.exp(-tau_y)) * np.sum(list(BB.values()), axis=0)
        return [tau_x,tau_y], [Ix,Iy]

    @auto_refresh
    def cal_pol_abs(self,Av=0.0,NH=0.0,verbose=True,save_output=False,filename_output=None,radiative_process=False):
        '''
        This function calculates the degree of starlight polarization (0D)
        The inputs are taken from the input datafile with the additional input parameters
        Inputs:
        -------
              1- Av: extinction in mag
              2- verbose: if True, print the information
              3- save_output: if True, write the output to a file in the output folder
              4- filename_output: if None, the output file will be named 'p_abs.dat' otherwise it will be named as filename_output+'_abs.dat'
              5- radiative_process: if True, update the radiation field based on the extinction Av
                 [This option is used when Av!=0.0 and only works for the uniform cloud exposed to the ISRF]
                 [If Av!=0 and radiative_process=False, the optical depth effect is accounted -- NOTE: radiation field will not be updated]
                 [If Av!=0 and radiative_process=True, the optical depth effect is accounted  -- NOTE: radiation field will be updated based on the extinction Av]
                 [Simiarly with the value of NH -- NOTE: Av and NH cannot be both non-zero!]
        Outputs:
        -------
              1- wavelength in micron
              2- pext/Ngas: for a sum over dust compositions 
                            [a single dust composition is allowed if only one dust type is used]
        '''
        self.verbose=verbose #transfer verbose to parent var to pass to pol_degree and align classes
        if Av < 0.0:
            raise ValueError('Av should be >= 0.0 mag [\033[1;5;7;91m failed \033[0m]')
        if NH < 0.0:
            raise ValueError('NH should be >= 0.0 cm-2 [\033[1;5;7;91m failed \033[0m]')
                
        elif Av!=0.0 or NH!=0.0:
            if (NH !=0.0) and (Av !=0.0):
                raise ValueError('Av and NH cannot be both non-zero! [\033[1;5;7;91m failed \033[0m]')

            # convert Av to Ngas from extinction curve
            w_,ext_ = self.extinction_curve(verbose=False,f_align=True)
            fextinction=interp1d(w_,ext_,bounds_error=False, fill_value="extrapolate")
            Rv = fextinction(0.55)/(fextinction(0.44)-fextinction (0.55)) #RV= A_V/(A_B-A_V)
                            
            if (Av != 0.0):
                Ngas = Av*5.8e21/Rv # convert Av to Ngas (in cm-2)
                Av   = Av
            if (NH != 0.0):
                Ngas = NH
                Av   = Ngas*Rv/5.8e21 # convert NH to Av (in mag)
            
            if (radiative_process):
                self.U = self.U_init             ## -->> reset self.U
                self.mean_lam=self.mean_lam_init ## -->> reset self.mean_lambda
                self.update_radiation_aISRF(Av)  ## -->> update self.U, self.mean_lambda            
        else:
            Ngas = None
            
        # Update verbose to self-wise because its value might be changed in the extinction_curve function
        self.verbose=verbose       
       
        if (self.verbose):
            self.sdist_info_to_print()

        w,_,dP_abs= pol_degree.pol_degree(self)._pol_degree_absorption_thin_(self)
        # dusttypes = self.dust_type.split('+')
        # Pabs_NH_dict = {dusttype: dP_abs_dict[dusttype] / self.ngas for dusttype in dusttypes}            
        
        # Optically thin assumption tau_y - tau_x <<1
        if Ngas is None:
            Pabs_NH = dP_abs / self.ngas
        # General case (either optically thin or optically thick)
        else:
            Pabs_NH = np.tanh(dP_abs/self.ngas/100 * Ngas) * 100
            Pabs_NH = Pabs_NH / Ngas
            # if not (isinstance(Ngas,float) or isinstance(Ngas,int)):
            #     raise ValueError('Value of Ngas is invalid!')
            # else:
            #     w, Pabs = pol_degree.pol_degree(self)._pol_degree_absorption_general_(self, Ngas)
            #     Pabs_NH = Pabs/Ngas
         
        if (save_output):
            #Save the output
            data_save={}
            data_save['wavelength'] = w*1e4

            # for dusttype in dusttypes:
            #     data_save[f'Pext/Ngas_{dusttype}'] = Pabs_NH_dict[dusttype]

            data_save['p(total)/Ngas (%/cm-2)'] = Pabs_NH

            _,ext_curve = self.extinction_curve(verbose=False)
            data_save['A/Ngas']=ext_curve
            # if filename_output is None:
            #     self.outputFile = 'p_abs.dat'
            # else:
            #     self.outputFile = filename_output+'_abs.dat'
            base = os.path.basename(filename_output or "p")
            self.outputFile = f"{base}_abs.dat"
            
            DustPOL_io.output(self,data_save)
        return w*1e4,Pabs_NH

    @auto_refresh
    def cal_pol_emi(self,Av=0.0,NH=0.0,Tdust=None,verbose=True,save_output=False,filename_output=None, intensity_output=False):
        '''
        This function calculates the degree of thermal dust polarization (0D)
        The inputs are taken from the input datafile with the additional input parameters
        Inputs:
        -------
              1- Av: extinction in mag
                 [if Av=0.0, the optically thin assumption is used]
                 [if Av!=0.0, the optical depth effect is used]
              2- Tdust: dust temperature in K, if None, it will be calculated from the radiation field
              3- verbose: if True, print the information about the dust size distribution
              4- intensity_output: if True, return the intensity for each dust type
              5- save_output: if True, write the output to a file in the output folder
              6- filename_output: if None, the output file will be named 'p_emi.dat' otherwise it will be named as filename_output+'_emi.dat'
        Outputs:
        -------
              if intensity_output==False: [default]
                  1- wavelength in micron
                  2- Pem: polarization degree
              else intensity_output==True:
                  1- wavelength in micron
                  2- Iem_dict: intensity for each dust type [dictionary type]
                  3- Ipol_dict: polarized intensity for each dust type [dictionary type]
                  4- Pem: polarization degree
        '''
        self.verbose=verbose #transfer verbose to parent var to pass to pol_degree and align classes
        if Av < 0.0:
            raise ValueError('Av should be >= 0.0 mag [\033[1;5;7;91m failed \033[0m]')
        if NH < 0.0:
            raise ValueError('NH should be >= 0.0 cm-2 [\033[1;5;7;91m failed \033[0m]')
        
        ## Call the Planck function
        for dusttype in self.dust_type.split('+'):
            if dusttype=='astro':
                if Tdust is None:
                    Tdust = 16.4* self.U**(1./6) * (self.a/1.e-5)**(-1./15) # np.ones(self.na)
                    if (self.verbose):
                        log.info('\033[1;7;34m U=%.3f : radiation -->> Tdust \033[0m   \t\t '%(self.U))
                elif isinstance(Tdust,(float,int)):
                    self.U = (Tdust/16.4)**6 # Update U based on Tdust
                    if (self.verbose):
                        log.info('\033[1;7;34m Tdust=%.3f (K) --> U=%.3f \033[0m   \t\t '%(Tdust,self.U))
                    Tdust = float(Tdust) * np.ones_like(self.a)* (self.a/1.e-5)**(-1./15) #
                    
                BB_ = self.get_Planck_function(Tdust,self.a)
                setattr(self, f'BB_{dusttype}', BB_)
                if self.fscale is not None:
                    BB_add_ = self.get_Planck_function(self.Tdust_add,self.a_add)
                    setattr(self, f'BB_add_{dusttype}', BB_add_)
                continue

            ## For other dust types, we need to calculate the Planck function with the temperature distribution
            a_, T_, dP_dlnT_ = self.dP_dT(dusttype)
            BB_ = self.get_Planck_function(T_,self.a,a_init=a_,dP_dlnT=dP_dlnT_)
            setattr(self, f'BB_{dusttype}', BB_)

            if self.fscale is not None:
                self.U = self.U_add  # Update U for the additional grain size
                a_add_, T_add_, dP_dlnT_add_ = self.dP_dT(dusttype)
                BB_add_ = self.get_Planck_function(T_add_,self.a_add, a_init=a_add_, dP_dlnT=dP_dlnT_add_)
                setattr(self, f'BB_add_{dusttype}', BB_add_)
                        
        if Av!=0.0 or NH!=0.0:
            if (Av !=0.0) and (NH !=0.0):
                raise ValueError('Av and NH cannot be both non-zero! [\033[1;5;7;91m failed \033[0m]')

            # self.U = self.U_init                 ## -->> reset self.U
            # self.mean_lam=self.mean_lam_init     ## --> reset self.mean_lambda

            # calculate the extinction curve and Rv
            w_,ext_ = self.extinction_curve(verbose=False,f_align=True)
            fextinction=interp1d(w_,ext_,bounds_error=False, fill_value="extrapolate")
            Rv = fextinction(0.55)/(fextinction(0.44)-fextinction (0.55)) #RV= A_V/(A_B-A_V)
                
            if (Av != 0.0):             
                # convert Av to Ngas from extinction curve
                Ngas = Av*5.8e21/Rv # convert Av to Ngas (in cm-2)
                # Av   = Av
            if (NH != 0.0):
                Ngas = NH
                # Av   = Ngas*Rv/5.8e21 # convert NH to Av (in mag)

            # Update verbose to self-wise because its value might be changed in the extinction_curve function
            self.verbose=verbose
        
            # Calculate the degree of thermal dust polarization                
            w,intensity_dict,intensity_tot,Pem = pol_degree.pol_degree(self)._pol_degree_emission_general(self,Ngas)
            Iem_dict,Ipol_dict = intensity_dict
            Iem_tot, Ipol_tot  = intensity_tot
            
        else:
            # self.U = self.U_init             ## -->> reset self.U
            # self.mean_lam=self.mean_lam_init ## --> reset self.mean_lambda

            # Calculate the degree of thermal dust polarization for the optically thin case
            w,intensity_dict,intensity_tot,Pem = pol_degree.pol_degree(self)._pol_degree_emission_thin(self) 
            Iem_dict,Ipol_dict = intensity_dict
            Iem_tot, Ipol_tot  = intensity_tot
                
        if (self.verbose):
            self.sdist_info_to_print()

        # Save the output
        if (save_output):
            data_save={}
            data_save['wavelength'] = w*1e4 #convert to micron

            if intensity_output:
                # Save the total intensity and polarized intensity
                for dusttype in self.dust_type.split('+'):
                    data_save[f'Iem_{dusttype}']  = Iem_dict[dusttype]
                    data_save[f'Ipol_{dusttype}'] = Ipol_dict[dusttype]
            # Save the polarization degree
            data_save['Pem(total)'] = Pem

            # Define the output file name
            # if filename_output is None:
            #     self.outputFile = 'p_emi.dat'
            # else:
            #     self.outputFile = filename_output+'_emi.dat'
            base = os.path.basename(filename_output or "p")
            self.outputFile = f"{base}_emi.dat"
        
            # Save to file                  
            DustPOL_io.output(self,data_save)
        if intensity_output:
            # return w*1e4, Iem_tot, Ipol_tot, Pem
            return w*1e4, Iem_dict, Ipol_dict, Pem
        else:
            return w*1e4, Pem
    
    @auto_refresh
    def isoCloud_los(self,r0,progress=False,get_info=True,save_output=False,filename_output=None):
        '''
        This function calculates the degree of starlight and thermal dust polarization (1D) for a given line of sight in starless core
        For the fundamentals, see [website] for insights
        The integration along a given line of sight 'r0'
        Input parameters are taken from the input datafile with the additional input parameters
        Inputs:
        -------
              1- r0: line of sight [distance from the center of the cloud]
              2- progress: if True, print the progress bar
              3- get_info: if True, print the information
              4- save_output: if True, write the output to a file in the output folder
              5- filename_output: if None, the output file will be named 'p_abs.dat' otherwise it will be named as filename_output+'_abs.dat'
        Outputs:
        -------
              1- wavelength in micron
              2- gas column density along the line of sight defined by r0
              3- extinction along the line of sight defined by r0
              4- total intensity along the line of sight defined by r0
              5- degree of starlight polarization along the line of sight defined by r0
              6- degree of thermal dust polarization along the line of sight defined by r0
        '''
        if (get_info):
            if (progress):
                self.verbose=False
            else:
                self.verbose=True
        else:
            self.verbose=False

        # ------- Initialization paramters  -------
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
        
        U_0       =self.U          ##hard copy of the initial radiation field
        ngas_0    =self.ngas       ##hard copy of the initial gas volume density
        mean_lam_0=self.mean_lam   ##hard copy of the initial mean wavelength

        #call the starless_profile
        isoCloud_exe = isoCloud_class.isoCloud_profile()#(self)
        coords,rr=isoCloud_exe.isoCloud_model(self) #the global parameter "self" has been updated...
        x_,y_,z_=coords
        max_radius = rr.max()

        # Av_ = isoCloud_exe.Av_func(self,r0)
        Av_ = isoCloud_exe.Av_los_by_dust(self,r0)
        
        if get_info:
            print('-----------Get radiation------------------')
            log.info('U=%.3f '%self.U)                

            print('-----------Get ngas------------------')
            log.info('ngas_0=%.3e (cm-3)'%self.ngas)                

            print('-----------Get distance params-------')
            log.info('rflat=%.3f (pc)'%(self.rflat/self.pc))
            log.info('max_radius=%.3f (pc)'%(max_radius/self.pc))

            print('-----------Get observed Av-----------')
            log.info('Av_los=%.3f (mag.)'%Av_)

        if (len(z_)<2*self.nsample+1):
            raise IOError('descritation of z axis is wrong!')

        # get_planck_option=True

        dp_abs_matrix = np.zeros((len(z_),len(self.w)))
        dpmix_abs_matrix = np.zeros((len(z_),len(self.w)))

        dIext_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIp_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIpmix_emi_matrix = np.zeros((len(z_),len(self.w)))

        nH_=np.zeros(len(z_))
        Av_compute=np.zeros(len(z_))
        ali_=np.zeros(len(z_))

        ##loop over z_ along a single LOS
        if (progress) & (get_info): 
            printProgressBar(0, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)

        for j in range(len(z_)):
        
            r_compute=np.sqrt(z_[j]*z_[j]+r0*r0)
            if (r_compute>max_radius):
                # self.ngas=np.nan
                dp_abs_matrix[j,:] =np.zeros(len(self.w))
                dpmix_abs_matrix[j,:] =np.zeros(len(self.w))
                dIext_emi_matrix[j,:] =np.zeros(len(self.w))
                dIp_emi_matrix[j,:]=np.zeros(len(self.w))
                dIpmix_emi_matrix[j,:]=np.zeros(len(self.w))

                nH_[j]=0.0
                Av_compute[j]=0.0
                ali_[j]=0.0
                if (get_info):
                    if (progress):
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('z_=%.3f (pc), Av_los=%.3f (mag), ngas=%.3e'%(z_[j]/self.pc,Av_,nH_[j]))
                continue        
            else:
                #print('-----------Get the local computed Av-----------')
                Av_compute[j]=isoCloud_exe.Av_2calcule(ngas_0,self.rflat,self.p,Rv=4.0)(r_compute)

                #get U from starless law
                self.U = isoCloud_exe.U_starless(U_0,Av_compute[j])
                # self.U = radiation.radiation_retrieve(self).retrieve(PAHs=False) ##useful when taking dP_dT into account

                #print('-----------Get Tgas-----------')
                #print('Tgas_init=',self.Tgas)
                self.Tgas=isoCloud_exe.Tgas_starless(U_0,Av_compute[j],16.4)

                #print('-----------Get Dust-----------')
                self.Tdust=isoCloud_exe.Tdust_starless(U_0,Av_compute[j],16.4,self.a)

                #print('-----------Get mean_lam-----------')
                #print('mean_lam_init=',self.mean_lam*1e4)
                self.mean_lam=isoCloud_exe.lamda_starless(mean_lam_0,Av_compute[j])

                # self.ngas=starless_exe.ngas_starless(ngas_0,self.rflat)(np.sqrt(z_[j]*z_[j]+r0*r0))                
                self.ngas=isoCloud_exe.ngas_starless(ngas_0,self.rflat,self.p)(r_compute) 
                # dtau=starless_exe.get_dtau(self,self.ngas)
                # dtau_850 = interp1d(self.w,dtau,axis=0)(850e-4)*abs(z_[j])  

                if (get_info):
                    if (progress):
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('\n')
                        log.info('z_=%.3e (pc), Av_los=%.3f (mag), Av_compute=%.3f, U=%.3f, ngas=%.3e, amax=%.3f \t\t'%(z_[j]/self.pc,Av_,Av_compute[j],self.U,self.ngas,self.a.max()*1e4))

                #absorption polarization
                # w_abs,dp_abs,dpmix_abs=self.cal_pol_abs(verbose=self.verbose)
                _,dp_nH_abs=self.cal_pol_abs(verbose=self.verbose)
                dp_abs = dp_nH_abs * self.ngas  
                # w_abs contains negative values (probably due to the interpolation)
                dp_abs[self.w<0] = 0.0
                
                #emission polarization
                _,dIem_dict,dIpol_dict,_ = self.cal_pol_emi(Tdust=self.Tdust,verbose=self.verbose,save_output=False, intensity_output=True)

                # dIem_dict and dIpol_dict should already be filled with values for each dusttype
                dIem_dict['total'] = np.sum([v for v in dIem_dict.values()], axis=0)
                dIpol_dict['total'] = np.sum([v for v in dIpol_dict.values()], axis=0)
                
                #save to the matrix for los (i.e. z_ direction)
                dp_abs_matrix[j, :]    = dp_abs
                dIext_emi_matrix[j, :] = dIem_dict['total']
                dIp_emi_matrix[j, :]   = dIpol_dict['total']

                nH_[j]=self.ngas

        # Colum density of gas along the line of sight
        NH_=integrate.simpson(nH_,z_)

        # absorption polarization along the line of sight
        p_abs   = integrate.simpson(dp_abs_matrix,z_,axis=0)

        # thermal dust polarization along the line of sight
        Iext_emi = integrate.simpson(dIext_emi_matrix,z_,axis=0)
        Ip_emi   = integrate.simpson(dIp_emi_matrix,z_,axis=0)
        p_emi    = Ip_emi/Iext_emi*100

        if (save_output):
            self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
            self.Av_array=Av_

            # Save the output for absorption
            data_abs={}
            data_abs['w']=self.w*1e4
            data_abs['pabs']=p_abs         
            if filename_output is None:
                self.outputFile = 'isoCloud_los_abs.dat'                
            else:
                self.outputFile = filename_output+'_abs.dat'
            DustPOL_io.output(self,data_abs)

            # Save the output for emission
            data_emi={}
            data_emi['w']=self.w*1e4
            data_emi['pemi']=p_emi
            data_emi['Iext']=Iext_emi
            if filename_output is None:
                self.outputFile = 'isoCloud_los_emi.dat'                
            else:
                self.outputFile = filename_output+'_emi.dat'
            DustPOL_io.output(self,data_emi)

        return self.w*1e4, NH_, Av_, Iext_emi, p_abs, p_emi

    @auto_refresh
    def isoCloud_pos(self,filename_output=None,progress=False):
        '''
        This function calculates the degree of starlight and thermal dust polarization (2D) for a starless core
        For the fundamentals, see [website] for insights
        Input parameters are taken from the input datafile with the additional input parameters
        Inputs:
        -------
            1- filename_output: if None, the output file will be named 'isoCloud_pos_abs.dat' and 'isoCloud_pos_emi.dat'
            2- progress: if True, print the progress bar
        '''
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters

        isoCloud_exe = isoCloud_class.isoCloud_profile()
        coords,_=isoCloud_exe.isoCloud_model(self)
        x_,y_,z_=coords

        data_abs={}
        data_abs['w']=self.w*1e4
        
        data_emi={}
        data_emi['w']=self.w*1e4
        
        Av_array=[];x_=x_[x_>=0]
        r0_range=x_[::2]#np.linspace(0,self.rout/2e3,30)#*constants.pc.cgs.value

        # Av_test=np.zeros((len(r0_range),len(z_)))
        # ali_test=np.zeros((len(r0_range),len(z_)))
        start_time=time.time()

        if (not self.parallel): #None parallelization
            for i,r0 in enumerate(r0_range):
                print('---------------------------------------------------')
                print('cell number=%d/%d'%(i,len(r0_range)), 'r0=%.3e (pc)'%(r0/self.pc))

                self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
            
                _,_,Av_,Iext_,pabs_,pemi_=self.isoCloud_los(
                                        r0,
                                        progress=progress
                                        )
                Av_array.append(Av_)

                data_abs['p(Av=%.3f)'%Av_]=pabs_
                data_emi['p(Av=%.3f)'%Av_]=pemi_
                data_emi['I(Av=%.3f)'%Av_]=Iext_

        else: #parallelization
            get_info=False
            progress=False
            log.info('Parallel computation with : \033[1;36m %d \033[0m CPU cores'%(self.max_workers))
            # printProgressBar(0, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30)
            # printProgressBar(0, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30)
            
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tasks to the executor
                    j_submit=0
                    j_process=0
                    futures = []
                    for r0 in r0_range:
                        # Reset initial parameters
                        # self.__init__(self.input_params_file,**self.kwargs)
                        try:
                            # future = executor.submit(
                            #     self.isoCloud_los,
                            #     r0,
                            #     progress=progress,
                            #     get_info=get_info
                            # )
                            args = (self.input_params_file, self.kwargs, r0, False, False)
                            future = executor.submit(_isoCloud_los_worker, args)
                            
                            if future is not None:
                                futures.append(future)
                                j_submit=j_submit+1
                                printProgressBar(j_submit, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30)
                            else:
                                print(f"Warning: executor.submit returned None for r0={r0}")
                        except Exception as e:
                            # print(f"Error submitting task for r0={r0}: {e}")
                            log.debug(f"Error submitting task for r0={r0}: {e}")
                            
                    # Ensure no NoneType in futures
                    if not futures:
                        raise RuntimeError("ProcessPoolExecutor: No valid futures were created --> switch to joblib")

                    # Process completed futures                
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()  # Retrieve the result of the future
                            # print('Futures were created!')
                            _,_,Av_,Iext_,pabs_,pemi_=result

                            Av_array.append(Av_)

                            data_abs['p(Av=%.3f)'%Av_]=pabs_
                            data_emi['p(Av=%.3f)'%Av_]=pemi_
                            data_emi['I(Av=%.3f)'%Av_]=Iext_

                            j_process=j_process+1            
                            printProgressBar(j_process, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30)

                        except Exception as e:
                            print(f"Task generated an exception: {e}")
            except Exception as e:
                print(f"ProcessPoolExecutor failed ({e}); falling back to joblib loky backend.")
                results = Parallel(n_jobs=self.max_workers, backend='loky', verbose=1)(
                    delayed(_isoCloud_los_worker)(
                            (self.input_params_file, self.kwargs, float(r0), False, False)
                            )
                        for r0 in r0_range
                        )
                for (_,_,Av_,Iext_,pabs_,pemi_) in results:
                    Av_array.append(Av_)

                    data_abs['p(Av=%.3f)'%Av_]=pabs_
                    data_emi['p(Av=%.3f)'%Av_]=pemi_
                    data_emi['I(Av=%.3f)'%Av_]=Iext_   

        self.Av_array=np.array(Av_array)
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
        #There is a draw back of this saveout method: 
        #  If the keys are the same (two exact values of Av)
        #  Save the last array!!!
        
        # if filename_output is None:
        #     self.outputFile = 'isoCloud_pos_abs.dat'                
        # else:
        #     self.outputFile = filename_output+'_abs.dat'
        base = os.path.basename(filename_output or "isoCloud_pos")
        self.outputFile = f"{base}_abs.dat"

        DustPOL_io.output(self,data_abs)
        
        # if filename_output is None:
        #     self.outputFile = 'isoCloud_pos_emi.dat'               
        # else:
        #     self.outputFile = filename_output+'_emi.dat'
        self.outputFile = f"{base}_emi.dat"
        DustPOL_io.output(self,data_emi)

        end_time=time.time()
        if end_time-start_time<60:
            log.info('  -> Time for execution is %.2f secs'%(end_time-start_time))
        elif end_time-start_time<3600:
            print('  -> Time for execution is %.2f mins'%((end_time-start_time)/60))
        else:
            print('  -> Time for execution is %.2f hrs'%((end_time-start_time)/60/60))


    @auto_refresh
    def isoProtostar_los(self,r0,progress=False,get_info=True,save_output=False,filename_output=None):
        '''
        This function calculates the degree of starlight and thermal dust polarization (1D) for a given line of sight in protostellar core
        For the fundamentals, see [website] for insights
        The integration along a given line of sight 'r0'
        Input parameters are taken from the input datafile with the additional input parameters
        Inputs:
        -------
              1- r0: line of sight [distance from the center of the cloud]
              2- progress: if True, print the progress bar
              3- get_info: if True, print the information
              4- save_output: if True, write the output to a file in the output folder
              5- filename_output: if None, the output file will be named 'p_abs.dat' otherwise it will be named as filename_output+'_abs.dat'
        Outputs:
        -------
              1- wavelength in micron
              2- gas column density along the line of sight defined by r0
              3- extinction along the line of sight defined by r0
              4- total intensity along the line of sight defined by r0
              5- degree of starlight polarization along the line of sight defined by r0
              6- degree of thermal dust polarization along the line of sight defined by r0
        '''
        if (get_info):
            if (progress):
                self.verbose=False
            else:
                self.verbose=True
        else:
            self.verbose=False

        # ------- Initialization paramters  -------
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
        
        # self.update_radiation_aISRF(Av=0.0) ##estimate the initial radiation field at the surface of the core without extinction
                                            ## --> self.U and self.mean_lam are updated

        a_init       = self.a.copy()        ##hard copy of the initial grain size
        ngas_0       = self.ngas            ##hard copy of the initial gas volume density
        self.U_ISRF  = self.U_init          ##hard copy of the initial radiation field from outside
        self.mean_lam_ISRF= self.mean_lam   ##hard copy of the initial mean wavelength
        
        #call the starless_profile
        isoProtostar_exe = isoProtostar_class.isoProtostar_profile()#(self)
        coords,rr=isoProtostar_exe.isoProtostar_model(self) #the global parameter "self" has been updated...
        _,_,z_=coords
        max_radius = rr.max()

        NH_,Av_ = isoProtostar_exe.Av_los_by_gas(r0)

        if get_info:
            print('-----------Get radiation------------------')
            log.info('U=%.3f '%self.U)                

            print('-----------Get ngas------------------')
            log.info('ngas_0=%.3e (cm-3)'%self.ngas)                

            print('-----------Get distance params-------')
            log.info('rflat=%.3f (pc)'%(self.rflat/self.pc))
            log.info('max_radius=%.3f (pc)'%(max_radius/self.pc))

            print('-----------Get observed Av-----------')
            log.info('Av_los=%.3f (mag.)'%Av_)

        if (len(z_)<2*self.nsample+1):
            raise IOError('descritation of z axis is wrong!')

        # get_planck_option=True

        dp_abs_matrix = np.zeros((len(z_),len(self.w)))
        dpmix_abs_matrix = np.zeros((len(z_),len(self.w)))

        dIext_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIp_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIpmix_emi_matrix = np.zeros((len(z_),len(self.w)))

        nH_=np.zeros(len(z_))
        Av_compute=np.zeros(len(z_))
        ali_=np.zeros(len(z_))

        if (progress) & (get_info): 
            printProgressBar(0, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)

        ##loop over z_ along a single LOS
        for j in range(len(z_)):
        
            r_compute=np.sqrt(z_[j]*z_[j]+r0*r0)
            if (r_compute>max_radius):
                # self.ngas=np.nan
                dp_abs_matrix[j,:] =np.zeros(len(self.w))
                dpmix_abs_matrix[j,:] =np.zeros(len(self.w))
                dIext_emi_matrix[j,:] =np.zeros(len(self.w))
                dIp_emi_matrix[j,:]=np.zeros(len(self.w))
                dIpmix_emi_matrix[j,:]=np.zeros(len(self.w))

                nH_[j]=0.0
                Av_compute[j]=0.0
                ali_[j]=0.0
                if (get_info):
                    if (progress):
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('z_=%.3f (pc), Av_los=%.3f (mag), ngas=%.3e'%(z_[j]/self.pc,Av_,nH_[j]))
                continue        
            else:
                #print('-----------Get the local computed Av-----------')
                Av_compute[j]=isoProtostar_exe.Av_surface2cell(ngas_0,self.rflat,self.p)(r_compute)

                #get mean_lam and U
                self.mean_lam, self.U, self.gamma = isoProtostar_exe.radiation(self,r_compute)
                # print('  [test] -> mean_lam=%.3f (micron), U=%.2e, gamma=%.2f'%(self.mean_lam*1e4,self.U,self.gamma))
                #print('-----------Get Tgas-----------')
                #print('Tgas_init=',self.Tgas)
                self.Tgas = isoProtostar_exe.Tdust_protostar(self.U) #Assume Tgas=Tdust

                #print('-----------Get Dust-----------')
                self.Tdust=self.Tgas #* (self.a/1e-5)**(-1./15)

                # self.ngas=starless_exe.ngas_starless(ngas_0,self.rflat)(np.sqrt(z_[j]*z_[j]+r0*r0))                
                self.ngas=isoProtostar_exe.ngas_protostar(ngas_0,self.rflat,self.p)(r_compute) 
                # dtau=starless_exe.get_dtau(self,self.ngas)
                # dtau_850 = interp1d(self.w,dtau,axis=0)(850e-4)*abs(z_[j])  
                
                # checking whether RAT-D occurs
                lmin = np.searchsorted(a_init, self.amin)
                lmax = np.searchsorted(a_init, min(self.amax, disrupt.radiative_disruption(self).a_disrupt(a_init))) if self.ratd else np.searchsorted(a_init, self.amax + 0.1 * self.amax)
                a_check  = a_init[lmin:lmax+1]
                
                # print(' -> a.max=%.3f (micron), a_init.max = %.3f (micron)'%(a_check.max()*1e4,a_init.max()*1e4))
                if a_check.max() < a_init.max():

                    # ------- Initialization wavelength, grain size from the file  ------- 
                    self.get_grainsize_wavelength() ## -->> wavelength (self.w) and grain size (self.a)
        
                    # ------- Update grain size (if RAT-D occurs)  ------- 
                    self.update_grain_size(self.a,verbose=False) ## -->> update self.a

                    # ------- Update grain-size distribution -------
                    self.grain_size_distribution()  # update self.a -->> update self.dnda

                    # ------- Update Qext,Qabs,Qpol and Qpol_abs -------
                    self.get_coefficients_data() ## self.a -->> self.Qdata
                
                # print('  -> mean_lam=%.3f (micron), U=%.2e, ngas=%.3e, amax=%.3f'%(self.mean_lam*1e4,self.U,self.ngas,self.a.max()*1e4))
            
                if (get_info):
                    if (progress):
                        self.verbose=False
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('\n')
                        self.verbose=True
                        log.info('z_=%.3e (pc), Av_los=%.3f (mag), Av_surface2cell=%.3f, U=%.3e, mean_lam=%.3f (micron), Tgas=%.3f (K), ngas=%.3e, amax=%.3f (micron) \t\t'%(z_[j]/self.pc,Av_,Av_compute[j],self.U,self.mean_lam*1e4,self.Tgas,self.ngas,self.a.max()*1e4))
                else:
                    self.verbose=False
                    
                #absorption polarization
                # w_abs,dp_abs,dpmix_abs=self.cal_pol_abs(verbose=self.verbose)
                _,dp_nH_abs=self.cal_pol_abs(verbose=self.verbose)
                dp_abs = dp_nH_abs * self.ngas  
                # w_abs contains negative values (probably due to the interpolation)
                dp_abs[self.w<0] = 0.0
                
                #emission polarization
                _,dIem_dict,dIpol_dict,_ = self.cal_pol_emi(Tdust=self.Tdust,verbose=self.verbose,save_output=False, intensity_output=True)

                # dIem_dict and dIpol_dict should already be filled with values for each dusttype
                dIem_dict['total'] = np.sum([v for v in dIem_dict.values()], axis=0)
                dIpol_dict['total'] = np.sum([v for v in dIpol_dict.values()], axis=0)
                
                #save to the matrix for los (i.e. z_ direction)
                dp_abs_matrix[j, :]    = dp_abs
                dIext_emi_matrix[j, :] = dIem_dict['total']
                dIp_emi_matrix[j, :]   = dIpol_dict['total']

                #need to reset the grain size distribution for the next point
                # self.a = a_init                 ## --> update self.a                
                # self.grain_size_distribution()  ## update self.a -->> update self.dnda
                # self.get_coefficients_data()    ## self.a -->> self.Qdata
                
                # Reset grain state and Q* in O(1) by restoring snapshots
                self._restore_initial_grain_state()
                self._restore_initial_Qcoeffs()
                
        # absorption polarization along the line of sight
        p_abs   = integrate.simpson(dp_abs_matrix,z_,axis=0)

        # thermal dust polarization along the line of sight
        # dIext_emi_matrix = np.nan_to_num(dIext_emi_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        # dIp_emi_matrix   = np.nan_to_num(dIp_emi_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        Iext_emi = integrate.simpson(dIext_emi_matrix,z_,axis=0)
        Ip_emi   = integrate.simpson(dIp_emi_matrix,z_,axis=0)
        p_emi    = Ip_emi/Iext_emi*100

        if (save_output):
            self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
            self.Av_array=Av_
            self.NH_array=NH_

            # Save the output for absorption
            data_abs={}
            data_abs['w']=self.w*1e4
            data_abs['pabs']=p_abs         
            if filename_output is None:
                self.outputFile = 'isoProtostar_los_abs.dat'                
            else:
                self.outputFile = filename_output+'_abs.dat'
            DustPOL_io.output(self,data_abs)

            # Save the output for emission
            data_emi={}
            data_emi['w']=self.w*1e4
            data_emi['pemi']=p_emi
            data_emi['Iext']=Iext_emi
            if filename_output is None:
                self.outputFile = 'isoProtostar_los_emi.dat'                
            else:
                self.outputFile = filename_output+'_emi.dat'
            DustPOL_io.output(self,data_emi)

        return self.w*1e4, NH_, Av_, Iext_emi, p_abs, p_emi

    @auto_refresh
    def isoProtostar_pos(self,filename_output=None,progress=False):
        '''
        This function calculates the degree of starlight and thermal dust polarization (2D) for a protostellar core
        For the fundamentals, see [website] for insights
        Input parameters are taken from the input datafile with the additional input parameters
        Inputs:
        -------
            1- filename_output: if None, the output file will be named 'isoProstar_pos_abs.dat' and 'isoProstar_pos_emi.dat'
            2- progress: if True, print the progress bar
        '''
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters

        isoProstar_exe = isoProtostar_class.isoProtostar_profile()
        coords,_=isoProstar_exe.isoProtostar_model(self)
        x_,y_,z_=coords

        data_abs={}
        data_abs['w']=self.w*1e4
        
        data_emi={}
        data_emi['w']=self.w*1e4
        
        Av_array=[];NH_array=[];x_=x_[x_>=0]
        r0_range=x_[::2]#np.linspace(0,self.rout/2e3,30)#*constants.pc.cgs.value

        # Av_test=np.zeros((len(r0_range),len(z_)))
        # ali_test=np.zeros((len(r0_range),len(z_)))
        start_time=time.time()

        if (not self.parallel): #None parallelization
            for i,r0 in enumerate(r0_range):
                print('---------------------------------------------------')
                print('cell number=%d/%d'%(i,len(r0_range)), 'r0=%.3e (pc)'%(r0/self.pc))

                self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters

                _,NH_,Av_,Iext_,pabs_,pemi_=self.isoProtostar_los(
                                        r0,
                                        progress=progress
                                        )
                Av_array.append(Av_)
                NH_array.append(NH_)

                data_abs['p(Av=%.3f)'%Av_]=pabs_
                data_emi['p(Av=%.3f)'%Av_]=pemi_
                data_emi['I(Av=%.3f)'%Av_]=Iext_

        else: #parallelization
            get_info=False
            progress=False
            log.info('Parallel computation with : \033[1;36m %d \033[0m CPU cores'%(self.max_workers))
            # printProgressBar(0, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30)
            # printProgressBar(0, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30)
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tasks to the executor
                    j_submit=0
                    j_process=0
                    futures = []
                    for r0 in r0_range:
                        # Reset initial parameters
                        # self.__init__(self.input_params_file,**self.kwargs)
                        try:
                            # future = executor.submit(
                            #     self.isoProtostar_los,
                            #     r0,
                            #     progress=progress,
                            #     get_info=get_info
                            # )
                            args = (self.input_params_file, self.kwargs, float(r0), False, False)
                            future = executor.submit(_isoProtostar_los_worker, args)
                            if future is not None:
                                futures.append(future)
                                j_submit=j_submit+1
                                printProgressBar(j_submit, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30)
                            else:
                                print(f"Warning: executor.submit returned None for r0={r0}")
                        except Exception as e:
                            # print(f"Error submitting task for r0={r0}: {e}")
                            log.debug(f"Submit failed for r0={r0}: {e}")

                    # Ensure no NoneType in futures
                    if not futures:
                        raise RuntimeError("ProcessPoolExecutor: No valid futures were created --> switch to joblib.")

                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()  # Retrieve the result of the future
                            # print('Futures were created!')
                            _,NH_,Av_,Iext_,pabs_,pemi_=result

                            Av_array.append(Av_)
                            NH_array.append(NH_)

                            data_abs['p(Av=%.3f)'%Av_]=pabs_
                            data_emi['p(Av=%.3f)'%Av_]=pemi_
                            data_emi['I(Av=%.3f)'%Av_]=Iext_

                            j_process=j_process+1            
                            printProgressBar(j_process, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30)

                        except Exception as e:
                            print(f"Task generated an exception: {e}")
                            
            except Exception as e:
                print(f"ProcessPoolExecutor failed ({e}); falling back to joblib loky backend.")
                results = Parallel(n_jobs=self.max_workers, backend='loky', verbose=1)(
                    delayed(_isoProtostar_los_worker)(
                            (self.input_params_file, self.kwargs, float(r0), False, False)
                            )
                        for r0 in r0_range
                        )
                for (_, NH_, Av_, Iext_, pabs_, pemi_) in results:
                    Av_array.append(Av_)
                    NH_array.append(NH_)

                    data_abs['p(Av=%.3f)'%Av_]=pabs_
                    data_emi['p(Av=%.3f)'%Av_]=pemi_
                    data_emi['I(Av=%.3f)'%Av_]=Iext_        

        self.Av_array=np.array(Av_array)
        self.NH_array=np.array(NH_array)
        self.__init__(self.input_params_file,**self.kwargs) ##reset the initial parameters
        #There is a draw back of this saveout method: 
        #  If the keys are the same (two exact values of Av)
        #  Save the last array!!!
        
        # if filename_output is None:
        #     self.outputFile = 'isoProtostar_pos_abs.dat'                
        # else:
        #     self.outputFile = filename_output+'_abs.dat'
        # DustPOL_io.output(self,data_abs)
        
        # if filename_output is None:
        #     self.outputFile = 'isoProtostar_pos_emi.dat'               
        # else:
        #     self.outputFile = filename_output+'_emi.dat'
        # DustPOL_io.output(self,data_emi)

        base = os.path.basename(filename_output or "isoProtostar_pos")
        self.outputFile = f"{base}_abs.dat"

        DustPOL_io.output(self,data_abs)
        
        # if filename_output is None:
        #     self.outputFile = 'isoCloud_pos_emi.dat'               
        # else:
        #     self.outputFile = filename_output+'_emi.dat'
        self.outputFile = f"{base}_emi.dat"
        DustPOL_io.output(self,data_emi)
        
        end_time=time.time()
        if end_time-start_time<60:
            log.info('  -> Time for execution is %.2f secs'%(end_time-start_time))
        elif end_time-start_time<3600:
            print('  -> Time for execution is %.2f mins'%((end_time-start_time)/60))
        else:
            print('  -> Time for execution is %.2f hrs'%((end_time-start_time)/60/60))
                    
    @auto_refresh
    def sdist_info_to_print(self):

        # log.info(f'Radiation field: \033[1;7;34m U={self.U:.3f} \033[0m   \t\t ')
        log.info(f'Grain composition: \033[1;7;36m {self.dust_type:s} \033[0m and Size distribution: \033[1;7;36m {self.GSD_law:s} \033[0m  \t\t ')
        dusttype = self.dust_type.lower()
        gsd_law  = self.GSD_law.lower().split('+')

        def log_mrn(prefix='Astrodust:'):
            log.info(f'{prefix} Grain-size distribution: MRN with power_index={self.power_index:.2f}')

        def log_hd23(prefix='Astrodust:'):
            log.info(
                f'{prefix} Grain-size distribution: HD23 with \n'
                f'\t\t BAd={self.BAd:.3e}, a0Ad={self.a0Ad:.3e}, sigmaAd={self.sigmaAd:.3f}, '
                f'A0={self.A0:.3e}, A1-A5={self.A1:.3f}-{self.A2:.3f}-{self.A3:.3f}-{self.A4:.3f}-{self.A5:.3f}'
            )

        def log_wd01_sil(prefix='Astrodust'):
            log.info(
                f'{prefix} Grain-size distribution: WD01_silicate with \n'
                f'\t\t alpha={self.alpha_wd01_sil:.3e}, beta={self.beta_wd01_sil:.3e}, at={self.at_wd01_sil*1e4:.3f}, '
                f'ac={self.ac_wd01_sil:.3e}, Cs={self.Cs_wd01_sil:.3e}, B01={getattr(self, f'B1_wd01_sil', np.nan)}, B02={getattr(self, f'B2_wd01_sil', np.nan)}'
            )

        def log_wd01_car(prefix='Astrodust'):
            log.info(
                f'{prefix} Grain-size distribution: WD01_carbon with \n'
                f'\t\t alpha={self.alpha_wd01_car:.3e}, beta={self.beta_wd01_car:.3e}, at={self.at_wd01_car*1e4:.3f}, '
                f'ac={self.ac_wd01_car:.3e}, Cs={self.Cs_wd01_car:.3e}, '
                f'a01={self.a01_wd01_car*1e4:.3f}, a02={self.a02_wd01_car*1e4:.3f}, '
                f'sigma={self.sigma_wd01_car:.3f}, B01={self.B1_wd01_car:.3e}, B02={self.B2_wd01_car:.3e}'
            )

        if dusttype=="sil" or check_variable_combination(dusttype, required_parts = {"sil", "pah"}):#dusttype in ('sil','sil+pah', 'pah+sil'):
            # if gsd_law not in ('mrn','wd01','wd01_ed'):
            if 'hd23' in gsd_law:
                raise ValueError(f'{self.GSD_law} is not supported for {self.dust_type}')            
            else:
                for gsd_law_i in gsd_law:
                    log_mrn(prefix='Silicate') if gsd_law_i=='mrn' else log_wd01_sil(prefix='Silicate')
                
                if 'pah' in dusttype:
                    log.info(f'PAH: Grain-size distribution with \n\t\t B1={self.B1:.3e} and B2={self.B2:.3e}')

        elif dusttype=="car" or check_variable_combination(dusttype, required_parts = {"car", "pah"}):#dusttype in ('sil','sil+pah', 'pah+sil'):
            # if gsd_law not in ('mrn','wd01','wd01_ed'):
            if 'hd23' in gsd_law:
                raise ValueError(f'{self.GSD_law} is not supported for {self.dust_type}')   
            else:
                for gsd_law_i in gsd_law:
                    log_mrn(prefix='Carbon') if gsd_law_i == 'mrn' else log_wd01_sil(prefix='Carbon')
                                         
                if 'pah' in dusttype:
                    log.info(f'PAH: Grain-size distribution with \n\t\t B1={self.B1:.3e} and B2={self.B2:.3e}')

        elif check_variable_combination(dusttype, required_parts = {"sil", "car"}):#dust_type=='sil+car':
            # if gsd_law not in ('mrn','wd01','wd01_ed'):
            if 'hd23' in gsd_law:
                raise ValueError(f'{self.GSD_law} is not supported for {self.dust_type}')  
            else:
                for gsd_law_i in gsd_law:
                    if gsd_law_i == 'mrn': 
                        log_mrn(prefix='Silicate+Carbon')
                    else: 
                        log_wd01_sil(prefix='Silicate+Carbon')
                        log_wd01_car(prefix='Silicate+Carbon')

        elif check_variable_combination(dusttype, required_parts = {"sil", "car", "pah"}):#dust_type=='sil+car':
            # if gsd_law not in ('mrn','wd01','wd01_ed'):
            if 'hd23' in gsd_law:
                raise ValueError(f'{self.GSD_law} is not supported for {self.dust_type}')  
            else:
                for gsd_law_i in gsd_law:
                    if gsd_law_i == 'mrn':
                        log_mrn(prefix='Silicate+Carbon')  
                    else: 
                        log_wd01_sil(prefix='Silicate+Carbon')
                        log_wd01_car(prefix='Silicate+Carbon')
                log.info(f'PAH: Grain-size distribution with \n\t\t B1={self.B1:.3e} and B2={self.B2:.3e}')

        elif dusttype=='astro' or check_variable_combination(dusttype, required_parts = {"astro", "pah"}):#dusttype in ('astro', 'astro+pah', 'pah+astro'):
            for gsd_law_i in gsd_law:
                if gsd_law_i == 'mrn':
                    log_mrn()
                elif gsd_law_i == 'hd23':
                    log_hd23()

            if 'pah' in dusttype:
                log.info(f'PAH: Grain-size distribution with \n\t\t B1={self.B1:.3e} and B2={self.B2:.3e}')

        elif dusttype=='pah':
            log.info(f'PAH: Grain-size distribution with \n\t\t B1={self.B1:.3e} and B2={self.B2:.3e}')

        else:
            raise ValueError(f'{self.dust_type} is not recognized!')

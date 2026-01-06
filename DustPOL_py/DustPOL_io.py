import numpy as np
import os, ast
from . import rad_func, constants 
from .constants import au,amu,Lsun
from .read import readD
from .check_vars import check_variable_name,check_variable_combination
from  .decorators import auto_refresh
from astropy import log
import importlib.resources as importlib_resources
from pathlib import Path

# Helper functions for boolean
def _to_bool(v):
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ('1','true','yes','y','on'): return True
    if s in ('0','false','no','n','off'): return False
    raise ValueError(f'Cannot coerce to bool: {v!r}')

# Helper function for float
def _to_float(v):
    if isinstance(v, (int,float)): return float(v)
    return float(ast.literal_eval(str(v)))

# Helper function to resolve output directory
def resolve_output_dir(path, base=None, create=True):
    p = Path(path if path is not None else ".")
    if base is not None:
        p = Path(base) / p
    p = p.expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return str(p)

# Resolve package root path robustly for both installed and editable/source modes
def _resolve_package_path():
    try:
        # Python 3.9+: returns a Traversable pointing at the package directory
        base = importlib_resources.files(__package__)
        return Path(base)
    except Exception:
        # Fallback to directory of this file
        return Path(__file__).resolve().parent

class WD01_GSD:
    def __init__(self,path_to_file,dusttype='silicate',INDEX=6):
        ##INDEX here is a function of Rv
        data = readD(f'{path_to_file}/data/WD01_sdist.dat',2,10)
        if dusttype=='silicate':
            self.alpha_wd01 = data[5,INDEX]
            self.beta_wd01  = data[6,INDEX]
            self.at_wd01    = data[7,INDEX]*1.E-4
            self.ac_wd01    = 1.E-5 #0.1um
            self.Cs_wd01    = data[8,INDEX]
            self.a01 = None
            self.a02 = None
            self.sigma=None
            self.B1=None
            self.B2=None
            self.params_set = [self.alpha_wd01,self.beta_wd01,self.at_wd01,self.ac_wd01,self.Cs_wd01,self.a01,self.a02,self.sigma,self.B1,self.B2]

        elif dusttype=='carbon':
            self.alpha_wd01 = data[0,INDEX]
            self.beta_wd01  = data[1,INDEX]
            self.at_wd01    = data[2,INDEX]*1.E-4
            self.ac_wd01    = data[3,INDEX]*1.E-4
            self.Cs_wd01    = data[4,INDEX]
            self.a01  = 3.5E-8
            self.a02  = 3.E-7
            self.sigma= 0.4
            self.B1   = 2.0496E-7
            self.B2   = 9.6005E-11
            self.params_set = [self.alpha_wd01,self.beta_wd01,self.at_wd01,self.ac_wd01,self.Cs_wd01,self.a01,self.a02,self.sigma,self.B1,self.B2]

        else:
            raise ValueError('dustype attributed to class WD01_GSD is either silicate or carbon!')

class Astrodust_GSD_Hensley:
    def __init__(self):
        self.BAd = 3.31e-10 #H-1
        self.a0Ad= 63.8*1e-8 #AA-->cm
        self.sigmaAd = 0.353
        self.A0 =  2.97e-5 #H-1
        self.A1 = -3.40
        self.A2 = -0.807
        self.A3 = 0.157
        self.A4 = 7.96e-3
        self.A5 = -1.68e-3
        self.params_set = [self.BAd,self.a0Ad,self.sigmaAd,self.A0,self.A1,self.A2,self.A3,self.A4,self.A5]
          
class PAH_GSD_Hensley:
    def __init__(self):
        self.B1= 7.52e-7 #H-1
        self.B2=8.09e-10 #H-1
        self.params_set=[self.B1,self.B2]

class input_params():
    _file_cache = {}  # Class-level cache to store already read files
    def __init__(self,input_file,pah_data_type, overwrites=False):            
        self.input_file = input_file
        # Use resolved package path instead of pkg_resources distribution location
        package_path = _resolve_package_path()
        self.path = str(package_path) + '/'  # keep trailing slash for existing path joins
        inputs = self.input_file
        q = np.genfromtxt(inputs,skip_header=1,dtype=None,names=['names','params'],\
        comments='!',usecols=(0,1),encoding='utf=8')
        self.input_params = q['params']
        self.output_dir   = self.input_params[0]
        self.ratd         = eval(self.input_params[1])

        ## In case of (for example: starless core), Lstar is set to 'None' or 0 or nan in the input file
        self.gamma      = eval(self.input_params[9])
        self.mean_lam   = eval(self.input_params[10])*1e-4 #cm

        self.ngas       = eval(self.input_params[11])
        self.Tgas       = eval(self.input_params[12])

        self.mgas       = 1.3*amu #90%H + 10%He
        self.dust_type  = self.input_params[13].lower()

        self.amin       = eval(self.input_params[14])*1e-4 #cm
        self.amax       = eval(self.input_params[15])*1e-4 #cm
        self.Tdust      = eval(self.input_params[16])
        self.rho        = eval(self.input_params[17])
        self.alpha      = eval(self.input_params[18])
        self.Smax       = eval(self.input_params[19])
        self.dust_to_gas_ratio=eval(self.input_params[20])
        self.GSD_law    = self.input_params[21].lower()

        self.RATalign = self.input_params[23].lower() # RAT or MRAT
        self.f_max    = eval(self.input_params[25])
        self.B_angle  = eval(self.input_params[27])*np.pi/180 # rad.
        self.Bfield   = eval(self.input_params[26])*1e-6 #Gauss

        # parameters for magnetic properties of grains
        self.Ncl    = eval(self.input_params[28]) #number of iron cluster
        self.phi_sp = eval(self.input_params[29]) #volume filling factor of iron cluster
        self.fp     = eval(self.input_params[30]) #fraction of paramagnetic atoms

        # parameters for the 2-layer model
        self.model_layer = eval(self.input_params[31])
        if isinstance(self.model_layer,int):
            if self.model_layer not in [1, 2]:
                raise ValueError(f"model_layer must be 1 or 2, got {self.model_layer} instead.")
            elif self.model_layer == 1:
                self.fheat      = None
                self.fscale     = None
                self.fscale_car = None
            else:
                self.fheat      = eval(self.input_params[32]) # fraction of the heating between two layers
                self.fscale     = eval(self.input_params[33]) # scaling factor between two layers
                self.fscale_car = eval(self.input_params[34]) # scaling factor for the carbon abundance
        else:
            raise ValueError(f"model_layer must be an integer, got {type(self.model_layer)} instead.")

        # parameters for parallel processing
        self.parallel   = eval(self.input_params[35])
        if (self.parallel):
            self.cpu     = eval(self.input_params[36])
            if (self.cpu==-1): 
                self.max_workers=os.cpu_count()#None
            else:
                if not isinstance(self.cpu,int): 
                    raise IOError('cpu number must be an integer!')
                if (self.cpu>os.cpu_count()):
                    log.warning('the input value of cpu > your cpu --> use all your cpu cores')
                    self.max_workers=os.cpu_count()
                else:
                    self.max_workers=self.cpu

        self.u_ISRF = 8.64e-13 #(ergcm-3) typical interstellar radiation field
 
        # Paramaters with special cases to handle
        self.pah_data_type = pah_data_type
        
        Lstar_val    = self.input_params[2]    
        p_val        = self.input_params[3]
        rin_val      = self.input_params[4]
        rout_val     = self.input_params[5]
        rflat_val    = self.input_params[6]
        nsample_val  = self.input_params[7]

        
        U_val = self.input_params[8]
        f_min_val = self.input_params[24]
        self.car_align = False
        self.align_func= 'L20'
        
        # --------------------------------------------------------------------------#
        if (overwrites): #<< -- to overwrite parameters via code from DustPOL_class
            self.apply_overrides(overwrites)
        # --------------------------------------------------------------------------#

        ## handle special cases for Lstar and Tstar
        if Lstar_val.lower() == 'none' or float(eval(Lstar_val)) == 0.0 or eval(Lstar_val) == np.nan:
            self.Lstar = None
            self.Tstar = None
        else:
            self.Lstar  = eval(Lstar_val)* Lsun
            self.Tstar  = self.get_Tstar()

        ## handle special cases for p   
        if p_val.lower() == 'none' or eval(p_val) == np.nan or p_val is None:
            self.p = None
        else:
            self.p = eval(p_val)               #density profile index

        ## handle special cases for rin
        if rin_val.lower() == 'none' or eval(rin_val) == np.nan or rin_val is None:
            self.rin = None
        else:
            self.rin = eval(rin_val) * au         #cm

        ## handle special cases for rout
        if rout_val.lower() == 'none' or eval(rout_val) == np.nan or rout_val is None:
            self.rout = None   
        else:
            self.rout = eval(rout_val) * au        #cm
        
        ## handle special cases for rflat
        if rflat_val.lower() == 'none' or eval(rflat_val) == np.nan or rflat_val is None:
            self.rflat = None
        else:
            self.rflat = eval(rflat_val) * au        #cm
        
        ## handle special cases for nsample
        if nsample_val.lower() == 'none' or eval(nsample_val) == np.nan or nsample_val is None:
            self.nsample = None
        else:
            self.nsample = eval(nsample_val)
            
        ## handle special cases for U
        ## if U=='Tdust': then U is estimated by the dust temperature
        if U_val.lower() == 'tdust':
            self.U = None  # dimensionless
        else:
            self.U = eval(U_val) # dimensionless
 
        ## handle special cases for f_min
        ## if f_min is set to DG, then it is calculated within the model
        if f_min_val.lower() == 'dg' or f_min_val == 'None':
            self.f_min = None
        else:
            if not isinstance(eval(f_min_val), (int, float)):
                raise ValueError('f_min must be a number!')
            self.f_min = eval(f_min_val)
     
        ## handle special case for alignment function
        if self.align_func.lower() != 'g18':
            self.pstiff = None
        else:
            self.pstiff = 1.0

        self.check_sdist()
        self.get_sdist()

        dust_type_comps=self.dust_type.split("+")  
        self.dusttype_1,self.dusttype_2,self.dusttype_3, self.dusttype_4 = (dust_type_comps + [None]*4)[:4]
        
        self.get_Qfiles()
        self.get_mass_fraction()
        

    def check_sdist(self):
        dusttype = self.dust_type.lower()
        gsd_law = self.GSD_law.lower()

        if check_variable_name(dusttype, required_parts = {'sil', 'car'}):
            if not any(gsd in gsd_law for gsd in ('mrn', 'wd01', 'wd01_ed')):
                raise ValueError(f'Grain-size distribution for {dusttype} must be MRN or WD01/WD01-ed or a combination!')
        if 'astro' in dusttype:
            if not any(gsd in gsd_law for gsd in ('mrn', 'hd23')):
                raise ValueError(f'Grain-size distribution for {dusttype} must be MRN or HD23 or a combination!')

    def reset_all_params(self):
        self.power_index= np.nan
        self.BAd = self.a0Ad = self.sigmaAd = np.nan
        self.A0 = self.A1 = self.A2 = self.A3 = self.A4 = self.A5 = np.nan
        self.B1 = self.B2 = np.nan
        # self.alpha_wd01=self.beta_wd01=self.at_wd01=self.ac_wd01=self.Cs_wd01=np.nan
        self.alpha_wd01_sil=self.beta_wd01_sil=self.at_wd01_sil=self.ac_wd01_sil=self.Cs_wd01_sil=np.nan
        self.a01_wd01_sil=self.a02_wd01_sil=self.sigma_wd01_sil=self.B1_wd01_sil=self.B2_wd01_sil=None
        self.alpha_wd01_car=self.beta_wd01_car=self.at_wd01_car=self.ac_wd01_car=self.Cs_wd01_car=np.nan
        self.a01_wd01_car=self.a02_wd01_car=self.sigma_wd01_car=self.B1_wd01_car=self.B2_wd01_car=np.nan

    def get_sdist(self):
        self.reset_all_params()
        dusttype = self.dust_type.lower()
        gsd_law = self.GSD_law.lower()

        if "pah" in dusttype:
            self.B1, self.B2 = PAH_GSD_Hensley().params_set

        for law in gsd_law.split("+"):
            if law=='hd23':
                self.BAd, self.a0Ad, self.sigmaAd,\
                self.A0, self.A1, self.A2, self.A3, self.A4, self.A5 = Astrodust_GSD_Hensley().params_set

            elif law=='mrn':
                self.power_index = eval(self.input_params[22])

            elif law=='wd01' or law=='wd01_ed':
                if 'sil' in dusttype:#dusttype=='sil':
                    self.alpha_wd01_sil,self.beta_wd01_sil,self.at_wd01_sil,self.ac_wd01_sil,self.Cs_wd01_sil,\
                    self.a01_wd01_sil,self.a02_wd01_sil,self.sigma_wd01_sil,self.B1_wd01_sil,self.B2_wd01_sil= WD01_GSD(self.path).params_set
                if 'car' in dusttype:#dusttype=='car':
                    self.alpha_wd01_car,self.beta_wd01_car,self.at_wd01_car,self.ac_wd01_car,self.Cs_wd01_car,\
                    self.a01_wd01_car,self.a02_wd01_car,self.sigma_wd01_car,self.B1_wd01_car,self.B2_wd01_car=WD01_GSD(self.path,dusttype='carbon').params_set
                # elif check_variable_combination(dusttype, required_parts = {"car", "sil"}) or check_variable_combination(dusttype, required_parts = {"car", "sil", "pah"}):
                    # self.alpha_wd01_sil,self.beta_wd01_sil,self.at_wd01_sil,self.ac_wd01_sil,self.Cs_wd01_sil,\
                    # self.a01_wd01_sil,self.a02_wd01_sil,self.sigma_wd01_sil,self.B1_wd01_sil,self.B2_wd01_sil= WD01_GSD(self.path).params_set
                    # self.alpha_wd01_car,self.beta_wd01_car,self.at_wd01_car,self.ac_wd01_car,self.Cs_wd01_car,\
                    # self.a01_wd01_car,self.a02_wd01_car,self.sigma_wd01_car,self.B1_wd01_car,self.B2_wd01_car=WD01_GSD(self.path,dusttype='carbon').params_set
            else:
                raise ValueError(f"dust-type: {self.dust_type:s} is not supported by {self.GSD_law:s} size-distribution!")


    def load_astro_data(self):
        hdr_lines=4;skip_lines=4;len_a=169;len_w=1129;num_cols=8

        data_file_key=f"Q_aAstro_{self.alpha:.3f}"
        if data_file_key in input_params._file_cache:
            Data=input_params._file_cache[data_file_key]
        else:
            file_path=f"{self.path}data/astrodust/Q_aAstro_{self.alpha:.3f}_P0.2_Fe0.00.DAT"
            Data=rad_func.readDC(file_path,hdr_lines,skip_lines,len_a,len_w,num_cols)
            input_params._file_cache[data_file_key]=Data # Cache result

        return Data

    def load_sil_data(self):
        hdr_lines = 4;skip_lines=4;len_a_sil=70;len_a_car=100;len_w=800;num_cols=8

        data_file_key=f"Q_aSil2001_{self.alpha}"
        if data_file_key in input_params._file_cache:
            Data=input_params._file_cache[data_file_key]
        else:
            if self.alpha != 0.3333:
                raise ValueError(f'Model receives the input with s={self.alpha} for silicate grain -- but the current model only accounts for s=0.3333')
            file_path = f"{self.path}data/sil_car/Q_aSil2001_{self.alpha}_p20B.DAT"
            Data = rad_func.readDC(file_path,hdr_lines,skip_lines,len_a_sil,len_w,num_cols)
            input_params._file_cache[data_file_key]=Data
        return Data

    def load_car_data(self):
        hdr_lines = 4;skip_lines=4;len_a_sil=70;len_a_car=100;len_w=800;num_cols=8

        data_file_key=f"Q_amCBE_{self.alpha}"
        if data_file_key in input_params._file_cache:
            Data = input_params._file_cache[data_file_key]
        else:
            if self.alpha != 0.3333:
                raise ValueError(f'Model receives the input with s={self.alpha} for silicate grain -- but the current model only accounts for s=0.3333')
            file_path = f"{self.path}data/sil_car/Q_amCBE_{self.alpha}.DAT"
            Data = rad_func.readDC(file_path,hdr_lines,skip_lines,len_a_car,len_w,num_cols)
            input_params._file_cache[data_file_key]=Data
        return Data

    def load_pah_data(self):
        if self.pah_data_type=='hd23':
            ##PAHs dust composition from Hensley & Draine 
            hdr_lines=4;skip_lines=4;len_a=167;len_w=1000;num_cols=4

            data_file_key=f"Q_aPAH_neutral_Hensley"
            if data_file_key in input_params._file_cache:
                Data=input_params._file_cache[data_file_key]
            else:
                file_path=f"{self.path}data/PAHs/Q_aPAH_neutral_Hensley.DAT"
                Data=rad_func.readDC(file_path,hdr_lines,skip_lines,len_a,len_w,num_cols)
                input_params._file_cache[data_file_key]=Data # Cache result

            return Data

        elif self.pah_data_type.lower()=='dl07':
            ##PAHs dust composition from DL07
            hdr_lines=4;skip_lines=4;len_a=30;len_w=800;num_cols=4

            data_file_key=f"Q_aPAHs_DL07"
            if data_file_key in input_params._file_cache:
                Data=input_params._file_cache[data_file_key]
            else:
                file_path=f"{self.path}data/PAHs/Q_aPAHs_DL07.DAT"
                if not os.path.exists(file_path):
                    raise ValueError(f'{file_path:s} is not existed!')
                Data=rad_func.readDC(file_path,hdr_lines,skip_lines,len_a,len_w,num_cols)
                input_params._file_cache[data_file_key]=Data # Cache result
            return Data

        else:
            raise ValueError(f'{self.pah_data_type:s} is not supported for PAHs!')        

    @auto_refresh
    def get_Qfiles(self):
        self.Data_Qfiles_1 = self.Data_Qfiles_2 = self.Data_Qfiles_3 = self.Data_Qfiles_4 = None
        dusttypes=[self.dusttype_1,self.dusttype_2,self.dusttype_3,self.dusttype_4]
        data_targets = ['Data_Qfiles_1', 'Data_Qfiles_2', 'Data_Qfiles_3', 'Data_Qfiles_4']

        dust_loader = {
            'pah':   self.load_pah_data, 
            'astro': self.load_astro_data,
            'sil':   self.load_sil_data,
            'car':   self.load_car_data,
        }

        for idx, dusttype in enumerate(dusttypes):
            if not dusttype:  # handles None, '', or other falsy values
                continue  # skip this one

            if dusttype in dust_loader:
                data = dust_loader[dusttype]()
                setattr(self, data_targets[idx], data)

            else:
                raise ValueError(f"Unknown dust type: {dusttype}")

    def get_mass_fraction(self):
        dusttypes=[self.dusttype_1,self.dusttype_2,self.dusttype_3,self.dusttype_4]
        dust_loader = {
            'pah':   1.0, 
            'astro': 1.0,
            'sil':   0.625,
            'car':   0.375,
        }
        for dusttype in dusttypes:
            if not dusttype:  # handles None, '', or other falsy values
                continue  # skip this one

            if dusttype in dust_loader:
                f_mass_i = dust_loader[dusttype]
                setattr(self, f'f_mass_{dusttype}', f_mass_i)

            else:
                raise ValueError(f"Unknown dust type: {dusttype}")
    
    def get_Tstar(self):
        if self.Lstar>=5.e5*Lsun and self.Lstar<=1.e6*Lsun:
            Tstar=4.e4
        elif  self.Lstar>=1.e5*Lsun and self.Lstar<5.e6*Lsun:
            Tstar=3.e4
        elif self.Lstar>=1e4*Lsun and self.Lstar<1e5*Lsun:
            Tstar=2.5e4
        elif self.Lstar>=1e3*Lsun and self.Lstar<1e4*Lsun:
            Tstar=2.e4
        elif self.Lstar>=1e2*Lsun and self.Lstar<1e3*Lsun:
            Tstar=1.e4
        elif self.Lstar>=1e1*Lsun and self.Lstar<1e2*Lsun:
            Tstar=8000.
        elif self.Lstar>=1.0*Lsun and self.Lstar<1e1*Lsun:
            Tstar=6000.
        else:
            raise ValueError(f'Lstar={self.Lstar:.2e} is out of our range!')
        return Tstar

    def apply_overrides(self, overrides: dict):
        """
        Apply a subset of DustPOL kwargs onto this params object.
        Only sets attributes that exist. Coerces common types.
        """
        # Map user kwarg names to params attribute names if needed
        alias = {
            "output_dir": "output_dir",
            "dust_type": "dust_type",
            "ratd": "ratd",
            "RATalign": "RATalign",
            "alpha": "alpha",
            "rho": "rho",
            "Lstar": "Lstar",
            "p": "p",
            "rin": "rin",
            "rout": "rout",
            "rflat": "rflat",
            "nsample": "nsample",
            "U": "U",
            "gamma": "gamma",
            "mean_lam": "mean_lam",
            "Tgas": "Tgas",
            "Tdust": "Tdust",
            "ngas": "ngas",
            "Smax": "Smax",
            "amin": "amin",
            "amax": "amax",
            "GSD_law": "GSD_law",
            "power_index": "power_index",
            "Bfield": "Bfield",
            "Ncl": "Ncl",
            "phi_sp": "phi_sp",
            "fp": "fp",
            "f_min": "f_min",
            "f_max": "f_max",
            "B_angle": "B_angle",
            "car_align": "car_align",
            "align_func": "align_func",
            "pstiff": "pstiff",
            "PAHs_data_type": "PAHs_data_type",
        }
        for k, v in overrides.items():
            if v is None:
                continue
            attr = alias.get(k)
            if not attr or not hasattr(self, attr):
                continue
            # Coerce types for a few known attributes
            if attr in ("ratd", "car_align"):
                v = _to_bool(v)
            elif attr in ("Lstar","p","rin","rout","rflat","nsample",
                           "U","gamma","mean_lam","Tgas","Tdust","ngas",
                          "Smax","amin","amax","alpha","rho","Bfield",
                          "Ncl","phi_sp","fp","power_index","B_angle"):
                v = _to_float(v)
            elif attr == "f_min":
                sval = str(v).strip().lower()
                if sval in ("dg","none",""):
                    v = None
                else:
                    v = _to_float(v)
            elif attr in ("output_dir"):
                v = resolve_output_dir(v)
            elif attr in ("dust_type","GSD_law","align_func","PAHs_data_type"):
                v = str(v).strip()
            setattr(self, attr, v)
                
class output():
    def __init__(self,parent,data):
        self.U=parent.U
        self.alpha= parent.alpha
        self.path = parent.path
        filename  = parent.outputFile
        # subpath = path+'output/starless/astrodust/'#U=%.2f'%(self.U)+'/'
        # if not os.path.exists(subpath):
        #     os.mkdir(subpath)
        # subsubpath = subpath+'U=%.2f_alpha=%.4f'%(self.U,self.alpha)+'/Av_fixed_amax/'
        # if not os.path.exists(subsubpath):
        #     os.mkdir(subsubpath)

        # subpath = 'output/'#self.path+'output/'
        out_dir = getattr(parent, 'output_dir', 'output')
        outfile = getattr(parent, 'outputFile', 'results.dat')

        self.filename = outfile if os.path.isabs(outfile) else os.path.join(out_dir, outfile)
        # self.filename=subpath+filename
        self.ngas=parent.ngas
        self.mean_lam=parent.mean_lam
        self.gamma=parent.gamma
        self.amax=parent.amax
        self.dust_type=parent.dust_type
        self.data=data
        try:
            self.Av_array=parent.Av_array
        except:
            self.Av_array=None
        try:
            self.NH_array=parent.NH_array
        except:
            self.NH_array=None
        # output_abs = path+'amax=%.2f'%(self.amax*1e4)+'_abs.dat'
        # output_emi = path+'amax=%.2f'%(self.amax*1e4)+'_emi.dat'

        self.file_save()

    def file_save(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        f=open(self.filename,'w')
        f.write('U=%.3f \n'%self.U)
        f.write('ngas=%.3e (cm-3) \n'%self.ngas)
        f.write('mean_lam=%.3f (um) \n'%(self.mean_lam*1e4))
        f.write('gamma=%.3f \n'%self.gamma)
        f.write('amax=%.3f (um) \n'%(self.amax*1e4))
        f.write('dust_composition=%s \n'%self.dust_type)
        f.write('! \n')
        # if self.Av_array is None:
        #     f.write('Av= ')
        #     f.write(",".join(str("{:.3f}".format(iAv)) for iAv in self.Av_array) + "\n")
        #     f.write('! \n')
        if self.Av_array is not None:
            if isinstance(self.Av_array,float):
                f.write('Av= %.3f'%self.Av_array + "\n")
                f.write('! \n')
            elif isinstance(self.Av_array,np.ndarray):
                f.write('Av= ')
                f.write(",".join(str("{:.3f}".format(iAv)) for iAv in self.Av_array) + "\n")
                f.write('! \n')
            
            if isinstance(self.NH_array,float):
                f.write('NH= %.5e'%self.NH_array + "\n")
                f.write('! \n')
            elif isinstance(self.NH_array,np.ndarray):
                f.write('NH= ')
                f.write(",".join(str("{:.5e}".format(iNH)) for iNH in self.NH_array) + "\n")
                f.write('! \n')
                
        #keys=sorted(data_save.keys())
        keys=list(self.data.keys())
        print('\t '.join(keys), end="\n",file=f)
        for i in range(len(self.data[keys[0]])):
            line=''
            for k in keys:
                # line=line+str(self.eformat(self.data[k][i],4,2))+'\t '
                line=line+str("{:.3e}".format(self.data[k][i]))+'\t '

            print(line,end="\n",file=f)
        f.close()

    def eformat(self,f, prec, exp_digits):
        s = "%.*e"%(prec, f)
        mantissa, exp = s.split('e')
        # add 1 to digits as 1 is taken by sign +/-
        return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

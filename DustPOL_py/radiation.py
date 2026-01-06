import numpy as np
import os,re
from astropy import log
# -------------------------------------------------------------------------
# Temperature Distribution (dPdT) Pre-calculated by DustEM code
# -------------------------------------------------------------------------
##[!Warning] we don't calculate here dPdT, but we instead tabulate dPdT for a
#set of radiation strength (U)
class radiation_retrieve():
    _Urange_tempdist_cache = {}

    def __init__(self,parent):
        self.U = parent.U
        self.path=parent.path
        
    def retrieve(self,PAHs=False):
        key = (self.path, PAHs)
        
        if key not in radiation_retrieve._Urange_tempdist_cache:
            if not PAHs:
                tempdist_path = self.path+'data/sil_car/dp_dlnT/'
            else:
                tempdist_path = self.path+'data/PAHs/dp_dlnT/'
                
            all_folders = os.listdir(tempdist_path)
            tempdist_tab  = [all_folders[i] for i in range(len(all_folders)) if 'U=' in all_folders[i]]            
            Urange_tempdist=[]
            for s in tempdist_tab:
                match =  re.search(r'TEMP_U=([0-9]*\.?[0-9]+)', s)
                if match:
                    number = float(match.group(1))
                    Urange_tempdist.append(number)
            Urange_tempdist.sort()
            radiation_retrieve._Urange_tempdist_cache[key] = Urange_tempdist
        else:
            Urange_tempdist = radiation_retrieve._Urange_tempdist_cache[key]

        ##MAKE A TRICK
        if self.U < min(Urange_tempdist):
            log.warning('*** [get dP/dT] Your value of U=%.3f < pre-computed Umin=%.3f --> \033[1;5;33m set U == %.3f \033[0m'%(self.U,min(Urange_tempdist),min(Urange_tempdist)))
            U_near=min(Urange_tempdist)
        elif self.U> max(Urange_tempdist):
            log.warning('*** [get dP/dT] Your value of U=%.3f > pre-computed Umax=%.3f --> \033[1;5;33m set U == %.3f \033[0m'%(self.U,max(Urange_tempdist),max(Urange_tempdist)))
            U_near=max(Urange_tempdist)
        else:
            U_near=self.U

        idx = abs(np.array(Urange_tempdist)-U_near).argmin()
        U_near = Urange_tempdist[idx]
        return U_near
        #
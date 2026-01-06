"""THIS ROUTINE IS FOR CHECKING THE MAP OF A_ALIGN AND AV
"""

import numpy as np 
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed#, Memory
from matplotlib.colors import LogNorm
from astropy import constants
pc = constants.pc.cgs.value

from DustPOL_py import DustPOL, isoProtostar_profile

##GLOBAL ARGUMENTS
args = DustPOL('input_template_protostar.dustpol')

model = isoProtostar_profile()
[x,y,z],radius=model.isoProtostar_model(args)

##---------------------------------------------------------------------------##
##Map of alignment size on the OXZ plane
align_=model.get_map_align(args)

fig,ax=plt.subplots(figsize=(8,8))
im = plt.imshow(align_*1e4,interpolation='bilinear',origin='lower',cmap='magma',
                extent=[x[0]/pc,x[-1]/pc,z[0]/pc,z[-1]/pc])
cbar=plt.colorbar(im,ax=ax,shrink=0.8)
cbar.set_label('$\\rm a_{align}\\, (\\mu m)$')
plt.xlabel('x/pc')
plt.ylabel('z/pc')
X, Z = np.meshgrid(x/pc, z/pc)
CS = ax.contour(X, Z, align_*1e4,levels=[0.1,0.2,0.5,0.7,1.0],colors='white')
ax.clabel(CS, inline=True, fmt='%.2f', fontsize=15)
# plt.xlim([-0.4,0.4])
# plt.ylim([-0.4,0.4])
##tick color --> white
ax.tick_params(axis='x',color='w', which='both')
ax.tick_params(axis='y',color='w', which='both')
cbar.ax.tick_params(axis='y', color='cyan', which='both', labelcolor='black')
ax.set_title('$a_{\\rm align}$ on OXZ plane')

##---------------------------------------------------------------------------##
##Av_map from surface to cell on the OXZ plane
Av_surface2cell = model.get_map_Av_surface2cell(args)

fig,ax=plt.subplots(figsize=(8,8))
im = plt.imshow(Av_surface2cell,interpolation='bilinear',origin='lower',
                cmap='magma_r',norm=LogNorm(vmin=3,vmax=1000),extent=[x[0]/pc,x[-1]/pc,y[0]/pc,y[-1]/pc])
# im = plt.imshow(Av_los,interpolation='bilinear',origin='lower',cmap='magma',extent=[x[0]/pc,x[-1]/pc,y[0]/pc,y[-1]/pc])
t=[1,10,20,50,100,300,500,1000]
cbar=plt.colorbar(im,ax=ax,ticks=t,format='%.0f',shrink=0.8)
# cbar=plt.colorbar(im,ax=ax,shrink=0.8)
cbar.set_label('$\\rm A^{surface2cell}_{V}\\, (mag.)$')
plt.xlabel('x/pc')
plt.ylabel('z/pc')
X, Z = np.meshgrid(x/pc, z/pc)
CS = ax.contour(X, Z, Av_surface2cell,levels=[20,50,100,300,500,1000],colors='white')
ax.clabel(CS, inline=True, fmt='%.0f', fontsize=15)
# plt.xlim([-0.4,0.4])
# plt.ylim([-0.4,0.4])
##tick color --> white
ax.tick_params(axis='x',color='w', which='both')
ax.tick_params(axis='y',color='w', which='both')
cbar.ax.tick_params(axis='y', color='cyan', which='both', labelcolor='black')
ax.set_title('$A^{\\rm surface2cell}_{\\rm V}$ on OXZ plane')

##---------------------------------------------------------------------------##
##Av_map on the OXY plane
Av_los = model.get_map_Av_los(args)

fig,ax=plt.subplots(figsize=(8,8))
im = plt.imshow(Av_los,interpolation='bilinear',origin='lower',cmap='magma_r',
                norm=LogNorm(vmin=3,vmax=500),extent=[x[0]/pc,x[-1]/pc,y[0]/pc,y[-1]/pc])
# im = plt.imshow(Av_los,interpolation='bilinear',origin='lower',cmap='magma',extent=[x[0]/pc,x[-1]/pc,y[0]/pc,y[-1]/pc])
# t=[3,10,20,50,100,300,500]
# cbar=plt.colorbar(im,ax=ax,ticks=t,format='%.0f',shrink=0.8)
cbar=plt.colorbar(im,ax=ax,shrink=0.8)
cbar.set_label('$\\rm A^{LOS}_{V}\\, (mag.)$')
plt.xlabel('x/pc')
plt.ylabel('y/pc')
X, Y = np.meshgrid(x/pc, y/pc)
CS = ax.contour(X, Y, Av_los,levels=[3,10,20,50,100,300,700],colors='white')
ax.clabel(CS, inline=True, fmt='%.0f', fontsize=12)
# plt.xlim([-0.4,0.4])
# plt.ylim([-0.4,0.4])
##tick color --> white
ax.tick_params(axis='x',color='black', which='both')
ax.tick_params(axis='y',color='black', which='both')
cbar.ax.tick_params(axis='y', color='cyan', which='both', labelcolor='black')
ax.set_title('$A^{\\rm LOS}_{\\rm V}$ on OXY plane')
plt.show()
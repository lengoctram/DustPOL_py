import numpy as np
from numpy import *

# read text data
def readD(filename,skipH,N_par):
    file=open(filename,'r')
    data=[]
    for ln in file:
        data.append(ln.split())
    dataf = np.zeros([N_par,len(data)-skipH])
    for j in range(skipH,len(data)):
        for i in range(N_par):
            dataf[i,j-skipH] = data[j][i]
    file.close()
    return dataf

# read text data - multile data combined
def readDC(filename,Header,skip,Nd,Nline,N_par):
    file=open(filename,'r')
    data=[]
    for ln in file:
        data.append(ln.split())
    data = data[Header:len(data)]
    dataf = np.zeros([N_par,Nline,Nd])
    for i in range(Nd):
        for j in range(Nline):
            for k in range(N_par):
                dataf[k,j,i] = data[i*(Nline+skip)+skip+j][k]
    file.close()
    return dataf
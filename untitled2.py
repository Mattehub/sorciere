# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 03:39:13 2022

@author: matte
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:10:23 2022

@author: matte
"""

import h5py


import numpy as np
import pandas as pd
from os.path import join as pjoin
from itertools import product
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
import os
import pickle
import utils_avalanches as av
import warnings 
import Utils_FC as fc
from lempel_ziv_complexity import lempel_ziv_complexity
import math
from collections import Counter
from scipy.stats import entropy

def spearmancoef(A):
    Ad=pd.DataFrame(A)
    corr_matrix=Ad.corr(method='spearman')
    return corr_matrix


def excess_entropy(data, n, overlap=True, biascorrect=True):
    
    #print(np.unique(data))
    
    
    n_regions=len(data)
    exen=np.zeros(n_regions)
    n_comb=2**n
    molt=np.zeros(n_comb)
    molt[0]=0
    molt[1]=1
    
    for i in np.arange(2,n_comb):
        molt[i]=molt[i-1]*2+1
    
    n_overlap=1
    
    if overlap:
        n_overlap=n
    
    else:
        n_overlap=1
        
    for m in range(n_overlap):
        
        dvb=data[:,m:int(len(data[0,m:])/n)*n+m].copy()
        #print(dvb[:,0:5])
        dvbb=np.reshape(dvb, (n_regions, int(len(dvb[0,:])/n),n))
        
        
        for i in range(n):
            dvbb[:,:,i]=(2**i)*dvbb[:,:,i]
        
        bbnd=np.sum(dvbb, axis=2)
        
        for i in np.arange(n_comb-1,1,-1):
            bbnd=np.where(bbnd==i, molt[i], bbnd)
            
        bbnd_diff=np.diff(bbnd, axis=1)
        
            
        if m==0:
            bnd=bbnd
            bnd_diff=bbnd_diff
        elif m!=0:
            bnd=np.concatenate((bnd, bbnd), axis=1)
            bnd_diff=np.concatenate((bnd_diff, bbnd_diff), axis=1)

        
        
    for k in range(n_regions):
        
        
        trans0=np.zeros((n_comb, n_comb))
        pp=np.zeros(n_comb)
        
        bn=bnd[k,:]
        
        
            
        
        #print(bn)
        bn_diff =bnd_diff[k,:]
            
        summy=np.zeros(n_comb)
        for i in range(n_comb):
                
            nn=len(np.where(bn==molt[i])[0])

                
            if nn!=0:
                
                #ccc=nn/len(bn)
                
                pp[i]=nn/len(bn)
                #print(nn)
                    
                for j in range(n_comb):
                    if i!=j:
                        
                        #aaa=len(np.where(bn_diff==molt[j]-molt[i])[0])/nn
                        trans0[j,i]=len(np.where(bn_diff==molt[j]-molt[i])[0])/nn
                        
                        if trans0[j,i]<0:
                            print(len(np.where(bn_diff==molt[j]-molt[i])[0]))
                            print(molt[j]-molt[i])
                            
                summy[i]=np.sum(trans0[:,i])
                #print(summy[i])
                        
        for i in range(n_comb):
            trans0[i,i]=1.0-summy[i]                  
            if trans0[i,i]<0.000000000000001:
                trans0[i,i]=0
                
        
        count=0
        
        for i in range(n_comb):
            for j in range(n_comb):
                count+=trans0[i,j]*pp[j]
                
        if count>1:
            print("Error!!!")
            print(count, k)
         
        count=0
        for i in range(n_comb):
            for j in range(n_comb):
                if trans0[i,j]!=0 and pp[i]!=0 and pp[j]!=0 :
                    count+=trans0[i,j]*pp[j]*np.log(trans0[i,j]/pp[i])
                    if math.isnan(trans0[i,j]*pp[j]*np.log(trans0[i,j]/pp[i]))==True:
                        print(trans0[i,j],pp[i])
                        print(trans0[i,j]<0)
                        print(i,j)
        #print(count)
        if biascorrect:
            count+= (n_comb -1)/len(bnd[0,:])
            
        exen[k]=count
    excess_entropy=exen/np.log(2**n)
    
    return excess_entropy

def excess_entropy_voila(data, bina, overlap=True, biascorrect=True):
    
    #print(np.unique(data))
    
    
    nregions=len(data)
    times=len(data[0,:])
    re_len=int(times/bina)*bina  
    redat=np.broadcast_to(data[np.newaxis, :, :], [bina, nregions, times])
    exx=[]
    
    for k in range(nregions):
        
        H_XY=entropy(list(Counter(zip(data[:,k,:-bina], data[:,k,bina:])).value()), base=2)
        H_X=entropy(list(Counter(data[0,k,:-bina]).values()), base=2)
        H_Y=entropy(list(Counter(data[0,k,bina:]).values()), base=2)
        
        exx.append(H_X+H_Y-H_XY)
    
    return np.array(exx)
        
datoz=np.rand((100,100))

print(excess_entropy_voila(datoz,3))

    
    
    
#In the following code: the functions to compute and study neuronal avalanches

import numpy as np
import pandas as pd
from os.path import join as pjoin
from itertools import product
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
import os
import pickle
import warnings 

def go_avalanches(data, binsize, thre=3., direc=0):
    
    #method can be simple or area. It depends on which criteria we want the algorithm to work.
    #the data is given time x regions
    
    if direc==1:
        Zb=np.where(stats.zscore(data)>thre,1,0)
    elif direc==-1:
        Zb=np.where(stats.zscore(data)<-thre,1,0)
    elif direc==0:
        Zb=np.where(np.abs(stats.zscore(data))>thre,1,0)
    else:
        print('wrong direc')
        
    nregions=len(data[0])
    
    #here we are changing the length of the Zb in such a way that can be binarized with the binsize.
    Zb=Zb[:(int(len(Zb[:,0])/binsize)*binsize),:]

    a=np.sum(Zb)/(Zb.shape[0]*Zb.shape[1])
    Zbin=np.reshape(Zb,(-1, binsize, nregions))
    Zbin=np.where(np.sum(Zbin, axis=1)>0,1,0)
    
    #The number of regions actives at each time steps
    dfb_ampl=np.sum(Zbin,axis=1).astype(float)
    #the number of regions actives at each time step, no zeros, no times where no regions are activated.
    dfb_a=dfb_ampl[dfb_ampl!=0]
    
    #for the bratio the formula used here is the exp of the mean of the log of the ratio between the number of region actives at time t_i/t_(i-1). 
    bratio=np.exp(np.mean(np.log(dfb_a[1:]/dfb_a[:-1])))
    #indices of no avalanches
    NoAval=np.where(dfb_ampl==0)[0]
    
    #here we plot the binarized matrix
    """plt.figure(figsize=(12,8))
    plt.imshow(Zbin.T[:,:1000], aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()
    plt.close()"""
    
    inter=np.arange(1,len(Zbin)+1); inter[NoAval]=0
    Avals_ranges=consecutiveRanges(inter)
    Avals_ranges=Avals_ranges[1:] #remove the first for avoiding boundary effects
    
    #plt.plot(inter[:3000])
    #plt.show()
    #plt.close()
    
    Naval=len(Avals_ranges)   #number of avalanches
    
    Avalanches={'dur':[],'siz':[],'IAI':[],'ranges':Avals_ranges[:-1],'Zbin': Zbin,'bratio':bratio, 'onespercentage': a} #duration and size
    for i in range(Naval-1): #It goes till the second last avalanche for avoiding bondaries effects
        xi=Avals_ranges[i][0];xf=Avals_ranges[i][1]; xone=Avals_ranges[i+1][0]
        Avalanches['dur'].append(xf-xi)
        Avalanches['IAI'].append(xone-xf)
        Avalanches['siz'].append(len(np.where(np.sum(Zbin[xi:xf],axis=0)>0)[0]))
        
    return Avalanches

def consecutiveRanges(a):
    n=len(a)
    length = 1;list = []
    if (n == 0):
        return list 
    for i in range (1, n + 1):
        if (i == n or a[i] - a[i - 1] != 1):
            if (length > 0):
                if (a[i - length]!=0):
                    temp = (a[i - length]-1, a[i - 1])
                    list.append(temp)
            length = 1
        else:
            length += 1
    return list

def min_siz_filt(avalanches, min_siz):
    
    #creating a new dictionary where we have only the avalanches bigger than the selected minsiz.
    dur=[]
    siz=[]
    ranges=[]
    Zbin_reduced=avalanches['Zbin'].copy()
    
    for i in range(len(avalanches['siz'])):
        #we want to select all the avalanches bigger than the minsize
        if avalanches['siz'][i]>=min_siz:
            dur.append(avalanches['dur'][i])
            siz.append(avalanches['siz'][i])
            ranges.append(avalanches['ranges'][i])
            
    #we create a new matrix where only the big avalanches are appearing 
    Zbin_reduced[:ranges[0][0],:]=0    
    Zbin_reduced[ranges[-1][1]:,:]=0
    for t in range(len(ranges)-1):
        Zbin_reduced[ranges[t][1]:ranges[t+1][0],:]=0
    
    #creating the new dictionary
    mins_avalanches={'dur':dur,'siz':siz,'IAI':[],'ranges':ranges,'Zbin':avalanches['Zbin'],'Zbin_reduced':Zbin_reduced, 'bratio':avalanches['bratio']}
    for i in range(len(ranges)-1): #It goes till the second last avalanche for avoiding bondaries effects
        xf=ranges[i][1]; xone=ranges[i+1][0]
        mins_avalanches['IAI'].append(xone-xf)
        
    return mins_avalanches
            
    
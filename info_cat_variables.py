


import h5py
import mne
import numpy as np
import scipy.stats as stats
import scipy
import pandas as pd
from os.path import join as pjoin
from itertools import product
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
import matplotlib.cm as cm
import random
import os
import pickle
import warnings 
import matplotlib
import seaborn as sns
import saving as sv
import mattepy as mp
import utils_avalanches as av
import Utils_FC as fc
from scipy.stats import entropy

def excess_entropy1(data, n, overlap=True, biascorrect=True):
    
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
                    #if math.isnan(trans0[i,j]*pp[j]*np.log(trans0[i,j]/pp[i]))==True:
                        #print(trans0[i,j],pp[i])
                        #print(trans0[i,j]<0)
                        #print(i,j)
        #print(count)
        if biascorrect:
            count+= (n_comb -1)/len(bnd[0,:])
            
        exen[k]=count
    excess_entropy=exen/np.log(2**n)
    
    return excess_entropy




def excess_entropy22(data, n, overlap=True, biascorrect=True):
    
    #print(np.unique(data))
    n_regions=len(data)
    exen=np.zeros(n_regions)
    n_comb=2**n
    molt=np.zeros(n_comb)
    molt[0]=0
    molt[1]=1

    for i in np.arange(2,n_comb):
        molt[i]=molt[i-1]*2+1
    
    #print(data.shape)
    data_newax=data[:,:,np.newaxis]
    dvb=data_newax[:,:-n,:]

    for i in np.arange(1,n,1):

        dvb=np.concatenate((dvb, data_newax[:, i:-(n-i), :]), axis=2)


        
    #print(dvb.shape) 
    for i in range(n):
        dvb[:,:,i]=(2**i)*dvb[:,:,i]
        
        bnd=np.sum(dvb, axis=2)
        
        for i in np.arange(n_comb-1,1,-1):
            bnd=np.where(bnd==i, molt[i], bnd)
            
        bnd_diff=np.diff(bnd, axis=1)

        
    for k in range(n_regions):
        
        
        trans0=np.zeros((n_comb, n_comb))
        pp=np.zeros(n_comb)
        
        bn=bnd[k,:]
        
        
            
        
        #print(bn)
        bn_diff = bnd_diff[k,:]
            
        summy=np.zeros(n_comb)
        for i in range(n_comb):
                
            nn=len(np.where(bn==molt[i])[0])

            if bn[-1]==molt[i]:
                nn-=1

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
                    
                    elif i==j:
                        trans0[i,i]=len(set(np.where(bn_diff==0)[0]).intersection(set(np.where(bn==molt[i])[0])))/nn
                            
                #summy[i]=np.sum(trans0[:,i])
                #print(summy[i])
                        
        """for i in range(n_comb):
            trans0[i,i]=1.0-summy[i]                  
            if trans0[i,i]<0.000000000000001:
                trans0[i,i]=0"""
                
        
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
                    #if math.isnan(trans0[i,j]*pp[j]*np.log(trans0[i,j]/pp[i]))==True:
                        #print(trans0[i,j],pp[i])
                        #print(trans0[i,j]<0)
                        #print(i,j)
        #print(count)
        
        if biascorrect:

            count+= (n_comb -1)/len(bnd[0,:])
            
        exen[k]=count

    excess_entropy=exen/np.log(2**n)
    
    return excess_entropy




def excess_entropy_multistates(data, n, biascorrect=True):

    n_regions=len(data)
    exen=np.zeros(n_regions)
    n_comb=2**n
    molt=np.zeros(n_comb)
    molt[0]=0
    molt[1]=1

    for i in np.arange(2,n_comb):
        molt[i]=molt[i-1]*2+1
    
    #print(data.shape)
    data_newax=data[:,:,np.newaxis]
    dvb=data_newax[:,:-n-1,:]

    for i in np.arange(1,n,1):
        print(dvb.shape, data_newax[:, i:-n-i+1, :].shape)
        dvb=np.concatenate((dvb, data_newax[:, i:-n-i+1, :]), axis=2)


        
    #print(dvb.shape) 
    for i in range(n):
        dvb[:,:,i]=(2**i)*dvb[:,:,i]
        
        bnd=np.sum(dvb, axis=2)
        
        for i in np.arange(n_comb-1,1,-1):
            bnd=np.where(bnd==i, molt[i], bnd)
    
    return excess_entropy_voila(bnd, biascorrect=biascorrect)


def excess_entropy_multistates(data, n, biascorrect=True):

    n_regions=len(data)
    exen=np.zeros(n_regions)
    n_comb=2**n
    molt=np.zeros(n_comb)
    molt[0]=0
    molt[1]=1

    for i in np.arange(2,n_comb):
        molt[i]=molt[i-1]*2+1
    
    #print(data.shape)
    data_newax=data[:,:,np.newaxis]

    for i in np.arange(1,n,1):
        print(dvb.shape, data_newax[:, i:-n-i+1, :].shape)
        dvb=np.concatenate((dvb, data_newax[:, i:-n-i+1, :]), axis=2)


        
    #print(dvb.shape) 
    for i in range(n):
        dvb[:,:,i]=(2**i)*dvb[:,:,i]
        
        bnd=np.sum(dvb, axis=2)
        
        for i in np.arange(n_comb-1,1,-1):
            bnd=np.where(bnd==i, molt[i], bnd)
    
    return excess_entropy_voila(bnd, biascorrect=biascorrect)

def excess_entropy_multistates_no_slide(data, n, biascorrect=True):

    nregions=len(data)
    n_time=len(data[0,:])
    new_n_time=int(n_time/n)*n
    n_comb=2**n
    molt=np.zeros(n_comb)
    molt[0]=0
    molt[1]=1

    for i in np.arange(2,n_comb):
        molt[i]=molt[i-1]*2+1
    
    #print(data.shape)
    data_newax=data[:, :new_n_time ,np.newaxis]
    dvb=np.reshape(data[:, :new_n_time], (nregions, int(new_n_time/n), n))

    #print(dvb.shape) 
    for i in range(n):
        dvb[:,:,i]=(2**i)*dvb[:,:,i]
        
        bnd=np.sum(dvb, axis=2)
        
        for i in np.arange(n_comb-1,1,-1):
            bnd=np.where(bnd==i, molt[i], bnd)
    
    return excess_entropy_voila(bnd, biascorrect=biascorrect)


            
    

def excess_entropy_voila(data,biascorrect=True):
    
    #data must be given in a regions x time matrix
    nregions=len(data)
    times=len(data[0,:])
    exx=np.zeros(nregions)
    for k in range(nregions):
        
        joint_data=np.concatenate((data[np.newaxis, k,:-1], data[np.newaxis, k,1:]), axis=0)
        #print(joint_data.shape)
        H_XY=entropy(np.unique(joint_data, return_counts=True, axis=1)[1])
        H_X=entropy(np.unique(data[k,:-1], return_counts=True)[1])
        H_Y=entropy(np.unique(data[k,1:], return_counts=True)[1])
        
        exx[k]=(H_X+H_Y-H_XY)
    
    return exx
        

def MI(x,y,biascorrect=True):
    
    if len(x.shape)<2:
        x=x[np.newaxis,:]
    if len(y.shape)<2:
        y=y[np.newaxis,:]
    
    
    joint_data=np.concatenate((x, y), axis=0)
    #print(joint_data.shape)
    H_XY=entropy(np.unique(joint_data, return_counts=True, axis=1)[1])
    H_X=entropy(np.unique(data[k,:-1], return_counts=True)[1])
    H_Y=entropy(np.unique(data[k,1:], return_counts=True)[1])
    
    return H_X+H_Y-H_XY

def syn_luppi(data, time_delay=1):

    n_regions, n_time = data.shape

    m=np.zeros((n_regions, n_time))

    for i in range(n_regions):
        for j in range(n_regions):
            m[i,j] = min(MI(),MI(),MI(),MI())

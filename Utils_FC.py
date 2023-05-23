#import h5py
import numpy as np
import pandas as pd
from os.path import join as pjoin
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors

import warnings 
warnings.simplefilter('ignore')



def trip(edges_arr,n):
    mat=np.zeros((n,n))
    mat[np.triu_indices(n,1)]=edges_arr
    mat=mat+mat.T+np.eye(n)
    return mat


def go_edge_list(tseries, edge_list):
    
    matrix_E=[]
    for edge in edge_list:
        i,j = edge
        E=np.multiply((tseries[i], tseries[j]))
        matrix_E.append(E)
        
    matrix_E=np.array(matrix_E)

    return(matrix_E)


def clean2(x1, N=3):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    absx=np.absolute(x)
    
    #mean and standard deviation of the absolute values in the raw data
    s=np.std(absx)
    m=np.mean(absx)
    
    #where the distance betwenn the absolute value of the activity is the mean absolute values of 
    #activities is bigger than N*std, we substitute with a nan value
    x[absx-m>N*s]=np.nan
    
    #We do the mean again without consider the nan values
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
    return x

def clean1(x1, N=5):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    #mean and standard deviation of the raw
    s=np.std(x)
    m=np.mean(x)

    #where the distance betwenn the data and the mean is bigger than N*std, we substitute with a nan value
    x[np.abs(x-m)>N*s]=np.nan
    
    #We do the mean again without consider the nan value
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
            
    return x


def compute_K(r, V):
    x = 1 + np.pi**2 * r**2 + V**2
    K = np.sqrt( (x - 2*np.pi*r ) / (x + 2*np.pi*r ) )

    return K

#Kuramoto
def Kuramotize(r,V,active=1):
    Z = np.empty_like(r+1j*r)
    
    for i in range(len(r[0])):
        Z[:,i]=(1-np.conj(np.pi*r[:,i]+1j*V[:,i]))/(1+np.conj(np.pi*r[:,i]+1j*V[:,i]))
    
    if active==1:
        Z=Z*r
    return Z

def enlarger(fmri_image):
    fmri_image = np.repeat(fmri_image, repeats=2, axis=0)
    fmri_image = np.repeat(fmri_image, repeats=2, axis=1)
    larger_fmri_image = np.repeat(fmri_image, repeats=2, axis=2)
    return larger_fmri_image

def go_edge(tseries):
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    
    return Eseries

def go_edge_names(names):
    nregions=len(names)
    iTriup= np.triu_indices(nregions,k=1) 
    Enames=[str(names[iTriup[0][i]])+ str(' - ') + str(names[iTriup[1][i]]) for i in range(len(iTriup[0]))]
    
    return Enames


def go_edge_list(tseries, edge_list):
    
    matrix_E=[]
    for edge in edge_list:
        i,j = edge
        E=np.multiply((tseries[i], tseries[j]))
        matrix_E.append(E)
        
    matrix_E=np.array(matrix_E)

    return(matrix_E)



def intervals(a):
    # Create an array of how long are all the intervals (zero sequences).
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return list(np.diff(ranges).flatten())

def compute_fcd(ts, win_len=30, win_sp=1):
    """
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
    """
    n_samples, n_nodes = ts.shape
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)

    fcs = np.array(fc_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs
    
def jn(mlist, char=None):

    """Summary

    Parameters
    ----------
    mlist : list
       List of strings to concatenate.
    join : str, optional
       Character for concatenation.

    Returns
    -------
    Concatenated string"""
    
    if char is None:
        string = '_'.join(mlist)
    else:
        string = char.join(mlist)

    return string


def flatten(nested):
    flatl = [item for sublist in nested for item in sublist]
    empty_l = [np.asarray(flatl).shape[i]
               for i in range(np.asarray(flatl).ndim)]

    tup = tuple(empty_l)
    flat_list = np.array(flatl).reshape(tup)
    
    return flat_list
    
    
def clean(x1, N=3):
    
    x=x1.copy()
    s=np.std(x)

    #x[x>N*s+np.mean(x)]=N*s+np.mean(x)
    #x[x<-N*s+np.mean(x)]=-N*s+np.mean(x)

    x[x>N*s]=N*s
    x[x<-N*s]=-N*s

            
    return x

def shifting(x, n=None):
    
    if n==None:
        n=np.random.choice(range(len(x)))
    
    x_new=np.concatenate((x[n:],x[:n]))
    
    return x_new

def shifting_matrix(A, n_list=None):
    
    A_new=A.copy()
    
    if n_list==None:
        for i in range(len(A)):
            A_new[i,:]=shifting(A[i,:], n=None)
        
    else:
        for i in range(len(A)):
            A_new[i,:]=shifting(A[i,:], n=n_list[i])
    
    return A_new



def clean2(x1, N=3):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    absx=np.absolute(x)
    
    #mean and standard deviation of the absolute values in the raw data
    s=np.std(absx)
    m=np.mean(absx)
    
    #where the distance betwenn the absolute value of the activity is the mean absolute values of 
    #activities is bigger than N*std, we substitute with a nan value
    x[absx-m>N*s]=np.nan
    
    #We do the mean again without consider the nan values
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
    return x

def clean1(x1, N=5):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    #mean and standard deviation of the raw
    s=np.std(x)
    m=np.mean(x)

    #where the distance betwenn the data and the mean is bigger than N*std, we substitute with a nan value
    x[np.abs(x-m)>N*s]=np.nan
    
    #We do the mean again without consider the nan value
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
            
    return x
    


    

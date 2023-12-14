#In this code, one function that allow to compute the edges time series from the time series of brain activity.

import h5py
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
    

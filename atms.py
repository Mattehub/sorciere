#In this code the functions to compute the activity transition matrices from the binarized data

import numpy as np

def transprob(aval):  # (t,r)
    nregions = len(aval.T)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN):  # (t,r)
    nregions = len(ZBIN.T)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > 2:
            mat += transprob(ZBIN[aout[iaut]])
            ifi += 1
    mat = mat / ifi
    return mat

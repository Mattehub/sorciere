
import h5py
import mne
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
#import Utils_FC as fc
#from lempel_ziv_complexity import lempel_ziv_complexity

from sklearn.manifold import SpectralEmbedding
from sklearn import manifold
import sksfa
from Montbrio_net_dimred_func_analysis import *

warnings.simplefilter('ignore')

#%%

path='/home/orlando/Documents/PhD project/data_dcp/'
#path='C:/Users/matte/OneDrive/Documenti/matteo/'
#CREATING THE LIST OF SUBJECTS

sound_list=['rest','music','speech']
arr_mu = os.listdir(path +'seeg_fif_data/music')
arr_rest = os.listdir(path +'seeg_fif_data/speech')
arr_speech = os.listdir(path +'seeg_fif_data/rest')

subject_set_mu=set()
subject_set_speech=set()
subject_set_rest=set()

for st in arr_mu: 
    #print(st)
    subject_set_mu.add(st.partition('_')[0])
    #print(st.partition('_')[0])
    
for st in arr_speech:
    subject_set_speech.add(st.partition('_')[0])
    
for st in arr_rest:
    subject_set_rest.add(st.partition('_')[0])

subject_list=list(subject_set_mu.intersection(subject_set_speech,subject_set_rest))

#Here I create a set of the  all channels
total_channels_set=set()

for subject in subject_list:
    with h5py.File(pjoin(path +'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        #print(f.keys())
        #print('chnames', f['chnames'].shape)
        
        chnames = f['chnames'][...].astype('U')
        total_channels_set.update(chnames)
        
#print(total_channels_set)


#Here I create a set of the H channels
ch_H=set()
for ch in total_channels_set:
    
    if "H" in ch:
        ch_H.add(ch)
        
ch_IM=set()
for ch in total_channels_set:
    
    if "IP" in ch:
        ch_IM.add(ch)
#print(ch_H)


#PARAMETERS

subject_list=subject_list


size_rest=[]
size_speech=[]
size_music=[]

mean_vector_speech=[]
mean_vector_music=[]
mean_vector_rest=[]

subject_iai_speech=[]
subject_iai_music=[]
subject_iai_rest=[]

fc_dict={}

final_channels_without_H={}

final_channels_H={}

final_channels_all={}

rss_speech=[]
rss_music=[]
rss_rest=[]

parameter_rest=[]
parameter_music=[]
parameter_speech=[]


#%%

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

#Transprob(ZBIN[i, ...].data.T)
    
#%%

trM = np.zeros((len(subject_list),3),dtype=object)
trM_sy = np.zeros((len(subject_list),3),dtype=object)

zb = {}
zb['speech'] = {}
zb['music'] = {}
zb['rest'] = {}

canali= {}
canali['H']={}
canali['w_H']={}

for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    
    fc_dict[subject]={}
    
    with h5py.File(pjoin(path+'high_gamma_down100_env/', subject + '_seeg_preproc.hdf5'), 'r') as f:

        data_m=f['music'][...]
        data_s=f['speech'][...]
        data_r=f['rest'][...]


    with h5py.File(pjoin(path+ 'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        
        chnames = f['chnames'][...].astype('U')

    with h5py.File(pjoin(path + 'seeg_data_h5py/h5_misc/', subject + '_misc.hdf5'), 'r') as f:
        
        bad_chans = f['outlier_chans']['strict_bads_names'][...].astype('U')
        mu_bad_epo = f['outlier_epochs']['music']['strict_bads_epochs'][...]
        sp_bad_epo = f['outlier_epochs']['speech']['strict_bads_epochs'][...]

## Cleaning from artifacts

    ch_i = [i for i, ch in enumerate(chnames) if ch in bad_chans]
    
    clean_chnames = [ch for i, ch in enumerate(chnames) if ch not in bad_chans]
    
    clean_music = np.delete(data_m, ch_i, axis=0)
    clean_speech = np.delete(data_s, ch_i, axis=0)
    clean_rest = np.delete(data_r, ch_i, axis=0)

#selecting only the channels we want, in this script H
    
    ch_H_i= [i for i, ch in enumerate(clean_chnames) if ch in ch_H]
    ch_H_w_i= [i for i, ch in enumerate(clean_chnames) if ch not in ch_H]
    
    canali['H'][subject]=ch_H_i
    canali['w_H'][subject]=ch_H_w_i

    final_channels_without_H[subject]=[ch for i, ch in enumerate(clean_chnames) if i not in ch_H_i]
    final_channels_H[subject]=[ch for i, ch in enumerate(clean_chnames) if i in ch_H_i]
    final_channels_all[subject]=clean_chnames
    
    final_channels=clean_chnames
    
    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    clean_music_without_H = np.delete(clean_music, ch_H_w_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_H_w_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_H_w_i, axis=0)
    
    clean_speech=clean_speech
    clean_music=clean_music
    clean_rest=clean_rest

    zb['speech'][subject]=av.go_avalanches(clean_speech.T,  direc=0, binsize=1)['Zbin']
    zb['music'][subject] =av.go_avalanches(clean_music.T,  direc=0, binsize=1)['Zbin']
    zb['rest'][subject]=av.go_avalanches(clean_rest.T,  direc=0, binsize=1)['Zbin']
    
    #fig_par('broad_band',subject,'all')
    
    
    trm_speech = Transprob(zb['speech'][subject])
    trM[isub,0]=trm_speech
    trM_sy[isub,0]=(trm_speech+trm_speech.T)*0.5
   
    trm_music = Transprob(zb['music'][subject])
    trM[isub,1]=trm_music
    trM_sy[isub,1]=(trm_music+trm_music.T)*0.5
 
    trm_rest = Transprob(zb['rest'][subject])
    trM[isub,2]=trm_rest
    trM_sy[isub,2]=(trm_rest+trm_rest.T)*0.5



# %%

#%%

for d in distances:
    
    for isub,subject in enumerate(subject_list):
        
        plt.plot([dist_sp_re[subject][d], dist_re_mu[subject][d], dist_sp_mu[subject][d]])
    plt.title(d)
    plt.xticks([0,1,2], ['sp_re', 'mu_re', 'sp_mu'])
    plt.xlabel('condition')
    plt.show()
    plt.close()

#%%

for d in distances:
    
    for isub,subject in enumerate(subject_list):
        ""
        plt.plot([dist_sp_re[subject][d], dist_re_mu[subject][d], dist_sp_mu[subject][d]])
    plt.title(d)
    plt.xticks([0,1,2], ['spech_rest', 'music_rest', 'speech_music'])
    plt.xlabel('condition')
    plt.show()
    plt.close()


#%% without cdist

from scipy.spatial import distance
from scipy.stats import pearsonr,zscore

dist_sp_mu = {}
dist_sp_re = {}
dist_re_mu = {}

dist_s_m = []
dist_s_r = []
dist_m_r = []

for isub,subject in enumerate(subject_list):
    
    dist_sp_mu[subject] = {}
    dist_sp_re[subject] = {}
    dist_re_mu[subject] = {}
    
    
    sp = trM_sy[isub,0]#
    mu = trM_sy[isub,1]#
    re = trM_sy[isub,2]#
    
    # sp = zscore(np.triu(0.5*(sp+sp.T),0).flatten())
    # mu = zscore(np.triu(0.5*(mu+mu.T),0).flatten())
    # re = zscore(np.triu(0.5*(re+re.T),0).flatten())

    sp = zscore(sp.flatten())
    mu = zscore(mu.flatten())
    re = zscore(re.flatten())

    dist_sp_mu[subject]['euclidean'] =distance.euclidean(sp,mu)
    dist_sp_re[subject]['euclidean'] =distance.euclidean(sp,re)
    dist_re_mu[subject]['euclidean'] =distance.euclidean(mu,re)

    dist_sp_mu[subject]['cosine'] =distance.euclidean(sp,mu)
    dist_sp_re[subject]['cosine'] =distance.euclidean(sp,re)
    dist_re_mu[subject]['cosine'] =distance.euclidean(mu,re)
    
    dist_sp_mu[subject]['correlation'] = 1 - abs(pearsonr(sp,mu)[0])
    dist_sp_re[subject]['correlation'] = 1 - abs(pearsonr(sp,re)[0])
    dist_re_mu[subject]['correlation'] = 1 - abs(pearsonr(mu,re)[0])
    
    dist_s_m.append(1 - abs(pearsonr(sp,mu)[0]))
    dist_s_r.append(1 - abs(pearsonr(sp,re)[0]))
    dist_m_r.append(1 - abs(pearsonr(mu,re)[0]))
    
dist_s_m=np.array(dist_s_m)
dist_s_r=np.array(dist_s_r)
dist_m_r=np.array(dist_m_r)

dist_s_m_mean=np.mean(dist_s_m)
dist_s_r_mean=np.mean(dist_s_r)
dist_m_r_mean=np.mean(dist_m_r)


#%%

ssmm=[]
ssrr=[]
mmrr=[]

for i in range(0,10000):
    
    s_m=[]
    s_r=[]
    m_r=[]
    
    for isub,subject in enumerate(subject_list):
        
        dist_sp_mu[subject] = {}
        dist_sp_re[subject] = {}
        dist_re_mu[subject] = {}
        
        
        sp = trM_sy[isub,0]#
        mu = trM_sy[isub,1]#
        re = trM_sy[isub,2]#
        
        sp = zscore(sp.flatten())
        mu = zscore(mu.flatten())
        re = zscore(re.flatten())
        
        aa = np.random.randint(0,len(sp),len(sp))
        new_sp_m = np.concatenate((sp[aa[0:int(len(aa)*0.5)]],mu[aa[int(len(aa)*0.5):len(aa)]]))
        new_sp_r = np.concatenate((sp[aa[0:int(len(aa)*0.5)]],re[aa[int(len(aa)*0.5):len(aa)]]))
        new_mu_s = np.concatenate((mu[aa[0:int(len(aa)*0.5)]],sp[aa[int(len(aa)*0.5):len(aa)]]))
        new_mu_r = np.concatenate((mu[aa[0:int(len(aa)*0.5)]],re[aa[int(len(aa)*0.5):len(aa)]]))
        new_re_s = np.concatenate((re[aa[0:int(len(aa)*0.5)]],sp[aa[int(len(aa)*0.5):len(aa)]]))
        new_re_m = np.concatenate((re[aa[0:int(len(aa)*0.5)]],mu[aa[int(len(aa)*0.5):len(aa)]]))

        s_m.append(1 - abs(pearsonr(new_sp_m,new_mu_s)[0]))
        s_r.append(1 - abs(pearsonr(new_sp_r,new_re_s)[0]))
        m_r.append(1 - abs(pearsonr(new_mu_r,new_re_m)[0]))

    ssmm.append(s_m)
    ssrr.append(s_r)
    mmrr.append(m_r)

#%%

ssmm=np.array(ssmm)
ssrr=np.array(ssrr)
mmrr=np.array(mmrr)

for isub,subject in enumerate(subject_list):
    plt.hist(mmrr[:,isub], bins=30)
    plt.axvline(dist_sp_re[subject]['correlation'])
    plt.title('mu-re')
    #plt.show()
    plt.close()
    

plt.hist(np.mean(mmrr, axis=1), bins=30)
plt.axvline(dist_m_r_mean)
plt.title('sp-re')
plt.show()
plt.close()

#%%

distances = ['correlation']

for d in distances:
    
    dist_sp_mu1 = []
    dist_sp_re1 = []
    dist_re_mu1 = []
    
    for isub,subject in enumerate(subject_list):
        ""
        dist_sp_mu1.append(dist_sp_mu[subject][d])
        dist_sp_re1.append(dist_sp_re[subject][d])
        dist_re_mu1.append(dist_re_mu[subject][d])

    fig, ax = plt.subplots()
    data=[dist_sp_mu1,dist_re_mu1,dist_sp_re1]
# build a violin plot
    ax.violinplot(data, showmeans=False, showmedians=True)
    #ax.set_ylim(0,0.2)
    plt.title(d)
    plt.xticks([1,2,3], ['speech_rest', 'music_rest', 'speech_music'])
    #plt.xlabel('condition')
    plt.show()
    plt.close()

#%%

import scipy

scipy.stats.wilcoxon(dist_sp_mu1,dist_sp_re1)


#%% ELEMENTI MISTI H - noH

from scipy.spatial import distance
from scipy.stats import pearsonr,zscore
from itertools import product

dist_sp_mu = {}
dist_sp_re = {}
dist_re_mu = {}


for isub,subject in enumerate(subject_list):
    
    dist_sp_mu[subject] = {}
    dist_sp_re[subject] = {}
    dist_re_mu[subject] = {}
    
    indici_mix  = list(product(canali['H'][subject], canali['H'][subject]))
    idx_i, idx_j = zip(*indici_mix)
    
    sp = trM[isub,0][idx_i,idx_j]
    mu = trM[isub,1][idx_i,idx_j]#
    re = trM[isub,2][idx_i,idx_j]#

    dist_sp_mu[subject]['euclidean'] = distance.euclidean(sp,mu)
    dist_sp_re[subject]['euclidean'] = distance.euclidean(sp,re)
    dist_re_mu[subject]['euclidean'] = distance.euclidean(mu,re)

    dist_sp_mu[subject]['cosine'] = distance.euclidean(sp,mu)
    dist_sp_re[subject]['cosine'] = distance.euclidean(sp,re)
    dist_re_mu[subject]['cosine'] = distance.euclidean(mu,re)
    
    dist_sp_mu[subject]['correlation'] = 1 - abs(pearsonr(sp,mu)[0])
    dist_sp_re[subject]['correlation'] = 1 - abs(pearsonr(sp,re)[0])
    dist_re_mu[subject]['correlation'] = 1 - abs(pearsonr(mu,re)[0])


#%% STATISTICHE H


from scipy.spatial import distance
from scipy.stats import pearsonr,zscore

dist_sp_mu = {}
dist_sp_re = {}
dist_re_mu = {}

dist_s_m = []
dist_s_r = []
dist_m_r = []

for isub,subject in enumerate(subject_list):
    
    dist_sp_mu[subject] = {}
    dist_sp_re[subject] = {}
    dist_re_mu[subject] = {}
    
    indici_mix  = list(product(canali['H'][subject], canali['w_H'][subject]))
    idx_i, idx_j = zip(*indici_mix)
    
    sp = trM[isub,0][idx_i,idx_j]
    mu = trM[isub,1][idx_i,idx_j]#
    re = trM[isub,2][idx_i,idx_j]#
    
    # sp = zscore(np.triu(0.5*(sp+sp.T),0).flatten())
    # mu = zscore(np.triu(0.5*(mu+mu.T),0).flatten())
    # re = zscore(np.triu(0.5*(re+re.T),0).flatten())

    sp = zscore(sp.flatten())
    mu = zscore(mu.flatten())
    re = zscore(re.flatten())

    dist_sp_mu[subject]['euclidean'] =distance.euclidean(sp,mu)
    dist_sp_re[subject]['euclidean'] =distance.euclidean(sp,re)
    dist_re_mu[subject]['euclidean'] =distance.euclidean(mu,re)

    dist_sp_mu[subject]['cosine'] =distance.euclidean(sp,mu)
    dist_sp_re[subject]['cosine'] =distance.euclidean(sp,re)
    dist_re_mu[subject]['cosine'] =distance.euclidean(mu,re)
    
    dist_sp_mu[subject]['correlation'] = 1 - abs(pearsonr(sp,mu)[0])
    dist_sp_re[subject]['correlation'] = 1 - abs(pearsonr(sp,re)[0])
    dist_re_mu[subject]['correlation'] = 1 - abs(pearsonr(mu,re)[0])
    
    dist_s_m.append(1 - abs(pearsonr(sp,mu)[0]))
    dist_s_r.append(1 - abs(pearsonr(sp,re)[0]))
    dist_m_r.append(1 - abs(pearsonr(mu,re)[0]))
    
dist_s_m=np.array(dist_s_m)
dist_s_r=np.array(dist_s_r)
dist_m_r=np.array(dist_m_r)

dist_s_m_mean=np.mean(dist_s_m)
dist_s_r_mean=np.mean(dist_s_r)
dist_m_r_mean=np.mean(dist_m_r)


#%%

ssmm=[]
ssrr=[]
mmrr=[]

for i in range(0,10000):
    
    s_m=[]
    s_r=[]
    m_r=[]
    
    for isub,subject in enumerate(subject_list):
        
        dist_sp_mu[subject] = {}
        dist_sp_re[subject] = {}
        dist_re_mu[subject] = {}
        
        
        indici_mix  = list(product(canali['H'][subject], canali['w_H'][subject]))
        idx_i, idx_j = zip(*indici_mix)
        
        sp = trM[isub,0][idx_i,idx_j]
        mu = trM[isub,1][idx_i,idx_j]#
        re = trM[isub,2][idx_i,idx_j]#
        
        sp = zscore(sp.flatten())
        mu = zscore(mu.flatten())
        re = zscore(re.flatten())
        
        aa = np.random.randint(0,len(sp),len(sp))
        new_sp_m = np.concatenate((sp[aa[0:int(len(aa)*0.5)]],mu[aa[int(len(aa)*0.5):len(aa)]]))
        new_sp_r = np.concatenate((sp[aa[0:int(len(aa)*0.5)]],re[aa[int(len(aa)*0.5):len(aa)]]))
        new_mu_s = np.concatenate((mu[aa[0:int(len(aa)*0.5)]],sp[aa[int(len(aa)*0.5):len(aa)]]))
        new_mu_r = np.concatenate((mu[aa[0:int(len(aa)*0.5)]],re[aa[int(len(aa)*0.5):len(aa)]]))
        new_re_s = np.concatenate((re[aa[0:int(len(aa)*0.5)]],sp[aa[int(len(aa)*0.5):len(aa)]]))
        new_re_m = np.concatenate((re[aa[0:int(len(aa)*0.5)]],mu[aa[int(len(aa)*0.5):len(aa)]])) 

        s_m.append(1 - abs(pearsonr(new_sp_m,new_mu_s)[0]))
        s_r.append(1 - abs(pearsonr(new_sp_r,new_re_s)[0]))
        m_r.append(1 - abs(pearsonr(new_mu_r,new_re_m)[0]))

    ssmm.append(s_m)
    ssrr.append(s_r)
    mmrr.append(m_r)

#%%

ssmm=np.array(ssmm)
ssrr=np.array(ssrr)
mmrr=np.array(mmrr)

for isub,subject in enumerate(subject_list):
    plt.hist(ssrr[:,isub], bins=30)
    plt.axvline(dist_sp_re[subject]['correlation'])
    plt.title('mu-re')
    #plt.show()
    plt.close()
    

plt.hist(np.mean(mmrr, axis=1), bins=30)
plt.axvline(dist_m_r_mean)
plt.title('music-rest no H')
plt.show()
plt.close()


#%%

C1 = {'s': '#93cb90', 'm': '#ffa96b', 'r': '#6da3cc'}
#C2 = {     'sr': '#803b7f',    'mr': '#c0896f',    'sm': '#b6d12d'}
plt.figure(figsize=(2.4*1.6,2.8*1.25))

#plt.axhline(0,alpha=0.3, linestyle=':',color='k')
box_plot_data=[dist_sp_re1,dist_re_mu1,dist_sp_mu1]#[left_increase_list['speech']-right_increase_list['speech'],left_increase_list['music']-right_increase_list['music']]
for i in range(len(dist_sp_mu1)):
    spmu=dist_sp_mu1[i]
    remu=dist_re_mu1[i]
    spre=dist_sp_re1[i]

    plt.scatter(1,spre,color='c',alpha=.5)#color=C2['sr']
    plt.scatter(2,remu,color='r',alpha=.5)#color=C2['mr'],
    plt.scatter(3,spmu,color='r',alpha=.5)#color=C2['sm']

    # plt.scatter(1,spre,color=C1['s'],alpha=.5)
    # plt.scatter(2,remu,color=C1['m'],alpha=.5)
    # plt.scatter(3,spmu,color=C1['m'],alpha=.9)
    # plt.scatter(1,spre,color=C1['r'],alpha=.5)
    # plt.scatter(2,remu,color=C1['r'],alpha=.5)
    # plt.scatter(3,spmu,color=C1['s'],alpha=.5)
    if spre-remu>0:
        plt.plot([1,2],[spre,remu],color='grey',alpha=0.4)
    else:
        plt.plot([1,2],[spre,remu],color='grey',alpha=0.4,linestyle='--')

    if spmu-remu>0:
        plt.plot([3,2],[spmu,remu],color='grey',alpha=0.4)
    else:
        plt.plot([3,2],[spmu,remu],color='grey',alpha=0.4,linestyle='--')

plt.boxplot(box_plot_data,patch_artist=False,labels=['d(speech,rest)','d(music,rest)','d(speech,music)'])
psrmr=stats.wilcoxon(dist_sp_re1,dist_re_mu1)[1]
psrsm=stats.wilcoxon(dist_sp_re1,dist_sp_mu1)[1]
pmrsm=stats.wilcoxon(dist_re_mu1,dist_sp_mu1)[1]

# p=Gtest(left_increase_list['speech']-right_increase_list['speech'],left_increase_list['music']-right_increase_list['music'])
# print('Gio test  ', p)
# print(stats.wilcoxon(left_increase_list['speech']-right_increase_list['speech'],left_increase_list['music']-right_increase_list['music']))
# print('p_value music - speech ', pp)
# print('p_value music - rest ', psr)
# print('p_value music - rest ', pmr)

to_min1=np.array([dist_sp_re1,dist_sp_mu1,dist_re_mu1])
bottom1, top1 = np.min(to_min1), np.max(to_min1)
bars_diff(psrmr, bottom1, top1,height=2.2)
bars_diff(psrsm, bottom1, top1,x1=1,x2=3,height=3.7)
bars_diff(pmrsm, bottom1, top1,x1=2,x2=3,height=1.25)
plt.tight_layout(pad=0.35)
plt.ylim((0,0.52))

# plt.savefig('left-right_increase_with_rest.pdf', dpi=600)
# plt.show()
# plt.close()

#%%

def bars(p, bottom, top):
    # Get info about y-axis
    yrange = top - bottom

    # Columns corresponding to the datasets of interest
    x1 = 1
    x2 = 2
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (yrange * 0.08 * level) + top
    bar_tips = bar_height - (yrange * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level

    if p < 0.001:
        sig_symbol = '*'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = ''
    text_height = bar_height + (yrange * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
    plt.ylim((bottom-top/8, top+top/4))

def bars_diff(p, bottom, top,x1=1, x2=2, height=1):
    # Get info about y-axis
    yrange = top - bottom

    # Columns corresponding to the datasets of interest
    
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (yrange * 0.08 * level)*height + top
    bar_tips = bar_height - (yrange * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level

    if p < 0.001:
        sig_symbol = '*'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = ''
    text_height = bar_height + (yrange * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
    #plt.ylim((-0.24,0.44))

    
def Gtest(a,b,N=5000):
    nsubs=len(a)
    diff=a-b
    diff_bin=np.where(diff>0,1,0)
    observed=np.sum(diff_bin)
    pool=[]
    for i in range(N):
        pool.append(np.sum(np.random.randint(2, size=nsubs)))
    if observed<np.mean(pool):
        p=np.sum(np.asarray(pool)<observed)/N
    else:
        p=np.sum(np.asarray(pool)>observed)/N
    return p

#%%

from scipy import stats

stats.kstest([dist_sp_mu[subject]['correlation'] for subject in subject_list],[dist_re_mu[subject]['correlation'] for subject in subject_list])




#%% VIOLIN

import seaborn as sns 
    
distances = ['euclidean', 'cosine', 'correlation']

for d in distances:
    
        ""
    sns.violinplot([dist_sp_re[subject][d], dist_re_mu[subject][d], dist_sp_mu[subject][d]])
    plt.title(d)
    plt.xticks([0,1,2], ['speech_rest', 'music_rest', 'speech_music'])
    #plt.xlabel('condition')
    plt.show()
    plt.close()
    
    

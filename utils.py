
import numpy as np
import matplotlib.pyplot as plt

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
    
    return np.array(A_new)


def split_hemis(chlist):
    
    left_names, right_names = [], []
    left_index, right_index = [], []
    for i, ch in enumerate(chlist):
        if "'" in ch:
            left_names.append(ch)
            left_index.append(i)
        else:
            right_names.append(ch)
            right_index.append(i)
    
    return(left_names, right_names, left_index, right_index)
            
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

def extract_amplitude(
        raw, # fif file from mne
        freqs, # these were the bands between 80-120
        normalize=True,
        n_hilbert=None,
        picks=None,
        n_jobs=-1):
    """Extract high gamma amplitude in several bands.
    Returns
    -------
    MNE raw instance high gamma amplitude filtered, plotting the envelope and the activity in a certain band.
    """
    from mne.io import Raw

    n_hilbert = raw.n_times if n_hilbert is None else n_hilbert

    # make sure total samples to power of 2, otherwise super slow
    if n_hilbert == 'auto':
        n_hilbert = int(2 ** np.ceil(np.log2(raw.n_times)))
    n_hilbert = int(n_hilbert)
    freqs = np.atleast_2d(freqs)
    picks = range(len(raw.ch_names)) if picks is None else picks

    # filter for hfa and extract amplitude
    bands = np.zeros([freqs.shape[0], len(raw.ch_names), raw.n_times])
    for i, (fmin, fmax) in enumerate(freqs):
        # lg.info('processing between %s and %s, %s filters left to go' %
        #         (fmin, fmax, len(freqs) - (i + 1)))
        raw_band = raw.copy()
        raw_band.filter(
            fmin,
            fmax,
            phase='zero-double',
            filter_length='auto',
            n_jobs=n_jobs)
        plt.plot(raw_band.get_data()[0,:100])
        raw_band.apply_hilbert(
            picks,
            n_fft=n_hilbert,
            envelope=True,
            n_jobs=n_jobs)
        plt.plot(raw_band.get_data()[0,:100])
        plt.show()
        plt.close()        

        if normalize is True:
            # Scale frequency band so that the ratios of all are the same
            raw_band_mn = raw_band._data.mean(1)[:, np.newaxis]
            raw_band._data /= raw_band_mn
        bands[i] = raw_band._data.copy()
    
    # Average across fbands
    raw._data[picks, :] = bands.mean(axis=0)
    
    plt.plot(bands.mean(axis=0)[0,:100])


    return raw
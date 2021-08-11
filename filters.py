# all kind of filters for time traces
# Fra 2015

# WARNING: spectra plotted here use a function here that is OBSOLETE. Do not trust the plots quantitively.
# TODO :  use scipy.signal.filtfilt instead of to apply the Butterworth filter. filtfilt is the forward-backward filter. It applies the filter twice, once forward and once backward, resulting in zero phase delay.

import numpy as np
from scipy.signal import butter, lfilter, freqz, bode, medfilt, filtfilt
import matplotlib.pyplot as plt
import scipy.ndimage.filters




def rm_interpolate(sig, p0=None, p1=None, pts=10, mode='spline', plot_signame='', plots=False):                                                                                         
    ''' remove interpolation made of pts from sig[p0:p1] 
        mode = 'spline' or 'linear' 
        see also: rm_interpolate_xy()
    '''
    # crop sig:
    sig = sig[p0:p1]
    # pick up pts points along sig by avg:
    idxs = np.linspace(0, len(sig)-1, pts).astype(int)
    sigpts = np.array([np.median(sig[i:i+np.diff(idxs)[0]]) for i in idxs])
    sigidx = range(0, len(sig))
    if mode == 'spline':
        from scipy.interpolate import splev
        from scipy.interpolate import splrep
        # interpolate by spline:
        spline = splrep(idxs, sigpts)
        interp = splev(sigidx, spline)
        # remove interpolation from sig :
        sig_out = sig - interp + sigpts[0]
    elif mode == 'linear':
        from scipy.signal import detrend
        sig_out = detrend(sig) + sigpts[0]
        interp = sig - sig_out + sigpts[0]
    if plots:
        if len(sig) > 1000000:
            dw = 10
        else:
            dw = 1
        plt.figure('rm_interpolate '+plot_signame, clear=True)
        plt.plot(sigidx[::dw], sig[::dw], label='raw data')
        plt.plot(idxs, sigpts, 'o', mfc='none')
        plt.plot(sigidx[::dw], interp[::dw], label='interpolation')
        plt.plot(sigidx[::dw], sig_out[::dw], alpha=0.5, label='corrected')
        plt.ylabel('sig', fontsize=14)
        plt.xlabel('index', fontsize=14)
        plt.legend()
    return sig_out



def rm_interpolate_xy(x, y, p0=None, p1=None, add='mean', npts=100, plots=False):
    ''' remove from y the interpolation made of "npts" points along x 
        like rm_interpolate, but x,y in input'''
    from scipy.interpolate import interp1d
    x = x[p0:p1]
    y = y[p0:p1]
    x_mn, y_mn, _, _ = binavg_xydata(x,y, nbins=npts, plots=False)
    f = interp1d(x_mn, y_mn, fill_value='extrapolate')
    if add == 'mean':
        _add = np.mean(y)
    elif add == 'max':
        _add = np.max(f(x))
    y_corr = y - f(x) + _add 
    if plots:
        plt.figure('rm_interpolate', clear=True)
        plt.subplot(211)
        plt.plot(x, y, ',', alpha=0.1, label='orig')
        plt.plot(x_mn, y_mn, '.', label='chosen')
        plt.plot(x_mn, f(x_mn), '-', label='interp.')
        plt.legend()
        plt.subplot(212)
        plt.plot(x, y_corr, ',', alpha=0.1, label='corr.')
        plt.legend()
    return y_corr, f(x)



def binavg_xydata(x, y, nbins=10, plots=False):  
    ''' average xy data on nibns bins '''
    bins = np.linspace(np.min(x), np.max(x), nbins+1, endpoint=True)
    idxs = np.digitize(x, bins)
    x_mn = []; x_sd = []
    y_mn = []; y_sd = []
    for i in range(1, len(bins)):
        if len(x[idxs == i]):
            x_mn = np.append(x_mn, np.nanmedian(x[idxs == i]))
            x_sd = np.append(x_sd, np.nanstd(x[idxs == i]))
            y_mn = np.append(y_mn, np.nanmedian(y[idxs == i]))
            y_sd = np.append(y_sd, np.nanstd(y[idxs == i]))
    if plots:
         plt.figure('bin_xydata', clear=True)
         plt.plot(x, y, 'o', ms=4, alpha=0.5, zorder=1)
         plt.errorbar(x_mn, y_mn, xerr=x_sd, yerr=y_sd, fmt='sk', ms=5, capsize=2, alpha=0.6, zorder=2)
    return x_mn, y_mn, x_sd, y_sd



def correct_sig_modulation(sig, angle_turns, method='poly', polydeg=10, add='mean', interp_pts=100, plots=False, plots_figname='', plot_ss=30, plots_test=False, return_all=False):
    '''correct the 1-turn periodic modulation (for a signal of the BFM eg: omega, z, radius), 
       by removing a polynome fit of sig Vs mod(angle_turns,1).
         sig, angle_turns: any signal and relative angle_turns trace (they have same length)
         method: 'poly' polynome fit, 'interp' interpolation
         polydeg : polyn degree to fit
         add : ['mean', 'max'] value to add to the corrected signal. If add==None, then np.mean(sig) is added. But if sig=radius of xy traj, it should  be different TODO what?
         interp_pts : numb of point to use to interpolate
         plot_ss: subsample when plotting
    return signal corrected
    '''
    if add == 'mean':
        _add = np.mean(sig)
    # angle turns in 0,1:
    am  = np.mod(angle_turns - angle_turns[0], 1)
    if method == 'poly':
        # polyn fit:
        pf = np.polyfit(am, sig, polydeg)
        po = np.poly1d(pf)
        if add == 'max':
            _add = np.max(po(am))
        # sig corrected:
        sig_corr = sig - po(am) + _add
    elif method == 'interp':
        sig_corr, interp_f = rm_interpolate_xy(am, sig, npts=interp_pts, add=add, plots=plots)
    if plots :
        if plots_figname:
            plt.figure(plots_figname, clear=True)
        else:
            plt.figure('correct_sig_modulation', clear=True)
        plt.subplot(321)
        plt.plot(sig[::plot_ss], '-', alpha=0.6, label='sig.raw')
        plt.plot(sig_corr[::plot_ss], '-', alpha=0.7, label='sig.corr')
        plt.legend()
        plt.subplot(322)
        plt.plot(angle_turns[::plot_ss], '-', label='angle_turns')
        plt.legend()
        plt.subplot(312)
        if method == 'poly':
            plt.plot(am[::plot_ss], sig[::plot_ss], ',', ms=1, alpha=0.3, label='sig.raw')
            plt.plot(am[::plot_ss], po(am)[::plot_ss], ',', ms=2, label='poly.fit')
        plt.xlabel('angle_turns mod 1')
        plt.legend()
        plt.subplot(313)
        if method == 'poly':
            plt.plot(am[::plot_ss], sig_corr[::plot_ss], ',', ms=1, alpha=0.3, label='sig.corr.')
        plt.xlabel('angle_turns mod 1')
        plt.legend()
        plt.tight_layout()

    if plots_test:
        FPS = 10000
        plt.figure('correct_sig_modulation TEST Z PAPER', clear=True)
        plt.subplot(321)
        plt.plot(np.arange(len(sig[::plot_ss]))/FPS, sig[::plot_ss], '-', alpha=0.9, label='raw')
        plt.plot(np.arange(len(sig[::plot_ss]))/FPS, sig_corr[::plot_ss], 'g-', alpha=0.6, label='corr.')
        plt.ylabel('z (nm)')
        plt.xlabel('time (s)')
        plt.xlim([5,5.05])
        plt.legend(fontsize=8, labelspacing=0)
        plt.subplot(322)
        plt.plot(np.arange(len(angle_turns[::plot_ss]))/FPS, angle_turns[::plot_ss], '-', label='turns')
        plt.xlabel('time (s)')
        plt.ylabel(r'$\phi$ (turns)')
        plt.subplot(312)
        if method == 'poly':
            plt.plot(am[::plot_ss], sig[::plot_ss], ',', ms=1, alpha=0.3)
            plt.plot(am[::plot_ss], po(am)[::plot_ss], '.', ms=1, label='poly.fit')
        plt.xlabel(r'mod($\phi$, 1 turn)')
        plt.ylabel('z raw (nm)')
        plt.subplot(313)
        if method == 'poly':
            plt.plot(am[::plot_ss], sig_corr[::plot_ss], 'g,', ms=1, alpha=0.3, label='corr.')
        plt.xlabel(r'mod($\phi$, 1 turn)')
        plt.ylabel('z corrected (nm)')
        plt.tight_layout()
    if return_all and method == 'poly':
        return sig_corr, am, po(am)
    elif return_all and method == 'interp':
        return sig_corr, am, interp_f
    else:
        return sig_corr



def outlier_smoother(x, m=3, win=3, plots=False, figname=''):
    ''' finds outliers in x (points > m*mdev(x)) [mdev:median deviation] 
    and replaces them with the median of win points around them 
    return x_corrected and number of outliers found '''
    x_corr = np.copy(x)
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    idxs_outliers = np.nonzero(d > m*mdev)[0]
    if len(idxs_outliers): print(f'outlier_smoother(): removing {len(idxs_outliers)} outliers [win:{win}]...')
    k = 0
    for i in idxs_outliers:
        if 100%100 == 0: print(f'outlier_smoother(): {k}/{len(idxs_outliers)}', end='\r')
        if i-win < 0:
            x_corr[i] = np.median(np.append(x[0:i], x[i+1:i+win+1]))
        elif i+win+1 > len(x):
            x_corr[i] = np.median(np.append(x[i-win:i], x[i+1:len(x)]))
        else:
            x_corr[i] = np.median(np.append(x[i-win:i], x[i+1:i+win+1]))
        k += 1
    if plots:
        fig = plt.figure(f'outlier_smoother '+figname, clear=True)
        ax1 = fig.add_subplot(211)
        ax1.plot(x, label='orig.', lw=2)
        ax1.plot(idxs_outliers, x[idxs_outliers], 'ro', label='outliers')
        ax1.legend()
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(x_corr, '-o', label='corrected')
        ax2.legend()
    return x_corr, len(idxs_outliers)



def savgol_filter(x, win, polyorder, mode=None, plots=0):
    ''' savgol filter of x,
    mode = 'valid' crop in [win:-win], 
    mode = None does nothing
    from https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way/26337730#26337730'''
    import scipy.signal as sig
    if polyorder >= win:
        polyorder = win-1
        print('savgol_filter(): bad polyorder fixed to win-1')
    if np.mod(win,2) == 0:
        win = win+1
        print('Warning savgol_filter, win must be odd: forced win = '+str(win))
    y = sig.savgol_filter(x, window_length=win, polyorder=polyorder)
    if mode == 'valid':
        y = y[win:-win]
        x = x[win:-win]
    if plots:
        plt.figure('savgol_filter()')
        plt.clf()
        plt.plot(x, '.')
        plt.plot(y)
    return y



def estimated_autocorrelation(x, fs=1, check=False, plots=False):
    """
    fs : sampling freq.
    check: double check with assert analyt expression
    from:
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    if check: 
        assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))             
    result = r/(variance*(np.arange(n, 0, -1)))
    timelag = np.arange(len(result))/fs
    if plots:
        plt.figure('estimated_autocorrelation')
        plt.subplot(211)
        plt.plot(timelag, result)
        plt.xlabel('time lag (s)')
        plt.ylabel('a.correlation')
        plt.subplot(212)
        plt.plot(np.arange(len(x))/fs, x)
        plt.xlabel('time (s)')
        plt.ylabel('signal')
    return result, timelag



def median_filter(x, win=10, fs=1, usemode='reflect', plots=False, plots_spectra=False):
    ''' median filter from scipy.ndimage '''
    from scipy.ndimage import median_filter
    y = median_filter(x, size=win, mode=usemode)
    if plots:
        plt.figure('median_filter()')
        plt.plot(np.arange(len(x))/fs, x, 'o')
        plt.plot(np.arange(len(y))/fs, y, '-')
        plt.xlabel('Time (s)')
        if plots_spectra: 
            plot_orig_filtered_spectra(x, y, fs)
    return y


#def median_filter(data, fs=1, win=3, usemode='valid', plots=0):
#    '''median filter. fs:sampling frequency. win:window kernel size for filter, usemode=['same' | 'valid'] '''
#    if np.mod(win,2) == 0:
#        win = win+1
#        print('Warning median_filter, win must be odd: forced win = '+str(win))
#    y = medfilt(data, kernel_size=win)
#    if usemode == 'valid':
#        y = y[int(np.ceil(win/2.)) : int(np.floor(-win/2.))]    
#    if plots:
#        plt.figure('median_filter()')
#        plt.plot(np.arange(len(data))/fs, data, 'bo')
#        plt.plot(np.arange(len(y))/fs, y, 'r-')
#        plt.xlabel('Time (s)')
#        plot_orig_filtered_spectra(data, y, fs)
#    return y


def run_win_smooth(data, win=10, fs=1, algorithm='cumsum', usemode='same', pad=10, plots=False):
    '''average running window filter, by convolution or cumsum (20x faster!)
            algorithm : 'conv', 'cumsum'
            win = pts of running window
            usemode: for 'conv': same as in np.convolve(). For 'cumsum': 'same','valid'
            pad [10]: for 'cumsum'+'same', pad the input with the median of data[:pad] and data[-pad:]
            fs [= 1] sampling freq. (only used in plots)
        return: y smoothed data 
    
    'cumsum' idea from :
    https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python/34387987#34387987
    '''
    if algorithm == 'conv':
        box = np.ones(int(win))/win
        y = np.convolve(data, box, mode=usemode)
    elif algorithm == 'cumsum':
        if usemode == 'valid':
            x_cs = np.cumsum(data)
            y = (x_cs[win:] - x_cs[:-win]) / win
        elif usemode == 'same':
            # win even number:
            win += win%2
            # insert copies of 1st and last value of data:
            p0 = np.median(data[:pad])
            p1 = np.median(data[-pad:])
            xpad = np.insert(data, 0, np.repeat(p0, win))
            xpad = np.insert(xpad[::-1], 0, np.repeat(p1, win))[::-1]
            xpad_cs = np.cumsum(xpad)
            y = (xpad_cs[win:] - xpad_cs[:-win]) / win
            y = y[int(win/2):int(-win/2)]

    if plots:
        freq_orig, spectrum_orig = calculate_spectrum(data, fs)
        freq_filtered, spectrum_filtered = calculate_spectrum(y, fs)
        plt.figure('run_win_smooth', clear=True)
        plt.subplot(311)
        plt.plot(np.arange(len(data))/fs, data, 'bo')
        plt.plot(np.arange(len(y))/fs, y, 'r-')
        plt.xlabel('Time (s)')
        plt.subplot(312)
        plt.loglog(freq_orig, spectrum_orig)
        plt.grid(True)
        plt.title('Original')
        plt.subplot(313)
        plt.loglog(freq_filtered, spectrum_filtered)
        plt.grid(True)
        plt.title('Running window average ('+str(win)+' pts, '+str(win/fs)+' sec)')
        plt.xlabel('Freq. (Hz)')
        plt.tight_layout()
    return y


def gaussian_filter(data, gsigma=1, plots=0, gmode='reflect'):
    ''' multi dim gaussian fileter of data '''
    y = scipy.ndimage.filters.gaussian_filter(data, gsigma, mode='reflect')
    if plots:
        plt.figure()
        plt.plot(data)
        plt.plot(y)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5, plots=True):
    ''' fs : sampling freq. (pts/s) '''
    lowcut = float(lowcut)
    highcut = float(highcut)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)    
    y = lfilter(b, a, data)
    if plots:
        plot_filter(a,b,fs,y,t,data, lowcut,highcut)
    return y




def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    return b, a

def bandstop_filter(data, lowcut, highcut, fs, order=5, plots=True):
    ''' fs : sampling freq. (pts/s) '''
    lowcut = float(lowcut)
    highcut = float(highcut)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)    
    y = lfilter(b, a, data)
    if plots:
        plot_filter(a,b,fs,y,t,data, lowcut,highcut)
    return y



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5, nodelay=False, plots=True):
    ''' fs : sampling freq. (pts/s) 
    if nodelay=True : use filtfilt() to remove delay '''
    cutoff = float(cutoff)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    t = t[:len(data)]
    b, a = butter_lowpass(cutoff, fs, order=order)
    if nodelay: 
        y = filtfilt(b, a, data)#, padlen=300)
    else:
        y = lfilter(b, a, data)
    w, mag, phase = bode((b,a))
    if plots:
        plot_filter(a,b,fs,y,t,data, cutoff)
    return y



def plot_filter(a,b,fs,y,t,data, f1=0, f2=0):
    ''' 
    a,b from filterfs = sampling freq.
    f1 = cutoff, f2 = 0 (butter_lowpass)
    f1 = lowcut, f2 = highcut (butter_bandpass) 
    '''
    plot_orig_filtered_spectra(data, y, fs)
    w, h = freqz(b, a, worN=8000)
    plt.figure('plot_filter')
    plt.subplot(2, 1, 1)
    plt.semilogx(0.5*fs*w/np.pi, np.abs(h), 'b')
    if f1:
        plt.plot(f1, 0.5*np.sqrt(2), 'ko')
        plt.axvline(f1, color='k')
    if f2:
        plt.plot(f2, 0.5*np.sqrt(2), 'ko')
        plt.axvline(f2, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    

 
def calculate_spectrum(data, fs):
    ''' OBSOLETE! TODO: change with scripts/powerspectrum/PSpectrum.py 
    calculates the spectra of data, downsampling data at power of 2 points 
    fs : sampling freq. (pts/s) 
    '''
    import numpy.fft as fft
    # dowsample at power of 2 points:
    n_pts = np.floor(np.log(len(data))/np.log(2)) 
    print('Spectra warning: considering '+str(2**n_pts)+' points (2**'+str(n_pts)+'), instead of len(data)='+str(len(data))) 
    data = data[:int(2**n_pts)]
    spectrum = np.abs(fft.fft(data))
    freq = fft.fftfreq(len(data), d=1./fs)
    freq = freq[0:int(len(spectrum)/2.)]
    spectrum = spectrum[0:int(len(spectrum)/2.)]
    return freq, spectrum


def plot_orig_filtered_spectra(data, data_filtered, fs):
    '''plot the filtered and original spectrum of data and of diff(data) '''
    # signal derivative : 
    data_dif = np.diff(data)
    data_filtered_dif = np.diff(data_filtered)
    freq_orig_dif, spectrum_orig_dif = calculate_spectrum(data_dif, fs)
    freq_filtered_dif, spectrum_filtered_dif = calculate_spectrum(data_filtered_dif, fs)
    # signal :
    freq_orig, spectrum_orig = calculate_spectrum(data, fs)
    freq_filtered, spectrum_filtered = calculate_spectrum(data_filtered, fs)
    #plots:
    plt.figure('plot_orig_filtered_spectra_1')
    plt.subplot(211)
    plt.loglog(freq_orig, spectrum_orig)
    plt.grid(True)
    plt.title('Signal Original')
    plt.subplot(212)
    plt.loglog(freq_filtered, spectrum_filtered)
    plt.grid(True)
    plt.title('Signal Filtered')
    plt.xlabel('Freq. (Hz)')
    plt.figure('plot_orig_filtered_spectra_2')
    plt.subplot(211)
    plt.loglog(freq_orig_dif, spectrum_orig_dif)
    plt.grid(True)
    plt.title('derivative ORIGINAL')
    plt.subplot(212)
    plt.loglog(freq_filtered_dif, spectrum_filtered_dif)
    plt.grid(True)
    plt.title('derivative FILTERED')
    plt.xlabel('Freq. (Hz)')





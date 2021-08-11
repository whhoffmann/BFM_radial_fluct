import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def spectrum(x, FPS, plots=0, new_fig=1, prints=True, downsample=False):
    '''finds and plots the single-side spectrum of x(t), 
    downsample=True : crop the first 2**N points of x (for a maximal N)
    spectrum_x[1:] = 2 * abs(fft(x))**2 / (n**2 * df) 
    spectrum_x[0]  =     abs(fft(x))**2 / (n**2 * df) 
    (as defined in labview "single sided Pw.Spectrum")
    return     freq, spectrum
    '''
    FPS = float(FPS)
    if downsample:
        # dowsample at power of 2 elements to speed up fft() :
        n_pts = int(np.floor(np.log(len(x))/np.log(2)))
        print(f'spectrum(): dwsampling from {len(x)} to 2**{n_pts}={2**n_pts} pts')
        x_cut = x[:int(2**n_pts)]
    else:
        x_cut = x
    # spectrum of x(t) :
    spectrum_x = np.abs(fft.fft(x_cut))**2
    freq_x = fft.fftfreq(len(x_cut), d=1./FPS)
    freq_x = freq_x[:int(len(spectrum_x)/2.)]
    spectrum_x = spectrum_x[:int(len(spectrum_x)/2.)]
    # NOTE!! Is this next line wrong ???!!!!
    # it seems ok, eg see https://stackoverflow.com/questions/31153563/normalization-while-computing-power-spectral-density
    spectrum_x = spectrum_x/(len(x_cut)**2 *freq_x[1])
    ##spectrum_x = spectrum_x *((1/FPS)**2/len(x_cut))
    # Ashley?: spectrum_x = spectrum_x/float(len(x_cut) *freq_x[1])
    spectrum_x[1:] = 2*spectrum_x[1:]
    if prints:
        print('var(x)*len(x)/FPS   = {:.9e}'.format(np.var(x_cut)*len(x_cut)/FPS))
        print('df                  = {:.9e}'.format(freq_x[1]))
        print('1/(len(x)*dt)       = {:.9e}'.format(1/(len(x)/FPS)))
        print('integral of PSD[1:] = {:.9e}'.format(np.sum(spectrum_x[1:])*freq_x[1]))
        print('var(x)              = {:.9e}'.format(np.var(x_cut)))
    if plots:
        if new_fig:
            plt.figure()
        plt.subplot(211)
        plt.plot(np.arange(len(x))/FPS, x)
        plt.xlabel('Time (s)')
        plt.ylabel('X(t)')
        plt.grid(True)
        plt.axis('tight')
        plt.subplot(212)
        plt.loglog(freq_x, spectrum_x, '-o', ms=2)
        plt.axis('tight')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectrum(X)')
        plt.grid(True)
    return freq_x, spectrum_x



def log_avg_spectrum(freq, spectrum, npoints=500):
    ''' average a spectrum on 'npoints' logarithmic bins of frequencies
    return logwin, spt_avg = logarithmic bins used, spectrum averaged '''
    spt_avg = np.array([])
    logbins = np.array([])
    # define bins 'logwin', is DC there? f[0]==0 ? :
    if freq[0] != 0:
        logwin = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), npoints)
    else:
        logwin = np.logspace(np.log10(freq[1]), np.log10(freq[-1]), npoints)
    freq_idx = np.digitize(freq, logwin)
    # find average on every bin:
    for i in range(len(logwin)):
        mn_idx = np.where(freq_idx == i)[0]
        if len(mn_idx):
            spt_avg = np.append(spt_avg, np.mean(spectrum[mn_idx]))
            logbins = np.append(logbins, logwin[i])
    return logbins, spt_avg




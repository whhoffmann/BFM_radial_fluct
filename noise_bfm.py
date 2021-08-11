#### RADIAL fluctuations of bead essay:
# xy_fluctuations* functions to study radial fluctuations, drags, and hook stiffness EI
# TODO correct drift by interpolation of the entire trace, not by subtraction?


#### SPIKES analysis:
# collaboration with Victor Nov 2019
# IDEA: analyze and find statistics of spikes in speed (better on torque) traces
# for a given filter do
# running window on speed trace, find speed spikes locally as points < s*sigma 
# find t1 t2 of each spike
# for each spike: get t1, t2, Dt=t2-t1, amplitude, stator level start, stator level stop(=start(int)-amplitude(float)),
# extrapolate results for filter = 0
# TODO 
# spikes_analysis: in each window fit gaussian, get error, skew to negative 
# get torque from xy circle

# Data:
# this does not have theta_deg, but has um/px (147.5nm/px):
# test_filename = '/home/francesco/lavoriMiei/cbs/data/AshleyData/resurrections/D_WT_1000Res.p'
# this one has theta_deg, but not um/px:
# test_filename = '/home/francesco/lavoriMiei/cbs/people/collaborations/Victor/BFM/data/D_WT_1000Res.p'


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import itertools
import sys
import warnings
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

import Drag.DragRotatingBead
import Drag.DragTranslationBead
import filters
import fitEllipse
import openTDMS
import utils
import PSpectrum




KT = 4.11e-21 # Joule
eta = 0.001   # [Pa*s] = [N*s/m^2] water viscosity


class BFMnoise():
    ''' 
        Analysis of radial fluctuations of BFM bead assay.
    '''


    def __init__(self, key=None, filter_name=None, filter_win=None, savgol_deg=5, umppx=None, filename=None):
        self.filename    = filename
        self.filter_name = filter_name
        self.savgol_deg  = savgol_deg
        self.filter_win  = filter_win
        self.key         = key
        if key != None and filename:
            d = self.get_dict()
            self.extract_from_dict(d, key, umppx=umppx) 



    def get_dict(self):
        ''' get dict in filename.  
        extract data in key with extract_from_dict() '''
        # get dict:
        print(f'BFMnoise.get_dict(): loading {self.filename}')
        d = np.load(self.filename, allow_pickle=1, encoding='latin1')
        # from unsized array(dic) to dic:
        if type(d) == np.ndarray and d.shape == ():
            d = d.item()
        print(f'BFMnoise.get_dict(): found dict with keys: {d.keys()}')
        return d



    def extract_from_dict(self, d, key=0, umppx=0.1475, prints=False):
        ''' extract and store data from dict (see get_dict())'''
        print(f'BFMnoise.extract_from_dict(): d[{key}] keys: {d[key].keys()}')
        for k in d[key].keys():
            if prints: print(f'BFMnoise.extract_from_dict(): key:{key} found: {k}')
            setattr(self, k, d[key][k])
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            print('BFMnoise.extract_from_dict(): Warning! x,y not found in dict.')
        if 'theta_deg' in d[key]:
            self.angle_turns = d[key]['theta_deg']/360
        else:
            print('BFMnoise.extract_from_dict(): Warning! theta_deg not found in dict.')
        if 'nm_per_pix' in d[key]:
            self.umppx = d[key]['nm_per_pix']*1e-3
        elif 'nm_per_pixel' in d[key]:
            self.umppx = d[key]['nm_per_pixel']*1e-3
        else:
            self.umppx = umppx # micron per pixel
            print(f'BFMnoise.extract_from_dict(): Warning! \'nm_per_pix\' not found in dict, using external value: {self.umppx}.')
        if 'cellnum' in d[key]:
            print(f'BFMnoise.extract_from_dict(): cellnum: {d[key]["cellnum"]}')



    def make_filtered_trace(self, c0=0, c1=-1, force_speed_from_xy=False, filter_win=None, filter_name=None, savgol_deg=5, xyspeed_n=1, stretch_xy=False, plots=False):
        ''' make, filter, and store the angular speed (omega) from xy.
                force_speed_from_xy : build speed from arctan of xy points
                filter_win : pts window for filer
                filter_name : 'savgol', 'run_win_smooth', None
                savgol_deg : degree of savgol filter
                xyspeed_n [1]: calc linear speed (um/s) from hypot(xy), xy at distance xyspeed_n
                stretch_xy : stretch trajectory to circle as 1st thing
                c0,c1 : crop (preprocessing)
            TODO: add outlier_smoother, remove speed periodic modulation
        '''
        if stretch_xy:
            x, y, _,_ = fitEllipse.stretch_ellipse(self.x.flatten()[c0:c1], self.y.flatten()[c0:c1], stretchit=True, plots=plots)
        else:
            x, y = self.x[c0:c1], self.y[c0:c1]
        # angular speed:
        if hasattr(self, 'angle_turns') and not force_speed_from_xy:
            print(f'BFMnoise.make_filtered_trace(): making speed from angle_turns')
        else:
            print(f'BFMnoise.make_filtered_trace(): making speed from x,y')
            self.angle_turns = np.unwrap(np.arctan2(y - np.mean(y), x - np.mean(x)))/(2*np.pi)
        self.speed_Hz = np.diff(self.angle_turns)*self.FPS
        # linear speed (um/s):
        self.xyspeed_ums = np.hypot(x[xyspeed_n:] - x[:-xyspeed_n], y[xyspeed_n:] - y[:-xyspeed_n])*self.umppx*self.FPS/xyspeed_n
        # store filter:
        if filter_name :
            self.filter_name = filter_name
        if filter_win:
            self.filter_win = filter_win
        if savgol_deg:
            self.savgol_deg = savgol_deg
        print(f'BFMnoise.make_filtered_trace(): Filtering by {self.filter_name} {self.filter_win} ...')
        # apply filter:
        if self.filter_name == 'savgol':
            self.speed_Hz_f    = filters.savgol_filter(self.speed_Hz   , self.filter_win, self.savgol_deg, plots=False)
            self.xyspeed_ums_f = filters.savgol_filter(self.xyspeed_ums, self.filter_win, self.savgol_deg, plots=False)
            self.angle_turns_f = filters.savgol_filter(self.angle_turns, self.filter_win, self.savgol_deg, plots=False)
        elif self.filter_name == 'run_win_smooth':
            self.speed_Hz_f    = filters.run_win_smooth(self.speed_Hz   , self.filter_win, usemode='same', plots=False)
            self.xyspeed_ums_f = filters.run_win_smooth(self.xyspeed_ums, self.filter_win, usemode='same', plots=False)
            self.angle_turns_f = filters.run_win_smooth(self.angle_turns, self.filter_win, usemode='same', plots=False)
        elif self.filter_name == None:
            self.speed_Hz_f    = self.speed_Hz
            self.xyspeed_ums_f = self.xyspeed_ums
            self.angle_turns_f = self.angle_turns
        else:
            raise Exception('BFMnoise.make_filtered_trace(): Error filter_name not valid')
        print(f'BFMnoise.make_filtered_trace(): Done.')
        if plots:
            speed_f_time_sec = np.arange(len(self.speed_Hz_f))/self.FPS
            plt.figure('make_speed', clear=True)
            plt.subplot(311)
            plt.plot(self.x[c0:c1:30], self.y[c0:c1:30], ',')
            plt.axis('image')
            plt.subplot(312)
            #plt.plot(speed_f_time_sec, self.speed_Hz_f, label=f'filter_win:{self.filter_win}')
            plt.plot(self.speed_Hz_f, label=f'filter_win:{self.filter_win}')
            plt.legend()
            plt.ylabel('speed_Hz_f')
            plt.subplot(313)
            plt.plot(self.xyspeed_ums_f, label=f'filter_win:{self.filter_win}')
            plt.legend()
            plt.ylabel('xyspeed_ums_f')



    def open_drift_file(self, locate=True, filename='', roi=0, c0=None, c1=None, invert_z=False, z_to0=False, print_found=False, plots=False):
        ''' finds filename in os, open and return the xyz traces in filename-roi, cropped in c0:c1, that can be used for drift correction in x,y,z , 
        filename : complete file name if locate=False, keyword to search if locate=True
        invert_z : [False] if True: z = -z
        z_to0    : [False] send min(z) to 0
        '''
        if locate:
            if filename == '':
                filename = self.cellnum[:-5]
                print(f'open_drift_file(): searching for {filename}')
                #raise RuntimeError('open_drift_file(): locate is True but filename is empty')
            ll = utils.find_files(filename)
            if ll == ['']:
                raise RuntimeError('open_drift_file(): no file found')
            found_filename = [l for l in ll if l.endswith('_Trckd.tdms')][0]
            print(f'open_drift_file(): drift file found from keyword "{filename}" (ROI{roi}) : {found_filename}')
        else:
            found_filename = filename
        ff = openTDMS.openTdmsFile(found_filename, print_found=print_found)
        xdrift = ff[f'/ROI{roi}_Trk/X{roi}'][c0:c1]
        ydrift = ff[f'/ROI{roi}_Trk/Y{roi}'][c0:c1]
        zdrift = ff[f'/ROI{roi}_Trk/Z{roi}'][c0:c1]
        if invert_z:
            zdrift = -zdrift
        if z_to0:
            zdrift -= np.min(zdrift)
        if plots:
            fig = plt.figure('open_drift_file', clear=True)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')
            ax1.plot(xdrift)
            ax1.plot(ydrift)
            ax1.plot(zdrift)
            ax1.set_ylabel('x,y,z')
            ax2.plot(xdrift, ydrift, zdrift, ',')
        return [xdrift, ydrift, zdrift]



    def correct_xyz_drift(self, x=None, y=None, z=None, c0=None, c1=None, roi=0, rm_outliers=False, rm_outliers_win=10, filename='', locate=False, print_found=False, plots=False):
        ''' correct the drift in x,y,z finding and opening the corresponding drift file, 
        taking ROI='roi', removing outliers in drift traces 
            x,y,z       : inputs, already cropped in [c0:c1]
            c0, c1      : indexes where x,y,z have been cropped
            locate      : in linux (with $updatedb done in advance), find automatically the file using 'filename' as keyword
            filename    : full path and filename if locate is False, otherwise keyword for search (eg, use self.cellnum[:-5]) 
            rm_outliers : remove outliers in xyz
            rm_outliers_win : pts win use to remove outliers
        ''' 
        if len(x) or len(y) or len(z):
            print('correct_xyz_drift(): opening reference bead file...')
            xyz_drift = self.open_drift_file(locate=locate, filename=filename, roi=roi, c0=c0, c1=c1, invert_z=False, z_to0=1, print_found=print_found, plots=False)
            x_drift = xyz_drift[0]
            y_drift = xyz_drift[1]
            z_drift = xyz_drift[2]
        if len(x): 
            if rm_outliers:
                # remove outliers from drift traces:
                x_drift, x_noutliers = filters.outlier_smoother(x_drift, m=5, win=rm_outliers_win, plots=False)
            x_corr = x - x_drift
            print('correct_xyz_drift(): x drift corrected.')
        else:
            x_corr = []
        if len(y): 
            if rm_outliers:
                # remove outliers from drift traces:
                y_drift, y_noutliers = filters.outlier_smoother(y_drift, m=5, win=rm_outliers_win, plots=False)
            y_corr = y - y_drift
            print('correct_xyz_drift(): y drift corrected.')
        else:
            y_corr = []
        if len(z): 
            if rm_outliers:
                z_drift, z_noutliers = filters.outlier_smoother(z_drift, m=5, win=rm_outliers_win, plots=False)
            z_corr = z - z_drift
            print('correct_xyz_drift(): z drift corrected.')
        else:
            z_corr = []
        if plots:
            fig = plt.figure('correct_xyz_drift', clear=True)
            ax11 = fig.add_subplot(331)
            ax12 = fig.add_subplot(332, sharex=ax11)
            ax13 = fig.add_subplot(333, sharex=ax11)
            ax21 = fig.add_subplot(334, sharex=ax11)
            ax22 = fig.add_subplot(335, sharex=ax11)
            ax23 = fig.add_subplot(336, sharex=ax11)
            ax31 = fig.add_subplot(337, sharex=ax11)
            ax32 = fig.add_subplot(338, sharex=ax11)
            ax33 = fig.add_subplot(339, sharex=ax11)
            ax11.plot(x)
            ax12.plot(x_drift)
            ax13.plot(x_corr)
            ax21.plot(y)
            ax22.plot(y_drift)
            ax23.plot(y_corr)
            ax31.plot(z)
            ax32.plot(z_drift)
            ax33.plot(z_corr)
            ax11.set_title('orig', fontsize=9)
            ax12.set_title('drift', fontsize=9)
            ax13.set_title('orig-drif', fontsize=9)
            ax11.set_ylabel('x')
            ax21.set_ylabel('y')
            ax31.set_ylabel('z')
            fig.tight_layout()
        return x_corr, y_corr, z_corr
   


    def allan_variance(self, sig, n_taus=10, FPS=None, plots=False, clear=True):
        ''' find the allan deviation of sig. See van Oene BPJ 2018'''
        def adtheo(tau, k, gamma):
            ''' sqrt of eq.3 theoretical AV(tau) '''
            A = (KT/k)
            tau_c = gamma/k
            t1 = tau/tau_c
            av = A*(1/t1)**2 * (2*t1 + 4*np.exp(-t1) - np.exp(-2*t1) - 3)
            return np.sqrt(av)
        import allantools
        if FPS == None: 
            FPS = self.FPS
        # time points for AD: 
        taus = np.logspace(np.log10(1), np.log10(len(sig)), n_taus, endpoint=True)/FPS 
        # experimental Allan Deviation:
        taus, ad, ade, ns = allantools.oadev(sig, rate=FPS, data_type="freq", taus=taus) 
        # fit exp - theory: 
        popt, _ = curve_fit(adtheo, taus, ad, p0=[1e-6, 1e-9])
        print(f'allan_variance(): k, gamma : {popt}')
        # store:
        self.allan_dic = {'taus':taus, 'ad':ad, 'popt':popt, 'adtheo':adtheo}
        if plots:
            plt.figure('allan_variance 0', clear=clear)
            a = 100 if len(sig)>1000000 else 1
            plt.subplot(211)
            plt.plot(np.arange(0,len(sig))[::a], sig[::a])
            plt.subplot(212)
            plt.loglog(taus, self.allan_dic['ad'], '-o', ms=5, alpha=0.4)
            x_taus = np.logspace(np.log10(np.min(taus)), np.log10(np.max(taus)), 1000)
            plt.loglog(x_taus, self.allan_dic['adtheo'](x_taus, *self.allan_dic['popt']), 'k--', lw=2, alpha=0.2)
            plt.xlabel('Time (s)')
            plt.ylabel('Allan Deviation (m)')                                                                                             
            plt.tight_layout()

 

    def kernel_density_histo(self, sig, kernel='gaussian', logscale=False, band=1, return_all=False, plots=False):
        ''' kernel density histogram of input sig.
        kernel: ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
        band  : kernel bandwidth (~ bin size)
        return (x, density) if return_all==False
        return (x, density, score, kde) if return_all==True
        '''
        from sklearn.neighbors import KernelDensity
        if logscale:
            # TODO ok?
            Xout = np.logspace(np.log10(np.min(sig)*0.9), np.log10(np.max(sig)*1.1), 1000)[:, np.newaxis] 
        else:
            Xout = np.linspace(np.min(sig)*0.9, np.max(sig)*1.1, 1000)[:, np.newaxis]
        kde = KernelDensity(kernel=kernel, bandwidth=band).fit(sig[:, np.newaxis])
        dens = np.exp(kde.score_samples(Xout))
        score = kde.score_samples
        if plots:
            plt.figure('kernel_density_histo', clear=True)
            plt.plot(Xout, dens)
        if return_all:
            return Xout, dens, score, kde
        else: 
            return Xout, dens



    def gauss(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def gauss_fit(self, x, y, return_err=False):
        ''' return popt of gaussain fit and mean square log-error of the fit '''
        popt, pcov = curve_fit(self.gauss, x, y, p0=[np.max(y), np.mean(x), np.std(x)])
        if return_err:
            idx = np.nonzero((y>0) * (self.gauss(x, *popt)>0))[0]
            err = np.mean((np.log10(y[idx]) - np.log10(self.gauss(x[idx], *popt)))**2)
            return popt, err
        else:
            return popt



    def xy_fluctuations_theta_find_offset(self, win_s=4, win_subsample=1, c0=None, c1=None, psd_c0=None, psd_c1=None, 
                                        stretch_xy=False, correct_win_mod=False, stretch_z=False, 
                                        speed_thr=0, negate_speed=False, separate_switching=False,
                                        use_xyz=False, correct_drift=False, drift_roi=0, correct_outliers_z=False ):
        ''' find the best offset in xy_fluctuations_theta() with offset as variable 
            by minimization of the difference between the drag found by fit and brenner theory (MSE_gamma_loren_brenner) '''
        def xy_fluctuations_theta_offset(offset, 
                                        win_s=win_s, win_subsample=win_subsample, c0=c0, c1=c1, psd_c0=psd_c0, psd_c1=psd_c1, 
                                        stretch_xy=stretch_xy, correct_win_mod=correct_win_mod, stretch_z=stretch_z, 
                                        speed_thr=speed_thr, negate_speed=negate_speed, separate_switching=separate_switching,
                                        use_xyz=use_xyz, correct_drift=correct_drift, drift_roi=drift_roi, correct_outliers_z=correct_outliers_z):
            ''' xy_fluctuations_theta with only "offset" as variable '''
            return self.xy_fluctuations_theta(
                    offset=offset, return_MSE_gamma_loren_brenner=True, # important: offset is the variable, and returns the MSE to minimize 
                    win_s=win_s, win_subsample=win_subsample, c0=c0, c1=c1, psd_c0=psd_c0, psd_c1=psd_c1, 
                    center_win_traj=True, correct_win_mod=correct_win_mod, stretch_xy=stretch_xy, stretch_z=stretch_z,
                    plots_lev=[], speed_thr=speed_thr, negate_speed=negate_speed, separate_switching=separate_switching,
                    use_xyz=use_xyz, correct_drift=correct_drift, drift_roi=drift_roi, correct_outliers_z=correct_outliers_z )
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(xy_fluctuations_theta_offset, bounds=[0e-9, 200e-9], method='bounded', options={'maxiter':20, 'xatol':.1e-9})
        print(f'xy_fluctuations_theta_find_offset(): {res}')
        return(res)



    def xy_fluctuations_theta(self, dic_ext={}, 
                                    win_n=None, win_s=None, win_subsample=1, 
                                    c0=None, c1=None, 
                                    psd_c0=0, psd_c1=-1,
                                    beta=None, Dbeta=None, 
                                    center_win_traj=True, 
                                    stretch_xy=False, 
                                    stretch_z=False,
                                    correct_win_mod=False, 
                                    correct_outliers_z=False,
                                    correct_outliers_xy=False,
                                    correct_outliers_z_win=10,
                                    correct_outliers_xy_win=10,
                                    plots_lev=[1], savefigs=False,
                                    offset=0,
                                    return_MSE_gamma_loren_brenner=False,
                                    no_lorentz=False,
                                    negate_speed=False,
                                    speed_thr=None,
                                    separate_switching=False,
                                    use_xyz=False,
                                    correct_drift_xy=False,
                                    correct_drift_z=False,
                                    correct_drift_roi=0):
        ''' Based on theta angle (angle on the plane perpendicular to the rotation plane), 
            analyse radial, tangential fluctuations, hook bending stiffness EI, drag corrections of one trace 
            by raw signal, Gauss fit, Lorentz fit.
            Can analyse xy only or xyz data.
        
        dic_ext             : [{}] external dict with params to overwrite self.(x,y,umppx,FPS,filename,key,dbead_nm)
        c0,c1               : [None, None] crop x,y in [c0:c1]
        win_n               : [None] analyse a number 'win_n' of time-windows
        win_s               : [None] seconds in a window, alternative to win_n
        win_subsample       : [1] float in (0,1] to go faster, subsample the n. of windows, taking only a random number of them of this factor 
        center_win_traj     : [True] center to 0,0 each win xy trajectory, by centering the fit ellipse, with no stretching
        stretch_xy        : [False] stretch total xy trajectory to a circle, bring circle center to 0,0
        stretch_z           : [False] stretch total z trace to make it flat
        correct_win_mod     : [False] correct radius trace for angle-periodic modulation (eg due to kinks)
        beta, Dbeta         : [None, None] consider only an arc (beta, Dbeta) of the xy trajectory
        plots_lev           : [[]] list of plots with elements from: [0,1,2,3,4,5, 'theta phi fluct', 'xyz', 'drift', 'zoutliers']
        savefigs            : [False] save figs with automatic name 
        offset              : [0] in meters. For xy: min z when rmax is found, called s_min in text. For xyz: offset for z. Also called 's'.
        return_MSE_gamma_loren_brenner : [False] to use for minimization by offset (see xy_fluctuations_theta_find_offset)
        no_lorentz          : [False] do not fit lorentzian, which is slow (all the analysis will be affected)
        negate_speed        : [None] invert speed, speed_Hz(_f) = -speed_Hz(_f)
        speed_thr           : cut out portions of abs(speed) < speed_thr (ex use with separate_switching=True)
        separate_switching  : [False] cut and append cw regions to end of speed trace to better analysis cw/ccw
        use_xyz             : [False] if True use xyz points to define theta, otherwise use only xy
        correct_drift_xy    : [False] automatically search for drift file and correct for xy drift (careful it can introduce high freq. noise)
        correct_drift_z     : [False] automatically search for drift file and correct for z drift
        correct_drift_roi   : [0] ROI from which extract the drift traces
        correct_outliers_z  : [False] smooth out outlier points in z trace, only valid if use_xyz=True
        correct_outliers_z_win : [10] windows of points to remove outliers
        correct_outliers_xy : [False] smooth out outlier points in z trace, only valid if use_xyz=True
        correct_outliers_xy_win : [10] windows of points to remove outliers
        '''
        def log_lorentzian(f, fc, gamma):
            ''' return log10(single-side lorentzian)
            f: frequency, fc: corner frequency, gamma:drag,  A0: amplitude '''
            return np.log10(KT/(np.pi**2 *gamma*(f**2 + fc**2)))
        def log_lorentzian_f2(f, fc, gamma):
            ''' return log10(single-side lorentzian * f**2)
            f: frequency, fc: corner frequency, gamma:drag,  A0: amplitude '''
            return np.log10(f**2 * KT/(np.pi**2 *gamma*(f**2 + fc**2)))
        def log_lorentzian_RMSE(params, *data):
            x, y = data
            return np.sqrt(np.mean((log_lorentzian(x, *params) - y)**2))
        def log_lorentzian_f2_RMSE(params, *data):
            x, y = data
            return np.sqrt(np.mean((log_lorentzian_f2(x, *params) - y)**2))

        if 1 in plots_lev:
            fig1 = plt.figure(f'xy_fluctuations_theta 1', figsize=(8,7), clear=True)
            ax11 = fig1.add_subplot(331)
            ax18 = fig1.add_subplot(332)
            ax12 = fig1.add_subplot(333)
            ax13 = fig1.add_subplot(334)
            ax14 = fig1.add_subplot(335)
            ax15 = fig1.add_subplot(336)
            ax16 = fig1.add_subplot(337)
            ax17 = fig1.add_subplot(338)
            ax19 = fig1.add_subplot(339)
            fig2 = plt.figure(f'xy_fluctuations_theta 2', figsize=(8,7), clear=True)
            ax21 = fig2.add_subplot(331)
            ax28 = fig2.add_subplot(332)
            ax22 = fig2.add_subplot(333)
            ax23 = fig2.add_subplot(334)
            ax24 = fig2.add_subplot(335)
            ax25 = fig2.add_subplot(336)
            ax26 = fig2.add_subplot(337)
            ax27 = fig2.add_subplot(338)
            ax29 = fig2.add_subplot(339)
        if 2 in plots_lev:
            fig0 = plt.figure(f'xy_fluctuations_theta 0', figsize=(8,7), clear=True)
            ax01 = fig0.add_subplot(221)
            ax02 = fig0.add_subplot(222)
            ax03 = fig0.add_subplot(223)
            ax04 = fig0.add_subplot(224)
            fig6 = plt.figure(f'xy_fluctuations_theta 6', clear=True)
            ax61 = fig6.add_subplot(211)
            ax62 = fig6.add_subplot(212)
        if 5 in plots_lev:
            fig3 = plt.figure('xy_fluctuations_theta 3', figsize=(5.8,7.3), clear=True)
            ax31 = fig3.add_subplot(421)
            ax33 = fig3.add_subplot(422)
            ax34 = fig3.add_subplot(423)
            ax35 = fig3.add_subplot(425)
            ax36 = fig3.add_subplot(427)
            ax37 = fig3.add_subplot(424)
            ax38 = fig3.add_subplot(426)
            ax39 = fig3.add_subplot(428)
        if 3 in plots_lev:
            fig4 = plt.figure('xy_fluctuations theta 4', clear=True)
            ax41 = fig4.add_subplot(221)
            ax42 = fig4.add_subplot(222, sharex=ax41, sharey=ax41)
            ax43 = fig4.add_subplot(223, sharey=ax41)
            ax44 = fig4.add_subplot(224, sharey=ax41)
        if 'theta phi fluct' in plots_lev:
            fig7 = plt.figure('xy_fluctuations 7', clear=True)
            ax71 = fig7.add_subplot(341)
            ax72 = fig7.add_subplot(342)
            ax73 = fig7.add_subplot(343)
            ax73a = fig7.add_subplot(344)
            ax74 = fig7.add_subplot(334)
            ax75 = fig7.add_subplot(335)
            ax76 = fig7.add_subplot(336)
            ax77 = fig7.add_subplot(337)
            ax78 = fig7.add_subplot(338)
            ax79 = fig7.add_subplot(339)
        if 'xyz' in plots_lev:
            fig_xyzLL = plt.figure('xy_fluctuations_theta xyz LL', clear=True)
            ax_xyzLL1 = fig_xyzLL.add_subplot(211)
            ax_xyzLL2 = fig_xyzLL.add_subplot(212)
            fig_xyz = plt.figure('xy_fluctuations_theta xyz 1', figsize=(10,3), clear=True)
            ax_xyz11 = fig_xyz.add_subplot(221, projection='3d')
            ax_xyz12 = fig_xyz.add_subplot(222, projection='3d')
            ax_xyz13 = fig_xyz.add_subplot(223)
            ax_xyz14 = fig_xyz.add_subplot(224)
            fig2_xyz = plt.figure('xy_fluctuations_theta xyz 2', figsize=(10,3), clear=True)
            ax_xyz21 = fig2_xyz.add_subplot(131, projection='3d')
            ax_xyz22 = fig2_xyz.add_subplot(132)
            ax_xyz23 = fig2_xyz.add_subplot(133)
        if 'ztheta' in plots_lev:
            fig_ztheta = plt.figure('xy_fluctuations_theta ztheta', clear=True)
            ax_ztheta1 = fig_ztheta.add_subplot(211)
            ax_ztheta2 = fig_ztheta.add_subplot(212)
        if beta!=None and Dbeta!=None:
            fig5 = plt.figure('xy_fluctuations theta 5', clear=True)
            ax51 = fig5.add_subplot(211)
            ax52 = fig5.add_subplot(212)
        # overwrite self using external dict:
        if dic_ext:
            self.dbead_nm = dic_ext['dbead_nm']
            self.x        = dic_ext['x']
            self.y        = dic_ext['y']
            self.z        = dic_ext['z'] if 'z' in dic_ext else None
            self.umppx    = dic_ext['umppx']
            self.FPS      = dic_ext['FPS']
            self.key      = dic_ext['key']
            self.filename = dic_ext['filename']
        # params:
        z_correction = 0.85                 # optical correction factor for z
        r_bead       = self.dbead_nm*1e-9/2 # bead radius m
        gamma        = 6*np.pi*eta*r_bead   # translational bulk drag of sphere
        L_hook       = 60e-9                # hook length
        filter_win   = 801                  # filter window to plot speed(t)
        plotlist3    = [2,15,30]            # windows to plot in fig3
        # init:
        thetavar_sig_arr  = []
        theta_stds_loren  = []
        speedmn_arr       = []
        speedvar_arr      = []
        radmn_arr         = []
        a_ellipse_arr     = []
        thetamn_sig_arr   = []
        thetamin_sig_arr  = []
        thetamax_sig_arr  = []
        freq_c            = []
        gamma_loren       = []
        gamma_loren_f2      = []
        gamma_loren_f2_pts  = []
        gamma_theta_brenner = []
        thetadists          = {}
        thetadists_popt     = {}
        thetadists_popt_err = {}
        EI_gauss            = []
        EI_lorentz          = []
        MSE_loren_f2        = []
        radbins             = {}
        dwt_var_arr         = []
        L_arr               = []
        # start analysis:
        print(f'\nxy_fluctuations(): ------------ start ------------')
        x = self.x[c0:c1]
        y = self.y[c0:c1]
        z = self.z[c0:c1] if hasattr(self,'z') else None
        if correct_outliers_z  and hasattr(self,'z'): #and use_xyz
            # correct outliers in z (it can be long):
            print('xy_fluctuations(): correcting z outliers...')
            z, _ = filters.outlier_smoother(z, m=4, win=correct_outliers_z_win, plots='zoutliers' in plots_lev, figname='z')
        if correct_outliers_xy:
            # correct outliers in xy:
            print('xy_fluctuations(): correcting xy outliers...')
            x, _ = filters.outlier_smoother(x, m=4, win=correct_outliers_xy_win, plots='xyzoutliers' in plots_lev, figname='x')
            y, _ = filters.outlier_smoother(y, m=4, win=correct_outliers_xy_win, plots='xyzoutliers' in plots_lev, figname='y')
        if correct_drift_xy or correct_drift_z:
            # find drift file automatically and correct drift in xyz   #TODO gives error later eg empty z if correct_drift_z=0
            try:
                if correct_drift_xy:
                    x,y,_ = self.correct_xyz_drift(x=x, y=y, z=[], c0=c0, c1=c1, roi=correct_drift_roi, rm_outliers=True, filename=self.cellnum[:-5], locate=True, plots='drift' in plots_lev)
                if correct_drift_z:
                    _,_,z = self.correct_xyz_drift(x=[], y=[], z=z, c0=c0, c1=c1, roi=correct_drift_roi, rm_outliers=True, filename=self.cellnum[:-5], locate=True, plots='drift' in plots_lev)
            except RuntimeError:
                print(f'xy_fluctuations(): ERROR: no drift correction possible. Continue...')
        # xy to um and center:
        x = x*self.umppx*1e-6 #[meters]
        y = y*self.umppx*1e-6 #[meters]
        x = x - np.mean(x)
        y = y - np.mean(y)
        angle_turns = np.unwrap(np.arctan2(y, x))/(2*np.pi)
        x_orig, y_orig = x,y
        # fit and stretch xy ellipse into a circle:
        if stretch_xy:
            print('xy_fluctuations(): xy stretched to a circle.')
            x,y,_,_ = fitEllipse.stretch_ellipse(x.flatten(), y.flatten(), stretchit=True)
        # case xyz, define theta from xyz and offset:
        if hasattr(self, 'z'): # and use_xyz:
            print('xy_fluctuations(): self.z found.')
            z = -z*1e-9       # invert z [meters]
            z -= np.min(z)
            z *= z_correction
            z_orig = z 
            # correct 1-turn periodic modulation in z:
            if stretch_z:
                print('xy_fluctuations(): z corrected from 1-turn periodic modulation.')
                z = filters.correct_sig_modulation(z, angle_turns, polydeg=15, interp_pts=100, method='interp', plots=True, plots_figname='correct_sig_modulation Z')
            # final z shift, z is the dist(bead_center, wall) + .1nm (to avoid gap=0.0):
            z = z - np.min(z) + offset + r_bead + 0.1e-9
            if use_xyz: 
                # find theta from xyz, offset inserted here:
                print(f'xy_fluctuations(): theta defined by xyz points.')
                theta = np.arctan2(z, np.sqrt(x**2 + y**2))
        # tangent speed:
        tan = -np.unwrap(np.arctan2(x, y))
        speed_Hz = np.diff(tan)/(2*np.pi)*self.FPS
        speed_Hz_f = filters.run_win_smooth(speed_Hz, filter_win, algorithm='cumsum', plots=False, usemode='same', pad=100) 
        if negate_speed:
            speed_Hz   = -speed_Hz
            speed_Hz_f = -speed_Hz_f
            print(f'xy_fluctuations(): Speed negated.')
        # threshold speed, remove where abs(speed) < speed_thr:
        if speed_thr:
            idx = np.nonzero(np.abs(speed_Hz_f) > speed_thr)[0]
            print(f'xy_fluctuations(): Careful, speed_thr! Cutting out regions of |speed|<{speed_thr}. Using {len(idx)/len(x)*100:.1f}% of tot.pts.')
            speed_Hz = speed_Hz[idx]
            speed_Hz_f = speed_Hz_f[idx]
            x,y = x[idx], y[idx]
            if hasattr(self, 'z'): 
                z = z[idx]
        # cut and append cw parts at the end of speed trace:
        if separate_switching:
            print(f'xy_fluctuations(): Careful, separate_switching! Appending CW intervals at end of traces.')
            idxs1 = speed_Hz_f > 0
            speed_Hz_f_ccw = speed_Hz_f[idxs1]
            speed_Hz_f_cw  = speed_Hz_f[~idxs1]
            speed_Hz_ccw   = speed_Hz[idxs1]
            speed_Hz_cw    = speed_Hz[~idxs1]
            if speed_thr:
                x_ccw, y_ccw = x[idxs1],  y[idxs1]
                x_cw,  y_cw  = x[~idxs1], y[~idxs1]
                if hasattr(self, 'z'):
                    z_ccw, z_cw = z[idxs1], z[~idxs1]
            else:
                x_ccw, y_ccw = x[:-1][idxs1],  y[:-1][idxs1]
                x_cw,  y_cw  = x[:-1][~idxs1], y[:-1][~idxs1]
                if hasattr(self, 'z'):
                    z_ccw, z_cw = z[:-1][idxs1], z[:-1][~idxs1]
            speed_Hz   = np.append(speed_Hz_ccw, speed_Hz_cw)
            speed_Hz_f = np.append(speed_Hz_f_ccw, speed_Hz_f_cw)
            x = np.append(x_ccw, x_cw)
            y = np.append(y_ccw, y_cw)
            if hasattr(self, 'z'):
                z = np.append(z_ccw, z_cw)
        # define windows:
        if win_n:
            idxs = np.linspace(0, len(x), win_n+1, endpoint=True).astype(int)
        elif win_s:
            idxs = np.arange(0, len(x), int(win_s*self.FPS))
        else:
            raise Exception('xy_fluctuations(): win_n or win_s must be defined.')
        didxs = np.diff(idxs)
        # subsample windows:
        ran = np.sort(np.random.choice(len(idxs)-1, size=int(win_subsample*(len(idxs)-1)), replace=False))
        idxs = idxs[ran]
        didxs = didxs[ran]
        if win_subsample > 1:
            raise RuntimeError('win_subsample must <= 1')
        print(f'xy_fluctuations(): tot.  len: {len(self.x)} pts = {len(self.x)/self.FPS:.3f} s')
        print(f'xy_fluctuations(): c0-c1 len: {len(x)} pts = {len(x)/self.FPS:.3f} s')
        print(f'xy_fluctuations(): {len(didxs)} windows, each of {didxs[0]/self.FPS:.3f} s')
        
        ### 1°loop) only for xy case, cycle all wins to find rmax (max radius in trace) and L, considering offset:
        k = 1
        for i, di in zip(idxs, didxs):
            print(f'xy_fluctuations(): {k}/{len(didxs)}', end='\r')
            xw = x[i:i+di]
            yw = y[i:i+di]
            if center_win_traj:
                xw,yw,_,_ = fitEllipse.stretch_ellipse(xw, yw, stretchit=False)
            # define radius in windows:
            radw = np.hypot(xw, yw)
            # decide to get only points in a portion (beta,Dbeta) of the circle:
            if beta!=None and Dbeta!=None:
                betaidxs = (np.arctan2(yw, xw) > beta) * (np.arctan2(yw,xw) < beta+Dbeta)
                radw = radw[betaidxs]
            foo, radbins[k] = np.histogram(radw, 40, density=False)
            k = k+1
        rmax = np.median([radbins[k][-1] for k in radbins.keys()])
        L = np.sqrt((offset + r_bead)**2 + rmax**2) # offset inserted here.
        theta_min = np.arctan((r_bead+offset)/rmax)
        self.L         = L 
        self.rmax      = rmax
        self.theta_min = theta_min
        print(f'xy_fluctuations(): 1st loop: Given offset:{offset*1e9:.3f}nm, R_b:{r_bead*1e9:.1f}nm. Found r_max:{rmax*1e9:.1f}nm, L:{L*1e9:.1f}nm, theta_min:{theta_min*180/np.pi:.1f}°, L_tether:{(L-r_bead)*1e9:.1f}nm')
        
        ### 2°loop) main analysis in each window:
        k = 1
        for i,di in zip(idxs, didxs):
            print(f'xy_fluctuations(): 2nd loop: {k}/{len(didxs)}', end='\r')
            # windowing:
            xw = x[i:i+di]
            yw = y[i:i+di]
            # in window, fit ellipse, center to 0,0:
            if center_win_traj:
                xw,yw,_,_ = fitEllipse.stretch_ellipse(xw, yw, stretchit=False)
            # get radius and tangent, projecting x,y on circle:
            tanw = np.unwrap(np.arctan2(yw, xw))    # angle phi in win
            radw = np.hypot(xw, yw)                 # radius r in win
            # tangent speed in win: 
            speedw_Hz = speed_Hz[i:i+di]            # motor speed omega
            speedw_Hz_f = speed_Hz_f[i:i+di]
            # correct 1-turn periodic modulation in radw:
            if correct_win_mod:
                radw = filters.correct_sig_modulation(radw, tanw/(2*np.pi), polydeg=15, plots=0)
            # decide to get only points in an arc (beta,Dbeta) of the circle:
            if beta!=None and Dbeta!=None:
                betaidxs = (np.arctan2(yw, xw) > beta) * (np.arctan2(yw,xw) < beta+Dbeta)
                print(f'xy_fluctuations(): getting pts in arc, #sub-windows~{len(np.nonzero(np.diff(np.array(betaidxs).astype(int))==1)[0])}')
                tanw = tanw[betaidxs]
                radw = radw[betaidxs]
                xw = xw[betaidxs]
                yw = yw[betaidxs]
                speedw_Hz = speedw_Hz[betaidxs]
                speedw_Hz_f = speedw_Hz_f[betaidxs]
            # define theta in win:
            if hasattr(self, 'z') and use_xyz:
                # case xyz:
                zw = z[i:i+di]
                thetaw = theta[i:i+di]
                if 'ztheta' in plots_lev:
                    ax_ztheta1.plot(i, np.median(zw), 'o')
                    ax_ztheta1.plot(i, np.mean(zw), '+')
                    ax_ztheta2.plot(i, np.median(thetaw), 'o')
                    ax_ztheta2.plot(i, np.mean(thetaw), '+')
            else:
                # case xy:
                if np.any(radw/L > 1): print(f'xy_fluctuations(): k:{k}, i:{i} Warning, bad value of radw/L in arccos. Clipped in [-1,1].')
                # theta in win from radius and L, clip for arccos:
                thetaw = np.arccos(np.clip(radw/L, -1, 1))    # hook angle theta
            # store sig stats in win:
            thetavar_sig_arr  = np.append(thetavar_sig_arr, np.var(thetaw))
            thetamn_sig_arr   = np.append(thetamn_sig_arr , np.mean(thetaw))
            thetamin_sig_arr  = np.append(thetamin_sig_arr, np.min(thetaw))
            thetamax_sig_arr  = np.append(thetamax_sig_arr, np.max(thetaw))
            speedmn_arr       = np.append(speedmn_arr     , np.median(speedw_Hz))
            speedvar_arr      = np.append(speedvar_arr    , np.var(speedw_Hz))
            radmn_arr         = np.append(radmn_arr       , np.mean(radw))
            # PSD of theta, logbin, crop:
            freq_theta, ps_theta         = PSpectrum.spectrum(thetaw - np.mean(thetaw), self.FPS, plots=False, prints=0, downsample=False)
            logbins_theta, log_psd_theta = PSpectrum.log_avg_spectrum(freq_theta[1:], np.log10(ps_theta[1:]), npoints=40)
            logbins_theta = logbins_theta[psd_c0:psd_c1] # Hz
            log_psd_theta = log_psd_theta[psd_c0:psd_c1]
            # PSD*f^2 of theta (it is = KT/(pi^2 gamma) for f=infty):
            psd_theta_f2 = logbins_theta**2 * 10**log_psd_theta
            log_psd_theta_f2 = np.log10(psd_theta_f2)
            # differential evolution fit to PSD (works better than curve_fit):
            de_bounds = [(1, 900), (0.01e-20, 100e-20)]
            if no_lorentz:
                de_log_lorentzian_f2 = 0
                popt_loren_f2 = [1,1]
            else:
                de_log_lorentzian_f2 = differential_evolution(log_lorentzian_f2_RMSE, de_bounds, args=(logbins_theta, log_psd_theta_f2), popsize=7)
                popt_loren_f2 = de_log_lorentzian_f2.x
            _MSE_loren_f2 = np.mean((log_psd_theta_f2 - log_lorentzian_f2(logbins_theta, *popt_loren_f2))**2)
            # store Lorentzian fit:
            freq_c             = np.append(freq_c,             np.abs(popt_loren_f2[0])) 
            gamma_loren_f2     = np.append(gamma_loren_f2,     popt_loren_f2[1])
            gamma_loren_f2_pts = np.append(gamma_loren_f2_pts, KT/(np.pi**2 * np.mean(psd_theta_f2[-4:]))) 
            MSE_loren_f2       = np.append(MSE_loren_f2,       _MSE_loren_f2)
            # theta prob. distributions, gaussian fit:
            thetadists[k] = np.histogram(thetaw , 40, density=False)
            try:
                thetadists_popt[k], thetadists_popt_err[k] = self.gauss_fit(thetadists[k][1][:-1], thetadists[k][0], return_err=True)
            except RuntimeError:
                thetadists_popt[k] = [0,0,0]
                thetadists_popt_err[k] = 0
                print(f'/nxy_fluctuations(): {k}/{len(didxs)} Warning! gauss fit failed')
            # stiffness from lorentz fit:
            _stif_loren = popt_loren_f2[1]*popt_loren_f2[0]*2*np.pi
            # std from lorentz fit stifness + equip.:
            theta_stds_loren = np.append(theta_stds_loren, np.sqrt(KT/_stif_loren))
            # prepare for angular Drag Coeff:
            if hasattr(self, 'z') and use_xyz:
                # case xyz:
                dist_beadcent_wall = zw
                LL = np.sqrt(dist_beadcent_wall**2 + radw**2)
                #print(f'xyz: L={LL*1e6:.3f} um   xy: L={np.sqrt(L**2 - np.mean(radw)**2)*1e6:.3f} um')
            else:
                # case xy:
                dist_beadcent_wall = np.sqrt(L**2 - radw**2) 
                dist_beadcent_wall = np.clip(dist_beadcent_wall, r_bead+1e-9, np.inf)
                if np.any(dist_beadcent_wall <= r_bead): print('xy_fluctuations(): Warning, dist_beadcent_wall <= r_bead')
                LL = L
            if 'xyz' in plots_lev:
                ax_xyzLL1.plot(i/self.FPS, np.mean(LL)*1e6, 'o')
                ax_xyzLL1.plot(i/self.FPS, np.median(LL)*1e6, '+')
                ax_xyzLL1.set_xlabel('#win')
                ax_xyzLL1.set_ylabel(r'L ($\mu$m)')
                ax_xyzLL2.plot(np.median(speedw_Hz_f), np.mean(LL)*1e6, 'o')
                ax_xyzLL2.set_xlabel('speed (Hz)')
                ax_xyzLL2.set_ylabel(r'L ($\mu$m)')
                fig_xyzLL.tight_layout()
            # store L as array:
            L_arr = np.append(L_arr, np.median(LL))
            # angular (theta) Drag Coeff, combining Brenner's corrections:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                _,_,_, gamma_parallel_brenner, gamma_perp_brenner = Drag.DragTranslationBead.calc_drag(r_bead, dist_beadcent_wall, prints=False)
            _gamma_theta_brenner = np.mean(LL**2 * np.sqrt(gamma_perp_brenner**2 * np.cos(thetaw)**2 + gamma_parallel_brenner**2 * np.sin(thetaw)**2))
            gamma_theta_brenner = np.append(gamma_theta_brenner, _gamma_theta_brenner)

            # plots in 2° cycle, for each window:
            if 1 in plots_lev:
                ax11.plot((i + np.arange(len(speedw_Hz_f)))/self.FPS, speedw_Hz_f, '-')
                ax11.plot(i/self.FPS, np.median(speedw_Hz_f), 'ko', ms=3)
            if 2 in plots_lev:
                ax01.plot(xw*1e9, yw*1e9, '.', ms=1)
                ax02.plot(np.arange(len(speedw_Hz_f))/self.FPS, speedw_Hz_f)
                ax03.plot(np.arange(len(radw))/self.FPS, radw)
                ax04.plot(np.arange(len(tanw))/self.FPS, tanw, '.', ms=1)
                ax01.set_xlabel('x (nm)')
                ax01.set_ylabel('x (nm)')
                p61, = ax61.semilogx(logbins_theta, log_psd_theta_f2, '-o', lw=1, ms=2, alpha=0.3)
                ax61.semilogx(logbins_theta, log_lorentzian_f2(logbins_theta, *popt_loren_f2), '--', color=p61.get_color(), alpha=0.4, lw=1)
                p62, = ax62.semilogx(logbins_theta, np.log10(10**log_psd_theta_f2/logbins_theta**2), '-o', lw=1, ms=2, alpha=0.3)
                ax62.semilogx(logbins_theta, log_lorentzian(logbins_theta, *popt_loren_f2), '--', color=p62.get_color(), alpha=0.4, lw=1)
                ax61.set_xlabel('Freq. (Hz)')
                ax61.set_ylabel(r'$PSD\times f^2$')
                ax62.set_xlabel('Freq. (Hz)')
                ax62.set_ylabel(r'$PSD$')
            if 5 in plots_lev:
                p31, = ax31.plot((i + np.arange(0, len(speedw_Hz_f), 1000))/self.FPS, speedw_Hz_f[::1000], '-', color='0.6', lw=1, zorder=0)
                ax31.set_xlabel('Time (s)')
                ax31.set_ylabel(r'$\omega$ (Hz)')
                ax37.semilogx(logbins_theta, log_psd_theta, '-', lw=1, alpha=0.2, color='0.7', zorder=0)
                ax37.set_xlabel(r'$f$ (Hz)')
                ax37.set_ylabel(r'$\log_{10}$ $PSD_{\theta}$')
                if k in plotlist3:
                    ax31.plot((i + np.arange(0, len(speedw_Hz_f), 1000))/self.FPS, speedw_Hz_f[::1000], 'o', ms=6, alpha=0.5, lw=1, zorder=1)
                    p37, = ax37.semilogx(logbins_theta, log_psd_theta, 'o', ms=3, mfc='none', alpha=1, zorder=1)
                    ax37.semilogx(logbins_theta, log_lorentzian(logbins_theta, *popt_loren_f2), '--', color=p37.get_color(), lw=1.5, zorder=2)
                if beta!=None and Dbeta!=None:
                    ax51.plot(xw, yw, '.', ms=1, alpha=0.2)
                    ax52.plot(radw)
                    ax51.axis('image')
            if 3 in plots_lev:
                ax41.plot(np.hypot(xw, yw) + 8*k*np.std(radw), '.', ms=1)
                ax42.plot(radw + 8*k*np.std(radw), '.', ms=1)
                ax43.plot(np.mod(tanw, 2*np.pi), np.hypot(xw, yw) + 8*k*np.std(radw), '.', ms=1)
                ax44.plot(np.mod(tanw, 2*np.pi), radw + 8*k*np.std(radw), '.', ms=1)
            if 4 in plots_lev:
                self.allan_variance(radw, n_taus=20, FPS=self.FPS, plots=True, clear=False)
            if 'theta phi fluct' in plots_lev:
                ax71.plot(np.median(speedw_Hz_f), np.var(thetaw), 'ko', alpha=0.3)
                ax71.set_xlabel(r'$\omega$ (Hz)')
                ax71.set_ylabel(r'$\sigma^2_{\theta}$ (rad$^2$)')
                ax72.plot(np.median(speedw_Hz_f), np.var(speedw_Hz), 'ko', alpha=0.3)
                ax72.set_xlabel(r'$\omega$ (Hz)')
                ax72.set_ylabel(r'$\sigma^2_{\omega}$ (Hz$^2$)')
                # find dwell times (to cross a fixed displacement) in win and their variance:
                import dwell_times
                dwt_win = 10 # degrees
                dwt,_ = dwell_times.dwellTimes(tanw*180/np.pi, self.FPS, 360/dwt_win, plots=0, progbar=0)
                dwt_var_arr = np.append(dwt_var_arr, np.var(dwt))
                ax73.loglog(np.median(speedw_Hz), np.var(dwt), 'ko', alpha=0.3)
                ax73.set_xlabel(r'$\omega$ (Hz)')
                ax73.set_ylabel(r'$\sigma^2_{dwt}$ ($s^2$)')
                # stepsize from randomness parameter:
                stepsize = np.var(dwt)*np.median(speedw_Hz*2*np.pi)**2/(dwt_win*np.pi/180) # rad
                ax73a.plot(np.median(speedw_Hz), 2*np.pi/stepsize, 'ko', alpha=0.3)
                ax73a.set_xlabel(r'$\omega$ (Hz)')
                ax73a.set_ylabel(r'Steps / Turn')
                ax74.plot(np.var(speedw_Hz), np.var(thetaw), 'ko', alpha=0.3)
                ax74.set_xlabel(r'$\sigma^2_{\omega}$ (Hz$^2$)')
                ax74.set_ylabel(r'$\sigma^2_{\theta}$ (rad$^2$)')
                ax75.plot(np.var(speedw_Hz), np.var(dwt), 'ko', alpha=0.3)
                ax75.set_xlabel(r'$\sigma^2_{\omega}$ (Hz$^2$)')
                ax75.set_ylabel(r'$\sigma^2_{dwt}$ (s$^2$)')
                ax76.plot(np.var(thetaw), np.var(dwt), 'ko', alpha=0.3)
                ax76.set_xlabel(r'$\sigma^2_{\theta}$ (rad$^2$)')
                ax76.set_ylabel(r'$\sigma^2_{dwt}$ (s$^2$)')
                
            k += 1

        # EI from gauss fit:        
        theta_stds_gaus = np.array([thetadists_popt[k][2] for k in thetadists_popt.keys()])
        theta_mns_gaus  = np.array([thetadists_popt[k][1] for k in thetadists_popt.keys()])
        EI_gauss = KT*L_hook/theta_stds_gaus**2
        # EI from var of lorentzian fit and equipartition (EI_lorentz = _stif_loren*L_hook):
        EI_lorentz = KT*L_hook/theta_stds_loren**2
        # EI from var of signal with no fit:
        EI_sig = KT*L_hook/thetavar_sig_arr
        # theta stiffness by signal equipartition: KT=k*variance_signal (theta stif = EI/L_hook):
        stif_sig = KT/thetavar_sig_arr
        # theta stiffness by Gaussian + equipartition: KT=k*variance_gauss:
        stif_gauss = KT/theta_stds_gaus**2
        # theta stiffness by Lorentzian corner freq: k=gamma*fc(*2pi)
        stif_loren = gamma_loren_f2_pts*freq_c*2*np.pi
        # log-error of gaussian fit of radial distributions, median and std:
        popt_err_std = np.std(list(thetadists_popt_err.values()))
        popt_err_med = np.median(list(thetadists_popt_err.values()))
        # store :
        self.EI_gauss            = EI_gauss
        self.EI_lorentz          = EI_lorentz
        self.EI_sig              = EI_sig
        self.speedmn_arr         = speedmn_arr
        self.speedvar_arr        = speedvar_arr
        self.a_ellipse_arr       = a_ellipse_arr
        self.thetamn_sig_arr     = thetamn_sig_arr
        self.thetamin_sig_arr    = thetamin_sig_arr
        self.thetamax_sig_arr    = thetamax_sig_arr
        self.thetavar_sig_arr    = thetavar_sig_arr
        self.gamma_theta_brenner = gamma_theta_brenner
        self.gamma_loren_f2_pts  = gamma_loren_f2_pts
        self.gamma_loren_f2      = gamma_loren_f2
        self.freq_c              = freq_c
        self.radmn_arr           = radmn_arr
        self.L_arr               = L_arr
        
        # Mean Square Diff (log) between gamma_theta_brenner and gamma_loren_f2, to minimize by offset:
        MSE_gamma_loren_brenner = np.log10(np.mean((gamma_loren_f2 - gamma_theta_brenner)**2))
        print(f'\nxy_fluctuations(): MSE_gamma_loren_brenner: {MSE_gamma_loren_brenner:.3f} with offset:{offset*1e9:.3f} nm')
        print()
        if return_MSE_gamma_loren_brenner:
            return MSE_gamma_loren_brenner
        
        # filter on fit:
        #idx_loren = MSE_loren < 0.17
        #idx_gauss = np.array(list(thetadists_popt_err.values())) < .1
        #idx_keep = idx_loren * idx_gauss
        
        if 'xyz' in plots_lev:
            ss = 70
            try: 
                z_orig
            except NameError:
                z_orig = np.zeros((len(x_orig)))
            ax_xyz11.scatter(x_orig[::-ss]*1e9, y_orig[::-ss]*1e9, z_orig[::-ss]*1e9, s=1, linewidths=0, alpha=0.6, c=(np.arange(len(x_orig[::ss])))[::-1])
            ax_xyz11.set_title('orig', fontsize=9)
            ax_xyz12.scatter(x[::-ss]*1e9, y[::-ss]*1e9, z[::-ss]*1e9, s=1, linewidths=0, alpha=0.6, c=(np.arange(len(x[::ss])))[::-1])
            ax_xyz12.set_title('modif', fontsize=9)
            ax_xyz13.scatter((np.arange(len(speed_Hz_f))/self.FPS)[::ss], filters.run_win_smooth(speed_Hz_f, 1501)[::ss], s=1, c=np.arange(len(speed_Hz_f[::ss])))
            ax_xyz14.semilogy(self.speedmn_arr, self.EI_lorentz, 'g.', label='Lorentz')
            ax_xyz14.semilogy(self.speedmn_arr, self.EI_sig, 'y.', label='signal')
            ax_xyz14.semilogy(self.speedmn_arr, self.EI_gauss, 'r.', label='Gauss.')
            ax_xyz11.set_xlabel('x (nm)')
            ax_xyz11.set_ylabel('y (nm)')
            ax_xyz11.set_zlabel('z (nm)')
            ax_xyz13.set_xlabel('Time (s)')
            ax_xyz13.set_ylabel(r'$\omega$ (Hz)')
            ax_xyz14.set_xlabel(r'$\omega$ (Hz)')
            ax_xyz14.set_ylabel(r'$EI$ ($Nm^2$)')
            #ax_xyz13.legend(labelspacing=0, fontsize=8)
            fig_xyz.tight_layout()
            #fig_xyz.savefig(f'/home/francesco/lavoriMiei/cbs/articles/xBFM_rad_fluct/xyz_omega_EI_Rb1000_k{self.key}.png')
            ax_xyz21.scatter(x[::-ss]*1e9, y[::-ss]*1e9, z[::-ss]*1e9, s=1, linewidths=0, alpha=0.6, c=(np.arange(len(x[::ss])))[::-1])
            ax_xyz22.scatter((np.arange(len(speed_Hz_f))/self.FPS)[::ss], filters.run_win_smooth(speed_Hz_f, 1501)[::ss], s=1, c=np.arange(len(speed_Hz_f[::ss])))
            ax_xyz23.semilogy(self.speedmn_arr, self.EI_lorentz, 'g.', label='Lorentz')
            #ax_xyz23.semilogy(self.speedmn_arr, self.EI_sig, 'y.', label='signal')
            #ax_xyz23.semilogy(self.speedmn_arr, self.EI_gauss, 'r.', label='Gauss.')
            ax_xyz21.set_xlabel('x (nm)')
            ax_xyz21.set_ylabel('y (nm)')
            ax_xyz21.set_zlabel('z (nm)')
            ax_xyz22.set_xlabel('Time (s)')
            ax_xyz22.set_ylabel(r'$\omega$ (Hz)')
            ax_xyz23.set_xlabel(r'$\omega$ (Hz)')
            ax_xyz23.set_ylabel(r'$EI$ ($Nm^2$)')
            fig2_xyz.tight_layout()
        if 'theta phi fluct' in plots_lev:
            try:
                pz = np.polyfit(np.log10(speedmn_arr), np.log10(dwt_var_arr), 1)
                po = np.poly1d(pz)
                px = np.linspace(np.min(speedmn_arr), np.max(speedmn_arr), 10)
                ax73.plot(px, 10**po(np.log10(px)), '--', alpha=0.5)
                ax73.set_title(f'dwelltimes in {dwt_win}°, slope:{pz[0]:.1f}', fontsize=8)
            except:
                print('xy_fluctuations(): speed - dwelltime fit failed!')
            from scipy import stats
            # rm outliers from thetavar_sig_arr before linear fit:
            idx = np.nonzero(np.abs(stats.zscore(thetavar_sig_arr))<2)[0]
            po = np.poly1d(np.polyfit(speedvar_arr[idx], thetavar_sig_arr[idx], 1))
            ax74.plot(np.linspace(speedvar_arr.min(), speedvar_arr.max(), 100), po(np.linspace(speedvar_arr.min(), speedvar_arr.max(), 100)), '--')
            ax77.plot(x[::50], y[::50], ',', alpha=0.1)
            ax78.plot(thetamn_sig_arr, '.')
            ax78.set_xlabel('win#')
            ax78.set_ylabel(r'$\langle \theta \rangle$ (rad)')
            ax79.plot(speedmn_arr, '.')
            ax79.set_xlabel('win#')
            ax79.set_ylabel(r'$\omega$ (Hz)')
            fig7.tight_layout()
        if 1 in plots_lev:
            ax12.plot(speedmn_arr, freq_c, 'gs', ms=3, alpha=0.3)
            ax13.plot(speedmn_arr, gamma_loren_f2/(gamma*L_arr**2), 'r+', ms=3, alpha=0.8, label='Lorentz_f2')
            ax13.plot(speedmn_arr, gamma_theta_brenner/(gamma*L_arr**2), 'b.', ms=3, alpha=0.8, label='Brenner')
            
            ax14.plot(speedmn_arr, np.sqrt(thetavar_sig_arr), 'yo', ms=3, label='sig.', alpha=0.3)
            ax14.plot(speedmn_arr, theta_stds_loren, 'go', ms=3, label='Lorentz.', alpha=0.3)
            ax14.plot(speedmn_arr, theta_stds_gaus, 'r+', ms=3, label='Gauss.', alpha=0.3)
            ax15.plot(speedmn_arr, MSE_loren_f2, 'k.')
            ax16.errorbar(speedmn_arr, thetamn_sig_arr, yerr=[thetamn_sig_arr-thetamin_sig_arr, thetamax_sig_arr-thetamn_sig_arr], capsize=0, fmt='o', ms=3, ecolor='0.6', alpha=0.3)
            for k in thetadists:
                p17, = ax17.semilogy(thetadists[k][1][:-1], thetadists[k][0], 'o', ms=2, alpha=0.3)
                ax17.plot(thetadists[k][1][:-1], self.gauss(thetadists[k][1][:-1], *thetadists_popt[k]), lw=1, alpha=0.3, color=p17.get_color())
            ax17.plot(thetadists[1][1][:-1], thetadists[1][0], 'r-', alpha=0.3)
            ax17.plot(thetadists[k][1][:-1], thetadists[k][0], 'k-', alpha=0.3)
            ax18.semilogy(speedmn_arr, EI_lorentz, 'gs', alpha=0.3, ms=3, label='lorentz.equip')
            ax18.semilogy(speedmn_arr, EI_sig, 'yo', alpha=0.3, ms=3, label='signal.equip')
            ax18.semilogy(speedmn_arr, EI_gauss, 'r+', alpha=.3, ms=3, label='gauss.equip')
            ax19.semilogy(speedmn_arr, list(thetadists_popt_err.values()), 'ko', ms=3, alpha=0.3)
            ax19.axhline(popt_err_med + popt_err_std, alpha=0.5)
            ax19.axhline(popt_err_med, alpha=0.5 )
            ax11.set_xlabel('Time (s)')
            ax11.set_ylabel('Speed (Hz)')
            ax12.set_ylabel(r'$f_c$ (Hz)')
            ax12.set_xlabel('Speed (Hz)')
            ax13.set_xlabel('speed (Hz)')
            ax13.set_ylabel(r'$\gamma_\theta / (\gamma_o L^2)$ ')
            ax13.legend(fontsize=5, labelspacing=0)
            ax13.set_title(f'offset:{offset*1e9:.1f}nm', fontsize=8)
            ax14.set_ylabel(r'$\sigma_\theta$')
            ax14.set_xlabel('Speed (Hz)')
            ax14.legend(fontsize=5, labelspacing=0)
            ax15.set_xlabel('Speed (Hz)')
            ax15.set_ylabel('Lorentz_fit MSE')
            ax16.set_xlabel('Speed (Hz)')
            ax16.set_ylabel(r'$<\theta>$')
            ax17.set_xlabel(r'$\theta$')
            ax17.set_ylabel('Counts')
            ax17.set_ylim(ymin=1, ymax=np.max(thetadists[1][0]*1.5))
            ax18.set_xlabel('Speed (Hz)')
            ax18.set_ylabel(r'$EI\,\,(Nm^2)$')
            #ax18.set_xlim(xmin=0)
            ax18.legend(fontsize=5, labelspacing=0)
            ax19.set_ylabel('gauss_fit MSE')
            ax19.set_xlabel('Speed (Hz)')
            ax19.set_title(f'med:{popt_err_med:.1e}', fontsize=8)
            ax11.set_title(f'b.diam:{self.dbead_nm:.0f}nm   key:{self.key}', fontsize=8)
            fig1.tight_layout()
            if savefigs:
                filename = 'xy_fluctuations_' + self.filename.rpartition('/')[-1][:-2] + '_k' + str(self.key) + '_f1.png'
                print(f'xy_fluctuations(): saving {filename}')
                fig1.savefig(filename )
            ax21.plot(x[::10]*1e9, y[::10]*1e9, ',', alpha=0.1)
            ax22.plot(freq_c, 'gs', ms=3)
            ax23.plot(gamma_loren_f2/(gamma*L_arr**2), 'r.', ms=3, alpha=0.8, label='Lorentz_f2')
            #ax23.plot(gamma_loren_f2_pts/(gamma*L_arr**2), 'g.', ms=3, alpha=0.8, label='Lorentz_f2_pts')
            ax23.plot(gamma_theta_brenner/(gamma*L_arr**2), 'b.', ms=3, alpha=0.8, label='Brenner')
            ax24.plot(np.sqrt(thetavar_sig_arr), 'yo', ms=3, label='sig')
            ax24.plot(theta_stds_loren, 'go', ms=3, label='Loren.')
            ax24.plot(theta_stds_gaus, 'r+', ms=3, label='Gauss')
            ax25.plot(stif_loren, 'gs', alpha=0.7, ms=3, label='loren.')
            ax25.plot(stif_sig, 'yo', alpha=0.7, ms=3, label='sig')
            ax25.plot(stif_gauss, 'r+', alpha=0.7, ms=3, label='gauss')
            ax26.plot(thetamn_sig_arr, 'o', ms=3, label=r'<\theta>')
            ax27.plot(MSE_loren_f2, 'ko', ms=3)
            ax28.semilogy(EI_gauss, 'r+', alpha=.7, ms=3, label='Gaus')
            ax28.semilogy(EI_sig, 'yo', alpha=.7, ms=3, label='sig')
            ax28.semilogy(EI_lorentz, 'gs', alpha=0.7, ms=3, label='Loren.')
            ax29.semilogy(list(thetadists_popt_err.values()), 'ko', ms=3)
            ax29.axhline(popt_err_med + popt_err_std)
            ax29.axhline(popt_err_med )
            ax21.set_xlabel('x (nm)')
            ax21.set_ylabel('y (nm)')
            ax21.axis('equal')
            ax22.set_ylabel(r'$f_c$ (Hz)')
            ax22.set_xlabel('win #')
            ax23.set_xlabel('win #')
            ax23.set_ylabel(r'$\gamma_\theta / (\gamma_o L^2)$')
            ax24.set_ylabel(r'$\sigma_\theta$')
            ax24.set_xlabel('win #')
            ax24.legend(fontsize=5, labelspacing=0)
            ax25.legend(fontsize=5, labelspacing=0)
            ax25.set_xlabel('win #')
            ax25.set_ylabel(r'$k_\theta$')
            ax26.set_xlabel('win #')
            ax26.set_ylabel(r'$<\theta>$ (rad)')
            ax27.set_xlabel('win #')
            ax27.set_ylabel('Loren MSE')
            ax28.set_xlabel('win #')
            ax28.set_ylabel(r'$EI\,\,(Nm^2)$')
            ax28.set_xlim(xmin=0)
            ax28.legend(fontsize=5, labelspacing=0)
            ax29.set_ylabel('gauss MSE')
            ax29.set_xlabel('win #')
            ax29.set_title(f'mn:{np.mean([thetadists_popt_err[k] for k in thetadists_popt_err.keys()]):.1e}', fontsize=8)
            ax21.set_title(f'{self.dbead_nm:.0f}nm bead  key:{self.key}', fontsize=8)
            fig2.tight_layout()
            if savefigs:
                filename = 'xy_fluctuations_' + self.filename.rpartition('/')[-1][:-2] + '_k' + str(self.key) + '_f2.png'
                print(f'xy_fluctuations(): saving {filename}')
                fig2.savefig(filename )

        if 2 in plots_lev:
            ax01.axis('image')
            ax02.set_xlabel('Time (s)')
            ax02.set_ylabel('speed (Hz)')
            ax03.set_xlabel('Time (s)')
            ax03.set_ylabel('radial')
            ax04.set_xlabel('Time (s)')
            ax04.set_ylabel('tangential')
            fig0.tight_layout()

        if 5 in plots_lev:
            ax33.semilogy(speedmn_arr, EI_sig,     'yo', alpha=0.4, ms=3, label='Signal')
            ax33.semilogy(speedmn_arr, EI_lorentz, 'go', alpha=0.4, ms=3, label='Lorentz.fit')
            ax33.semilogy(speedmn_arr, EI_gauss,   'ro', alpha=0.4, ms=3, label='Gauss.fit')
            ax33.legend(fontsize=6, labelspacing=0, loc='lower right')
            ax35.plot(speedmn_arr, thetamn_sig_arr, 'o', color='0.6', zorder=0, alpha=0.5)
            ax35.set_xlabel(r'$\omega$ (Hz)')
            ax35.set_ylabel(r'$\langle \theta \rangle$ (rad)')
            ax36.plot(speedmn_arr, np.sqrt(thetavar_sig_arr), 'o', label='sig.', color='0.6', alpha=0.5)
            ax36.set_xlabel(r'$\omega$ (Hz)')
            ax36.set_ylabel(r'$\sigma_{\theta}$ (rad)')
            ax38.plot(speedmn_arr, freq_c, 'o', color='0.6', alpha=0.5)
            ax38.set_xlabel(r'$\omega$ (Hz)')
            ax38.set_ylabel(r'$f_c$ (Hz)')
            ax39.plot(speedmn_arr, gamma_loren_f2/(gamma*L**2), 'o', color='0.6', alpha=0.6, label='Lorentz_f2')
            ax39.set_xlabel(r'$\omega$ (Hz)')
            ax39.set_ylabel(r'$\gamma_\theta \,/\, (\gamma_o L^2)$ ')
            for k in thetadists:
                ax34.plot(thetadists[k][1][:-1], thetadists[k][0], '-', lw=1, color='0.7', alpha=0.3, zorder=0)
                if k in plotlist3:
                    p34, = ax34.plot(thetadists[k][1][:-1], thetadists[k][0], 'o', ms=3, mfc='none', alpha=1, zorder=1)
                    ax34.plot(thetadists[k][1][:-1], self.gauss(thetadists[k][1][:-1], *thetadists_popt[k]), '--', lw=1.5, alpha=1, color=p34.get_color(), zorder=2)
                    ax35.plot(speedmn_arr[k], thetamn_sig_arr[k], 'o', color=p34.get_color(), zorder=10)
                    ax36.plot(speedmn_arr[k], np.sqrt(thetavar_sig_arr[k]), 'o', label='sig.', color=p34.get_color())
                    ax38.plot(speedmn_arr[k], freq_c[k], 'o', color=p34.get_color())
                    ax39.plot(speedmn_arr[k], gamma_loren_f2[k]/(gamma*L**2), 'o', color=p34.get_color(), alpha=1, label='Lorentz_f2')
            ax34.set_xlabel(r'$\theta$ (rad)')
            ax34.set_ylabel('Counts')
            ax34.set_xlim([1.08, 1.4])
            ax33.set_xlabel(r'$\omega$ (Hz)')
            ax33.set_ylabel(r'$EI\,\,(Nm^2)$')
            ax33.set_xlim(left=0)
            ax31.text(-0.3, 0.96, 'a'     , fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax31.transAxes)
            ax33.text(-0.3, 0.96, 'b'     , fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax33.transAxes)
            ax34.text(-0.32, 0.96, r'c$_1$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax34.transAxes)
            ax35.text(-0.32, 0.96, r'c$_2$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax35.transAxes)
            ax36.text(-0.32, 0.96, r'c$_3$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax36.transAxes)
            ax37.text(-0.3, 0.96, r'd$_1$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax37.transAxes)
            ax38.text(-0.3, 0.96, r'd$_2$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax38.transAxes)
            ax39.text(-0.3, 0.96, r'd$_3$', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax39.transAxes)
            ax31.grid(False)
            #ax32.grid(False)
            ax33.grid(False)
            ax34.grid(False)
            ax35.grid(False)
            ax36.grid(False)
            ax37.grid(False)
            ax38.grid(False)
            ax39.grid(False)
            fig3.tight_layout()
            fig3.subplots_adjust(top=0.977, bottom=0.069, left=0.125, right=0.98, hspace=0.401, wspace=0.384)

            if savefigs:
                filename0 = 'xy_fluctuations_' + self.filename.rpartition('/')[-1][:-2] + '_k' + str(self.key) + '_f0.png'
                filename3 = 'xy_fluctuations_' + self.filename.rpartition('/')[-1][:-2] + '_k' + str(self.key) + '_f3.png'
                print(f'xy_fluctuations(): saving {filename0}')
                print(f'xy_fluctuations(): saving {filename3}')
                fig0.savefig(filename0)
                fig3.savefig(filename3)

        if 'ztheta' in plots_lev:
            ax_ztheta1.set_ylabel('zw')
            ax_ztheta2.set_ylabel('thetaw')
        





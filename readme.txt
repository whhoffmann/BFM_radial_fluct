python 3.7.11
ipython 7.26.0
matplotlib 3.4.2
numpy 1.20.3
scipy 1.6.2



import noise_bfm

nb = noise_bfm.BFMnoise(filename='file2000res.npy', key=29)
nb.xy_fluctuations_theta(win_s=3, center_win_traj=1, stretch_xy=1, correct_win_mod=1, plots_lev=[1,2,5], correct_drift_xy=0, use_xyz=0, c0=265000, c1=-500000, offset=80e-9, psd_c1=-5)

nb = noise_bfm.BFMnoise(filename='file2000res.npy', key=20)
nb.xy_fluctuations_theta(win_s=3, center_win_traj=1, stretch_xy=1, correct_win_mod=1, plots_lev=[1,2,5], correct_drift_xy=0, use_xyz=0, c0=100000, c1=1500000, offset=60e-9, psd_c1=-5)





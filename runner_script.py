'''Script created by Wil Hoffmann to process data utilising the Dynamic 
Stiffening of the Flagellar Motor Repository
'''
import openTDMS
import numpy as np
import noise_bfm

#Their data
#nb = noise_bfm.BFMnoise(filename = 'file2000res.npy', key= 29)

#nb.xy_fluctuations_theta(win_s=3, center_win_traj=1, stretch_xy=1, correct_win_mod=1, plots_lev=[1,2,5], correct_drift_xy=0, use_xyz=0, c0=265000, c1=-500000, offset=40e-9, psd_c1=-5)


#My data
directory = 'C:/Users/CBS/Documents/GitHub/BFM_radial_fluct/'
tracked_file = 'CL_220405_190053_Trckd.tdms'
npy_file = 'dic.npy'

dic = openTDMS.tdms_Trckd_to_pydic(path = directory, include_ROI0 = True, inspect = False, dbead_nm = 500)
np.save(npy_file, dic)
data_dic = np.load(npy_file, allow_pickle=1).item()

#Analysis
nb = noise_bfm.BFMnoise(filename = npy_file, key=list(data_dic.keys())[0])

nb.make_filtered_trace(stretch_xy = True, plots = True)


#nb.xy_fluctuations_theta(win_s=3, center_win_traj=1, stretch_xy=1, correct_win_mod=1, plots_lev=[1,2,5], correct_drift_xy=0, use_xyz=0, c0=100000, c1=1500000, offset=23e-9, psd_c1=-5)


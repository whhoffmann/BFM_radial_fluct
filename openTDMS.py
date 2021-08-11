import matplotlib.pyplot as plt
import numpy as np
import re
import nptdms
import os




def tdms_Trckd_to_pydic(path='', filelist=[], excludefiles=[], inspect=True, include_ROI0=False, plot_sub=100, stator='WT', dbead_nm=1000, nm_per_pix=91.3, strain='MT02 deltaCheBR'):
    ''' Build and return a python dict, inserting in it the info from tracked ("_Trckd.tdms") files.
        Each ROI of each _Trckd.tdms file becomes a key of the output dict. 
        The keys in the dic are : 
        ['x', 'y', 'z', 'FPS', 'cellnum', 'strain', 'stator', 'dbead_nm', 'nm_per_pix']

            path           : directory with the .Trckd.tdms files. 
            filelist       : use only the files in this list (must be *_Trckd.tdms). If empty, use all Trckd.tdms files in path 
            inspect [True] : inspect each file, and decide which ROIs to include in the dict
            plot_sub [100] : plot subsampled traces
            stator ['WT'], 
            dbead_nm [1100],  
            nm_per_pix [91.3], 
            strain         : dict parameters to add by hand, all the same for all the files (TODO: variable?)
    '''
    # output dic:
    dout = {}
    k = 0
    # find Trckd.tdms files in path:
    if path and not filelist:
        ll = os.listdir(path)
        filelist = [f for f in ll if f.endswith('Trckd.tdms')]
    filelist = [path+f for f in filelist if f not in excludefiles]
    print(f'tdms_Trckd_to_pydic(): {filelist}')
    # check size of _index files to be small:
    indexfilelist = [path+f for f in ll if f.endswith('Trckd.tdms_index')]
    for i in indexfilelist:
        if os.stat(i).st_size > 2000:
            print(f'tdms_Trckd_to_pydic(): WARNING! tdms_index file')
            print(f'tdms_Trckd_to_pydic():     {i} ')
            print(f'tdms_Trckd_to_pydic(): is big ({os.stat(i).st_size} B). Defrag it first maybe?')
    # calc tot. file size:
    totsize = 0
    for f in filelist:
        totsize += os.stat(f).st_size
    cont = input(f'tdms_Trckd_to_pydic(): Trckd.tdms files tot = {totsize/1e9} GB. Continue [Y]? : ')
    if cont == '': cont = 'Y'
    if cont != 'Y': return
    # open each Trckd.tdms file:
    for f in filelist:
        exclude = []
        # find filename:
        filename = f[::-1].partition('/')[0][::-1] 
        print(f'\ntdms_Trckd_to_pydic(): opening file {filename}')
        print(f'tdms_Trckd_to_pydic(): size = {os.stat(path+filename).st_size/1e9} GB')
        tdms = openTdmsFile(f, print_found=0)
        # find config string:
        CLconfig = tdms['/CL-config/#X'][0]
        # find numb. or ROIs:
        re_nrois = re.search(r'Number of ROI : ([1-4])', CLconfig)
        nrois = int(re_nrois.groups()[0])
        # find FPS: 
        re_fps = re.search(r'Frame Rate : (\d+)', CLconfig)
        FPS = int(re_fps.groups()[0])
        print(f'tdms_Trckd_to_pydic(): nROIs {nrois}')
        print(f'tdms_Trckd_to_pydic(): FPS {FPS}')
        # xyz in ROIs:
        x = {}
        y = {}
        z = {}
        if '/ROI0_Trk/X0' in tdms:
            x[0] = tdms['/ROI0_Trk/X0']
            y[0] = tdms['/ROI0_Trk/Y0']
        if '/ROI0_Trk/Z0' in tdms:
            z[0] = tdms['/ROI0_Trk/Z0']
        if '/ROI1_Trk/X1' in tdms:
            x[1] = tdms['/ROI1_Trk/X1']
            y[1] = tdms['/ROI1_Trk/Y1']
        if '/ROI1_Trk/Z1' in tdms:
            z[1] = tdms['/ROI1_Trk/Z1']
        if '/ROI2_Trk/X2' in tdms:
            x[2] = tdms['/ROI2_Trk/X2']
            y[2] = tdms['/ROI2_Trk/Y2']
        if '/ROI2_Trk/Z2' in tdms:
            z[2] = tdms['/ROI2_Trk/Z2']
        if '/ROI3_Trk/X3' in tdms:
            x[3] = tdms['/ROI3_Trk/X3']
            y[3] = tdms['/ROI3_Trk/Y3']
        if '/ROI3_Trk/Z3' in tdms:
            z[3] = tdms['/ROI3_Trk/Z3']
        # plot each file to inspect:
        if inspect:
            mos = plt.subplot_mosaic([np.arange(nrois), np.arange(nrois, 2*nrois), np.arange(2*nrois, 3*nrois), np.arange(3*nrois,4*nrois)], subplot_kw={} ,num='tdms_Trckd_to_pydic', clear=True)
            for i in np.arange(nrois):
                mos[1][i].plot(x[i][::plot_sub], y[i][::plot_sub], ',', alpha=.3)
                mos[1][i].set_title(f'ROI{i}')
                mos[0].suptitle(filename, fontsize=9)
                mos[0].tight_layout()
            for i in np.arange(nrois, 2*nrois):
                mos[1][i].plot(x[i-nrois][::plot_sub], alpha=0.5) 
                mos[1][i].plot(y[i-nrois][::plot_sub], alpha=0.5)
                mos[1][i].set_title('x,y', fontsize=8)
            for i in np.arange(2*nrois, 3*nrois):
                if f'/ROI{i-2*nrois}_Trk/Z{i-2*nrois}' in tdms:
                    mos[1][i].plot(z[i-2*nrois][::plot_sub], alpha=0.5) 
                    mos[1][i].set_title('z', fontsize=8)
            for i in np.arange(3*nrois, 4*nrois):
                if f'/ROI{i-3*nrois}_Trk/Z{i-3*nrois}' in tdms:
                    mos[1][i].plot(x[i-3*nrois][::plot_sub], z[i-3*nrois][::plot_sub], ',', alpha=0.3) 
                    mos[1][i].set_title('x,z', fontsize=8)
            plt.pause(0.1)
            # user chooses ROIs to exclude:
            exclude = input('tdms_Trckd_to_pydic(): Insert ROIs to exclude.. ?\n    [Ex:`01`. Default: None (just hit Enter)] : ')
        # exclude ROIs chosen:
        include = '0123' if include_ROI0 else '123'
        include = [int(i) for i in include if int(i) in x and i not in exclude]
        # include in dout all ROIs as new keys:
        for i in include:
            dout[k] = {}
            cellnum = filename.partition('_Trckd.tdms')[0] + f'_ROI{i}'
            print(f'tdms_Trckd_to_pydic(): including {cellnum}')
            # update dout:
            dout[k]['x'] = x[i]
            dout[k]['y'] = y[i]
            dout[k]['z'] = z[i]
            dout[k]['cellnum'] = cellnum
            dout[k]['FPS'] = FPS
            dout[k]['stator'] = stator
            dout[k]['dbead_nm'] = dbead_nm
            dout[k]['nm_per_pix'] = nm_per_pix
            dout[k]['strain'] = strain
            k += 1
    return dout
    


def openTdmsFile(filename, print_found=True):
    ''' OLD nptdms versions 
    open a tdms file and put all its structure into a dictionary "d".
    to access the data of one channel, e.g.:
    plot(d[d.keys()[3]], '.')
    
    example of .tdms file porganization:
     -TDMS file
     --group1
     ---group1/channel1
     ---group1/channel2
     --group2
     ---group2/channel1
     ---group2/channel2
     ---group2/channel3
    '''
    if nptdms.__version__ == '1.1.0':
        dout = {}
        tdms_file = nptdms.TdmsFile.read(filename)
        for group in tdms_file.groups():
            group_name = '/'+group.name
            for channel in group.channels():
                channel_name = '/'+channel.name
                data = channel[:]
                dout[group_name + channel_name] = data
        return dout
    else:
        # previous nptdms versions
        f = nptdms.TdmsFile(filename)
        # all the groups in file:
        fg = f.groups()
        # empty dict, will contain all data
        d = {}
        # for each group:
        for grp in fg:
            # channels in grp:
            #fc = f.group_channels(grp)
            fc = f.channels(grp)
            # for each channel:
            for chn in fc:
                if chn.has_data:
                    chname = chn.path.replace('\'','')
                    d[chname] = chn.data
                    if print_found:
                        print('openTdmsFile(): Found '+str(chname) )
        return d




def openTdmsOneROI(filename, prints=False):
    ''' open a movie in tdms file "filename", 
    using the config in "/CL-config/#X" (which must be in the .tdms file)
    returns one single image(t) with all the ROIs 
    '''
    d = openTdmsFile(filename, print_found=prints)
    # find camera configuration string:
    if '/CL-config/#X' in d:
        CLconfig = d['/CL-config/#X'][0]
    else:
        print('        openTdmsOneROI : ERROR, configuration not found')
    if prints:
        print(CLconfig)
    # find frame size (tuple) :  
    re_framesize = re.search(r'Frame Size : (\d+),(\d+)', CLconfig)
    if re_framesize:
        framesize = (int(re_framesize.groups()[0]), int(re_framesize.groups()[1]))
        if prints: print('        Frame size = '+str(framesize))
    else:
        print('        openTdmsOneROI : ERROR framesize')
    # find number of ROIs (int):
    re_nrois = re.search(r'Number of ROI : ([1-4])', CLconfig)
    nrois = int(re_nrois.groups()[0])
    if prints: print('        Num. ROIs = ' + str(nrois))
    # find all images and reshape to 2D:
    if '/CLImg/ROIs' in d:
        imgs = np.array(d['/CLImg/ROIs'])
        imgs = np.reshape(imgs, (int(len(imgs)/(framesize[0]*framesize[1]*nrois)), framesize[1]*nrois, framesize[0]))
        if prints: print('        Num. of frames = '+str(imgs.shape[0]))
    else:
        print('        openTdmsOneROI : ERROR no images found')
    return imgs




def show_movie(imgs,n=1):
    ''' show a simple movie of imgs (imgs from openTdmsOneROI() )
    downsample with n>1'''
    i = 0
    try:
        while True:
            plt.figure(68769)
            plt.clf()
            plt.imshow(imgs[i,::n,::n])
            plt.title(str(i)+'/'+str(len(imgs)))
            i = np.mod(i+1, len(imgs))
            plt.pause(0.001)
    except KeyboardInterrupt:
        print('Stopped')
    



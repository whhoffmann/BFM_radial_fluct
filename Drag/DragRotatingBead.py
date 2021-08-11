# finds drag of a bead rotating on an axis offset from its center, + faxen corrections

import matplotlib.pyplot as plt
import numpy as np

eta = 0.001 # [Pa*s]=[N*s/m^2]]     water viscosity


def calc_drag(bead_radius=0.25e-6, axis_offset=0.05e-6, dist_beadsurf_glass=0.02e-6, k_hook=400.,prints=1, return_all=False):
    ''' returns the FAXEN drag (in pN nm s) on a bead of radius "bead_radius" [m], 
    rotating on a circular trajectory whose center is displaced 
    of "axis_offset" [m] from the bead center,
    at a distance from the glass surface (bead surface - glass surface) of
    "dist_beadsurf_glass" [m]
    block polyhooks: <k_hook>=1.52e-12 dyne cm/rad = 152 pN nm/rad
    see: Comparison of Faxen s correction for a microsphere translating or rotating near a surface PRE 2009 '''
	#drag_mot = 2.0e-4          # [pN nm s] /(rad^2?)   drag Motor, from PNAS paper
	#drag_cyl = 5.              # [pN nm s] (rad^2?)    drag cylinder measured 
	#bead_radius = 0.250e-6     # [m]                   bead radius
	#axis_offset = 0.050e-6     # [m]                   rot.axis offset from center
        #k_hook = 4.0e2             # [pN nm /rad^2],       hook spring constant 
    dist_beadcent_glass = bead_radius + dist_beadsurf_glass   # [m] dist(bead center, surface)
    # correction beta_parallel (torque)
    faxen_1 = 1 - (1/8.)*(bead_radius/dist_beadcent_glass)**3 
    # correction gamma_parallel (force):
    faxen_2 = 1 - (9/16.)*(bead_radius/dist_beadcent_glass) + (1./8)*(bead_radius/dist_beadcent_glass)**3
    faxen_2_1 = 1 - (9/16.)*(bead_radius/dist_beadcent_glass) + (1./8)*(bead_radius/dist_beadcent_glass)**3 - (45/256.)*(bead_radius/dist_beadcent_glass)**4 - (1./16)*(bead_radius/dist_beadcent_glass)**5
    rot_bulk = 8*np.pi*eta*bead_radius**3
    transl_bulk = 6*np.pi*eta*bead_radius*axis_offset**2 
    rot   = 8*np.pi*eta*bead_radius**3/faxen_1 
    trans = 6*np.pi*eta*bead_radius*axis_offset**2/faxen_2
    drag_bead = rot + trans #8*np.pi*eta*bead_radius**3/faxen_1 + 6*np.pi*eta*bead_radius*axis_offset**2/faxen_2
    #drag_bead = 8*np.pi*eta*bead_radius**3/faxen_1 + 6*np.pi*eta*bead_radius*axis_offset**2/faxen_2_1
    drag_bead_pNnms = drag_bead*1e21
    if prints:	    
        print('using faxen_2 correction')
        print("bead drag : "+str(drag_bead_pNnms)+" pN nm s")
        print("charact.time on hook: "+str(1000*drag_bead_pNnms/k_hook)+" ms")
        print("charact.freq. on hook: "+str(1./(drag_bead_pNnms/k_hook))+" Hz")
        print('faxen 1 = '+str(faxen_1))
        print('faxen 2_1 = '+str(faxen_2_1))
        print('faxen 2 = '+str(faxen_2))
        print(f'bulk rot drag = {rot_bulk *1e21}')
        print(f'bulk trasl drag = {transl_bulk*1e21}')
        print(f'bulk trasl + rot  = {(transl_bulk + rot_bulk)*1e21}')

    if return_all:
        return drag_bead_pNnms, rot, trans
    else:
        return drag_bead_pNnms



def calc_drag_brenner(bead_radius, axis_offset, dist_beadsurf_glass, return_all=False):
    ''' BRENNER formula 
    return gamma_pNnms '''
    z3 = 1.20205 # Riemann Z(3)
    rot = (8 * np.pi * eta * bead_radius**3) * (z3 - 3*(np.pi**2/6 - 1)*(dist_beadsurf_glass/bead_radius)) 
    trans = np.abs((6 * np.pi * eta * axis_offset**2 * bead_radius) * (8/15 * np.log(dist_beadsurf_glass/bead_radius) - 0.9588))
    gamma = rot + trans
    gamma_pNnms = gamma*1e21
    if return_all:
        return gamma_pNnms, rot, trans
    else:
        return gamma_pNnms

    

def compare_drags(bead_radius=500e-9, axis_offset=200e-9):
    ''' compare FAXEN and BRENNER drag '''
    #dist_beadsurf_glass = np.linspace(0.001*bead_radius, 10*bead_radius, 10000)
    dist_beadsurf_glass = np.linspace(1e-9, 20*bead_radius, 10000)
    gamma_bulk_rot = (8 * np.pi * eta * bead_radius**3) 
    gamma_bulk_transl = ((6 * np.pi * eta * axis_offset**2 * bead_radius))
    gamma_bulk_tot = gamma_bulk_transl + gamma_bulk_rot
    gamma_manouk_pNnms, rot_manouk, trans_manouk = calc_drag_brenner(bead_radius, axis_offset, dist_beadsurf_glass, return_all=True)
    gamma_faxen_pNnms, rot_faxen, trans_faxen = calc_drag(bead_radius=bead_radius, axis_offset=axis_offset, dist_beadsurf_glass=dist_beadsurf_glass, prints=False, return_all=True)
    gamma_faxen = gamma_faxen_pNnms*1e-21
    gamma_manouk = gamma_manouk_pNnms*1e-21
    plt.figure('compare_drags 1 phi', clear=True)
    plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_tot, len(dist_beadsurf_glass)), 'k-', label='bulk tot.', lw=1)
    plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_rot, len(dist_beadsurf_glass)), 'g-', label='bulk rot', lw=1)
    plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_transl, len(dist_beadsurf_glass)), 'y-', label='bulk tranls', lw=1)
    plt.plot(dist_beadsurf_glass, gamma_faxen , 'k'    , label='faxen tot'    ,lw=3)
    plt.plot(dist_beadsurf_glass, rot_faxen   , 'g'    , label='faxen rot'    ,lw=2)
    plt.plot(dist_beadsurf_glass, trans_faxen , 'y'    , label='faxen transl' ,lw=2)
    plt.plot(dist_beadsurf_glass, gamma_manouk, 'k--'  , label='manouk tot'   ,lw=3)
    plt.plot(dist_beadsurf_glass, rot_manouk  , 'g--'  , label='manouk rot'   ,lw=2)
    plt.plot(dist_beadsurf_glass, trans_manouk, 'y--'  , label='manouk transl',lw=2)
    plt.legend(fontsize=8)
    plt.xlabel('bead surface to surface (m)')
    plt.ylabel('Drag (Nms)')
    plt.xscale('log')
    plt.title(f'$r_b$={bead_radius} m  $r_e$={axis_offset} m')
    plt.ylim(ymin=0)
    plt.figure('compare_drags 2 phi', clear=True)
    plt.subplot(121)
    plt.plot(dist_beadsurf_glass*1e9, np.repeat(gamma_bulk_tot/gamma_bulk_tot, len(dist_beadsurf_glass)), 'k:', label=r'$\beta_0=$'+f'{gamma_bulk_tot*1e21:.2f} pN nm s', lw=1)
    #plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_rot, len(dist_beadsurf_glass)), 'g-', label='bulk rot', lw=1)
    #plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_transl, len(dist_beadsurf_glass)), 'y-', label='bulk tranls', lw=1)
    plt.plot(dist_beadsurf_glass*1e9, gamma_faxen/gamma_bulk_tot , 'k--'    , label='faxen', lw=3)
    #plt.plot(dist_beadsurf_glass, rot_faxen   , 'g'    , label='faxen rot'    ,lw=2)
    #plt.plot(dist_beadsurf_glass, trans_faxen , 'y'    , label='faxen transl' ,lw=2)
    plt.plot(dist_beadsurf_glass*1e9, gamma_manouk/gamma_bulk_tot, 'k-'  , label='Brenner', lw=3)
    #plt.plot(dist_beadsurf_glass, rot_manouk  , 'g--'  , label='manouk rot'   ,lw=2)
    #plt.plot(dist_beadsurf_glass, trans_manouk, 'y--'  , label='manouk transl',lw=2)
    plt.legend(fontsize=8)
    plt.xlabel('bead surface to surface (nm)')
    plt.ylabel(r'$\beta/\beta_0$')
    plt.xscale('log')
    plt.title(f'$R_b$={bead_radius} m  $r_e$={axis_offset} m')
    plt.ylim(ymin=0)
    plt.subplot(122)
    plt.plot(dist_beadsurf_glass*1e9, np.repeat(gamma_bulk_tot, len(dist_beadsurf_glass)), 'k:', label=r'$\beta_0=$'+f'{gamma_bulk_tot*1e21:.2f} pN nm s', lw=1)
    #plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_rot, len(dist_beadsurf_glass)), 'g-', label='bulk rot', lw=1)
    #plt.plot(dist_beadsurf_glass, np.repeat(gamma_bulk_transl, len(dist_beadsurf_glass)), 'y-', label='bulk tranls', lw=1)
    plt.plot(dist_beadsurf_glass*1e9, gamma_faxen, 'k--'    , label='faxen', lw=3)
    #plt.plot(dist_beadsurf_glass, rot_faxen   , 'g'    , label='faxen rot'    ,lw=2)
    #plt.plot(dist_beadsurf_glass, trans_faxen , 'y'    , label='faxen transl' ,lw=2)
    plt.plot(dist_beadsurf_glass*1e9, gamma_manouk, 'k-'  , label='Brenner', lw=3)
    #plt.plot(dist_beadsurf_glass, rot_manouk  , 'g--'  , label='manouk rot'   ,lw=2)
    #plt.plot(dist_beadsurf_glass, trans_manouk, 'y--'  , label='manouk transl',lw=2)
    plt.legend(fontsize=8)
    plt.xlabel('bead surface to surface (nm)')
    plt.ylabel(r'$\beta$')
    plt.xscale('log')
    plt.title(f'$R_b$={bead_radius} m  $r_e$={axis_offset} m')
    plt.ylim(ymin=0)
    
    plt.figure('compare_drags 3 phi', clear=True, figsize=(5,4))
    plt.subplot(111)
    plt.plot(dist_beadsurf_glass*1e9, np.repeat(gamma_bulk_tot, len(dist_beadsurf_glass))*1e21, 'k:', label=r'$\gamma_{\phi\,0}=$'+f'{gamma_bulk_tot*1e21:.3f} pN nm s', lw=3)
    plt.plot(dist_beadsurf_glass*1e9, gamma_faxen*1e21, 'k--'    , label='Faxen', lw=3)
    plt.plot(dist_beadsurf_glass*1e9, gamma_manouk*1e21, 'k-'  , label='Brenner', lw=3)
    plt.legend(fontsize=10)
    plt.xlabel(r'$s$ (dist. bead surface to wall, nm)',fontsize=12)
    plt.ylabel(r'$\gamma_{\phi}$ (pN nm s)',fontsize=12)
    plt.xscale('log')
    plt.title(f'$R_b$={bead_radius*1e9:.0f} nm  $\langle r \\rangle $={axis_offset*1e9:.0f} nm')
    #plt.ylim([3,6])
    plt.tight_layout()



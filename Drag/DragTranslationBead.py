
import matplotlib.pyplot as plt
import numpy as np


def calc_drag(bead_radius, dist_beadcent_wall, prints=True):
    '''
    Calculate drag for a translating bead gamma0, plus faxen and brenner corrections.
        bead_radius : [m]
        dist_beadcent_wall : distance center of bead to wall, careful to be >= bead_radius

    return gamma_0 (=6 pi eta radius), gamma_parallel_faxen, gamma_perp_faxen, gamma_parallel_brenner, gamma_perp_brenner
    in [Pa s]
    
    See: 
    Comparison of Faxén’s correction for a microsphere translating or rotating near a surface PRE 2009
    The slow motion of a sphere through a viscous fluid towards a plane surface 1961
    '''
    eta = 0.001 # [Pa*s]=[N*s/m^2]]     water viscosity
    a_s = np.clip(bead_radius/dist_beadcent_wall, 0,1)#0.85)
    # Faxen (from "Comparison of Faxén correction ..." 2009):
    gamma0 = 6*np.pi*eta*bead_radius
    gamma_parallel_faxen = gamma0/(1 - (9/16)*(a_s) + (1/8)*(a_s)**3)
    gamma_perp_faxen     = gamma0/(1 - (9/8 )*(a_s) + (1/2)*(a_s)**3)
    # Brenner (from "The slow motion of ..." 1961):
    sinh = np.sinh
    h = dist_beadcent_wall
    b = bead_radius
    dist_beadsurf_glass = h - b
    alpha = np.log(h/b + np.sqrt((h/b)**2 - 1))
    SUM = 0
    for n in range(1,20):
        A = 2*sinh((2*n+1)*alpha) + (2*n+1)*sinh(2*alpha)
        B = 4*sinh((n+0.5)*alpha)**2 - (2*n+1)**2*sinh(alpha)**2
        S1 = n*(n+1)/((2*n-1)*(2*n+3))
        S2 = A/B - 1
        SUM = SUM + (S1*S2)
    lambd = 4/3 * sinh(alpha) * SUM 
    gamma_perp_brenner = gamma0 * lambd    
    gamma_parallel_brenner = np.abs(gamma0 * (8/15 * np.log(dist_beadsurf_glass/bead_radius) - 0.9588))

    if prints and type(a_s) != np.ndarray:
        print(f'dist_beadcent_wall = {bead_radius/a_s:.2e}')
        print(f'gamma0                   = {gamma0:.2e}')
        print(f'gamma_parallel_faxen     = {gamma_parallel_faxen:.2e} = {gamma_parallel_faxen/gamma0:.3f} gamma0')
        print(f'gamma_perp_faxen         = {gamma_perp_faxen:.2e} = {gamma_perp_faxen/gamma0:.3f} gamma0')
        print(f'gamma_parallel_brenner   = {gamma_parallel_brenner:.2e} = {gamma_parallel_brenner/gamma0:.3f} gamma0')
        print(f'gamma_perp_brenner       = {gamma_perp_brenner:.2e} = {gamma_perp_brenner/gamma0:.3f} gamma0')

    return gamma0, gamma_parallel_faxen, gamma_perp_faxen, gamma_parallel_brenner, gamma_perp_brenner




def plot_drags(bead_radius=[1000e-9, 505e-9, 250e-9]):
    ''' plot drags from calc_drag '''
    cols = iter(['b','g','r'])
    fig1 = plt.figure('plot_drags 1',clear=True)
    fig2 = plt.figure('plot_drags 2',clear=True)
    fig3 = plt.figure('plot_drags 3',clear=True)
    ax11 = fig1.add_subplot(111)
    ax21 = fig2.add_subplot(111)
    ax31 = fig3.add_subplot(111)

    for _bead_radius in bead_radius:
        dist_beadcent_wall = np.linspace(_bead_radius*1.001, _bead_radius*100, 10000)
        gamma_parallel_brenner = []
        gamma_perp_brenner     = []
        gamma_perp_faxen       = []
        gamma_parallel_faxen   = []
        for d in dist_beadcent_wall:
            _gamma0, _gamma_parallel_faxen, _gamma_perp_faxen, _gamma_parallel_brenner, _gamma_perp_brenner = calc_drag(_bead_radius, d, prints=False)
            gamma_parallel_faxen   = np.append(gamma_parallel_faxen, _gamma_parallel_faxen)
            gamma_perp_faxen       = np.append(gamma_perp_faxen, _gamma_perp_faxen)
            gamma_parallel_brenner = np.append(gamma_parallel_brenner, _gamma_parallel_brenner)
            gamma_perp_brenner     = np.append(gamma_perp_brenner, _gamma_perp_brenner)
            gamma0 = _gamma0
        
        col = next(cols)
        ax11.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_perp_faxen/gamma0,       label=f'$\gamma_\perp/\gamma_o$ ($R_b=${_bead_radius*1e9:.0f} nm)',     color=col, ls='-', lw=2)
        ax11.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_parallel_faxen/gamma0,   label=f'$\gamma_\parallel/\gamma_o$ ($R_b=${_bead_radius*1e9:.0f} nm)', color=col, ls='--', lw=2)
        ax21.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_perp_brenner/gamma0,     label=f'$\gamma_\perp/\gamma_o$ ($R_b=${_bead_radius*1e9:.0f} nm)',     color=col, ls='-', lw=2)
        ax21.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_parallel_brenner/gamma0, label=f'$\gamma_\parallel/\gamma_o$ ($R_b=${_bead_radius*1e9:.0f} nm)', color=col, ls='--', lw=2)

        ax31.loglog((dist_beadcent_wall - _bead_radius)*1e9, gamma_perp_faxen/gamma0,     label=r'$\gamma_{\perp,F}\,/\gamma_o$',     color='k', ls='--', lw=2)
        ax31.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_parallel_faxen/gamma0,   label=r'$\gamma_{\parallel,F}\,/\gamma_o$', color='0.4', ls='--', lw=2)
        ax31.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_perp_brenner/gamma0,     label=r'$\gamma_{\perp,B}\,/\gamma_o$',     color='k', ls='-', lw=2)
        ax31.plot((dist_beadcent_wall - _bead_radius)*1e9, gamma_parallel_brenner/gamma0, label=r'$\gamma_{\parallel,B}\,/\gamma_o$', color='0.4', ls='-', lw=2)
        ax31.axhline(1, ls=':', lw=2, color='k', label=f'$\gamma_o$ = {gamma0:.1e}')
        
    ax11.set_title('Translation Faxen corrections')
    ax21.set_title('Translation Brenner corrections')
    ax11.axhline(1, ls=':', label=r'$\gamma_o$')
    ax21.axhline(1, ls=':', label=r'$\gamma_o$')
    ax11.legend()
    ax21.legend()
    ax31.legend()
    ax11.set_ylabel(r'$\gamma/\gamma_o$')
    ax11.set_xlabel(r'Distance(plane - bead surface) (nm)')
    ax21.set_ylabel(r'$\gamma/\gamma_o$')
    ax21.set_xlabel(r'Distance(plane - bead surface) (nm)')
    ax31.set_ylabel(r'$\gamma/\gamma_o$')
    ax31.set_xlabel(r'$s$ (dist. bead surface to wall) (nm)')
    ax31.grid(0)
    ax31.set_xlim([1,6000])
    ax31.set_ylim([.5,90])
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()



def xy_fluctuations_drag(bead_radius=500e-9, L=700e-9):
    ''' Brenner angular drag for a bead translating along an arc of radius L perpendicular to a surface.
    done for the radial fluctuations and stiffness of the hook. 
    see noise_bfm.xy_fluctuations_theta()
    '''
    gamma_parallel_brenner = []
    gamma_perp_brenner     = []
    theta_min = np.arcsin(bead_radius/L)
    theta = np.linspace(theta_min, np.pi/2, 5000)[1:-1]
    dist_beadcent_wall = L*np.sin(theta)
    print(dist_beadcent_wall)
    for d in dist_beadcent_wall:
        gamma0, _, _, _gamma_parallel_brenner, _gamma_perp_brenner = calc_drag(bead_radius, d, prints=False)
        gamma_parallel_brenner = np.append(gamma_parallel_brenner, _gamma_parallel_brenner)
        gamma_perp_brenner     = np.append(gamma_perp_brenner, _gamma_perp_brenner)
    # combine Brenner corrections:
    gamma_theta = L**2 * np.sqrt(gamma_perp_brenner**2 * np.cos(theta)**2 + gamma_parallel_brenner**2 * np.sin(theta)**2)
    # translation drag:
    #import DragRotatingBead
    # TODO
    plt.figure('xy_fluctuations_drag 1 theta', clear=True)
    plt.subplot(121)
    plt.semilogy((dist_beadcent_wall - bead_radius)*1e9, gamma_theta/(gamma0*L**2), label=r'$\gamma_\theta/(\gamma_o L^2)$')
    plt.xlabel(r'|bead surf., plane| (nm)')
    plt.legend()
    plt.title(f'$R_b:{bead_radius*1e9:.1f}$  L:{L*1e9:.1f}')
    plt.subplot(122)
    plt.semilogy((dist_beadcent_wall - bead_radius)*1e9, gamma_theta, label=r'$\gamma_\theta$')
    plt.xlabel(r'|bead surf., plane| (nm)')
    plt.legend()
    plt.tight_layout()

    plt.figure('xy_fluctuations_drag theta 2', clear=True)
    plt.subplot(221)
    plt.semilogy(theta, gamma_theta/(gamma0*L**2), label=r'$\gamma_\theta/(\gamma_o L^2$)')
    plt.title(f'$R_b:{bead_radius*1e9:.0f} nm$  $L:{L*1e9:.0f} nm$')
    plt.xlabel(r'$\theta$')
    plt.legend()
    plt.subplot(222)
    plt.semilogy((dist_beadcent_wall - bead_radius)*1e9, gamma_theta/(gamma0*L**2), label=r'$\gamma_\theta/(\gamma_o L^2)$')
    plt.xlabel(r'|bead surf., plane| (nm)')
    plt.legend()
    plt.subplot(223)
    plt.semilogy(theta, gamma_perp_brenner/gamma0, label=r'$\gamma_\perp/\gamma_o$')
    plt.plot(theta, gamma_parallel_brenner/gamma0, label=r'$\gamma_\parallel/\gamma_o$')
    plt.xlabel(r'$\theta$')
    plt.legend()
    plt.subplot(224)
    plt.semilogy((dist_beadcent_wall - bead_radius)*1e9, gamma_perp_brenner/gamma0    , label=r'$\gamma_\perp/ \; \gamma_o$')
    plt.plot((dist_beadcent_wall - bead_radius)*1e9, gamma_parallel_brenner/gamma0, label=r'$\gamma_\parallel/ \; \gamma_o$')
    plt.xlabel(r'|bead surf., plane| (nm)')
    plt.legend()
    plt.tight_layout()
    
    return gamma0, gamma_parallel_brenner, gamma_perp_brenner




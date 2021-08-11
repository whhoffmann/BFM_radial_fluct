# fitEllipse.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
import multiprocessing as mulpr
import time



def stretch_ellipse(x,y, stretchit=True, plots=False):
    ''' fit ellipse, decide if stretch it to be circular, center to origin. 
    When stretchit=True stretches the short axis to become equal to the long axis '''
    # fit ellipse on xy :
    xx0,yy0,center0,a,b,phi = makeBestEllipse(x,y, nel=50)
    # rotate xy so major axis is vertical: (check if this is always the case :NO!)
    x_rot, y_rot = rotateArray(np.array((x,y)), -phi)
    # fit again an ellipse on the rotated xy:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # translate to 0,0 and scale x to make a circle from the ellipse:
    if a>b:
        x_rot = (x_rot - center[0])
        if stretchit:
            y_rot = (y_rot - center[1])*a/b
        else:
            y_rot = (y_rot - center[1])
    else:
        if stretchit:
            x_rot = (x_rot - center[0])*b/a
        else:
            x_rot = (x_rot - center[0])
        y_rot = (y_rot - center[1])
    # fit again on the scaled circular x_rot y_rot data:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    if plots:
        plt.figure('stretch_ellipse', clear=True)
        plt.subplot(211)
        plt.plot(x, y, ',', ms=1, alpha=0.1)
        plt.plot(xx0, yy0,'-')
        plt.title('orig.', fontsize=9)
        plt.axis('equal')
        plt.subplot(212)
        plt.plot(x_rot, y_rot, ',', ms=1, alpha=0.1)
        plt.plot(xx,yy,'-')
        plt.title(f'rotated and {"" if stretchit else "NOT"} stretched', fontsize=9)
        plt.axis('equal')
    return x_rot, y_rot, a, b



def fitEllipse(x,y):
    """Algorithm from Fitzgibbon et al 1996, Direct Least Squares Fitting of Ellipsees.  
    Formulated in terms of Langrangian multipliers, rewritten as a generalized eigenvalue problem. """
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a



def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])



def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c)) # getting warning from (a-c)=0



def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b-a*c)*((c-a)*np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a)) # getting warning from (a-c)=0
    down2 = (b*b-a*c)*((a-c)*np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
    res1 = np.sqrt(abs(up/down1))
    res2 = np.sqrt(abs(up/down2))
    return np.array([res1, res2])



def makeBestEllipse(x,y, nel=100):
    ''' x,y : input position data 
    nel: number of pts in best ellipse
    returns: xx,yy,center,a,b,phi parameters of best ellipse'''
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    a,b = ellipse_axis_length(a)
    epts = np.arange(0, 2*np.pi, 2*np.pi/nel)
    xx = center[0] + a*np.cos(epts)*np.cos(phi) - b*np.sin(epts)*np.sin(phi)
    yy = center[1] + a*np.cos(epts)*np.sin(phi) + b*np.sin(epts)*np.cos(phi)
    return xx,yy,center,a,b,phi



def rotateArray(a,th):
    ''' rotates the array a of the angle theta (rad)'''
    R = np.array(([np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]))
    rota = np.dot(R,a)
    return rota



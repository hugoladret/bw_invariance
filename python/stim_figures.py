# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:38:20 2020

@author: Admin
This is to generate the illustration of stimulations

"""

import matplotlib.pyplot as plt
import numpy as np
import MotionClouds as mc

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

B_thetas = np.linspace(np.pi/2, 0.05, 3)/ 2.5
colors = plt.cm.inferno(np.linspace(.8, .2, len(B_thetas)))

# Spatial frequency, in cpd
sf = 0.0616279 # see stimulation notebooks
sf_0 = sf
B_sf = sf_0


def generate_cloud(theta, b_theta, phase,
                   N_X, N_Y, seed, contrast=1.,
                   transition=False):
    
    
    fx, fy, ft = mc.get_grids(N_X, N_Y, 1)
    disk = mc.frequency_radius(fx, fy, ft) < .5

    if b_theta == 0 : 
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                 V_X=0., V_Y=0., B_V=0.,
                                 sf_0=sf_0, B_sf=B_sf,
                                 theta=0, B_theta=b_theta)
        mc_i = np.rot90(mc_i)
    else :
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                 V_X=0., V_Y=0., B_V=0.,
                                 sf_0=sf_0, B_sf=B_sf,
                                 theta=theta, B_theta=b_theta)
      
    im_ = np.zeros((N_X, N_Y, 1))
    im_ += mc.rectif(mc.random_cloud(mc_i, seed=seed),
                     contrast=2)
    im_ += -.5
    return im_[:,:,0]

def generate_gratings(n_sins, imsize, div):
    sinwave = np.sin(np.linspace(0, np.pi * n_sins, imsize))
    grating = np.tile(sinwave, (imsize, 1))
    
    return grating/div
    

# Generate the MotionClouds
fig, axs = plt.subplots(figsize = (16,8), ncols = len(B_thetas), nrows = 1,
                        gridspec_kw = {'wspace':0.01, 'hspace':0.05})

theta = np.pi/4
N_X, N_Y = 512, 512

for ibt in range(0,len(B_thetas)) :
    ax = axs[ibt]
    img = generate_cloud(theta = theta, b_theta = B_thetas[ibt], phase = 0,
                       N_X = N_X, N_Y = N_Y, seed = 42, contrast=1.,
                       transition=False)
    im = ax.imshow(img, cmap = 'gray', interpolation = 'bilinear')
    im.set_clim(-1,1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect('equal')

    for edge in ['top', 'bottom', 'left', 'right'] :
        ax.spines[edge].set_color(colors[ibt])
        ax.spines[edge].set_linewidth(2)  
        
fig.savefig('./figs/fig1e.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)


# Generate the MotionClouds distributions (fig1e)
from scipy.special import i0 as I0

def vm(theta, amp, theta0, Btheta):
     return amp * np.exp((np.cos(2*(theta-theta0))-1) / 4 / Btheta**2)

fig, ax = plt.subplots(figsize = (8,6))

# These aren't exactly the values of Bthetas we used, but 
# they are on the same illustrative range (can't do 0 in a real VM)
B_thetas = np.linspace(np.pi/2, 0.115 , 3) / 2.5
lab_bt = np.linspace(np.pi/2, 0. , 8) / 2.5

labels = [r'$\frac{\pi}{5}$',
          r'$\frac{\pi}{2.5}$',
          '0']
for i, bt in enumerate(B_thetas) :
    xs = np.linspace(0, np.pi, 15000)
    vonmises = vm(theta = xs,
              theta0 = np.pi/2,
              Btheta = bt,
              amp = 1)
    ax.plot(np.linspace(0, np.pi, 15000),
            vonmises,
           color = colors[i],
           label = labels[i])

ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels(['-90', '-45', r'$\theta_{0}$', '+45', '+90'])
ax.tick_params(axis='both', labelsize=12)

ax.set_xlabel('Stimulation orientation (°)', fontsize = 14)
ax.set_ylabel('Distribution energy (u.a.)', fontsize = 14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(-.05, 1.1)
fig.legend(ncol = 1, fontsize = 14, frameon = True, title = r'B$_\theta$')
fig.tight_layout()
fig.savefig('./figs/fig1f.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)





# Generate the gratings (fig1a)
fig, axs = plt.subplots(figsize = (16,8), ncols = len(B_thetas), nrows = 1,
                        gridspec_kw = {'wspace':0.01, 'hspace':0.05})
contrasts = [1, 2, 3]
for ibt in range(0,len(B_thetas)) :
    ax = axs[ibt]
    grat = generate_gratings(n_sins = 20, imsize = 500, div = contrasts[ibt])
    im = ax.imshow(grat, cmap = 'gray', interpolation = 'bilinear')
    
    im.set_clim(-1,1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect('equal')
    
fig.savefig('./figs/fig1a.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)
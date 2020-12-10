# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:18:04 2020

@author: Hugo
"""


import stim
import plots
import LIF
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Parameters
## Simulation parameters 
T = 50 # total simtime ; ms
dt = 0.01 #timestep ; ms
n_repeat = 2 # number of time we rerun the whole stimulation set, used for trial-to-trial variance

## LIF parameters, note that we scale everything later to speed up computations
## You probably don't want to touch this
Rm = 1 # resistance (kOhm)
Cm = 10 # capacitance (uF)
tau_m = Rm*Cm # time constant (msec)
refrac_time = 1. # refractory period (msec)
Vth = 1. # spike threshold (V)

## Stimulation parameters
n_pars = 10 #number of parameters, either contrast or bandwidth
contrasts = np.linspace(1., 8., n_pars) #stimulation contrast, max = 5 is a good idea
bandwidths = np.linspace(.3, .8, n_pars) # stimulation bandwidth, it's sigma of gaussian

## Finn parameters
k = 3.5 # power law scale 
a = -.5 # power law exponent
loc = .8 # noise normal law center
scale = .5 # noise normal law var

## Bandwidth parameters
k_bw = 3.5 # other neurons' power law scale
a_bw = -.5 # multiplier of bw on other neurons

## Plotting parameters
labels = bandwidths #rescale for actual stim values
colors = plt.cm.inferno(np.linspace(.9, .2, len(bandwidths))) #tc colormap

# Initialization
tot_steps = int(T/dt) # length of the time vector 
time = np.linspace(0, T+dt, tot_steps) # time vector

all_hwhhs = []
a_bws = np.linspace(-0., -4.5, 5)
for a_bw in tqdm(a_bws) :
    # Stimulations
    input_tcs = []  
    mod_tcs = []
    pwlaws = []  
    for i, bw in enumerate(bandwidths) :
        inp = stim.generate_stim(mu = 0., sig = bw, max_amp = np.max(contrasts))
        
        # Compute the right part
        new_pwlaw = stim.power_law(k = np.max(inp), 
                                    x = np.linspace(1, 3, len(inp)//2),
                                    a = a_bw*np.exp(bw))
        mult = inp[len(inp)//2:]-(1/new_pwlaw)
        mult[mult<0] = 0
        
        # Compute the left part 
        mult_left = mult[::-1]
        
        mod_tcs.append(np.concatenate((mult_left, mult)))
        pwlaws.append(new_pwlaw)
        input_tcs.append(inp)
    
    
    # Simulation
    out_vms, out_spikes = [], []
    for inp in mod_tcs :
        vm, spikes, noise = LIF.simulate(input_tc = inp,
                                time = time, tot_steps = tot_steps, n_repeat = n_repeat,
                                dt = dt, 
                                Rm = Rm, Cm = Cm, tau_m = tau_m, refrac_time = refrac_time,
                                Vth = Vth,
                                loc = loc, scale = scale)
        out_vms.append(vm)
        out_spikes.append(spikes)
        
    out_vms = np.asarray(out_vms) # shape stims, ori, repeats, timesteps
    out_spikes = np.asarray(out_spikes) # shape stims, ori, repeats

        
    plot_spike = True 
    hwhhs = []
    if plot_spike:
        fig, ax = plt.subplots(figsize = (8,6))
        for i in range(n_pars) :
           hwhh = plots.plot_spike_tc(ax = ax, 
                                all_spiketimes = out_spikes[i,:,:],
                                lab = labels[i], col = colors[i])
           #ax.plot()
           hwhhs.append(hwhh)
        
        plt.close(fig)
    all_hwhhs.append(hwhhs)
    
    
fig, ax = plt.subplots(figsize = (8,6))
colors = plt.cm.viridis(np.linspace(.2, .9, len(a_bws)))
for i_bw, a_bw in enumerate(a_bws) :
    coef= np.polyfit(bandwidths, all_hwhhs[i_bw],2)
    poly1d_fn = np.poly1d(coef)
    ax.scatter(bandwidths, all_hwhhs[i_bw], color = colors[i_bw])
    ax.plot(bandwidths, poly1d_fn(bandwidths), color = colors[i_bw],
            linestyle = '--', label = -a_bw)

ax.legend(title = 'Exponant', fontsize = 14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_ylabel('Orientation bandwidth', fontsize = 18)
ax.set_xlabel('Tuning curve bandwidth', fontsize = 18)

fig.savefig('./figs/fig2e.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)
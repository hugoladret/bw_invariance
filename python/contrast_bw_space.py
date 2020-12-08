# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:24:50 2020

@author: Hugo

Source for the sim 
https://medium.com/analytics-vidhya/modeling-the-simplest-biological-neuron-with-python-adda892c8384 
https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
http://tips.vhlab.org/techniques-and-tricks/matlab/integrate-and-fire

Choice for the default params
http://neuralensemble.org/docs/PyNN/reference/neuronmodels.html
"""

import stim
import plots
import LIF
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D 
import fit

def get_3d_quantifs(data):
    mean_spiketrains = []
    std_spiketrains = []
    for trial in data :
        mean_spiketrains.append(np.mean([len(x) for x in trial]))
        std_spiketrains.append(np.std([len(x) for x in trial]))
        
    tc_pars = fit.fit_gaussian(mean_spiketrains)
    fit_tc = fit.gaussian(np.linspace(-3, 3, 1000),
                         tc_pars['mu'], tc_pars['sig'], tc_pars['scale'])
                         
    return tc_pars['sig'], np.max(fit_tc)

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
n_pars = 12 #number of parameters, either contrast or bandwidth
contrasts = np.linspace(1., 8., n_pars) #stimulation contrast, max = 5 is a good idea
bandwidths = np.linspace(.3, .8, n_pars) # stimulation bandwidth, it's sigma of gaussian

## Finn parameters
k = 3.5 # power law scale 
a = -.5 # power law exponent
loc = .8 # noise normal law center
scale = .5 # noise normal law var

## Bandwidth parameters
k_bw = 3.5 # other neurons' power law scale
a_bw = -4.5 # multiplier of bw on other neurons

## Plotting parameters
labels = bandwidths #rescale for actual stim values
colors = plt.cm.inferno(np.linspace(.9, .2, len(bandwidths))) #tc colormap

# Initialization
tot_steps = int(T/dt) # length of the time vector 
time = np.linspace(0, T+dt, tot_steps) # time vector




# Stimulation, contrasts and bandwidths
pwlaw = stim.power_law(k = k, x = contrasts, a = a)
input_tcs = np.zeros((n_pars, n_pars), dtype = object)
for i0, max_amp in enumerate(contrasts) :
    for i1, bw in enumerate(bandwidths) :
        
        inp = stim.generate_stim(mu = 0., sig = bw, max_amp = max_amp)
        inp *= pwlaw[i0]
        
        new_pwlaw = stim.power_law(k = np.max(inp), 
                                    x = np.linspace(1, 3, len(inp)//2),
                                    a = a_bw*np.exp(bw))
        mult = inp[len(inp)//2:]-(1/new_pwlaw)
        mult[mult<0] = 0
        mult_left = mult[::-1]
        
        input_tcs[i0, i1] =  np.concatenate((mult_left, mult))
    
      
    
    
    
# Simulation
out_spikes = np.zeros((n_pars, n_pars), dtype = object)
for i0, max_amp in tqdm(enumerate(contrasts), total = len(contrasts)) :
    for i1, bw in enumerate(bandwidths) :
        inp = input_tcs[i0, i1]
        vm, spikes, noise = LIF.simulate(input_tc = inp,
                                time = time, tot_steps = tot_steps, n_repeat = n_repeat,
                                dt = dt, 
                                Rm = Rm, Cm = Cm, tau_m = tau_m, refrac_time = refrac_time,
                                Vth = Vth,
                                loc = loc, scale = scale)
        out_spikes[i0, i1] = spikes
   
    


# HWHH plot        
hwhhs = np.zeros((n_pars, n_pars))
tcmaxs = np.zeros((n_pars, n_pars))       
for i0, max_amp in enumerate(contrasts) :
    for i1, bw in enumerate(bandwidths) :
        data = out_spikes[i0, i1]
        hwhh, tcmax = get_3d_quantifs(data)
        hwhhs[i0, i1] = hwhh
        tcmaxs[i0, i1] = tcmax
        
        
fig, ax = plt.subplots(figsize = (8,6))
im = ax.matshow(hwhhs, origin = 'lower')
ax.set_xticklabels(np.round(np.concatenate(([0], contrasts)), 2))
ax.set_yticklabels(np.round(np.concatenate(([0], bandwidths)), 2))
ax.xaxis.set_ticks_position('bottom')

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylabel('Orientation bandwidth', fontsize = 14)
ax.set_xlabel('Orientation contrast', fontsize = 14)

cbar = fig.colorbar(im, ax = ax)
cbar.set_label('Tuning curve bandwidth')

fig.savefig('./figs/fig3a.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)


# TC max
fig, ax = plt.subplots(figsize = (8,6))
im = ax.matshow(tcmaxs, origin = 'lower')
ax.set_xticklabels(np.round(np.concatenate(([0], contrasts)), 2))
ax.set_yticklabels(np.round(np.concatenate(([0], bandwidths)), 2))
ax.xaxis.set_ticks_position('bottom')

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylabel('Orientation bandwidth', fontsize = 14)
ax.set_xlabel('Orientation contrast', fontsize = 14)

cbar = fig.colorbar(im, ax = ax)
cbar.set_label('Tuning curve amplitude')

fig.savefig('./figs/fig3b.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)
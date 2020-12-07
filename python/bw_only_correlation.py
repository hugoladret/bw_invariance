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
n_pars = 20 #number of parameters, either contrast or bandwidth
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


# Stimulation, contrasts
pwlaw = stim.power_law(k = k, x = contrasts, a = a)
input_tcs = []
# for i, max_amp in enumerate(contrasts) :
#     inp = stim.generate_stim(mu = 0., sig = .4, max_amp = max_amp)
#     inp *= pwlaw[i]
#     input_tcs.append(inp)
    
for i, bw in enumerate(bandwidths) :
    inp = stim.generate_stim(mu = 0., sig = bw, max_amp = np.max(contrasts))
    inp *= pwlaw[i]
    input_tcs.append(inp)

# Simulation
out_vms, out_spikes = [], []
for inp in tqdm(input_tcs, 'Simulating') :
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

hwhhs  = []
fig, ax = plt.subplots(figsize = (8,6))
plot_spike = True 
if plot_spike:
    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(n_pars) :
        hwhh = plots.plot_spike_tc(ax = ax, 
                            all_spiketimes = out_spikes[i,:,:],
                            lab = labels[i], col = colors[i])
        hwhhs.append(hwhh)



fig, ax = plt.subplots(figsize = (8,6))
coef= np.polyfit(bandwidths, hwhhs,1)
poly1d_fn = np.poly1d(coef)
ax.scatter(bandwidths, hwhhs, color = 'k')
ax.plot(bandwidths, poly1d_fn(bandwidths), color = 'k', linestyle = '--')

ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylabel('Orientation bandwidth', fontsize = 14)
ax.set_xlabel('Tuning curve bandwidth', fontsize = 14)

fig.savefig('./figs/fig2c.pdf' , format = 'pdf', dpi = 100, bbox_inches = 'tight', transparent = True)
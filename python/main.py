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

# Parameters
## Simulation parameters 
T = 50 # total simtime ; ms
dt = 0.01 #timestep ; ms
n_repeat = 5 # number of time we rerun the whole stimulation set, used for trial-to-trial variance

## LIF parameters, note that we scale everything later to speed up computations
## You probably don't want to touch this
Rm = 1 # resistance (kOhm)
Cm = 10 # capacitance (uF)
tau_m = Rm*Cm # time constant (msec)
refrac_time = 1. # refractory period (msec)
Vth = 1. # spike threshold (V)

## Stimulation parameters
max_amps = np.linspace(1., 10., 3) #stimulation contrast, max = 5 is a good idea

## Finn parameters
k = 3.5 # power law scale 
a = -.5 # power law exponent
loc = .8 # noise normal law center
scale = .5 # noise normal law var

## Plotting parameters
labels = max_amps
colors = plt.cm.gray(np.linspace(.8, .3, len(max_amps)))

# Initialization
tot_steps = int(T/dt) # length of the time vector 
time = np.linspace(0, T+dt, tot_steps) # time vector


# Stimulation
pwlaw = stim.power_law(k = k, x = max_amps, a = a)
input_tcs = []
for i, max_amp in enumerate(max_amps) :
    inp = stim.generate_stim(mu = 0., sig = .4, max_amp = max_amp)
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



# Plotting
plot_st = False
if plot_st :
    fig, ax = plt.subplots(figsize = (8,6))
    plots.plot_single_trial(ax = ax,
                            time = time, Vm = out_vms[-1, 50, 0, :], 
                            spiketimes = out_spikes[-1, 50, 0])

plot_vm = False 
if plot_vm :
    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(len(max_amps)) :
        plots.plot_vm_tc(ax = ax,
                          all_vms = out_vms[i,:,:,:],
                          tot_steps = tot_steps,
                          lab = labels[i])
    
plot_spike = True 
if plot_spike:
    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(len(max_amps)) :
        plots.plot_spike_tc(ax = ax, 
                            all_spiketimes = out_spikes[i,:,:],
                            lab = labels[i], col = colors[i])

plot_stim = True
if plot_stim :
    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(len(max_amps)) :
        plots.plot_stimulation(ax, input_tc = input_tcs[i],
                               lab = labels[i])
#plots.plot_stimulation(figsize = (8,6), input_tc = input_tc)
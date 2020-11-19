#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo

Source for the sim 
https://medium.com/analytics-vidhya/modeling-the-simplest-biological-neuron-with-python-adda892c8384 
https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
http://tips.vhlab.org/techniques-and-tricks/matlab/integrate-and-fire

Choice for the default params
http://neuralensemble.org/docs/PyNN/reference/neuronmodels.html
"""

import matplotlib.pyplot as plt
import numpy as np

def scale_values(a, tgt_max, tgt_min) :
    return ((a - np.min(a)) / (np.max(a) - np.min(a))) * (tgt_max - tgt_min) + tgt_min
    
# Parameters
## Simulation parameters 
T = 50 # total simtime ; ms
dt = 0.1 #timestep ; ms
n_repeat = 1 # number of time we rerun the whole stimulation set, used for trial-to-trial variance

## LIF parameters, note that we scale everything later to speed up computations
Rm = 1 # resistance (kOhm)
Cm = 10 # capacitance (uF)
tau_m = Rm*Cm # time constant (msec)
refrac_time = 2. # refractory period (msec)
Vth = 1 # spike threshold (V)


# Initialization
t_refrac = 0 # refractory period counter
tot_steps = int(T/dt) # length of the time vector 
time = np.linspace(0, T+dt, tot_steps) # time vector
Vm = np.zeros(len(time)) # membrane potential vector


# Stimulation
I = np.zeros(len(time)) # input vector
I[int(.25*tot_steps):int(.6*tot_steps)] = 3. # stim from 25 to 50% of trial


# Simulation        
spiketimes = np.zeros(len(time)) # time of spikes vector
for i, t in enumerate(time):
    noise = np.random.normal(0,1)
    Vm[i] = Vm[i-1] + (-Vm[i-1] + I[i]*Rm + noise) / tau_m * dt # we can simplify everything with v_rest = 0
    if Vm[i] >= Vth and t_refrac > refrac_time: #if above threshold and we can spike
        spiketimes[i] = 1.
        Vm[i] = 0.
        t_refrac = 0.
        
    if t_refrac < refrac_time : # if we are refractory
        t_refrac +=dt # increase refractory time
        if Vm[i] >= Vth :
            Vm[i] = Vth
    

Vm = scale_values(Vm, -50, -70) #we scale post run, just like BRIAN2 does


# Plotting
fig, ax= plt.subplots(figsize = (12,5))

ax.plot(time, Vm, color = 'k')
for i, st in enumerate(spiketimes) :
    if st == 1 :
        ax.plot((time[i-1], time[i-1]), (-50, 20), c = 'k')

ax.axhline(-50, color = 'gray', linestyle = '--', zorder = -1)

ax.set_title('Leaky Integrate-and-Fire Neuron')
ax.set_ylabel('Membrane Potential (V)')
ax.set_xlabel('Time (msec)')

ax.set_xlim(0, T)

plt.show()
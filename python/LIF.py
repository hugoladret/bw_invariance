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
import stim
import plots
from tqdm import tqdm 

def scale_values(a, tgt_max, tgt_min) :
    return ((a - np.min(a)) / (np.max(a) - np.min(a))) * (tgt_max - tgt_min) + tgt_min
    
# Parameters
## Simulation parameters 
T = 50 # total simtime ; ms
dt = 0.1 #timestep ; ms
n_repeat = 3 # number of time we rerun the whole stimulation set, used for trial-to-trial variance

## LIF parameters, note that we scale everything later to speed up computations
Rm = 1 # resistance (kOhm)
Cm = 10 # capacitance (uF)
tau_m = Rm*Cm # time constant (msec)
refrac_time = 2. # refractory period (msec)
Vth = 1 # spike threshold (V)


# Initialization
tot_steps = int(T/dt) # length of the time vector 
time = np.linspace(0, T+dt, tot_steps) # time vector


# Stimulation
input_tc = stim.generate_stim(mu = 0., sig = .35, max_amp = 5.)

all_vms = np.zeros( (len(input_tc), n_repeat, len(time) ) )
all_spiketimes = np.zeros( (len(input_tc), n_repeat ), dtype = object )
for i0, ori in enumerate(input_tc) :
    
    
    for i1 in range(n_repeat) :
        I = np.zeros(len(time))
        I[int(.25*tot_steps):int(.5*tot_steps)] = ori
        
        t_refrac = 0 
        Vm = np.zeros(len(time))
    
        # Simulation        
        spiketimes = [] # vector of time at which a spike was emitted
        for i2, _ in enumerate(time):
            noise = np.random.normal(0,1)
            Vm[i2] = Vm[i2-1] + (-Vm[i2-1] + I[i2]*Rm + noise) / tau_m * dt # we can simplify everything with v_rest = 0
            if Vm[i2] >= Vth and t_refrac > refrac_time: #if above threshold and we can spike
                spiketimes.append(i2)
                Vm[i2] = 0.
                t_refrac = 0.
                
            if t_refrac < refrac_time : # if we are refractory
                t_refrac +=dt # increase refractory time
                if Vm[i2] >= Vth :
                    Vm[i2] = Vth
        
        Vm = scale_values(Vm, -50, -70) #we scale post run, just like BRIAN2 does
        
        all_vms[i0, i1, :] = Vm
        all_spiketimes[i0, i1] = spiketimes
     

# Plotting
plots.plot_single_trial(figsize = (8,6),
                        time = time, Vm = all_vms[50, 0, :], 
                        spiketimes = all_spiketimes[50, 0])

plots.plot_vm_tc(figsize = (8,6),
                 all_vms = all_vms)
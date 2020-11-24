#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo


"""

import numpy as np

def scale_values(a, tgt_max, tgt_min) :
    return ((a - np.min(a)) / (np.max(a) - np.min(a))) * (tgt_max - tgt_min) + tgt_min
    
def simulate(input_tc,
             time, tot_steps, n_repeat, dt,
             Rm, Cm, tau_m, refrac_time, Vth,
             loc , scale):
    
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
                noise = np.random.normal(loc = loc, scale = scale)
                Vm[i2] = Vm[i2-1] + (-Vm[i2-1] + I[i2]*Rm + noise) / tau_m * dt # we can simplify everything with v_rest = 0
                if Vm[i2] >= Vth and t_refrac > refrac_time: #if above threshold and we can spike
                    spiketimes.append(i2)
                    Vm[i2-1] = 1.
                    Vm[i2] = 0.
                    t_refrac = 0.
                    
                if t_refrac < refrac_time : # if we are refractory
                    t_refrac +=dt # increase refractory time
                    if Vm[i2] >= Vth :
                        Vm[i2] = Vth
                        
            Vm = scale_values(Vm, -50, -70) #we scale post run, just like BRIAN2 does
            
            all_vms[i0, i1, :] = Vm
            all_spiketimes[i0, i1] = spiketimes
    
    return all_vms, all_spiketimes, noise
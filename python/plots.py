# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:06:31 2020

@author: Hugo
"""

import numpy as np
import matplotlib.pyplot as plt
import fit

def plot_single_trial(ax,
                      time, Vm, spiketimes) :

    ax.plot(time, Vm, color = 'k')
    for st in spiketimes :
        ax.plot((time[st-1], time[st-1]), (-51, 20), c = 'k')
    
    ax.axhline(-50, color = 'gray', linestyle = '--', zorder = -1)
    
    ax.set_title('Single trial example')
    ax.set_ylabel('Membrane potential (mV)')
    ax.set_xlabel('Time (ms')
    
    ax.set_xlim(0, max(time))
    
    #ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def plot_vm_tc(ax,
               all_vms, tot_steps, 
               lab) :
    

    avg_vm = np.mean(all_vms, axis = 1)
    #mean_2nd = np.mean(avg_vm[:,int(.25*tot_steps):int(.5*tot_steps)], axis = -1)
    mean_2nd = np.mean(avg_vm, axis = -1)
    
    ax.plot(1/mean_2nd, label = lab)
    
    ax.set_title('Membrane tuning curve')
    ax.set_ylabel('Membrane potential (mV)')
    ax.set_xlabel('Orientation')
    
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def plot_spike_tc(ax,
                  all_spiketimes,
                  lab, col) :

    mean_spiketrains = []
    std_spiketrains = []
    for trial in all_spiketimes :
        mean_spiketrains.append(np.mean([len(x) for x in trial]))
        std_spiketrains.append(np.std([len(x) for x in trial]))
        
    tc_pars = fit.fit_gaussian(mean_spiketrains)
    fit_tc = fit.gaussian(np.linspace(-3, 3, 1000),
                         tc_pars['mu'], tc_pars['sig'], tc_pars['scale'])
    
    ax.plot(np.linspace(-3, 3, 1000),
            fit_tc, color = col)
    ax.scatter(np.linspace(-3, 3, len(mean_spiketrains)),
               mean_spiketrains, color = col, label = lab,
               marker = 'X', s = 5.)
    
    print(tc_pars['sig'])
    
    ax.set_title('Spiking tuning curve')
    ax.set_ylabel('Firing rate(spikes)')
    ax.set_xlabel('Orientation')
    
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def plot_stimulation(ax, input_tc,lab):
    
    ax.plot(input_tc, label = lab)
    
    ax.set_title('Input')
    ax.set_ylabel('Input value')
    ax.set_xlabel('Orientation')
    
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax
    
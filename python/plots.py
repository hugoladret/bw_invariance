# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:06:31 2020

@author: Hugo
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_single_trial(figsize,
                      time, Vm, spiketimes) :
    fig, ax = plt.subplots(figsize = figsize)
    
    ax.plot(time, Vm, color = 'k')
    for st in spiketimes :
        ax.plot((time[st-1], time[st-1]), (-51, 20), c = 'k')
    
    ax.axhline(-50, color = 'gray', linestyle = '--', zorder = -1)
    
    ax.set_title('Single trial example')
    ax.set_ylabel('Membrane potential (mV)')
    ax.set_xlabel('Time (ms')
    
    ax.set_xlim(0, max(time))
    
    return fig, ax

def plot_vm_tc(figsize,
               all_vms) :
    
    fig, ax = plt.subplots(figsize = figsize)
    
    avg_vm = np.mean(all_vms, axis = 1)
    max_avg_vm = np.mean(avg_vm, axis = -1)
    
    plt.plot(max_avg_vm)
    print(avg_vm.shape)
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:06:31 2020

@author: Hugo
"""

import numpy as np
import matplotlib.pyplot as plt
import fit
import stim


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
            fit_tc, color = col, label = lab)
    ax.scatter(np.linspace(-3, 3, len(mean_spiketrains)),
               mean_spiketrains, color = col,
               marker = 'X', s = 7.)

    
    ax.set_ylabel('Firing rate(spikes)', fontsize = 14)
    ax.set_xlabel('Orientation', fontsize = 14)
    
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    
    return tc_pars['sig']

def plot_stimulation(ax, input_tc,lab,col
                     ):
    
    input_tc = stim.scale_values(input_tc, 1, 0.)
    ax.plot(input_tc, label = lab, color = col)
    
    ax.set_ylabel('Normalized input value', fontsize = 14)
    ax.set_xlabel('Orientation', fontsize = 14)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return ax
    
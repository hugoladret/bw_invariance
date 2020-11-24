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

## Stimulation parameters
n_pars = 3 #number of parameters, either contrast or bandwidth
contrasts = np.linspace(1., 8., n_pars) #stimulation contrast, max = 5 is a good idea
bandwidths = np.linspace(.2, .6, n_pars) # stimulation bandwidth, it's sigma of gaussian

## Finn parameters
k = 3.5 # power law scale 
a = -.5 # power law exponent

# plt.figure()
# pwlaw = stim.power_law(k = 1, 
#                 x = np.linspace(1, 3, 50),
#                 a = a)
# plt.plot(1/pwlaw)


input_tcs = []  
mod_tcs = []
pwlaws = []  
for i, bw in enumerate(bandwidths) :
    inp = stim.generate_stim(mu = 0., sig = bw, max_amp = np.max(contrasts))
    
    # Compute the right part 
    new_pwlaw = stim.power_law(k = np.max(inp), 
                                x = np.linspace(1, 3, len(inp)//2),
                                a = -4.5*np.exp(bw))
    mult = inp[len(inp)//2:]-(1/new_pwlaw)
    mult[mult<0] = 0
    
    # And swap with the left part, TC aren't asymetric
    mult_left = mult[::-1]
    
    mod_tcs.append(np.concatenate((mult_left, mult)))
    pwlaws.append(new_pwlaw)
    input_tcs.append(inp)


# for i, bw in enumerate(bandwidths) :
#     plt.plot(input_tcs[i])
# plt.show()

colors = plt.cm.viridis(np.linspace(.1, .8, n_pars))
plt.figure()
for i, bw in enumerate(bandwidths) :
    plt.plot(input_tcs[i], color = colors[i])
    plt.plot(mod_tcs[i], linestyle = '--', color = colors[i])
    
# plt.plot(input_tcs[1][50:], label = 'input')
# plt.plot(3/pwlaws[1], label = 'inverse powerlaw')
# plt.plot(mod_tcs[1], label = 'mod')
# plt.legend()

# plt.plot(input_tcs[1][50:], label = 'input')
# plt.plot(new_pwlaw, label = 'powerlaw')

# plt.plot(multi, label = 'product')
# plt.legend()
#plt.plot(np.convolve(input_tcs[0][51:], new_pwlaw, mode = 'same'))
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

import numpy as np

def scale_values(a, tgt_max, tgt_min) :
    return ((a - np.min(a)) / (np.max(a) - np.min(a))) * (tgt_max - tgt_min) + tgt_min

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def power_law(k,x,a) :
    return k*x**a

def generate_stim(mu = 0., sig = .35, max_amp = 1.) :
    
    xs = np.linspace(-3, 3, 60)
    ys = gaussian(x = xs, mu = mu, sig = sig)
    ys = scale_values(ys, max_amp, 0)

    return ys

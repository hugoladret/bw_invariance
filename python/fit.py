# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:04:23 2020

@author: Hugo
"""

import numpy as np
from lmfit import Model, Parameters 
import stim

def gaussian(x, mu, sig, scale):
    gauss = (1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2))
    return stim.scale_values(gauss, tgt_max = scale, tgt_min = 0.)

def fit_gaussian(array):
    x = np.linspace(-3, 3., len(array), endpoint = False)
    y = array
    
    mod = Model(gaussian)
    pars = Parameters()
    pars.add_many(('mu', 0., True, -3., 3.),
                  ('sig', 1., True,  0.01, 3.),
                  ('scale', np.max(array), True, 0.01, 100))

    out = mod.fit(y, pars, x=x, nan_policy='omit')

    return out.best_values
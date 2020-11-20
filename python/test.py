# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:18:04 2020

@author: Hugo
"""

import numpy as np
import matplotlib.pyplot as plt

loc = -.5
scale = 1.
noise = np.random.normal(loc = loc, scale = scale, size = 500)

noise2 = []
for i in range(500) :
    noise2.append(np.random.normal(loc = loc, scale = scale, size = 1)[0])
    

plt.hist(noise)
plt.hist(noise2)
plt.show()
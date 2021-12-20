
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# import modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

path = "/home/marcus/Desktop/COND"
calibration = path+"/calibration_mn.txt"
pb = path + "/pure_pb.txt"
pb05 = path + "/pb5pin.txt"
pb07 = path + "/pb7pin.txt"

calibration_dat = pd.read_csv(calibration, sep = " ")
pb_dat = pd.read_csv(pb, sep = " ") 
pb05_dat = pd.read_csv(pb05, sep = " ")
pb07_dat = pd.read_csv(pb07, sep = " ")

# some plots

# current  = current * precision resistor

# plt.figure(dpi=350)
# plt.plot(pb_dat["current"], pb_dat["voltageInt"])
# plt.grid()

plt.figure()
start = 1665
end = 8450
plt.plot(pb_dat["current"][start:end], pb_dat["voltageInt"][start:end])

### calculate B -field

N = 4008 # winding number
l = 0.15 # m
R = 0.05 # Ohm -> precision resistance

H = N * pb_dat["current"] / (R*l)

plt.figure()
plt.plot(H, pb_dat["voltageInt"])

# plt.figure()   # -> current on the x-axis
# plt.plot(pb_dat["current"]/R, pb_dat["voltageInt"])


### fitting

def fit_func(xdata, ydata):
    model = lambda x, a: a*x 
    popt, pcov = curve_fit(model, xdata, ydata)
    return (popt, pcov)

popt, pcov = fit_func(pb_dat["current"][start:end], pb_dat["voltageInt"][start:end])
new_model = lambda x: popt[0] * x 

plt.figure()
# plt.plot(pb_dat["current"], pb_dat["voltageInt"])
# plt.plot(pb_dat["current"], new_model(pb_dat["current"]))
plt.plot(H, pb_dat["voltageInt"] - new_model(pb_dat["current"]))
plt.grid()
idx = np.argwhere(pb_dat["voltageInt"] == max(pb_dat["voltageInt"]))
plt.scatter(H[idx], pb_dat["voltageInt"][idx], c="red")
# popty, pcovy = fit_func(pb_dat["current"][0:start], pb_dat["voltageInt"][0:start])
# new_modely = lambda x: popty[0] * x
# plt.plot(H[0:start], new_modely(pb_dat["current"][0:start]))




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
from uncertainties import ufloat
import uncertainties.umath as unp  # sin(), etc.
plt.style.use(['science', 'grid'])


## correction fitting

def fit_func(xdata, ydata):
    model = lambda x, a: a*x
    popt, pcov = curve_fit(model, xdata, ydata)
    return (popt, pcov)


if __name__ == '__main__':
    ## Magnetic field
    N = 4008 # winding number
    l = 0.15 # m
    R = 0.05 # Ohm -> precision resistance
    H = lambda current: N * current / (R*l)
    H = np.vectorize(H)


    # Defining paths    
    path = "C:\\Users\\marcu\\OneDrive\\Desktop\\PraktikumIII\\Superconductivity"
    plots = path + "\\plots"
    data = path + "/data/exp_pro"
    calibration = data+"/calibration_mn.txt"
    pb = data + "/pure_pb.txt"
    pb05 = data + "/pb5pin.txt"
    pb07 = data + "/pb7pin.txt"

    calibration_dat = pd.read_csv(calibration, sep = " ")
    pb_dat = pd.read_csv(pb, sep = " ")
    pb05_dat = pd.read_csv(pb05, sep = " ")
    pb07_dat = pd.read_csv(pb07, sep = " ")
    
    
    ## Correction with calibration
    pb_dat["voltageInt"] = pb_dat["voltageInt"] - calibration_dat["voltageInt"]
    pb05_dat["voltageInt"] = pb05_dat["voltageInt"] - calibration_dat["voltageInt"]
    pb07_dat["voltageInt"] = pb07_dat["voltageInt"] - calibration_dat["voltageInt"]
    
    ## Add computed H-field 
    pb_dat["Hfield"] = H(pb_dat["current"])
    pb05_dat["Hfield"] = H(pb05_dat["current"])
    pb07_dat["Hfield"] = H(pb07_dat["current"])

    # current  = current * precision resistor

    ## Initial plotting
    if False:
        fig, ax = plt.subplots(2,2)
        fig.tight_layout()
        fig.set_dpi(300)
        fig.set_size_inches(12,6)
        ax = ax.ravel()
        ax[0].plot(calibration_dat["current"], calibration_dat["voltageInt"])
        ax[0].set_title("Calibration")
        ax[1].plot(pb_dat["current"], pb_dat["voltageInt"])
        
        ax[2].plot(pb05_dat["current"], pb05_dat["voltageInt"])
        ax[1].set_title("Pb")
        ax[2].set_title("Pb, 5 \% In")
        ax[3].set_title("Pb, 7 \% In")
        ax[3].plot(pb07_dat["current"], pb07_dat["voltageInt"])
        
        
        for axis in ax:
            axis.set_xlabel("Voltage drop [V]")
            axis.set_ylabel("Integrated Voltage [Vs]")

        # plt.savefig(plots + "\\InitialObservations.pdf")

    ## Plotting the data with points signifying the start and end points    
    pbpts = [20, np.where(pb_dat["voltageInt"] == np.max(pb_dat["voltageInt"]))[0][0], 1663, 8425]
    pb05pts = [20, np.where(pb05_dat["voltageInt"] == np.max(pb05_dat["voltageInt"]))[0][0], 3170, 8435]
    pb07pts = [20, np.where(pb07_dat["voltageInt"] == np.max(pb07_dat["voltageInt"]))[0][0], 3680, 8000]


    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        
        ax[0].plot(pb_dat["current"], pb_dat["voltageInt"])
        for i in pbpts:
            ax[0].scatter(pb_dat["current"][i], pb_dat["voltageInt"][i], color="red",
                          zorder=3, s=3.5
            )
    
        ax[1].plot(pb05_dat["current"], pb05_dat["voltageInt"])
        for i in pb05pts:
            ax[1].scatter(pb05_dat["current"][i], pb05_dat["voltageInt"][i], color="red",
                          zorder=3, s=3.5
            )
    
        ax[2].plot(pb07_dat["current"], pb07_dat["voltageInt"])
        for i in pb07pts:
            ax[2].scatter(pb07_dat["current"][i], pb07_dat["voltageInt"][i], color="red",
                          zorder=3, s=3.5
            )
        ax[0].set_ylabel("Integrated Voltage [Vs]")
        for axis in ax:
            axis.set_xlabel("Voltage drop [V]")
        ax[0].set_title("Pb")
        ax[1].set_title("Pb, 5 \% In")
        ax[2].set_title("Pb, 7 \% In")
        # plt.savefig(plots + "\\DataWithDots.pdf")
        
    

    
    ## Plot with current on x-axis
    # plt.figure()   # -> current on the x-axis
    # plt.plot(pb_dat["current"]/R, pb_dat["voltageInt"])
    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.suptitle("")
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        ax[0].set_ylabel("Integrated Voltage [Vs]")
        ax[0].plot(pb_dat["current"]/R, pb_dat["voltageInt"])
        ax[1].plot(pb05_dat["current"]/R, pb05_dat["voltageInt"])
        ax[2].plot(pb07_dat["current"]/R, pb07_dat["voltageInt"])
        ax[0].set_ylabel("Integrated Voltage [V]")
        
        for axis in ax:
            axis.set_xlabel("Current [A]")
       


    ## plot H-field
    # plt.figure()
    # plt.plot(H, pb_dat["voltageInt"])
    if False:         
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        ax[0].set_ylabel("Integrated Voltage [Vs]")
        ax[0].plot(pb_dat["Hfield"], pb_dat["voltageInt"])
        ax[1].plot(pb05_dat["Hfield"], pb05_dat["voltageInt"])
        ax[2].plot(pb07_dat["Hfield"], pb07_dat["voltageInt"])
        ax[0].set_ylabel("Integrated voltage [Vs]")
        for axis in ax:
            axis.set_xlabel("H-field [A/m]")

    ## Fitting correction terms for the slope after H_c(2)
    slope_mod = lambda x, a: a * x
    
    # pb
    pb_1 = pbpts[-2]
    pb_2 = pbpts[-1]
    pbp, pbcov = fit_func(pb_dat["Hfield"][pb_1:pb_2], pb_dat["voltageInt"][pb_1:pb_2])
    
    # pb05
    pb5_1 = pb05pts[-2]
    pb5_2 = pb05pts[-1]
    pbp5, pb5cov = fit_func(pb05_dat["Hfield"][pb5_1:pb5_2], pb05_dat["voltageInt"][pb5_1:pb5_2])
    
    # pb07
    pb7_1 = pb07pts[-2]
    pb7_2 = pb07pts[-1]
    pbp7, pb7cov = fit_func(pb07_dat["Hfield"][pb7_1:pb7_2], pb07_dat["voltageInt"][pb7_1:pb7_2])
    
    
    ## Plot the fits for removing background
    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(450)
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        
        
        ax[0].plot(pb_dat["Hfield"][:pb_2], pb_dat["voltageInt"][:pb_2], label="Recorded data")
        ax[0].plot(pb_dat["Hfield"][pb_1:pb_2], slope_mod(pb_dat["Hfield"][pb_1:pb_2], pbp[0]), label="Fit to noise data")
        ax[0].plot(pb_dat["Hfield"][:pb_2], pb_dat["voltageInt"][:pb_2] - slope_mod(pb_dat["Hfield"][:pb_2], 
                                                                                    pbp[0]), ls=":", lw=0.5, label="Recalibrated data")
        ax[0].set_ylabel("Integrated Voltage [Vs]")
        
        ax[1].plot(pb05_dat["Hfield"][:pb5_2], pb05_dat["voltageInt"][:pb5_2], label="Recorded data")
        ax[1].plot(pb05_dat["Hfield"][pb5_1:pb5_2], slope_mod(pb05_dat["Hfield"][pb5_1:pb5_2], pbp5[0]), label="Fit to noise data")
        ax[1].plot(pb05_dat["Hfield"][:pb5_2], pb05_dat["voltageInt"][:pb5_2] - slope_mod(pb05_dat["Hfield"][:pb5_2], 
                                                                                          pbp5[0]), ls=":", lw=0.5, label="Recalibrated data")
        
        ax[2].plot(pb07_dat["Hfield"][:pb7_2], pb07_dat["voltageInt"][:pb7_2], label="Recorded data")
        ax[2].plot(pb07_dat["Hfield"][pb7_1:pb7_2], slope_mod(pb07_dat["Hfield"][pb7_1:pb7_2], pbp5[0]), label="Fit to noise data")
        ax[2].plot(pb07_dat["Hfield"][:pb7_2], pb07_dat["voltageInt"][:pb7_2] - slope_mod(pb07_dat["Hfield"][:pb7_2], 
                                                                                          pbp7[0]), ls=":", lw=0.5, label="Recalibrated data")
        ax[0].set_title("Pb")
        ax[1].set_title("Pb, 5 \% In")
        ax[2].set_title("Pb, 7 \% In")
        
        for axis in ax:
            axis.set_xlabel("H-field [A/m]")
            axis.legend()
        # plt.savefig(plots + "\\RemovedBackground.pdf")

        
    
    ## Correcting with slopes post critical H-field
    pb_dat["voltageInt"] = pb_dat["voltageInt"] - slope_mod(pb_dat["Hfield"], pbp[0])
    pb05_dat["voltageInt"] = pb05_dat["voltageInt"] - slope_mod(pb05_dat["Hfield"], pbp5[0])
    pb07_dat["voltageInt"] = pb07_dat["voltageInt"] + slope_mod(pb07_dat["Hfield"], -pbp7[0])

    # Plotting corrected voltages
    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        
        ax[0].set_xlabel("H-field [A/m]")
        ax[0].plot(pb_dat["Hfield"][:pb_2], pb_dat["voltageInt"][:pb_2])
        ax[1].plot(pb05_dat["Hfield"][:pb5_2], pb05_dat["voltageInt"][:pb5_2])
        ax[2].plot(pb07_dat["Hfield"][:pb7_2], pb07_dat["voltageInt"][:pb7_2])


    ## Correcting slopes to get a rise of 1 on the initial peak
    # pb
    pb_init = pbpts[0]
    pb_top = pbpts[1]
    pbpC, pbcovC = fit_func(pb_dat["Hfield"][pb_init:pb_top], pb_dat["voltageInt"][pb_init:pb_top])
    
    #pb5
    pb5_init = pb05pts[0]
    pb5_top = pb05pts[1]
    pbp5C, pb5covC = fit_func(pb05_dat["Hfield"][pb5_init:pb5_top], pb05_dat["voltageInt"][pb5_init:pb5_top])
    
    # pb07
    pb7_init = pb07pts[0]
    pb7_top = pb07pts[1]
    pbp7C, pb7covC = fit_func(pb07_dat["Hfield"][pb7_init:pb7_top], pb07_dat["voltageInt"][pb7_init:pb7_top])
    
    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.set_size_inches(12,4)
        ax = ax.ravel()
        
        ax[0].plot(pb_dat["Hfield"][pb_init:pb_2], pb_dat["voltageInt"][pb_init:pb_2])
        ax[0].plot(pb_dat["Hfield"][pb_init:pb_top], slope_mod(pb_dat["Hfield"][pb_init:pb_top], pbpC[0]), lw=2)
        # ax[0].scatter(pb_dat["Hfield"][pb_top], pb_dat["voltageInt"][pb_top], color="red", zorder=5)
        
        ax[1].plot(pb05_dat["Hfield"][pb5_init:pb5_2], pb05_dat["voltageInt"][pb5_init:pb5_2])
        ax[1].plot(pb05_dat["Hfield"][pb5_init:pb5_top], slope_mod(pb05_dat["Hfield"][pb5_init:pb5_top], pbp5C[0]), lw=2)
        
        ax[2].plot(pb07_dat["Hfield"][:pb7_2], pb07_dat["voltageInt"][:pb7_2])
        ax[2].plot(pb07_dat["Hfield"][:pb7_top], slope_mod(pb07_dat["Hfield"][:pb7_top], pbp7C[0]), lw=2)  
        # ax[0].set_ylabel("Magnetisation [A/m]")
        ax[0].set_ylabel("Voltage drop [V]")
        ax[0].set_title("Pb")
        ax[1].set_title("Pb, 5 \% In")
        ax[2].set_title("Pb, 7 \% In")
        # plt.savefig(plots + "\\FirstSlopeCal.pdf")
       
        
    ## Correcting the initial slopes to have slope 1
    pb_dat["voltageInt"] = pb_dat["voltageInt"]/pbpC[0]
    pb05_dat["voltageInt"] = pb05_dat["voltageInt"] /pbp5C[0]
    pb07_dat["voltageInt"] = pb07_dat["voltageInt"]/ pbp7C[0]
    
    
    ## Calculate H_c for type two superconductors ()
    from scipy.integrate import cumulative_trapezoid
    
    
    H_c_pb5 = np.sqrt(2*cumulative_trapezoid(pb05_dat["Hfield"][:pb5_1], pb05_dat["Hfield"][:pb5_1])[-1])
    H_c_pb5s = ufloat(H_c_pb5, H_c_pb5 * 0.03)
    
    H_c_pb7 = np.sqrt(2*cumulative_trapezoid(pb05_dat["Hfield"][:pb7_1], pb05_dat["Hfield"][:pb7_1])[-1])
    H_c_pb7s = ufloat(H_c_pb7, H_c_pb7 * 0.03)
    
    
    
    if False:
        fig, ax = plt.subplots(1,3)
        fig.set_dpi(300)
        fig.set_size_inches(12,4)
        ax = ax.ravel()

        confirmation, _ = fit_func(pb_dat["Hfield"][pb_init:pb_top], pb_dat["voltageInt"][pb_init:pb_top])
        confirmation5, _ = fit_func(pb05_dat["Hfield"][pb5_init:pb5_top], pb05_dat["voltageInt"][pb5_init:pb5_top])
        confirmation7, _ = fit_func(pb07_dat["Hfield"][pb7_init:pb7_top], pb07_dat["voltageInt"][pb7_init:pb7_top])
        
        ax[0].set_ylabel("Magnetisation [A/m]")
        ax[0].plot(pb_dat["Hfield"][:pb_2], pb_dat["voltageInt"][:pb_2], label="Data w. slope 1")
        ax[0].vlines(pb_dat["Hfield"][pb_top+20], 0, 50000, label="$H_c$", color="red")
        
        ax[1].plot(pb05_dat["Hfield"][:pb5_2], pb05_dat["voltageInt"][:pb5_2], label="Data w. slope 1")
        ax[1].vlines(pb05_dat["Hfield"][pb5_top], 0, 60000, label="$H_{c1}$", color="red")
        ax[1].vlines(pb05_dat["Hfield"][pb5_1], 0, 60000, label="$H_{c2}$", color="green")
        ax[1].vlines(H_c_pb5, 0, 60000, label="$H_{c}$", color="purple")        
        
        ax[2].plot(pb07_dat["Hfield"][:pb7_2], pb07_dat["voltageInt"][:pb7_2], label="Data w. slope 1")
        ax[2].vlines(pb07_dat["Hfield"][pb7_top], 0, 60000, label="$H_{c1}$", color="red")
        ax[2].vlines(pb07_dat["Hfield"][pb7_1], 0, 60000, label="$H_{c2}$", color="green")
        ax[2].vlines(H_c_pb7, 0, 60000, label="$H_{c}$", color="purple")

        
        ax[0].set_title("Pb")
        ax[1].set_title("Pb, 5 \% In")
        ax[2].set_title("Pb, 7 \% In")
        
        for axis in ax:
            axis.legend()
            axis.set_xlabel("H-field [A/m]")
            
        plt.savefig(plots + "\\FinishedCalib_CriticalValues.pdf")
    
    Hc_0 = ufloat(pb_dat["Hfield"][pb_top+20], pb_dat["Hfield"][pb_top+20] - pb_dat["Hfield"][pb_top])
    
    Hc1_5 = ufloat(pb05_dat["Hfield"][pb5_top], pb05_dat["Hfield"][pb5_top] - pb05_dat["Hfield"][pb5_top+5])
    Hc1_7 = ufloat(pb07_dat["Hfield"][pb7_top], pb07_dat["Hfield"][pb7_top+32] - pb07_dat["Hfield"][pb7_top])

    Hc2_5 = ufloat(pb05_dat["Hfield"][pb5_1], pb05_dat["Hfield"][pb5_1+30] - pb05_dat["Hfield"][pb5_1])
    Hc2_7 = ufloat(pb07_dat["Hfield"][pb7_1], pb07_dat["Hfield"][pb7_1+36] - pb07_dat["Hfield"][pb7_1])
    
    
    ## Calculate the Energy band gap and condensation energy of the curves. 
    m_e = 9.10938356 * 10**(-31) #kilograms 
    pi = np.pi
    hbr = 6.62607015 * 10**(-34) #J*Hz−1	
    mu0 = 4*pi *1e-7  #N/A^2
    E_FPb = 1.517261e-18 # J
    
    Z = m_e / (pi**2 * hbr**2) * np.sqrt(2*m_e*E_FPb)
    Z /= 2
    
    E_CON_pb5 = H_c_pb5**2 / 2 * mu0
    E_CON_pb5s = H_c_pb5s**2 / 2 * mu0
    E_CON_pb7 = H_c_pb7**2 / 2 * mu0
    E_CON_pb7s = H_c_pb7s**2 / 2 * mu0
    
    DeltaE_pb5 = np.sqrt(2*E_CON_pb5 / Z) # J)
    DeltaE_pb5s = unp.sqrt(2*E_CON_pb5s / Z)
    DeltaE_pb7 = np.sqrt(2*E_CON_pb7 / Z) # J
    DeltaE_pb7s = unp.sqrt(2*E_CON_pb7s / Z)
    
    ## Ginzburg Landau
    k_pb5 = np.sqrt(pb05_dat["Hfield"][pb5_1] / (2* pb05_dat["Hfield"][pb5_top]))
    k_pb5s = unp.sqrt(Hc2_5 / (2*Hc1_5))
    k_pb7 = np.sqrt(pb07_dat["Hfield"][pb7_1] / (2* pb07_dat["Hfield"][pb7_top]))
    k_pb7s = unp.sqrt(Hc2_7 / (2*Hc1_7))
    
    
    ## calculate the weird derivative thing
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    d5y = smooth(pb05_dat["Hfield"][pb5_top:pb5_1], 250)
    d5x = pb05_dat["voltageInt"][pb5_top:pb5_1]
    
    fig, ax = plt.subplots(2,1, dpi=350, figsize=(6,4))
    fig.tight_layout()
    ax[0].plot(d5x[:-150], d5y[:-150], '.-', linewidth=0.5, c="red", label="Data to be fit")
    fct = lambda x, a, b: a*x + b
    a5 = curve_fit(fct, d5x[:-150], d5y[:-150])[0]
    ax[0].plot(d5x, a5[0]*d5x+a5[1], label="Fit data")
    ax[0].set_title("Pb, 5 \% In")
    ax[1].set_title("Pb, 7 \% In")
    ax[1].set_xlabel("H-field [A/m]")
    
    d7y = smooth(pb05_dat["Hfield"][pb7_top:pb7_1], 250)
    d7x = pb05_dat["voltageInt"][pb7_top:pb7_1]
    ax[1].plot(d7x[200:-150], d7y[200:-150], '.-', linewidth=0.5, c="red", label="Data to be fit")
    fct2 = lambda x, a, b, c: a + b*x + c*x**2
    a7 = curve_fit(fct2, d7x[200:-150], d7y[200:-150])[0]
    ax[1].plot(d7x, a7[0] + a7[1]*d7x + a7[2]*d7x**2, label="Fit data")
    # ax[1].plot(d7x, a7[0]/d7x + a7[1], label="Fit data")
    
    for axis in ax:
        axis.set_ylabel("Magnetization [A/m]")
        axis.legend()
    
    plt.savefig(plots + "\\fitted_derivative.pdf")


    deriv1 = -1*np.full_like(d5x, a5[0])
    deriv2 = a7[1] + 2*d7x*a7[2]
    
    fig, ax = plt.subplots(2,1, dpi=350, figsize=(6,4))
    fig.tight_layout()
    ax[0].plot(d5x, deriv1 / Hc2_5.nominal_value, label="Derivative of the first curve")
    ax[1].plot(d7x, deriv2 / Hc2_7.nominal_value, label="Derivative of the second curve")
    for axis in ax:
        axis.set_ylabel("Magnetization [A/m]")
        axis.legend()
    ax[0].set_title("Pb, 5 \% In")
    ax[1].set_title("Pb, 7 \% In")
    ax[1].set_xlabel("H-field [A/m]")

    plt.savefig(plots + "\\magnetization_throughHc2.pdf")

    
    
    
   



#################### DEPRECATION
    # plt.figure()
    # # plt.plot(pb_dat["current"], pb_dat["voltageInt"])
    # # plt.plot(pb_dat["current"], new_model(pb_dat["current"]))
    # plt.plot(H, pb_dat["voltageInt"] - new_model(pb_dat["current"]))
    # idx = np.argwhere(pb_dat["voltageInt"] == max(pb_dat["voltageInt"]))
    # plt.scatter(H[idx], pb_dat["voltageInt"][idx], c="red")
    # # popty, pcovy = fit_func(pb_dat["current"][0:start], pb_dat["voltageInt"][0:start])
    # # new_modely = lambda x: popty[0] * x
    # # plt.plot(H[0:start], new_modely(pb_dat["current"][0:start]))

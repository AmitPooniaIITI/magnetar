# profile_fitting.py - Pulse Profile Fitting using Sinusoidal Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from astropy.io import fits
from scipy.optimize import curve_fit
from tqdm import tqdm
from pylab import *

# Open the event file
lc = fits.open('output1.evn')
lc.info()

# Read time and PHA column data from the FITS file
tbdata = lc[1].data
time = tbdata['TIME'][:]
time = time - time[0]  # Normalize time
pha = tbdata['PHA'][:]

# Define frequency search range
f_r = np.arange(0.11506, 0.11510, 10**(-7))
ampmax = 0
amplitud = []
param = []

# Define model function for fitting
def f_x(phivale, a_0, a_1, phi_0, a_2, phi_1):
    return a_0 + a_1 * np.sin(2 * np.pi * (phivale - phi_0)) + a_2 * np.sin(4 * np.pi * (phivale - phi_1))

# Loop through frequency range to find best-fit frequency
for nu in tqdm(f_r):
    phi = nu * time
    phi1 = phi - phi.astype(int)
    ph = np.histogram(phi1, 16)  # Histogram in phase vs photon counts
    
    phi2 = np.ones(len(phi1))
    phi3 = phi1 + phi2
    phi4 = np.hstack((phi1, phi3))
    ph2 = np.histogram(phi4, 32)
    counts = ph2[0]
    phival = np.delete(ph2[1], 0)
    
    # Compute photon noise
    photon_noise = np.sqrt(counts)
    
    # Fit the sinusoidal model using scipy.optimize.curve_fit
    popt, pcov = curve_fit(f_x, phival, counts, p0=None, sigma=photon_noise, bounds=(0, [10548175, 10548175, 1, 10548175, 1]))
    
    # Store fit parameters
    param.append(popt)
    a0 = popt[0]
    a = max(popt[1], popt[3])
    amplitude = a / a0
    amplitud.append(amplitude)

# Find best-fit frequency index
max_amp = np.argmax(amplitud)
phi = time * f_r[max_amp]
phi1 = phi - phi.astype(int)
ph1 = np.histogram(phi1, 16)

phi2 = np.ones(len(phi1))
phi3 = phi1 + phi2
phi4 = np.hstack((phi1, phi3))
ph2 = np.histogram(phi4, 32)
pc = ph2[0]
phival = np.delete(ph2[1], 0)
y_err = np.sqrt(pc)
pc_shift = np.roll(pc, 2)
photon_noise = np.sqrt(pc_shift)
counts = pc_shift

# Plot the pulse profile fit
plt.figure(figsize=[8, 8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=10)
plt.figure(figsize=(12, 6))

print("Best-fit parameters:", param[max_amp])
print("Best-fit frequency:", f_r[max_amp])

plt.plot(phival, f_x(phival, *param[max_amp]))
plt.step(phival, counts, where='mid')
plt.errorbar(phival, counts, yerr=photon_noise, fmt='o')
plt.xlabel('Phase $\phi$')
plt.ylabel('Photon Count')
plt.title('Pulse profile fitting Observation-1')
plt.savefig('pf1.png')
plt.show()

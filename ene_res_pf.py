# ene_res_pf.py - Energy-Resolved Pulse Profile Analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import math
from astropy.io import fits

# Open the event file
lc = fits.open('output1.evn')
lc.info()

# Extract data from the FITS file
tbdata = lc[1].data

# Apply filtering based on PHA values (energy bands)
mask_data = (tbdata['PHA'] < 9)
newdata1 = tbdata[mask_data]
mask_data2 = newdata1['PHA'] > 6
newdata2 = newdata1[mask_data2]

# Extract and normalize time column
time2 = newdata2['TIME']
time2 = time2 - time2[0]
time = time2

# Define pulsation frequency
nu = 0.1150849
phi = nu * time
phi1 = phi - phi.astype(int)

# Compute histogram for pulse profile
phi2 = np.ones(len(phi1))
phi3 = phi1 + phi2
phi4 = np.hstack((phi1, phi3))
ph2 = np.histogram(phi4, 32)
ph1 = np.histogram(phi1, 16)

# Extract photon count and shift for phase alignment
pc = ph2[0]
phival = np.delete(ph2[1], 0)
pc_shift = np.roll(pc, 16)

# Plot the pulse profile for the selected energy band
plt.figure(figsize=[8, 8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=15)
plt.figure(figsize=(12, 6))

plt.step(phival, pc_shift, where='mid')
y_err = np.sqrt(pc_shift)
plt.errorbar(phival, pc_shift, yerr=y_err, fmt='o')
plt.ylabel('Counts')
plt.xlabel('Phase ($\phi$)')
plt.title('5-6 KeV Band')

# Calculate pulse fraction
total_time = max(time) - min(time)
time_bin = total_time / 16
pc1 = ph1[0]
count_rate = pc1 / time_bin
PF = (max(pc1) - min(pc1)) / (max(pc1) + min(pc1))

# Compute photon count rate error
y_error = []
N = 16
for i in range(N):
    prop_err = ((pc1[i])**0.5) / time_bin
    y_error.append(prop_err)

# Compute photon count rate variation
pc_ave2 = len(time) / total_time
PCR = np.sqrt(sum(((count_rate - pc_ave2) ** 2) - np.array(y_error) ** 2) / len(count_rate))

print("Photon Count Rate Variation:", PCR)
print("Pulse Fraction:", PF)

plt.show()

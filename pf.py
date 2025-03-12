# pf.py - Pulse Fraction Calculation

from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from astropy.io import fits

# Open the event file
lc = fits.open('4U0142.evt')
lc.info()

# Extract time column from the FITS file
tbdata = lc[1].data
time = tbdata['TIME'][:]
time = time - min(time)  # Normalize time

# Create a histogram of photon arrival times
ph = np.histogram(time, 118048)
x = np.where(ph[0] == max(ph[0]))
time = np.delete(time, x)  # Remove maximum peak

print(time)

# Define pulsation frequency
nu = 0.1150849
phi = nu * time
phi1 = phi - phi.astype(int)  # Phase computation

# Compute histogram of phase
ph1 = np.histogram(phi1, 16)
phi2 = np.ones(len(phi1))
phi3 = phi1 + phi2
phi4 = np.hstack((phi1, phi3))
ph2 = np.histogram(phi4, 32)

# Extract photon count and compute error
pc = ph2[0]
phival = np.delete(ph2[1], 0)
y_err = np.sqrt(pc)
pc_shift = np.roll(pc, 7)
y_err = np.sqrt(pc_shift)

# Plot the pulse profile
plt.step(phival, pc_shift, where='mid')
plt.errorbar(phival, pc_shift, yerr=y_err, fmt='o')
plt.text(1.8, 661500, 'f=0.115082')
plt.ylabel('Counts --->')
plt.xlabel(r'$\phi$-->')

# Calculate pulse fraction

total_time = max(time) - min(time)
time_bin = total_time / 16
pc1 = ph1[0]
count_rate = pc1 / time_bin
PF = (max(pc1) - min(pc1)) / (max(pc1) + min(pc1))
print("Pulse Fraction:", PF)

# Calculate photon count rate error
N = 16
y_error = []
prop_err = 0
for i in range(N):
    prop_err += (pc1[i])**0.5
    prop_err /= time_bin
    y_error.append(prop_err)
print("Photon Count Rate Error:", y_error)

# Compute photon count rate variation
pc_ave2 = sum(pc1) / total_time
PCR = np.sqrt(sum(((count_rate - pc_ave2) ** 2) - np.array(y_error) ** 2) / len(count_rate))
print("Photon Count Rate Variation:", PCR)

plt.show()

# z_test.py - Z-Squared Test for Pulsation Frequency Estimation

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from pylab import *

# Open the event file
lc = fits.open('output1.evn')
lc.info()

# Extract time column
tbdata = lc[1].data
time = tbdata['TIME'][:]
time = time - time[0]  # Normalize time
N = len(time)

print("Time Data:", time)
print("Max Time:", max(time))

# Initialize list to store Z-squared values
zns = []

# Define frequency search range
f_r = np.arange(0.11506, 0.11510, 10**(-7))

# Compute Z-squared statistic for each frequency
for nu in tqdm(f_r):
    phi = 2 * math.pi * nu * time
    
    sinval = np.sum(np.sin(phi))
    cosval = np.sum(np.cos(phi))
    
    ans = np.float128(sinval**2 + cosval**2)
    ans = np.float128(ans * (2 / len(time)))
    zns.append(ans)

# Plot the Z-squared test results
plt.figure(figsize=[8, 8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=10)
plt.figure(figsize=(16, 8))

plt.plot(f_r, zns)
plt.xlabel('Rotational Frequency (Hz)')
plt.ylabel(r'$Z^2$')
plt.title('Z-Squared Test for Pulsation Frequency')
plt.savefig('Z_test_harm1.png')
plt.show()

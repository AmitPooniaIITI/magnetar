# light_curve.py - Light Curve Generation

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Open the event file
lc = fits.open('4U0142.evt')
lc.info()

# Extract time column
tbdata = lc[1].data
time = tbdata['TIME'][:]

# Compute time range for light curve
t_max = max(time)
t_min = min(time)
t_s = t_max - t_min

print("Max Time:", t_max)
print("Min Time:", t_min)
print("Time Span:", t_s)

# Create a histogram of photon arrival times
ph = np.histogram(time, 118048)
x = np.where(ph[0] == max(ph[0]))
print("Max Photon Count Index:", x)

# Remove peak value from time data
new_time = np.delete(time, x)

# Plot the light curve
plt.hist(new_time, 118048)
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.title('Light Curve')
plt.show()

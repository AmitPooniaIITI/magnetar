import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
lc=fits.open('4U0142.evt')
lc.info()
tbdata=lc[1].data
time=tbdata['TIME'][:]
t_max=max(time)
t_min=min(time)
t_s=t_max-t_min
print(t_max)
print(t_min)
print(t_s)
ph=np.histogram(time ,118048)
x=np.where(ph[0]==max(ph[0]))
print(x)
new_time=np.delete(time,x)
plt.hist(new_time,118048)
plt.xlabel('time-s')
plt.ylabel('counts')
plt.show()

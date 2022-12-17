from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from pylab import *
lc=fits.open('output1.evn')
lc.info()
tbdata=lc[1].data
time=tbdata['TIME'][:]
time=time-time[0]
N=len(time)
print(time)
print(max(time))
zns = []
f_r=np.arange(0.11506,0.11510,10**(-7))
for nu in tqdm(f_r):
    phi=2*math.pi*nu*time
   # for i in range(1,3):
    sinval = np.sin(phi)
    sinval = np.sum(sinval)

    cosval = np.cos(phi)
    cosval = np.sum(cosval)
    ans =np.float128(sinval**2 + cosval**2)
    ans = np.float128(ans *(2/len(time)))
    zns.append(ans)

plt.figure(figsize=[8,8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=10)
plt.figure(figsize=(16,8))

plt.plot(f_r,zns)
plt.xlabel('Rotational Frequency(Hz)')
plt.ylabel(r'$Z^2$')
plt.savefig('Z_test_harm1.png')
plt.show()


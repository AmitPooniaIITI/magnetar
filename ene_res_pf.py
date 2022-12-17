from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import math
from astropy.io import fits
lc=fits.open('output1.evn')
lc.info()
tbdata=lc[1].data
mask_data=(tbdata['PHA']<9)
newdata1=tbdata[mask_data]
mask_data2=newdata1['PHA']>6
newdata2=newdata1[mask_data2]
time2=newdata2['TIME']
time2=time2-time2[0]
time=time2

nu=0.1150849
phi=nu*time
phi1=phi-phi.astype(int)
phi2=np.ones(len(phi1))
phi3=phi1+phi2
phi4=np.hstack((phi1,phi3))
ph2=np.histogram(phi4,32)
ph1=np.histogram(phi1,16)
pc=ph2[0]
phival=np.delete(ph2[1],0)
pc_shift=np.roll(pc,16)



plt.figure(figsize=[8,8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=15)
plt.figure(figsize=(12,6))






ph3=plt.step(phival,pc_shift,where='mid')
y_err=np.sqrt(pc_shift)
del_pc=plt.errorbar(phival,pc_shift,yerr=y_err,fmt='o')
#plt.text(1.7,80400,'f=0.1150849')
plt.ylabel('Counts ')
plt.xlabel('Phase ($\phi$)')
plt.title('5-6 KeV Band')



total_time=(max(time)-min(time))
time_bin=total_time/16
pc1=ph1[0]
count_rate=pc1/time_bin
PF=(max(pc1)-min(pc1))/((max(pc1)+min(pc1)))

y_error=[]
prop_err=0



N=16
for i in range(0,N):
     prop_err=((pc1[i])**0.5)/time_bin
     y_error.append(prop_err)

temp_ave=0

pc_ave2=len(time)/total_time

y_err2=np.sqrt(count_rate)
y_err2=0
PCR=[]
for i in range (0,N,1):
      temp_ave=temp_ave+((count_rate[i]-pc_ave2)**2)-((y_error[i])**2)
PCR = (temp_ave/len(count_rate))**0.5
print(PCR)
print(PF)
plt.show()

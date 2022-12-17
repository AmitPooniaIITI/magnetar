from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from astropy.io import fits
lc=fits.open('4U0142.evt')
lc.info()
tbdata=lc[1].data
time=tbdata['TIME'][:]
time=time-min(time)
ph=np.histogram(time ,118048)
x=np.where(ph[0]==max(ph[0]))
time=np.delete(time,x)

print(time)
nu=0.1150849
phi=nu*time
phi1=phi-phi.astype(int)
ph1=np.histogram(phi1,16)
phi2=np.ones(len(phi1))
phi3=phi1+phi2
phi4=np.hstack((phi1,phi3))
ph2=np.histogram(phi4,32)
pc=ph2[0]
phival=np.delete(ph2[1],0)
y_err=np.sqrt(pc)
pc_shift=np.roll(pc,7)
ph3=plt.step(phival,pc_shift,where='mid')
y_err=np.sqrt(pc_shift)
del_pc=plt.errorbar(phival,pc_shift,yerr=y_err,fmt='o')
#plt.ylim(656500,662000)
plt.text(1.8,661500,'f=0.115082')
plt.ylabel('Counts --->')
plt.xlabel(r'$\phi$-->')


#calculation of different parametres of pulse profile

#caluclation of pulse fraction 
total_time=(max(time)-min(time))
time_bin=total_time/16
pc1=ph1[0]
count_rate=pc1/time_bin
PF=(max(pc1)-min(pc1))/((max(pc1)+min(pc1)))
print(PF)
y_error=[]
prop_err=0

#calculation of photon count rate

N=16
for i in range(0,16):
     prop_err=prop_err+(pc1[i])**0.5
     prop_err=prop_err/time_bin
     y_error.append(prop_err)
print(y_error)
temp_ave=0
#pc_ave2=(sum(count_rate))/len(count_rate)
pc_ave2=sum(pc1)/total_time

y_err2=np.sqrt(count_rate)
y_err2=0
PCR=[]
for i in range (0,N,1):
      temp_ave=temp_ave+((count_rate[i]-pc_ave2)**2)-((y_error[i])**2)
PCR = (temp_ave/len(count_rate))**0.5
print(PCR)


plt.show()





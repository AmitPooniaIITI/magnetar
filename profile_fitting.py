#import Library
from hashlib import algorithms_available
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from astropy.io import fits
from scipy.optimize import curve_fit
from numpy import append, fmax
from tqdm import tqdm
from pylab import *
#Calling output event file 

lc=fits.open('output1.evn')
lc.info()

#reading time column data from input fits file
tbdata=lc[1].data
time=tbdata['TIME'][:]
time=time-time[0]
pha=tbdata['PHA'][:]
#Setup frequency range 
f_r = np.arange(0.11506, 0.11510, 10**(-7))

ampmax = 0
amplitud = []
param=[]
a0=[]
a1=[]
a2=[]
#defining Model function
def f_x(phivale, a_0, a_1, phi_0, a_2, phi_1):
    return a_0 + a_1*np.sin(2*np.pi*(phivale-phi_0)) + a_2*np.sin(4*np.pi*(phivale-phi_1))

#loop running in frequency range
for nu in tqdm(f_r):
    phi = nu*time
    phi1 = phi-phi.astype(int)
    ph = np.histogram(phi1, 16)                              #Histogram in phi vs photon counts 
    #counts = (ph[0])                                         #Calling Photon counts             
   # phival = np.delete(ph[1], 0)
    


    phi2=np.ones(len(phi1))
    phi3=phi1+phi2
    phi4=np.hstack((phi1,phi3))
    ph2=np.histogram(phi4,32)
    counts=ph2[0]
    phival=np.delete(ph2[1],0)
   # y_err=np.sqrt(counts)
    counts=np.roll(counts,2)
   # ph3=plt.step(phival,pc_shift,where='mid')
   # y_err=np.sqrt(pc_shift)







    photon_noise = (counts)**0.5                              
    
    #fitting curve using scipy curve fitting libraby 
    popt, pcov = curve_fit(f_x, phival, counts,p0=None,sigma=photon_noise,bounds=(0,[ 10548175,10548175,1,10548175,1]))
       
      
     #calling amplitude of curve fit
    param.append(popt)
    a0=popt[0]
    a=max(popt[1],popt[3])
    amplitude=a/a0
    amplitud.append(amplitude)
     #index of maximum ampitude
                          

max_amp=np.argmax(amplitud)
phi=time*f_r[max_amp]                              #Histogram in phi vs photon counts                                          #Calling Photon count
phi1=phi-phi.astype(int)
ph1=np.histogram(phi1,16)
phi2=np.ones(len(phi1))
phi3=phi1+phi2
phi4=np.hstack((phi1,phi3))
ph2=np.histogram(phi4,32)
pc=ph2[0]
phival=np.delete(ph2[1],0)
y_err=np.sqrt(pc)
pc_shift=np.roll(pc,2)
#ph3=plt.step(phival,pc_shift,where='mid')
photon_noise=np.sqrt(pc_shift)
counts=pc_shift





#printing results

plt.figure(figsize=[8,8])
rc('axes', linewidth=2)
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
rc('font', weight='bold')
plt.rc('font', size=10)
plt.figure(figsize=(12,6))

print(param[max_amp])
print(f_r[max_amp])
plt.plot(phival,f_x(phival ,*param[max_amp]))
plt.step(phival,counts,where='mid')
plt.errorbar(phival,counts,yerr=photon_noise,fmt='o')
plt.xlabel('Phase $\phi$')
plt.ylabel('Photon Count')
#plt.text(0.8,662000,'freq=0.1150849')
plt.title('Pulse profile fitting Observation-1')
plt.savefig('pf1.png')
plt.show()





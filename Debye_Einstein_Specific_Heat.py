# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:01:46 2024

@author: USER
"""

import numpy as np
from matplotlib import pylab as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit


kk=np.genfromtxt('CpvsT.dat')
xx=kk[:,0]
yy=kk[:,1]



def model_func(x, th_d, th_e,r):
    
    def integrand(z):
        s1=(z**4)*np.exp(z)
        s2=(np.exp(z)-1)**2
        return s1/s2
    cdlist=[]
    celist=[]
    for T in x:
        result, error = quad(integrand, 0, th_d/T)
        nn=9*8.3145*((T/th_d)**3)*result
        cdlist.append(nn)
        #ss1=np.exp(th_e/i)
        #ss2=(np.exp(th_e/i)-1)**2
        #cc=3*8.3145*th_e/i*ss1/ss2
        #celist.append(cc)

    ss1=np.exp(th_e/x)
    ss2=(np.exp(th_e/x)-1)**2
    ss=ss1/ss2
    ce=3*8.3145*((th_e/x)**2)*ss
    
    #celist.append(cc)
    cd=np.array(cdlist)
    #ce=np.array(celist), 6 is the number of  atoms in unit cell.
    f=(6-r)*cd+r*ce
    #f=6*ce
    return f




# Generate some data
x_data =xx[1:len(xx)]
y_data =yy[1:len(xx)]

param_bounds = ([0, 0, 0], [2000, 2000, 6])

# Perform the curve fit with bounds
popt, pcov = curve_fit(model_func, x_data, y_data, bounds=param_bounds)



# Fit the model to the data
#popt, pcov = curve_fit(model_func, x_data, y_data)

# Plot the data and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, model_func(x_data, *popt), label='Fitted curve', color='red')
plt.legend()
plt.show()

# Extract standard deviations of the parameters
perr = np.sqrt(np.diag(pcov))
#%%
perr = np.sqrt(np.diag(pcov))

print("Optimal parameters:", popt)
print("Parameter uncertainties:", perr)

# @author: USER

import numpy as np
from matplotlib import pylab as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

# Load the data
kk = np.genfromtxt('CpvsT.dat')
xx = kk[:, 0]
yy = kk[:, 1]

# Define the Debye model function
def debye_model(x, th_d):
    def integrand(z):
        s1 = (z**4) * np.exp(z)
        s2 = (np.exp(z) - 1)**2
        return s1 / s2
    
    cdlist = []
    for T in x:
        result, _ = quad(integrand, 0, th_d / T)
        nn = 9 * 8.3145 * ((T / th_d)**3) * result
        cdlist.append(nn)

    cd = np.array(cdlist)
    f = 6 * cd  # Assuming 6 atoms per unit cell
    return f

# Prepare the data
x_data = xx[1:]
y_data = yy[1:]

# Fit only one parameter: Debye temperature (th_d)
param_bounds = ([0], [2000])
popt, pcov = curve_fit(debye_model, x_data, y_data, bounds=param_bounds)

# Plot the data and fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, debye_model(x_data, *popt), label='Debye Fit', color='red')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/mol·K)')
plt.title('Heat Capacity vs Temperature (Debye Model)')
plt.show()

# Extract standard deviation of the parameter
perr = np.sqrt(np.diag(pcov))
print("Optimal Debye temperature (Theta_D):", popt[0])
print("Uncertainty:", perr[0])

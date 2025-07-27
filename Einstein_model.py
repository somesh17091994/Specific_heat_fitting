# @author: USER

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import curve_fit

# Load the data
kk = np.genfromtxt('CpvsT.dat')
xx = kk[:, 0]
yy = kk[:, 1]

# Define the Einstein model function
def einstein_model(x, th_e):
    R = 8.3145  # Gas constant in J/mol·K
    ss1 = np.exp(th_e / x)
    ss2 = (np.exp(th_e / x) - 1)**2
    ce = 3 * R * (th_e / x)**2 * ss1 / ss2
    f = 6 * ce  # Assuming 6 atoms per unit cell
    return f

# Prepare the data
x_data = xx[1:]
y_data = yy[1:]

# Fit only one parameter: Einstein temperature (th_e)
param_bounds = ([0], [2000])
popt, pcov = curve_fit(einstein_model, x_data, y_data, bounds=param_bounds)

# Plot the data and fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, einstein_model(x_data, *popt), label='Einstein Fit', color='green')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/mol·K)')
plt.title('Heat Capacity vs Temperature (Einstein Model)')
plt.show()

# Extract standard deviation of the parameter
perr = np.sqrt(np.diag(pcov))
print("Optimal Einstein temperature (Theta_E):", popt[0])
print("Uncertainty:", perr[0])

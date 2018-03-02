# Python packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# User-defined functions
import radiative_transfer as rt
from rms import rms


# Defining data
X = np.linspace(800, 1200, 100)
# T = np.arange(270, 320, 10)
T = 300 + 5 * np.random.randn(320,256)

# -----Radiance-----
B = rt.planckian(X, T)

# Radiance plot
plt.figure(1)													# Initializing figure
plt.plot(X, B[1,1,:], lw=1.)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))	# Forcing scientific notation on y-axis
plt.xlabel(r'Wavenumber [cm $^{-1}$]')
plt.ylabel(r'Radiance [W/(cm$^2$ sr cm$^{-1}$)]')
plt.title('Planckian at Different Temperatures')


# -----Brightness Temperature-----
Tb = brightnessTemperature(X, B)

# Brightness Temp plot
plt.figure(2)
plt.plot(X, Tb, lw=1.)
plt.xlabel(r'Wavenumber [cm$^{-1}$]')
plt.ylabel(r'Brightness Temperature [K]')
plt.title('Planckian Brightness Temperature')



# -----Fitting-----

# Generating data:
emis = .75
T1 = 303
Y = emis * planckian(X, T1)
Y = Y + np.mean(Y) * np.random.standard_normal(Y.shape) / 100


# Noisy data plot
plt.figure(3)
plt.plot(X, brightnessTemperature(X, Y), lw=1.)
plt.xlabel(r'Wavenumber [cm$^{-1}$]')
plt.ylabel(r'Brightness Temperature [K]')


# Fitting function
def f(p):
	return p[0] * planckian(X, p[1])

# Sum-of-squared-errors used to minimize
def e(p):
	return np.sum((Y - f(p)) ** 2)

# Initial guess
p0 = np.array((1., 300.))

# Optimization
p = opt.fmin(e, p0)


# Data fit plot
plt.figure(4)
plt.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)							# Create plot grid and first subplot
plt.plot(X, Y,'.', ms=2.5, label='Data')										# Plotting Data set
plt.plot(X, f(p), '-', lw=1., label='Fit')										# Plotting Fit
plt.title('Best fit parameters: %0.3f, temp %0.3f' % (p[0], p[1]))
plt.ylabel(r'Radiance [W/(cm$^2$ sr cm$^{-1}$)]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))					# Forcing scientific notation on y-axis
plt.legend()																	# Plot legend (lines labeled when first called)
plt.subplot2grid((5,4), (4,0), rowspan=1, colspan=4)							# Creating second subplot
plt.plot(X, Y - f(p), lw=1.)
plt.ylim([-5e-7,5e-7])															# Forcing y-axis limits
plt.title(r'RMS error: %0.3e W/cm(cm$^2$ sr cm$^{-1}$)' % (rms(Y - f(p))))
plt.ylabel('Resid.')
plt.xlabel(r'Wavenumbers [cm$^{-1}$]')


plt.show()		# Displays figures sequentially in separate windows
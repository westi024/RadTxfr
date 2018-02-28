"""
AFIT/CTISR radiative transfer module.

Code for generating the Planckian distribution and brightness temperature. Also
includes various helper functions for manipulating ND-arrays.

Contact information
-------------------
Kevin C. Gross (AFIT/ENP)
Air Force Institute of Technology
Wright-Patterson AFB, Ohio
Kevin.Gross@afit.edu
grosskc.afit@gmail.com
(937) 255-3636 x4558
Version 0.3.0 -- 28-Feb-2018

Version History
---------------
V 0.1.0 05-Jun-2017 Initial code.
V 0.2.0 13-Feb-2018 Added comments.
V 0.3.0 28-Feb-2018 Convert python numeric inputs to numpy arrays. Fixed
  comments to pass muster with the linter. Ensured 1D vectors are really 1D.
  Simplified array reshaping. Changed convention to spectral axis as the *first*
  dimension. Calculations now performed in SI units. Simplified units conversion.

TODO
____
* Add 1D RTE solution (radiance, transmittance)
* Add absorption cross-section database handling
* Add LBLRTM hooks
* Improve testing
"""

import numpy as np

# Module constants
# h  = 6.6260689633e-34 # [J s]       - Planck's constant
# c  = 299792458        # [cm/s]      - speed of light
# k  = 1.380650424e-23  # [J/K]       - Boltzman constant
c1 = 1.19104295315e-16  # [J m^2 / s] - 1st radiation constant, c1 = 2*h*c^2
c2 = 1.43877736830e-02  # [m K]       - 2nd radiation constant, c2 = h * c / k


def rs1D(y):
    """
    Reshape ND-array into a 1D vector.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 1D version of the array

    """
    y = np.array(y)
    dims = y.shape
    return y.flatten(), dims


def rs2D(y):
    """
    Reshape ND-array into 2D-array by flattening 2nd thru Nth dimensions.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 2D version of the array

    """
    y = np.array(y)
    dims = y.shape
    return y.reshape((dims[0], np.prod(dims[1:]))), dims


def rsND(y, dims):
    """
    Reshape a 1D- or 2D-array back into an ND-array.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array
    dims : tuple
      The shape of the ND-array

    Returns
    _______
    y : numpy array
      A reshaped, ND version of the array

    """
    return y.reshape(dims)


def planckian(X, T, f=False):
    """
    Compute the Planckian radiance distribution.

    Computes the spectral radiance L at wavenumber(s) X for a system at
    temperature(s) T using Planck's distribution function. X must be a scalar
    or a vector. T can be of arbitrary dimensions. The shape of output L will
    be ``(X.size, *T.shape)``.

    Parameters
    ----------
    X : numpy array
      spectral input in wavenumbers [1/cm]
    T : numpy array
      temperature imput in Kelvin [K]
    f : logical
      if true, spectral input X is given in wavelength [micron]

    Returns
    -------
    L : numpy array
      spectral radiance in [W / (cm^2 sr cm^-1)], or if f=True, spectral
      radiance in [micro W / (cm^2 sr micron)] (microflick)

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import radiative_transfer as rt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = rt.planckian(X,T)
    >>> plt.plot(X,L)

    """
    # Ensure inputs are NumPy arrays with correct shapes
    T = np.array(T)  # can be scalar, vector, ND-array
    X = np.array(X).squeeze()  # X *must* be a 1D vector
    if X.ndim > 1:
        raise TypeError('X must be a scalar or 1D array')
    elif X.ndim == 0:
        X = np.array([X])

    # Ensure X is column vector for outer products
    X = X[:, np.newaxis]

    # Ensure T is a row vector for outer products
    T, dimsT = rs1D(T)
    T = T[np.newaxis, :]

    # Compute Planck's spectral radiance distribution
    if f or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not f:
            print('Assumes X given in µm; returning L in µF')
        X *= 1e-6  # convert to m from µm
        L = c1 / (X**5 * (np.exp(c2 / (X * T)) - 1))  # [W/(m^2 sr m)]
        L *= 1e-4  # convert to [µW/(cm^2 sr µm^{-1})]
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L = c1 * X**3 / (np.exp(c2 * X / T) - 1)  # [W/(m^2 sr m^{-1})]
        L /= 100  # convert to [W/(cm^2 sr cm^{-1})]

    # Reshape L if necessary and return
    return np.reshape(L, (X.size, *dimsT)).squeeze()


def brightnessTemperature(X, L, f=False):
    """
    Compute brightness temperature at given spectral radiance.

    The brightness temperature is the temperature at which a perfect blackbody
    would need to be to produce the same spectral radiance L at each specified
    wavenumber X. The shape of output T will be ``(X.size, *L.shape)``.

    Parameters
    ----------
    X : numpy array (must be a vector)
      spectral input in wavenumbers [cm^{-1}]
    L : numpy array
      spectral radiance in [W/(cm^2 sr cm^-1)]
    f : logical
      if true, spectral input X is given in wavelength [µm] and L is given
      in [µW / (cm^2 sr µm)] (microflick, µF)

    Returns
    -------
    T : numpy array
      brightness temperature in [K]

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import radiative_transfer as rt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = rt.planckian(X,T)
    >>> T = rt.brightnessTemperature(X,L)
    >>> plt.plot(X,T)

    """
    # Ensure inputs are NumPy arrays with correct shapes
    X = np.array(X).squeeze()  # X *must* be a 1D vector
    if X.ndim > 1:
        raise TypeError('X must be a scalar or 1D array')
    L = np.array(L)

    # Ensure X is row vector for outer products
    X = X[:, np.newaxis]

    # Make L a column vector or 2D array w/ spectral axis as 1st dim
    L, dimsL = rs2D(L)

    # Evaluate brightness temperature
    if f or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not f:
            print('Assumes X given in µm and L given in µF')
        X *= 1e-6  # convert to m from µm
        L *= 1e+4  # convert to [W/(m^2 sr m)] from [µW/(cm^2 sr µm)]
        T = c2 / (X * np.log(1 + c1 / (X**5 * L)))
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L *= 100  # convert to [W/(m^2 sr m^{-1})] from [W/(cm^2 sr cm^{-1})]
        T = c2 * X / np.log(c1 * X**3 / L + 1)

    # NaN-ify garbage results
    ixBad = np.logical_or(np.real(L) <= 0, np.abs(np.imag(T)) > 0)
    T[ixBad] = np.nan

    # Reshape T if necessary
    return np.reshape(T, (X.size, *dimsL[1:])).squeeze()


if __name__ == "__main__":
    # Simple test script
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Set plotting defaults
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['text.latex.preamble'] = r'\usepackage[adobe-utopia]{mathdesign}, \usepackage{siunitx}'

    # Define strings containing LaTeX formatted stuff for plots

    s_rad_wn = r'Spectral Radiance, $L(\tilde{\nu})$ $\left[ \si{ {\micro}W/(cm^2.sr.cm^{-1}) } \right]$'
    s_rad_wl = r'Spectral Radiance, $L(\lambda)$ $\left[ \si{ {\micro}W/(cm^2.sr.\um) } \right]$'
    s_Tb_wn = r'Brightness Temperature, $T_B(\tilde{\nu})$ $\left[\si{K}\right]$'
    s_Tb_wl = r'Brightness Temperature, $T_B(\lambda)$ $\left[\si{K}\right]$'
    s_wn = r'Wavenumbers, $\tilde{\nu}$ $\left[\si{cm^{-1}}\right]$'
    s_wl = r'Wavelength, $\lambda$ $\left[\si{{\micro}m}\right]$'

    def s_leg_rad(T): return ["$T = {0}$ K".format(TT) for TT in T]

    # Test at known temperatures and wavenumbers / wavelengths
    T = 296
    wn = 500  # wavenumber
    wl = 10000 / wn  # equivalent wavelength
    d_wn = 1  # differential wavenumber
    d_wl = (d_wn / wn) * wl  # equivalent differential wavelength
    L_wn = planckian(wn, T)
    L_wl = planckian(wl, T, f=True)
    s1 = "L(X = {0}/cm, T = {1}K) = {2:0.6e} W/(cm^2 sr 1/cm)\n".format(wn, T, float(L_wn))
    s2 = "L(X = {0}µm, T = {1}K) = {2:0.6e} µW/(cm^2 sr µm)\n".format(wl, T, float(L_wl))
    sa = "L(X = {0}/cm, T = {1}K) * (ΔX = {2:0.2e}/cm) = {3:0.6e} W/(cm^2 sr)\n".format(wn, T, d_wn, float(L_wn * d_wn))
    sb = "L(X = {0}µm, T = {1}K) * (ΔX = {2:0.2e}µm) = {3:0.6e} W/(cm^2 sr)\n".format(wl, T, d_wl, float(1e-6 * L_wl * d_wl))  # convert to W from µW
    print(s1 + s2 + sa + sb)

    # Compute radiance and brightness temperature
    T = np.arange(250, 375, 25)
    X1 = np.linspace(100, 2500, 500)
    X2 = 10000 / X1
    L1 = planckian(X1, T)
    L2 = planckian(X2, T, f=True)
    Tb1 = brightnessTemperature(X1, L1)
    Tb2 = brightnessTemperature(X2, L2, f=True)

    # Plot results for wavenumbers
    plt.figure(figsize=(7.5, 10.5), dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(X1, L1 * 1e6)
    plt.xlabel(s_wn)
    plt.ylabel(s_rad_wn)
    plt.legend(s_leg_rad(T))
    plt.title('Planckian Spectral Radiance Distribution in Wavenumbers')
    plt.subplot(2, 1, 2)
    plt.plot(X1, Tb1)
    plt.title('Spectral Brightness Temperature Distribution in Wavelength')
    plt.yticks(T)
    plt.xlabel(s_wn)
    plt.ylabel(s_Tb_wn)
    plt.show()

    # Plot results for wavelengths
    plt.figure(figsize=(7.5, 10.5), dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(X2, L2)
    plt.xlabel(s_wl)
    plt.ylabel(s_rad_wl)
    plt.legend(s_leg_rad(T))
    plt.title('Planckian Spectral Radiance Distribution in Wavelength')
    plt.subplot(2, 1, 2)
    plt.plot(X2, Tb2)
    plt.title('Spectral Brightness Temperature Distribution in Wavelength')
    plt.yticks(T)
    plt.xlabel(s_wl)
    plt.ylabel(s_Tb_wl)
    plt.show()

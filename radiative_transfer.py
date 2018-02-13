"""
A radiative transfer module

---------------------------------
Kevin C. Gross (AFIT/ENP)
Air Force Institute of Technology
Wright-Patterson AFB, Ohio
Kevin.Gross@afit.edu
(937) 255-3636 x4558
05-Jun-2017  --  Version 0.1
---------------------------------

"""

import numpy as np


def rs1D(y):
    """
    Reshape a NumPy ND-array into a 1D vector

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 1D version of the array

    """

    dims = y.shape
    return y.reshape(np.prod(dims)), dims


def rs2D(y):
    """
    Reshape a NumPy ND-array into a 2D-array by collapsing the first N-1 dims

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 2D version of the array

    """

    dims = y.shape
    return y.reshape((np.prod(dims[:-1]), dims[-1])), dims


def rsND(y, dims):
    """
    Reshape a NumPy 2D-array back into an ND-array

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
    Compute the Planckian distribution.

    Computes the spectral radiance L at wavenumber(s) X for a system at
    temperature(s) T using Planck's distribution function. X must be a scalar
    or a vector. T can be of arbitrary dimensions. The shape of output L will
    be ``(*T.shape X.size)``.

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

    # Ensure inputs are correct
    if isinstance(X, float):
        X = np.array([X])
    if isinstance(T, float):
        T = np.array([T])
    if not isinstance(X, np.ndarray):
        raise RuntimeError('Wavenumber X must be a NumPy vector.')
    if X.shape[0] != X.size:
        raise RuntimeError('Wavenumber X cannot be a matrix.')

    # Ensure X shape is col vec so outer products work with row vec T
    X = X[:, np.newaxis]

    # Make T a vector for computation -- reshape output later
    dimsT = T.shape
    if len(dimsT) > 1:
        T, _ = rs1D(T)

    # Defining the first and second radiation constants (2014 CODATA)
    c1 = 1.19104295315e-16  # [W m^2 / sr]
    c2 = 1.43877736830e-02  # [m K]

    # Spectral radiance as a function of wavelength
    if f or np.mean(X) < 50:
        if np.mean(X) < 50:
            print('Assuming microflicks for output spectral units.')
        hold1 = c2 / np.outer(1e-6 * X, T)
        L = 1e-4 * c1 / np.multiply(np.exp(hold1) - 1, (1e-6 * X) ** 5)

    # Spectral radiance as a function of wavenumber
    else:
        c1 *= 100**2
        c2 *= 100
        hold2 = c2 * np.outer(X, 1 / T)
        L = c1 * np.multiply(1 / (np.exp(hold2) - 1), X ** 3)

    # Reshape L if necessary
    if len(dimsT) > 1:
        L = np.reshape(L.T, (*dimsT, X.size))

    return L


def brightnessTemperature(X, L, f=False):
    """
    Computes the brightness temperature, i.e. the temperature at which a
    perfect blackbody would need to be to produce the same spectral radiance
    at each specified wavenumber X.

    Parameters
    ----------
    X : numpy array
      spectral input in wavenumbers [1/cm]
    L : numpy array
      spectral radiance in [W/(cm^2 sr cm^-1)]
    f : logical
      if true, spectral input X is given in wavelength [micron] and L is given
      in [micro W / (cm^2 sr micron)] (microflick)

    Returns
    -------
    T   : numpy array
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

    # Ensure inputs are correct
    if isinstance(X, float):
        X = np.array([X])
    if isinstance(L, float):
        L = np.array([L])
    if not isinstance(X, np.ndarray):
        raise RuntimeError('Wavenumber X must be a NumPy vector.')
    if X.shape[0] != X.size:
        raise RuntimeError('Wavenumber X cannot be a matrix.')

    # Convert into 2D array if necessary
    dims = L.shape
    if len(dims) > 2:
        L, dims = rs2D(L)

    # Defining the first and second radiation constants  (2014 CODATA)
    c1 = 1.19104295315e-16  # [W m^2 / sr]
    c2 = 1.43877736830e-02  # [m K]

    # Evaluate brightness temperature:
    # Function of wavelength:
    if f or np.mean(X) < 50:
        if not f and np.mean(X) < 50:
            print('Assuming microflicks for output spectral units.')

        # Holding function for clarity and error testing:
        hold1 = np.multiply(1e4 * L, (1e-6 * X) ** 5)
        T = 1e6 * c2 / np.multiply(1 + c1 / hold1, X)

    # Function of wavenumber:
    else:
        c1 *= 100 ** 2
        c2 *= 100

        # Holding function for clarity and error testing
        hold2 = np.divide(X ** 3, L)
        T = c2 * np.divide(X, np.log(c1 * hold2 + 1))

    # NaN-ify garbage results
    ixBad = np.logical_or(np.real(L) <= 0, np.abs(np.imag(T)) > 0)
    T[ixBad] = np.nan

    # Convert back to ND array if necessary
    if len(dims) > 2:
        T = rsND(T, dims)

    return T

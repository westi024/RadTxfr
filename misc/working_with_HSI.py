import numpy as np
import scipy as sp
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import numpy.polynomial.polynomial as poly

import radiative_transfer as rt
import spectral as spc
importlib.reload(rt)

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)


def outlier_index(y, thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    madL = np.median(abs_dev[y <= m])
    madR = np.median(abs_dev[y >= m])
    y_mad = madL * np.ones(len(y))
    y_mad[y > m] = madR
    robust_z_score = 0.6745 * abs_dev / y_mad
    robust_z_score[y == m] = 0
    return robust_z_score > thresh


@jit
def q_n(a):
    """Rousseeuw & Croux's (1993) Q_n, an alternative to MAD.

    ``Qn := Cn first quartile of (|x_i - x_j|: i < j)``

    where Cn is a constant depending on n.

    Finite-sample correction factors must be used to calibrate the
    scale of Qn for small-to-medium-sized samples.

        n   E[Qn]
        --  -----
        10  1.392
        20  1.193
        40  1.093
        60  1.064
        80  1.048
        100 1.038
        200 1.019

    """
    if not len(a):
        return np.nan

    # First quartile of: (|x_i - x_j|: i < j)
    vals = []
    for i, x_i in enumerate(a):
        for x_j in a[i+1:]:
            vals.append(abs(x_i - x_j))
    quartile = np.percentile(vals, 25)

    # Cn: a scaling factor determined by sample size
    n = len(a)
    if n <= 10:
        # ENH: warn when extrapolating beyond the data
        # ENH: simulate for values up to 10
        #   (unless the equation below is reliable)
        scale = 1.392
    elif 10 < n < 400:
        # I fitted the simulated values (above) to a power function in Excel:
        #   f(x) = 1.0 + 3.9559 * x ^ -1.0086
        # This should be OK for interpolation. (Does it apply generally?)
        scale = 1.0 + (4 / n)
    else:
        scale = 1.0

    return quartile / scale


def nrm(x):
    return (x-min(x)) / max(x-min(x))


def estimate_tau(L):
    from scipy.interpolate import splev, splrep
    L_med = np.median(L, axis=0)
    L_mad = np.median(np.abs(L - L_med), axis=0)
    tau_est = nrm(L_mad / L_med)
    x = np.arange(tau_est.size) / tau_est.size
    w = np.ones(x.shape)
    for ii in range(10):
        spl = splrep(x, tau_est, w=w, k=2)
        w = tau_est - splev(x, spl)
        w[w < 0] = w[w < 0] / 100
        w = abs(w)
    tau_est = nrm(tau_est / splev(x, spl))
    return tau_est


# Import data
fdir = '/Users/grosskc/Dropbox (AFIT RSG)/Students/2017S-OKeefeDaniel/matlab/'
fname = '076_160816_000130_UI_WSurveyBi_12k000_77_Whisk71_L2S'
data = spc.envi.open(fdir+fname+'.hdr',fdir+fname+'.dat')
img = data.load()
L, dims = rt.rs2D(img)
X = np.array([float(i) for i in data.bands.centers])

# Filter out outlier spectra
L_s = mad(L.T, axis=0)
ixB1 = outlier_index(L_s, 3)
L_m = median(L, axis=1)
ixB2 = outlier_index(L_m, 3)
ixB = np.logical_or(ixB1, ixB2)
ixG = np.logical_not(ixB)

L2 = L.copy()
L2[ixB,:]=0
L = L[ixG,:]

tau_est = mad(L, axis=0) / median(L, axis=0)
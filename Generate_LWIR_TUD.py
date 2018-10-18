import multiprocessing as mp
import ctypes
from datetime import datetime

import numpy as np
import scipy.interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py

import radiative_transfer as rt
from smooth import smooth

# Load MAT file
atmos = loadmat("AtmosphericInputs-TIGR.mat")

# Load standard atmosphere
StdAtmos = np.genfromtxt("StandardAtmosphere.csv", dtype=float, delimiter=',', names=True)
P = StdAtmos["P_Pa"]
T = StdAtmos["T_K"]
z = StdAtmos["Z0_km"]
CO2 = StdAtmos["CO2"]
N2O = StdAtmos["N2O"]
CO = StdAtmos["CO"]
CH4 = StdAtmos["CH4"]
O2 = StdAtmos["O2"]
N2 = StdAtmos["N2"]
Ar = StdAtmos["Ar"]
T_SA = T.copy()
Z_SA = z.copy()

# atmopspheric state variables
_P = atmos["P"].flatten()*100  # pressure [Pa] from [hPa]
_T = atmos["T"]                # temperature profile, [K]
_H2O = atmos["H2O"]/1e6        # water profile, [ppmv] from mixing fraction
_O3 = atmos["O3"]              # ozone profile, [ppmv]
_z = atmos["z"]                # altitude [km]
nAtm = _T.shape[0]             # number of atmospheric states

# Interpolate onto the StdAtmos sampling grid
ix = np.arange(_T.shape[0])
T, H2O, O3 = [np.zeros((ix.size, StdAtmos["Z0_km"].size)) for _ in range(3)]
interp = lambda x, y, x0: scipy.interpolate.interp1d(x, y, kind='cubic')(x0)
z = StdAtmos["Z0_km"]
for ii, idx in enumerate(ix):
    T[ii, :] = interp(_z[idx,:], _T[idx,:], z)
    H2O[ii, :] = interp(_z[idx,:], _H2O[idx,:], z)
    O3[ii, :] = interp(_z[idx,:], _O3[idx,:], z)

# Determine mean profile
Tm, H2Om, O3m = tuple(map(lambda x: np.mean(x, axis=0), (T, H2O, O3)))

# Define Jacobian helper function
def JacIn(X, dX, rel=False):
    X_out = np.tile(X, (X.shape[-1], 1))
    if np.isscalar(dX):
        if rel:
            dX *= np.ones_like(X_out) * np.sqrt(np.mean(X_out**2))
        else:
            dX += np.zeros_like(X_out)
    for ii in np.arange(len(X_out)):
        X_out[ii, ii] = X_out[ii, ii] + dX[ii, ii]
    return X_out

# Create inputs to compute TUD Jacobian about mean atmospheric profiles
relStep = 0.001
varsJ = JacIn(np.hstack((Tm, H2Om, O3m)), relStep, rel=True)
var = []
l = len(Tm)
for ii in range(3):
    ix = np.arange(ii*l, (ii+1)*l)
    var.append(varsJ[ix][:,ix])
Tm_J, H2Om_J, O3m_J = tuple(map(lambda x: np.tile(x, (x.shape[-1] + 1, 1)), (Tm, H2Om, O3m)))
Tm_J[1:,:], H2Om_J[1:,:], O3m_J[1:,:] = var[0], var[1], var[2]
nAtm = Tm_J.shape[0]

# Extract subset of the 2311 atmospheric states
# N = 100
# ix = np.round(np.linspace(0, _T.shape[0]-1, N, endpoint=True)).astype(np.int)
# ixT = np.argsort(_T[:, 0])[ix]
# ixW = np.argsort(np.sum(_H2O, axis=1))[ix]
# ix = np.unique([ixT, ixW])
# nAtm = len(ix)
# H2O = H2O[ix, :]
# O3 = O3[ix, :]
# T = T[ix, :]

# Altitudes at which to compute tau and La
Altitudes = np.concatenate((np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000]) * 0.3048 /1e3, [z.max()])) # [km] from [ft]

# Compute TUD for StdAtmos
Xmin = 690
Xmax = 1410
DV = 0.0005
DV_out = 0.25
XX, OD_SA, La_SA, Ld_SA = rt.compute_TUD(Xmin, Xmax, DVOUT=DV, returnOD=True)
X, OD_SA = rt.reduceResolution(XX, OD_SA, DV_out)
f = lambda X_in, Y_in: rt.reduceResolution(X_in, Y_in, DV_out, X_out=X)
La_SA = f(XX, La_SA)
Ld_SA = f(XX, Ld_SA)

# plt.figure()
# plt.plot(X, np.exp(-OD_SA))
# plt.figure()
# plt.plot(X, La_SA, label="La")
# plt.plot(X, Ld_SA, label="Ld")

# Compute TUD for scenario
OD, La = [np.zeros((len(X), len(Altitudes), nAtm)) for _ in range(2)]
Ld = np.zeros((len(X), nAtm))

def make_shared(A):
    """Form a shared memory numpy array."""
    shared_array_base = mp.Array(ctypes.c_double, int(np.prod(A.shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    A = shared_array.reshape(*A.shape)
    return A

OD = make_shared(OD)
La = make_shared(La)
Ld = make_shared(Ld)

lock = mp.Lock()
fname = datetime.now().strftime('%Y%m%d-%H%M%S') + '-LWIR-TUD-Jacobian.npz'

def parallel_function(jj):
    """Function that operates on shared memory."""

    # MFs_VAL = 1e6*np.asarray([H2O[jj, :], CO2, O3[jj, :]]).T
    # nu, OD_, La_, Ld_ = rt.compute_TUD(Xmin, Xmax, MFs_VAL=MFs_VAL, MFs_ID=[1, 2, 3],
    #                                    Ts=T[jj, :], DVOUT=DV, Altitudes=Altitudes, returnOD=True)
    MFs_VAL = 1e6*np.asarray([H2Om_J[jj, :], CO2, O3m_J[jj, :]]).T
    nu, OD_, La_, Ld_ = rt.compute_TUD(Xmin, Xmax, MFs_VAL=MFs_VAL, MFs_ID=[1, 2, 3],
                                       Ts=Tm_J[jj, :], DVOUT=DV, Altitudes=Altitudes, returnOD=True)

    OD_ = f(nu, OD_)
    La_ = f(nu, La_)
    Ld_ = f(nu, Ld_)

    lock.acquire()
    OD[:, :, jj] = OD_
    La[:, :, jj] = La_
    Ld[:, jj] = Ld_
    lock.release()

    print("Completed {0:04d} of {1:04d}".format(jj + 1, nAtm))
    return jj

if __name__ == '__main__':
    nProc = 6
    nChunk = 4
    N = nProc * nChunk
    nBatch = np.ceil(nAtm / N).astype(np.int)
    vals = np.arange(N)
    with mp.Pool(processes=nProc) as pool:
        for ii in np.arange(nBatch):
            idx = ii * N + vals
            idx = (idx[idx < nAtm]).astype(np.int)
            print(f"Batch {ii+1:0d} of {nBatch:0d}, indices {idx[0]:d} through {idx[-1]:d}")
            print(idx)
            pool.map(parallel_function, idx)
            np.savez(fname, X=X, OD=OD, La=La, Ld=Ld, Altitudes=Altitudes)

    # Save as HDF5 file
    # hf = h5py.File('LWIR_TUD.h5', 'w')
    hf = h5py.File('LWIR_TUD_JACOBIAN.h5', 'w')
    d = hf.create_dataset('X', data=X)
    d.attrs['units'] = 'cm^{-1}'
    d.attrs['name'] = 'Wavenumbers'
    d.attrs['info'] = 'Spectral axis for tau, La, Ld'
    d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

    d = hf.create_dataset('OD', data=OD)
    d.attrs['units'] = 'none'
    d.attrs['name'] = 'Optical Depth'
    d.attrs['info'] = 'For nadir-viewing path. tau = np.exp(-OD)'
    d.attrs['label'] = r'$\tau(\tilde{\nu})$'

    d = hf.create_dataset('La', data=La)
    d.attrs['units'] = 'µW/(cm^2 sr cm^{-1})'
    d.attrs['name'] = 'Atmospheric Path Spectral Radiance'
    d.attrs['info'] = 'For nadir-viewing path, earth-to-space'
    d.attrs['label'] = r'$L_a(\tilde{\nu})\,\,\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]$'

    d = hf.create_dataset('Ld', data=Ld)
    d.attrs['units'] = 'µW/(cm^2 sr cm^{-1})'
    d.attrs['name'] = 'Atmospheric Downwelling Spectral Radiance'
    d.attrs['info'] = 'Hemispherically-averaged, space-to-earth'
    d.attrs['label'] = r'$L_d(\tilde{\nu})\,\,\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]$'

    d = hf.create_dataset('SensorAltitude', data=Altitudes)
    d.attrs['units'] = 'km'
    d.attrs['name'] = 'Sensor Altitude'
    d.attrs['info'] = 'Sensor height above surface'
    d.attrs['label'] = r'$z_s \,\, \left[ \si{km} \right]$'

    d = hf.create_dataset('StdAtmos', data=StdAtmos)
    d.attrs['units'] = 'N.A.'
    d.attrs['name'] = 'Standard Atmosphere Table'
    d.attrs['info'] = '1976 US Std. Atmos. with CO2 scaled to modern value'
    d.attrs['label'] = r'N.A.'

    d = hf.create_dataset('z', data=z)
    d.attrs['units'] = 'km'
    d.attrs['name'] = 'Altitude'
    d.attrs['info'] = 'z=0 at sea level'
    d.attrs['label'] = r'$z \,\, \left[ \si{km} \right]$'

    d = hf.create_dataset('T', data=T)
    d.attrs['units'] = 'K'
    d.attrs['name'] = 'Temperature profile'
    d.attrs['info'] = ''
    d.attrs['label'] = r'$T(z) \,\, \left[ \si{K} \right]$'

    d = hf.create_dataset('P', data=P)
    d.attrs['units'] = 'Pa'
    d.attrs['name'] = 'Pressure profile'
    d.attrs['info'] = ''
    d.attrs['label'] = r'$P(z) \,\, \left[ \si{Pa} \right]$'

    d = hf.create_dataset('H2O', data=H2O)
    d.attrs['units'] = 'ppmv'
    d.attrs['name'] = 'Water vapor VMR profile'
    d.attrs['info'] = 'VMR - volume mixing ratio'
    d.attrs['label'] = r'$\mathrm{H_2O}(z)\,\,\left[\mathrm{ppm}_v\right]$'

    d = hf.create_dataset('O3', data=O3)
    d.attrs['units'] = 'ppmv'
    d.attrs['name'] = 'Ozone VMR profile'
    d.attrs['info'] = 'VMR - volume mixing ratio'
    d.attrs['label'] = r'$\mathrm{O_3}(z)\,\,\left[\mathrm{ppm}_v\right]$'

    hf.close()


# --------------------
# Atmospheric Modeling
# --------------------
# import sklearn
# from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, SparsePCA, FactorAnalysis
# from sklearn.cross_decomposition import CCA
#
# def pca_mdl(pca, p):
#     p = np.array(p)
#     if len(p.shape) == 1:
#         n = p.size
#         m = 1
#     else:
#         n = p.shape[1]
#         m = p.shape[0]
#     k = np.min([pca.n_components_, pca.n_samples_])
#     p0 = np.zeros((m, k))
#     p0[0:m, 0:n] = p
#     return pca.inverse_transform(p0)


# pcaATM = PCA(whiten=True)
# N_level = T.shape[1]
# data = np.concatenate((T.T, H2O.T, O3.T)).T
# pcaATM.fit(data)

# def atm_mdl(p):
#     out = pca_mdl(pcaATM, p).flatten()
#     T = out[0:N_level]
#     H2O = out[N_level:2 * N_level]
#     O3 = out[2 * N_level:3 * N_level]
#     return [T, H2O, O3]

# T_mdl = lambda p: (atm_mdl(p)[0]).flatten()
# H2O_mdl = lambda p: (atm_mdl(p)[1]).flatten()
# O3_mdl = lambda p: (atm_mdl(p)[2]).flatten()

# ixZ = z < 8
# pcaT = PCA(whiten=True)
# pcaT.fit(T[:,ixZ])
# pcaH2O = PCA(whiten=True)
# pcaH2O.fit(H2O[:,ixZ])
# pcaO3 = PCA(whiten=True)
# pcaO3.fit(O3)

# T_mdl = lambda p: pca_mdl(pcaT, p).flatten()
# H2O_mdl = lambda p: pca_mdl(pcaH2O, p).flatten()
# O3_mdl = lambda p: pca_mdl(pcaO3, p).flatten()



# scaleHeight = 8.0
# Z0 = 8
# T2 = T.copy()
# W = np.ones(z.size)
# W[z>=Z0] = np.exp(-(z[z>=Z0]-Z0)/scaleHeight)
# T2 = W[np.newaxis,:]*T2 + ((1-W)*T2.mean(axis=0))[np.newaxis,:]

# pcaT = PCA(whiten=True)
# pcaT.fit(T2[:, z < 11])
# T_mean=T2[:,z>=11].mean(axis=0)
# Tm = lambda p: np.concatenate((pcaT.inverse_transform(np.concatenate((np.array(p), np.zeros(np.sum(z<11)-4)))), T_mean))

# ixZ = z < 40
# p0 = [0, 0, 0, 0]
# bnds = ((-5, 5), (-5, 5), (-5, 5), (-5, 5))

# p = np.zeros((T.shape[0], 4))
# errVal = np.zeros(T.shape[0])
# W = P/np.sum(P)
# for ii in range(T.shape[0]):
#     err = lambda p: np.sum(W[ixZ]*( (T[ii, ixZ] - T_mdl(p)[ixZ])**2 / T[ii,ixZ].mean()**2 + (H2O[ii, ixZ] - H2O_mdl(p)[ixZ])**2 / H2O[ii,ixZ].mean()**2) )
#     out = scipy.optimize.minimize(err, p0, bounds=bnds, options={'maxiter': 500})
#     p[ii,:] = out['x']
#     errVal[ii] = out['fun']

# ix = 20; plt.plot(T[ix, ixZ], z[ixZ], '-r', T_mdl(p[ix,:])[ixZ], z[ixZ], '-b')
# ix = 20; plt.plot(H2O[ix, ixZ], z[ixZ], '-r', H2O_mdl(p[ix,:])[ixZ], z[ixZ], '-b')

# # Fit parametric model to atmospheric profiles
# def T_mdl(z, T0=290, lapseRate=6.4, anomalousLapseRate=0, T_pause=216.78, Z_pause=11.0, scaleHeight=8.0, scaleFactor=1.0):
#     T = T0 - lapseRate * (z - z[0]) + anomalousLapseRate * (z - z[0]) * np.exp(-(z - z[0]) / scaleHeight)
#     T_linear = (T_pause - T0) / (Z_pause - z[0]) * (z-z[0]) + T0
#     W = 1.0/(z[0]-Z_pause) * z + 1.0
#     W[W<0] = 0
#     W *= np.exp(-(z - z[0]) / (Z_pause))
#     T = W*T + (1-W)*T_linear
#     Ts = np.interp(z, Z_SA, T_SA)
#     T_z0 = np.interp(Z_pause, z, T)
#     Ts_z0 = np.interp(Z_pause, Z_SA, T_SA)
#     TT = Ts[z>=Z_pause][0]
#     T[z > Z_pause] = (scaleFactor * (Ts[z > Z_pause]-TT) + TT) - (Ts_z0 - T_z0)
#     return T



# def H2O_mdl(z, a, b):
#     return a*np.exp(-(z-z[0])/b)

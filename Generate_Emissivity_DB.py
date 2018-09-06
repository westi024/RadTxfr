import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.io, scipy.optimize
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline, splrep
import h5py

em = scipy.io.loadmat("EmissivityDatabase.mat")
X = em["X"].flatten()
emis = em["emis"].T
ixSrt = np.argsort(np.mean(emis, axis=0))
emis = emis[:, ixSrt]

em_mean = np.mean(emis, axis=0)
em_vals = np.linspace(0, 1, 400)
ix = np.unique(np.argmin(np.abs(em_mean[np.newaxis,:] - em_vals[:,np.newaxis]),axis=1))

emis = emis[:, ix]
plt.plot(X, emis)
plt.show()

# Expand the emissivity database via mixtures
mixFrac = np.arange(0, 1.1, 0.1)
nX = X.size
nE = emis.shape[1]
nF = mixFrac.size
emisMix = np.zeros((nX,nE,nE,nF))
for ii in range(emis.shape[1]):
    for jj in range(ii+1, emis.shape[1]):
        emisMix[:, ii, jj, :] = mixFrac[np.newaxis,:] * emis[:, ii][:,np.newaxis] + (1 - mixFrac[np.newaxis,:]) * emis[:, jj][:,np.newaxis]
emisMix = np.reshape(emisMix, (emisMix.shape[0], np.product(emisMix.shape[1:])))
emis = np.unique(emisMix,axis=1)


em_mean = np.mean(emis, axis=0)
em_vals = np.linspace(0, 1, 504)
ix = np.unique(np.argmin(np.abs(em_mean[np.newaxis,:] - em_vals[:,np.newaxis]),axis=1))
emis = emis[:, ix]

ixSrt = np.argsort(np.mean(emis,axis=0))
emis = emis[:,ixSrt]

# bound emissivity 0 < emis < 1
TOL = 1e-4
emis[emis < TOL] = TOL
emis[emis > 1 - TOL] = 1 - TOL

plt.plot(X, emis)
plt.show()

# Save as HDF5 file
hf = h5py.File('LWIR_Emissivity_DB.h5', 'w')
d = hf.create_dataset('X', data=X)
d.attrs['units'] = 'cm^{-1}'
d.attrs['name'] = 'Wavenumbers'
d.attrs['info'] = 'Spectral axis for emis'
d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

d = hf.create_dataset('emis', data=emis)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Emissivity'
d.attrs['info'] = 'Hemispherically-averaged emissivity'
d.attrs['label'] = r'$\varepsilon(\tilde{\nu})$'

hf.close()

# Convolve emissivity profiles with MAKO lineshape
X_HI = X.copy()
emis_HI = emis.copy()
X, _ = ILS_MAKO(X_HI, emis[:, 0])
emis = np.zeros((X.size, emis_HI.shape[1]))
for ii in range(emis_HI.shape[1]):
    _, emis[:, ii] = ILS_MAKO(X_HI, emis_HI[:, ii])
    print("{0:02d} of {1:02d}".format(ii+1, emis_HI.shape[1]))

# Save as HDF5 file
hf = h5py.File('LWIR_Emissivity_DB_MAKO.h5', 'w')
d = hf.create_dataset('X', data=X)
d.attrs['units'] = 'cm^{-1}'
d.attrs['name'] = 'Wavenumbers'
d.attrs['info'] = 'Spectral axis for emis'
d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

d = hf.create_dataset('emis', data=emis)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Emissivity'
d.attrs['info'] = 'Hemispherically-averaged emissivity'
d.attrs['label'] = r'$\varepsilon(\tilde{\nu})$'

hf.close()

# Reduce emissivity to spline coefficient features
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, SparsePCA, FactorAnalysis
from sklearn.cross_decomposition import CCA

f = h5py.File("LWIR_Emissivity_DB_MAKO.h5")
emis = f["emis"][...]
X = f["X"][...]
f.close()

TOL = 1e-4
emis[emis < TOL] = TOL
emis[emis > 1 - TOL] = 1 - TOL
ix = np.argsort(X)
X = X[ix]
emis = emis[ix,:]

OD = -np.log(1 - emis)
pcaOD = PCA(whiten=True, n_components=48)

ica = FastICA(n_components=36, max_iter=5000)
ODIR = ica.fit_transform(OD) # Reconstruct signals
OD2 = ica.inverse_transform(ODIR)
emis2 = 1-np.exp(-OD2)  # Reconstruct signals
A_ = ica.mixing_ # Get estimated mixing matrix

nmf = NMF(n_components=48)
ODNR = nmf.fit_transform(OD)
OD2 = nmf.inverse_transform(ODNR)
emis2 = 1 - np.exp(-OD2)


N = 48
knots = np.linspace(X.min(), X.max(), N)[1:-1]
tck = splrep(X, -np.log(emis[:, 350]), t=knots)

t = tck[0]
c = np.zeros((emis.shape[-1], tck[1].size))
k = tck[2]
for ii in range(emis.shape[-1]):
    tck = splrep(X, -np.log(emis[:, ii]), t=knots)
    c[ii,:] = tck[1]

def emisFcn(X, tck):
    sp = BSpline(tck[0], tck[1], tck[2])
    return np.exp(-np.abs(sp(X)))

emisFit = np.zeros(emis.shape)
for ii in range(emis.shape[-1]):
    emisFit[:,ii] = emisFcn(X, (t, c[ii,:], k))

def emisFcn2(X, *p):
    X0_ = p[0::2]
    OD_ = np.abs(p[1::2])
    OD = scipy.interpolate.interp1d(X0_, OD_, kind='cubic', fill_value="extrapolate")
    return 1 - np.exp(-OD(X))

N = 24
X0 = np.linspace(X.min(), X.max(), N)
dX = np.mean(np.diff(X0))
OD = -np.log(0.5) + np.abs(np.random.randn(X0.size))
p0 = np.array([X0, OD]).T.flatten()
X0_L=np.linspace(X.min(), X.min() + (X.max() - X.min())/4, N)
X0_U=np.linspace(X.min() + (X.max() - X.min())/4, X.max(), N)
X0_L[-1] = X.max()
X0_U[0] = X.min()
OD_L=np.zeros(X0.shape)
OD_U=7 * np.ones(X0.shape)
pL=np.array([X0_L, OD_L]).T.flatten()
pU=np.array([X0_U, OD_U]).T.flatten()
bnd=scipy.optimize.Bounds(pL, pU)
ii=355
err=lambda p: np.sum( (emis[:, ii] - emisFcn2(X, *p))**2)
for _ in range(3):
    res = scipy.optimize.minimize(err, p0, bounds=bnd)
    p0 = res.x
plt.figure()
plt.plot(X, emis[:, ii], X, emisFcn2(X, *p0))



N = 48
knots = np.linspace(X.min(), X.max(), N)[1:-1]

rmsErr = 0.01
w = 1.0/rmsErr**2 * np.ones(X.size)
e = np.sum(w * (emis[:,255]-emisFit[:,255])**2)
tck = splrep(X, -np.log(emis[:,255]), s=10.0)
e = np.sum(w[:,np.newaxis] * (emis-emisFit)**2,axis=1)

t = tck[0]
c = np.zeros((emis.shape[-1], tck[1].size))
k = tck[2]
for ii in range(emis.shape[-1]):
    tck = splrep(X, -np.log(emis[:, ii]), t=knots)
    c[ii,:] = tck[1]

emisFit = np.zeros(emis.shape)
for ii in range(emis.shape[-1]):
    emisFit[:,ii] = emisFcn(X, (t, c[ii,:], k))

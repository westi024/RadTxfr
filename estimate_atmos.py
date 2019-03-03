import numpy as np
import scipy
from scipy.io import loadmat
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rv_continuous, gaussian_kde
import matplotlib.pyplot as plt

from neupy.algorithms import GRNN
from wpca import WPCA, EMPCA

from smooth import smooth

import h5py

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, Normalizer, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, SparsePCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import LocalOutlierFactor

import radiative_transfer as rt

def reduceResolution(X, Y, dX, N=4, window='hanning', X_out=None):
    dX_in = np.mean(np.diff(X))
    smFactor = np.int(np.round(dX/dX_in))
    smFcn1 = lambda y: smooth(y, window_len=smFactor, window=window)
    smFcn = lambda y: 0.5*(smFcn1(y) + smFcn1(y[::-1])[::-1])
    interpFcn = lambda x, y, x0: scipy.interpolate.interp1d(x, y, kind='cubic')(x0)
    X_ = smFcn(X)
    nPts = np.int(np.ceil(N * (X_[-smFactor-1] - X_[smFactor]) / dX))+1
    returnX_out = False
    if X_out is None:
        X_out = np.linspace(X_[smFactor], X_[-smFactor-1], nPts)
        returnX_out = True
    if len(Y.shape) > 1:
        Y_out = np.zeros((X_out.size, Y.shape[-1]))
        for ii in range(Y.shape[-1]):
            Y_out[:, ii] = interpFcn(X_, smFcn(Y[:, ii]), X_out)
    else:
        Y_out = interpFcn(X_, smFcn(Y), X_out)
    if returnX_out:
        return X_out, Y_out
    else:
        return Y_out

# TIGR Data from MODTRAN
# TUD = loadmat("/Users/grosskc/Documents/Students/2017S-LaneCory/TIGR LW and SW TUD MAT files/TIGR_LW_TUD_Mono_03KM_100.mat")
TUD = loadmat("/Users/grosskc/Documents/Students/2017S-LaneCory/toy problem/generate_LW_HSI/TIGR2311_LWTUD.mat")
# TIGR atmospheric variable inputs
z = TUD["TIGR"]["z"][0,0].flatten(); z[0]=0 # [km]
T_z = TUD["TIGR"]["T"][0, 0] # [K]
P_z = TUD["TIGR"]["P"][0,0].flatten() * 100 # [Pa]
H2O_z = TUD["TIGR"]["H2O"][0, 0] / 1e6 # [ppm]
O3_z=TUD["TIGR"]["O3"][0, 0] # [ppm]

# TIGR radiative transfer outputs
X = TUD['X'].flatten().astype(np.float)
tau = TUD['tau_T'].T
La = TUD['la_T'].T * 1e6
Ld = TUD['ld_T'].T * 1e6

# TIGR data from LBLRTM
TUD = np.load("20180802-225537-LWIR-TUD.npz")
X = TUD["X"]
OD = TUD["OD"][:,-1,:].T
La = TUD["La"][:,-1,:].T
Ld = TUD["Ld"].T
ix = OD.sum(axis=1) > 0
OD = OD[ix,:]
La = La[ix,:]
Ld = Ld[ix,:]


def smooth_trends(data_, N=4):
    data = data_.copy()
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pcaSm = PCA(n_components=N, whiten=True)
    ix = np.argsort(data.T)
    data2 = np.zeros(data.shape)
    for ii in range(ix.shape[0]):
        data2[:,ii] = data[ix[ii,:].flatten(),ii].T
    pcaSm.fit(data2.T)
    data3 = pcaSm.inverse_transform(pcaSm.transform(data2.T)).T
    for ii in range(ix.shape[0]):
        ix2 = np.zeros(ix.shape[1],dtype=np.int)
        ix2[ix[ii,:]] = np.arange(ix.shape[1]).astype(np.int)
        data3[:, ii] = data3[ix2.flatten(), ii].T
    return scaler.inverse_transform(data3)


# X2, tau = reduceResolution(X, tau.T, 5 * np.mean(np.diff(X)), N=3); tau = tau.T
# _, La = reduceResolution(X, La.T, 5 * np.mean(np.diff(X)), N=3); La = La.T
# _, Ld = reduceResolution(X, Ld.T, 5 * np.mean(np.diff(X)), N=3); Ld = Ld.T
# X = X2.copy(); del X2

ix = (X >= 800) & (X <= 1200)
X = X[ix]
OD = np.abs(OD[:,ix])
La = La[:,ix]
Ld = Ld[:, ix]
tau = np.exp(-OD)

TUDin = np.concatenate((OD.T, La.T, np.exp(-OD.T)*Ld.T)).T
TUD = np.concatenate((np.exp(-OD.T), La.T, np.exp(-OD.T)*Ld.T)).T
TUDin = np.concatenate((OD.T, La.T, Ld.T)).T
TUD = np.concatenate((np.exp(-OD.T), La.T, Ld.T)).T
# pcaTUD = PCA(whiten=True, n_components=100)
# pcaTUD.fit(TUD)
# TUD = pcaTUD.inverse_transform(pcaTUD.transform(TUD))

def TUD_transform(X, TUD):
    TUD_ = TUD.copy()
    if len(TUD_.shape) == 1:
        TUD_ = TUD_[None,:]
    ix0 = np.arange(0, X.size)
    ix1 = np.arange(X.size, 2*X.size)
    ix2 = np.arange(2*X.size, 3*X.size)
    # TUD_[:,ix0] = -np.log(np.abs(TUD_[:,ix0]))
    # TUD_[:,ix1] = rt.brightnessTemperature(X, np.abs(TUD_[:,ix1]).T).T
    # TUD_[:,ix2] = rt.brightnessTemperature(X, np.abs(TUD_[:,ix2]).T).T
    return TUD_

def TUD_inverse_transform(X, TUD):
    TUD_ = TUD.copy()
    if len(TUD_.shape) == 1:
        TUD_ = TUD_[None,:]
    ix0 = np.arange(0, X.size)
    ix1 = np.arange(X.size, 2*X.size)
    ix2 = np.arange(2*X.size, 3*X.size)
    TUD_[:,ix0] = np.exp(-np.abs(TUD_[:,ix0]))
    # TUD_[:,ix1] = rt.BT2L(X, TUD_[:,ix1].T).T
    # TUD_[:,ix2] = rt.BT2L(X, TUD_[:,ix2].T).T
    return TUD_

rms = lambda x: np.mean(x**2, axis=1)**0.5


def fit_pca_mdl(X_, N=5, scaler=None):
    pca = PCA(whiten=True, n_components=N)
    X = X_.copy()
    if not scaler:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X)
    pca.fit(scaler.transform(X))
    Xr = pca.transform(scaler.transform(X))
    Xe = scaler.inverse_transform(pca.inverse_transform(Xr))
    def X_mdl(p_):
        p = np.asarray(p_.copy())
        if len(p.shape) == 1:
            p = p.reshape(1, -1)
        X_out = scaler.inverse_transform(pca.inverse_transform(p))
        return X_out
    X2 = X.copy()
    SS = StandardScaler()
    SS.fit(X2)
    err = rms(SS.transform(X2) - SS.transform(Xe))
    return X_mdl, err, Xe, Xr

OD_mdl, errOD, ODe, ODr = fit_pca_mdl(OD, N=6)
La_mdl, errLa, Lae, Lar = fit_pca_mdl(La, N=6)
Ld_mdl, errLd, Lde, Ldr = fit_pca_mdl(Ld, N=6)
TUDrr = np.concatenate((ODr.T, Lar.T, Ldr.T)).T
pca = PCA(n_components=6, whiten=True)
qt = QuantileTransformer(output_distribution='normal')
qt.fit(TUDrr)
pca.fit(qt.transform(TUDrr))
TUDr2 = pca.transform(qt.transform(TUDrr))
plt.figure()
deg = 13
mdlOD = Pipeline([('poly', PolynomialFeatures(degree=deg)), ('linear', LinearRegression(fit_intercept=False, normalize=True))])
mdlOD.fit(TUDr2, ODr); plt.plot(np.sort(rms(mdlOD.predict(TUDr2)-ODr)),label="OD")
mdlLa = Pipeline([('poly', PolynomialFeatures(degree=deg)), ('linear', LinearRegression(fit_intercept=False, normalize=True))])
mdlLa.fit(TUDr2, Lar); plt.plot(np.sort(rms(mdlLa.predict(TUDr2)-Lar)),label="La")
mdlLd = Pipeline([('poly', PolynomialFeatures(degree=deg)), ('linear', LinearRegression(fit_intercept=False, normalize=True))])
mdlLd.fit(TUDr2, Ldr); plt.plot(np.sort(rms(mdlLd.predict(TUDr2)-Ldr)),label="Ld")
plt.legend()
f = lambda x: np.percentile(x, 90)

grnnOD = GRNN(std=0.1, verbose=True)
grnnOD.fit(TUDr2, ODr)

D = cdist(TUDr2, TUDr2)
D[D==0] = 10*D.max()
ix=np.unravel_index(np.argmin(D.flatten()),D.shape)
ix=np.unravel_index(np.argsort(D.flatten())[int(0.01*D.size)],D.shape)
mdlLa.predict(0.5*(TUDr2[ix[0],:] + TUDr2[ix[1],:]).reshape(1,-1)), 0.5*(mdlLa.predict(TUDr2[ix[0],:].reshape(1,-1)) + mdlLa.predict(TUDr2[ix[1],:].reshape(1,-1)))
La3 = 0.5*(La[ix[0],:]+La[ix[1],:])
La3e = La_mdl(mdlLa.predict(0.5*(TUDr2[ix[0],:] + TUDr2[ix[1],:]).reshape(1,-1)))

[f(rms(mdlOD.predict(TUDr2) - ODr)), f(rms(mdlLa.predict(TUDr2) - Lar)), f(rms(mdlLd.predict(TUDr2) - Ldr))]

[np.sum((mdlOD.predict(TUDr2)-ODr)**2), np.sum((mdlLa.predict(TUDr2)-Lar)**2), np.sum((mdlLd.predict(TUDr2)-Ldr)**2)]

tau2 = np.exp(-np.abs(OD_mdl(mdlOD.predict(TUDr2))))
La2 = La_mdl(mdlLa.predict(TUDr2))
Ld2 = Ld_mdl(mdlLd.predict(TUDr2))

TUD2 = np.concatenate((tau2.T, La2.T, Ld2.T)).T
SS = StandardScaler()
SS.fit(TUD)
err2 = rms(SS.transform(TUD2)-SS.transform(TUD))

plt.figure()
plt.plot(np.sort(rms(tau - tau2) / rms(tau)), label="tau")
plt.plot(np.sort(rms(La - La2) / rms(La)), label="La")
plt.plot(np.sort(rms(Ld - Ld2) / rms(Ld)), label="Ld")
plt.legend()

plt.figure()
plt.plot(np.sort(rms(tau - tau2)), label="tau")
plt.plot(np.sort(rms(La - La2)), label="La")
plt.plot(np.sort(rms(Ld - Ld2)), label="Ld")
plt.legend()


plt.plot(np.sort(rms((La - La2) / La)), label="La")
plt.plot(np.sort(rms((Ld - Ld2) / Ld)), label="Ld")
plt.legend()

ix = 400
plt.figure()
plt.plot(X, La[ix,:], X, La_mdl(mdlLa.predict(TUDr2[ix,:].reshape(1, -1)).flatten()).T)
plt.figure()
plt.plot(X, Ld[ix,:], X, Ld_mdl(mdlLd.predict(TUDr2[ix,:].reshape(1, -1)).flatten()).T)
plt.figure()
plt.plot(X, np.exp(-OD[ix,:]), X, np.exp(-np.abs(OD_mdl(mdlOD.predict(TUDr2[ix,:].reshape(1, -1)).flatten()).T)))

qtOD = QuantileTransformer()
qtLa = QuantileTransformer()
qtLd = QuantileTransformer()

qtOD.fit(ODr)
qtLa.fit(Lar)
qtLd.fit(Ldr)

pX = (ODr + Lar + Ldr) / 3.0

model = Pipeline([('poly', PolynomialFeatures(degree=5)), ('linear', LinearRegression(fit_intercept=False, normalize=True))])
model.fit(ODr, Lar)
model.predict(pX[0,:].reshape(1, -1))
Lar[0,:]
plt.figure()
plt.plot(np.sort(errOD),label='OD')
plt.plot(np.sort(errLa),label='La')
plt.plot(np.sort(errLd),label='Ld')
plt.legend()

def fit_atm_mdl(X, TUD_, N=5, scaler=None, norm=True):
    pca = PCA(whiten=True, n_components=N)
    TUD = TUD_.copy()
    if not scaler:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        # scaler = StandardScaler()
        scaler.fit(TUD)
    pca.fit(scaler.transform(TUD))
    TUDr = pca.transform(scaler.transform(TUD))
    TUDe = scaler.inverse_transform(pca.inverse_transform(TUDr))
    def TUD_mdl(p):
        TUD_out = scaler.inverse_transform(pca.inverse_transform(np.asarray(p).reshape(1, -1)))
        TUD_out[0:X.size] = np.exp(-np.abs(TUD_out[0:X.size]))
        return TUD_out
    TUD2 = TUD.copy()
    TUD2[:,0:X.size] = np.exp(-TUD2[:,0:X.size])
    TUDe[:,0:X.size] = np.exp(-np.abs(TUDe[:,0:X.size]))
    SS = StandardScaler()
    SS.fit(TUD2)
    err = rms(SS.transform(TUD2) - SS.transform(TUDe))
    return TUD_mdl, err, TUDe, TUDr

TUD_mdl, err, TUDe, TUDr = fit_atm_mdl(X, TUDin, N=5)
_, err, TUDe, TUDr = fit_atm_mdl(X, TUDin, N=5, robust=False, ica=True)
_, err, TUDe, TUDr = fit_atm_mdl(X, TUDin, N=4, kernel=True, robust=False)

tau2 = TUDe[:, 0:X.size]
La2 = TUDe[:, X.size:2 * X.size]
Ld2 = TUDe[:, 2*X.size:3*X.size]

plt.figure()
plt.plot(np.sort(err))

plt.figure()
ix = np.argmax(err)
plt.plot(TUD[ix,:], label='truth')
plt.plot(TUDe[ix,:], label='model')
plt.title('Largest Error')
plt.legend()

plt.figure()
ix = np.argsort(err)[int(0.90*err.size)]
plt.plot(TUD[ix,:], label='truth')
plt.plot(TUDe[ix,:], label='model')
plt.title('90th Percentile Error')
plt.legend()


plt.figure()
plt.plot(np.sort(rms(tau - tau2) / rms(tau)), label="tau")
plt.plot(np.sort(rms(La - La2) / rms(La)), label="La")
plt.plot(np.sort(rms(Ld - Ld2) / rms(Ld)), label="Ld")
plt.legend()

plt.figure()
plt.plot(TUDr[:, 0], TUDr[:, 1], '.')

kpcaTUD.fit(TUDr)
TUDrr = kpcaTUD.transform(TUDr)
TUDe = SS.inverse_transform(pcaTUD.inverse_transform(kpcaTUD.inverse_transform(TUDrr)))

# Kernel PCA model
kpcaTUD = KernelPCA(n_components=4, degree=3, kernel='poly', fit_inverse_transform=True)
SS = StandardScaler()
SS.fit(TUD)
TUDss = SS.transform(TUD)
kpcaTUD.fit(TUDss)
TUDr = kpcaTUD.transform(TUDss)
TUDe = SS.inverse_transform(kpcaTUD.inverse_transform(TUDr))

err = rms(SS.transform(TUD) - SS.transform(TUDe))
ix = np.argmax(err)

plt.figure()
plt.plot(np.sort(err))

plt.figure()
plt.plot(TUD[ix,:], label='truth')
plt.plot(TUDe[ix,:], label='model')
plt.legend()

ix = np.argsort(err)[int(0.95*err.size)]
plt.figure()
plt.plot(TUD[ix,:],label='truth')
plt.plot(TUDe[ix,:], label='model')
plt.legend()

plt.figure()
plt.plot(TUDr[:, 0], TUDr[:, 1], '.')


kernel = gaussian_kde(TUDr.T, bw_method=0.1)
P = kernel(TUDr.T)
TUD_mdl = lambda p: SS.inverse_transform(pcaTUD.inverse_transform(np.asarray(p).reshape(1,-1)))

## GARBAGE below this line

# Import data

tmp = np.load("LWIR_HSI_MAKO.npz")
X = tmp["X"].astype(np.float32)
L = tmp["L"].astype(np.float32)
T = tmp["T"].astype(np.float32)
emis = tmp["emis"].astype(np.float32)
ixE = tmp["ixE"]
ixA = tmp["ixA"]
ixT = tmp["ixT"]

tau = tmp["tau"].astype(np.float32)
La = tmp["La"].astype(np.float32)
Ld = tmp["Ld"].astype(np.float32)

ix = np.arange(73,75)
tau0 = tau[ix,:].mean(axis=0)
La0 = La[ix,:].mean(axis=0)
Ld0 = Ld[ix,:].mean(axis=0)

def whiten(X, eps=1e-5, returnMatrix=True, pca=False):
    """
    Function to compute ZCA (Mahalanobis) or PCA whitening matrix.
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: W: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X - X.mean(axis=1)[:, None], rowvar=True)  # [M x M]
    # sigma = np.corrcoef(X-X.mean(axis=1)[:,None], rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    if pca:
        # PCA Whitening matrix: U * Lambda
        W = np.dot(U, np.diag(1.0 / np.sqrt(S + eps)))  # [M x M]
    else:
        # ZCA Whitening matrix: U * Lambda * U'
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))  # [M x M]
    if returnMatrix:
        return W @ (X - X.mean(axis=1)[:, None]), W
    else:
        return W @ (X-X.mean(axis=1)[:,None])

# TUD = np.concatenate((tau.T, La.T, Ld.T)).T
# SS = StandardScaler()
# SS.fit(TUD)
# TUDss = SS.transform(TUD)
# TUDw, ZCA = whiten(TUDss)
# pcaTUDw = PCA(n_components=4, whiten=True)
# pcaTUDw.fit(TUDw)
# TUDr = pcaTUDw.transform(TUDw)
# TUDe = SS.inverse_transform(np.linalg.inv(ZCA) @ pcaTUDw.inverse_transform(TUDr))

TUD = np.concatenate((tau.T, La.T, Ld.T)).T
TUDw, ZCA = whiten(TUD)
pcaTUDw = PCA(n_components=8)
pcaTUDw.fit(TUDw)
TUDr = pcaTUDw.transform(TUDw)
TUDe = np.linalg.inv(ZCA) @ pcaTUDw.inverse_transform(TUDr)


tauM = tau.mean()
tauS = tau.std()
LaM = La.mean()
LaS = La.std()
LdM = Ld.mean()
LdS = Ld.std()


pcaTUD = PCA(whiten=True, n_components=4)
kpcaTUD = KernelPCA(n_components=4, kernel='poly', fit_inverse_transform=True)
# kpcaTUD = KernelPCA(n_components=4, kernel='poly', fit_inverse_transform=True)
TUD = np.concatenate((tau.T, La.T, Ld.T)).T
# TUD = np.concatenate((La.T, Ld.T)).T
# TUD = tau
SS = StandardScaler()
SS.fit(TUD)
pcaTUD.fit(SS.transform(TUD))
TUDr = pcaTUD.transform(SS.transform(TUD))
# TUDk = kpcaTUD.fit(TUDr)
N_comp = pcaTUD.n_components
# w = pcaTUD.explained_variance_ratio_
TUDe = SS.inverse_transform(pcaTUD.inverse_transform(TUDr))
TUD_mdl = lambda p: SS.inverse_transform(pcaTUD.inverse_transform(np.asarray(p).reshape(1,-1)))

ix0 = np.arange(0,X.size)
ix1 = np.arange(X.size,2*X.size)
ix2 = np.arange(2*X.size,3*X.size)

r = TUD-TUDe

err = lambda p: np.sum( (SS.transform(TUD[544,:].reshape(1,-1)) - SS.transform(TUD_mdl(p).reshape(1,-1)))**2)

kernel = gaussian_kde(TUDr.T, bw_method=0.1)
P = kernel(TUDr.T)

TUD
if N_comp == 2:
    x, y = np.mgrid[-3:3:100j, -3:3:100j]
    pos = np.vstack([x.ravel(), y.ravel()])
    Z = np.reshape(kernel(pos), x.shape)
    plt.imshow((np.rot90(Z[:,:].T)), extent=(-3, 3, -3, 3))
elif N_comp == 3:
    x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]
    pos = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    Z = np.reshape(kernel(pos), x.shape)
    plt.figure()
    plt.plot(TUDr[:,1],TUDr[:,0],'.')
    plt.imshow(np.log((np.rot90(Z[:,:,:].sum(axis=2).T))), extent=(-3, 3, -3, 3))
    plt.figure()
    plt.plot(TUDr[:,1],TUDr[:,0],'.')
    plt.imshow((np.rot90(Z[:,:,:].sum(axis=2).T)), extent=(-3, 3, -3, 3))

D = cdist(TUDr, TUDr)
# D = cdist(TUDr[:, 0:N_comp], TUDr[:, 0:N_comp], metric='minkowski', w=w, p=2)
D[D==0] = 10*D.max()
D_thresh = np.percentile(D.min(axis=0),95)
def my_dist(X, y):
    y = np.array(y)
    y = np.array([y, y])
    return cdist(X, y)[:,0].min()
    # return cdist(X, y, 'minkowski', w=w, p=2)[:,0]


def TUD_mdl(p):
    p = np.array(p)
    if len(p.shape) == 1:
        n = p.size
        m = 1
    else:
        n = p.shape[1]
        m = p.shape[0]
    k = np.min([pcaTUD.n_components_, pcaTUD.n_samples_])
    p0 = np.zeros((m, k))
    p0[0:m, 0:n] = p
    return pcaTUD.inverse_transform(p0)

def tau_mdl(p):
    return (TUD_mdl(p).flatten())[0:X.size]

def La_mdl(p):
    return (TUD_mdl(p).flatten())[X.size:2*X.size]

def Ld_mdl(p):
    return (TUD_mdl(p).flatten())[2*X.size:3*X.size]

err = lambda p: np.sum( (tau0 - tau_mdl(p))**2 / rms(tau0)**2 + (La0 - La_mdl(p))**2 / rms(La0)**2 + (Ld0 - Ld_mdl(p))**2 / rms(Ld0)**2)
out = scipy.optimize.minimize(err, [0, 0, 0]); pA = out['x']

plt.figure(); plt.plot(X, tau[ix,:].T,'-k', X,tau0,'-r',X,tau_mdl(pA).T,'-b')
plt.figure(); plt.plot(X, La[ix,:].T,'-k', X,La0,'-r',X,La_mdl(pA).T,'-b')
plt.figure(); plt.plot(X, Ld[ix,:].T, '-k', X, Ld0, '-r', X, Ld_mdl(pA).T, '-b')

print(rms(tau0-tau_mdl(pA)))
print(rms(La0-La_mdl(pA)))
print(rms(Ld0-Ld_mdl(pA)))


rms = lambda x,y: np.mean( (x-y)**2 )**0.5
plt.plot(X,La[1,:],X,pls_La.predict(tau[1,:].reshape(1,-1)).T)

# Build low-dimensional model of atmosphere
atm = np.concatenate((tau.T, La.T, Ld.T)).T
pca = PCA(whiten=True)
pca.fit(atm)
TUD_mdl = lambda p: np.concatenate((pcaT.inverse_transform(np.concatenate((np.array(p), np.zeros(np.sum(z<11)-4)))), T_mean))

# Explore stdev of surface-leaving radiance
ixTmp = ixA == 74
Ls = emis[ixE[ixTmp],:] * rt.planckian(X,T[ixA[ixTmp],ixT[ixTmp]].flatten()).T + (1-emis[ixE[ixTmp],:])*Ld[74,:][np.newaxis,:]
eM = np.mean(emis[ixE[ixTmp],:], axis=1)
ix = (ixA == 25) & (emis[ixE,:].mean(axis=1) > 0.75)

ix = ixA == 74
L = L[ix,:]
q = np.linspace(95, 99.9, 20) / 100
Lq = np.percentile(L, q * 100, axis=0)
Tq = lambda Tm, Ts, q: Tm + Ts * np.sqrt(2) * scipy.special.erfinv(2*q-1)
Lsq = lambda X, Tm, Ts, e, idx, q: e * rt.planckian(X, Tq(Tm, Ts, q)).T + (1 - e) * Ld[idx,:]
LqMdl = lambda p, idx: tau[idx,:] * Lsq(X, p[0], p[1], p[2], idx, q) + La[idx,:]

Tb = np.percentile(rt.brightnessTemperature(X, LL.T).T, 99)

p0 = [Tb, 5, 0.98]
bnds = ((Tb-10, Tb+10), (1, 25), (0.5, 1.0))
AtmErr = np.zeros(tau.shape[0])
AtmP = np.zeros((tau.shape[0],3))
for ix in range(tau.shape[0]):
    err = lambda p: np.sum(np.abs(Lq - LqMdl(p, ix)))
    Tb = T[ix,:].mean()
    p0 = [Tb, 5, 0.98]
    bnds = ((Tb-5, Tb+5), (2, 15), (0.9, 1.0))
    out = scipy.optimize.minimize(err, p0, bounds=bnds)
    AtmErr[ix] = out['fun']
    AtmP[ix,:] = out['x']

# Directly fit statistical model to get tau, La
ixAA=74
ix = ixA == ixAA
LL = L[ix,:]
q = np.linspace(95, 99.95, 50) / 100
Lq = rt.brightnessTemperature(X,np.percentile(LL, q * 100, axis=0).T).T
Tq = lambda Tm, Ts, q: Tm + Ts * np.sqrt(2) * scipy.special.erfinv(2*q-1)
Lsq = lambda X, Tm, Ts, q: rt.planckian(X, Tq(Tm, Ts, q)).T

def Lsq(X, q, Tm=296.15, Ts=5, e=0.95, Ld=[0]):
    Ld = np.asarray(Ld)
    q = np.asarray(q)
    Tq = Tm + Ts * np.sqrt(2) * scipy.special.erfinv(2*q-1)
    return e * rt.planckian(X,Tq).T + (1-e)*Ld[np.newaxis,:]

def LqMdl(X, q, p):
    TUD = TUD_mdl(p[0:3]).flatten()
    tau = TUD[0:X.size]
    La = TUD[X.size:2*X.size]
    Ld = TUD[2*X.size:3*X.size]
    return tau * Lsq(X, q, Tm=p[3], Ts=p[4], e=p[5], Ld=Ld) + La

q = np.linspace(85, 99, 100)/100
Lq = np.percentile(L, 100 * q, axis=0)

Tb = np.percentile(rt.brightnessTemperature(X,Lq.T),99)
p0 = np.array([0,0,0,Tb,5,0.95])
bnds=[]
bnds.append((-5, 5))
bnds.append((-5, 5))
bnds.append((-5, 5))
bnds.append((Tb - 20, Tb + 20))
bnds.append((0.5, 25.0))
bnds.append((0.0, 1.0))


err = lambda p: np.sum(np.abs(Lq - LqMdl(X, q, p))** 2)
out = scipy.optimize.minimize(err, p0, bounds=bnds, options={'maxiter':5000})
while not out['success']:
    out = scipy.optimize.minimize(err, out["x"], bounds=bnds, options={'maxiter': 5000})
    print(out['x'])
out = scipy.optimize.minimize(err, p0, method='Nelder-Mead', options={'maxiter':500})
while not out['success']:
    out = scipy.optimize.minimize(err, p0, method='Nelder-Mead', options={'maxiter': 500})
    print(out['x'])
while not out['success']:
    out = scipy.optimize.minimize(err, out["x"], bounds=bnds, options={'maxiter':5000})


def cdf(x, m=0.75,s=0.25):
    cdf0 = lambda x: 0.5*(1+scipy.special.erf((x - m) / s / np.sqrt(2.0)))
    pdf0 = lambda x: np.sqrt(2 * np.pi * s ** 2)**(-1) * np.exp(-0.5*(x-m)**2 / s ** 2) / (cdf0(1)-cdf0(0))
    pdf = np.zeros(x.size)
    ix = (x >= 0) & (x <= 1)
    pdf[ix] = pdf0(x[ix])
    return np.cumsum(pdf)*(x[1]-x[0])


N = 10000
em = lambda eL, eU: eL + (eU - eL) * np.random.rand(N)
Tr = lambda Tm, Ts: Tm + Ts*np.random.randn(N)
Lsq = lambda X, eL, eU, Tm, Ts: em(eL, eU) * rt.planckian(X, Tr(Tm, Ts)).flatten()


err = lambda p: np.sum(np.abs(Lq - LqMdl(p))**2)
out = scipy.optimize.minimize(err, p0, bounds=bnds, options={'maxiter':500})
while not out['success']:
    out = scipy.optimize.minimize(err, out["x"], bounds=bnds, options={'maxiter':5000})

plt.figure(); plt.plot(X,Lq.T,'-r',X,LqMdl(out['x']).T,'-b')
plt.figure(); plt.plot(X, tau[ixAA,:],label="Truth"); plt.plot(X,out['x'][0:X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X, La[ixAA,:], label="Truth"); plt.plot(X,out['x'][X.size:2*X.size], label="Est"); plt.legend()

# Directly fit statistical model to get tau, La, Ld, emis
def TUD_mdl(p):
    p = np.array(p)
    if len(p.shape) == 1:
        n = p.size
        m = 1
    else:
        n = p.shape[1]
        m = p.shape[0]
    k = np.min([pca_tauLa.n_components_, pca_tauLa.n_samples_])
    p0 = np.zeros((m, k))
    p0[0:m, 0:n] = p
    tauLa = pca_tauLa.inverse_transform(p0).flatten()
    tau = tauLa[0:X.size]
    La = tauLa[X.size:2 * X.size]
    Ld = krr_Ld.predict(np.concatenate((tau,La)).reshape(1,-1))
    return [tau,La,Ld]

q = np.linspace(5, 95, 100) / 100
f = lambda x: np.percentile(x, 100 * q, axis=0)
ixAA = 74
ix = ixA == ixAA
Ls = emis[ixE[ix],:] * rt.planckian(X, T[ixA[ix], ixT[ix]].flatten()).T + (1 - emis[ixE[ix],:]) * Ld[ixAA,:][np.newaxis,:]
LL = L[ix,:]
Lq = f(LL)
LsQ = f(Ls)
LsQ2 = f(emis[ixE[ix],:] * rt.planckian(X,T[ixA[ix],ixT[ix]].flatten()).T) + f((1-emis[ixE[ix],:])*Ld[ixAA,:][np.newaxis,:])
Tq = lambda Tm, Ts, q: Tm + Ts * np.sqrt(2) * scipy.special.erfinv(2 * q - 1)
def eq(eL, eU, q):
    emis = np.abs(eL + (eU - eL) * q[:, np.newaxis])
    emis[emis < 0] = 0
    emis[emis > 1] = 1
    return emis

Lsq = lambda X, Tm, Ts, eL, eU, Ld, q: eq(eL, eU, q)*rt.planckian(X, Tq(Tm, Ts, q)).T + (1-eq(eL, eU, q)) * Ld
LqMdl = lambda p: p[0:X.size] * Lsq(X, p[-2], p[-1], p[3*X.size:4*X.size], p[4*X.size:5*X.size], p[2*X.size:3*X.size], q) + p[X.size:2*X.size]
p0 = np.concatenate((tau.mean(axis=0), La.mean(axis=0), Ld.mean(axis=0), 0.25*np.ones(X.size), 0.95*np.ones(X.size), [Tb], [5]))
Tmax = 320
bnds=[]
for _ in range(X.size):
    bnds.append((0.0, 1.0))
for ii in range(X.size):
    bnds.append((0.0, rt.planckian(X[ii],Tmax)))
for _ in range(X.size):
    bnds.append((0.0, rt.planckian(X[ii],Tmax)))
for _ in range(X.size):
    bnds.append((0.0, 1.0))
for _ in range(X.size):
    bnds.append((0.0, 1.0))
bnds.append((Tb - 15, Tb + 15))
bnds.append((0.5, 25))

err = lambda p: np.sum(np.abs(Lq - LqMdl(p))**2)
out = scipy.optimize.minimize(err, p0, bounds=bnds, options={'maxiter':500})
while not out['success']:
    out = scipy.optimize.minimize(err, out["x"], bounds=bnds, options={'maxiter':5000})

plt.figure(); plt.plot(X,Lq.T,'-r',X,LqMdl(out['x']).T,'-b')
plt.figure(); plt.plot(X, tau[ixAA,:],label="Truth"); plt.plot(X,out['x'][0:X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X, La[ixAA,:], label="Truth"); plt.plot(X, out['x'][X.size:2 * X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X, Ld[ixAA,:], label="Truth"); plt.plot(X, out['x'][2 * X.size:3 * X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X,out['x'][3*X.size:4*X.size],X,out['x'][4*X.size:5*X.size])



# Newest approach
pcaTUD = PCA(whiten=True)
pcaTUD.fit(np.concatenate((tau.T, La.T, Ld.T)).T)

def TUD_mdl(p):
    p = np.array(p)
    if len(p.shape) == 1:
        n = p.size
        m = 1
    else:
        n = p.shape[1]
        m = p.shape[0]
    k = np.min([pcaTUD.n_components_, pcaTUD.n_samples_])
    p0 = np.zeros((m, k))
    p0[0:m, 0:n] = p
    return pcaTUD.inverse_transform(p0)

def randomEmisSample(N=1000, m=[0.75], s=[0.25], seed=None):
    m=np.asarray(m)
    s=np.asarray(s)
    cdf0=lambda x: 0.5*(1+scipy.special.erf((x - m[np.newaxis,:]) / s[np.newaxis,:] / np.sqrt(2.0)))
    pdf=lambda x: np.sqrt(2 * np.pi * s[np.newaxis,:] ** 2)**(-1) * np.exp(-0.5*(x-m[np.newaxis,:])**2 / s[np.newaxis,:] ** 2) / (cdf0(1) - cdf0(0))
    x=np.linspace(0, 1, 10000)[:,np.newaxis]
    cdf=np.cumsum(pdf(x), axis=0) * (x[1] - x[0])
    randSamp=np.zeros((N, m.size))
    if seed is not None:
        np.random.seed(seed)
    randInt = np.random.rand(N)
    for ii in range(m.size):
        randSamp[:,ii] = np.interp(randInt, cdf[:,ii], x[:,0])
    return randSamp

def Ls(X, Tm=296.15,Ts=5,em=[0.75],es=[0.25],Ld=[0],N=5000):
    em=np.asarray(em)
    es=np.asarray(es)
    Ld=np.asarray(Ld)
    T=Tm + Ts * np.random.randn(N)
    T[T < 200] = 200
    T[T > 350] = 350
    emis=randomEmisSample(N, m=em, s=es)
    return emis * rt.planckian(X,T).T + (1-emis)*Ld[np.newaxis,:]

# ixX=X < 900
# X=X[ixX]
# tau=tau[:, ixX]
# La=La[:, ixX]
# Ld=Ld[:, ixX]
# L=L[:,ixX]
ix=ixA == 74
L=L[ix,:]
q = np.linspace(5, 95, 91) / 100
Lq = np.percentile(L, q * 100, axis=0)

def LqMdl(p,N=10):
    TUD = TUD_mdl(p[-5:-2]).flatten()
    tau = TUD[0:X.size]
    La = TUD[X.size:2 * X.size]
    Ld = TUD[2 * X.size:3 * X.size]
    X0 = np.linspace(X[0],X[-1],N)
    emF = scipy.interpolate.interp1d(X0, p[0:N], kind='cubic')
    esF = scipy.interpolate.interp1d(X0, p[N:2*N], kind='cubic')
    L = tau * Ls(X, Tm=p[-2], Ts=p[-1], em=emF(X), es=esF(X), Ld=Ld) + La
    return np.percentile(L, 100 * q, axis=0)


# LqMdl=lambda p: np.percentile(p[0:X.size] * Ls(X, Tm=p[-2], Ts=p[-1], em=p[3 * X.size:4 * X.size], es=p[4 * X.size:5 * X.size], Ld=p[2 * X.size:3 * X.size]) + p[X.size:2 * X.size], 100 * q, axis=0)

Tb = np.percentile(rt.brightnessTemperature(X, L.T).T, 99)
p0 = np.concatenate((0.75 * np.ones(X.size), 0.25 * np.ones(X.size), [0, 0, 0], [Tb], [5]))
N = 10
p0 = np.concatenate((0.5*np.ones(N), 0.5*np.ones(N), [-0.33,1.1,0.66], [Tb], [5]))
bnds = []
for _ in range(N):
    bnds.append((-1.0, 2.0))
for _ in range(N):
    bnds.append((0.1, 2.0))
bnds.append((-5, 5))
bnds.append((-5, 5))
bnds.append((-5, 5))
bnds.append((Tb - 15, Tb + 15))
bnds.append((0.5, 25))

err=lambda p: np.sum(np.abs(Lq - LqMdl(p,N))** 2)
errMin = err(p0)
def err(p):
    val = np.sum(np.abs(Lq - LqMdl(p))** 2)
    global errMin
    if val < errMin:
        print(f"""Error: {val:0.5e}""")
        v0 = [p[0], p[0:N].mean()]
        v1 = [p[N], p[N:2*N].mean()]
        v2 = [p[-5], p[-4], p[-3]]
        v3 = [p[-2], p[-1]]
        print(f"""{v0[0]:0.3e}, {v0[1]:0.7e}, {v1[0]:0.3e}, {v1[1]:0.7e}""")
        print(f"""{v2[0]:0.3e}, {v2[1]:0.3e}, {v2[2]:0.3e}, {v3[0]:0.3e}, {v3[1]:0.3e}""")
        errMin = val
    return val

out = scipy.optimize.minimize(err, p0, method='Nelder-Mead', options={'maxiter':500})
out = scipy.optimize.minimize(err, p0, bounds=bnds, options={'maxiter':500})
while not out['success']:
    out = scipy.optimize.minimize(err, out["x"], bounds=bnds, options={'maxiter':5000})

plt.figure(); plt.plot(X,Lq.T,'-r',X,LqMdl(out['x']).T,'-b')
plt.figure(); plt.plot(X, tau[ixAA,:],label="Truth"); plt.plot(X,out['x'][0:X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X, La[ixAA,:], label="Truth"); plt.plot(X, out['x'][X.size:2 * X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X, Ld[ixAA,:], label="Truth"); plt.plot(X, out['x'][2 * X.size:3 * X.size], label="Est"); plt.legend()
plt.figure(); plt.plot(X,out['x'][3*X.size:4*X.size],X,out['x'][4*X.size:5*X.size])


# Ratio approach

sigL = L[ix,:].std(axis=0)

idxN, idxD = [], []
for ii in range(len(X)-15):
    for jj in range(1,15):
        idxN.append(ii)
        idxD.append(ii + jj)
idxN = np.asarray(idxN)
idxD = np.asarray(idxD)

def pctDiff(x, LB=95, UB=99):
    return np.diff(np.percentile(x, [LB, UB], axis=0), axis=0).flatten()

def estTau(X, L, LB=90, UB=98):
    Tb = rt.brightnessTemperature(X, L.T).T
    tauEst = np.diff(np.percentile(Tb, [LB, UB], axis=0), axis=0).flatten()
    # tauEst = (tauEst - tauEst.min()) / (tauEst.max()-tauEst.min())
    return tauEst

f = lambda x: scipy.stats.zscore(x)
g = lambda x: scipy.stats.zscore(x,axis=1)



ix = ixA == 74
tauE = estTau(X,L[ix,:])
tmp = cdist(np.broadcast_to(f(tauE), (2, tauE.size)), g(tau),'canberra')[0,:]


ix = ixA == 74
tauE = estTau(X,L[ix,:])
tauR_est = tauE[idxN] / tauE[idxD] #/ ( B[idxN] / B[idxD] )
tauR = tau[:, idxN] / tau[:, idxD]
tmp = cdist(np.broadcast_to(tauR_est, (2, tauR_est.size)), tauR)[0,:]

ix = ixA == 74
tauE = f(estTau(X, L[ix,:]))
# tauE = f(L[ix,:].std(axis=0))
tauR_est = tauE[idxN] - tauE[idxD] #/ ( B[idxN] / B[idxD] )
tauR = g(tau)[:, idxN] - g(tau)[:, idxD]
tmp = cdist(np.broadcast_to(tauR_est, (2, tauR_est.size)), tauR,'canberra')[0,:]

def mad(x, axis=None):
    return np.median(np.abs(x-np.median(x,axis=axis)),axis=axis)

ix = ixA == 74
# ix = (ixA == 74) & (np.mean(emis[ixE],axis=1) > 0.9)
Tb = np.percentile(rt.brightnessTemperature(X, L[ix,:].T), 99)
Tb_max = np.percentile(rt.brightnessTemperature(X,L[ix,:].T), 99, axis=0)
B = rt.planckian(X, Tb_max).T
# B = rt.planckian(X, Tb)
emisEst = L[ix,:] / B[np.newaxis,:]
# emisEst = L[ix,:] / B
w = np.std(emisEst, axis=0)
w = w / np.sum(w)
thresh = lambda x, p: x > np.percentile(x, p)
ix = (np.where(ix)[0])[thresh(np.average(emisEst, axis=1, weights=w), 90)]
# tauE = f(L[ix,:].std(axis=0))
tauE = f(mad(rt.brightnessTemperature(X,L[ix,:].T).T,axis=0))
# sigL = (L[ix,:] / B[np.newaxis,:]).std(axis=0)
# tauR_est = sigL[idxN] / sigL[idxD]  #/ ( B[idxN] / B[idxD] )
tauR_est = tauE[idxN] / tauE[idxD] #/ ( B[idxN] / B[idxD] )
tauR = tau[:, idxN] / tau[:, idxD]
tmp = cdist(np.broadcast_to(tauR_est, (2, tauR_est.size)), tauR)[0,:]

tmp = cdist(np.broadcast_to(L[ix,:].std(axis=0), (2, sigL.size)), tau, 'cosine')[0,:]
tmp = cdist(np.broadcast_to(f(L[ix,:].std(axis=0)), (2, sigL.size)), g(tau), 'cosine')[0,:]

tauEst = np.average(tau, axis=0, weights=tmp ** (-2))
LaEst = np.average(La, axis=0, weights=tmp ** (-2))
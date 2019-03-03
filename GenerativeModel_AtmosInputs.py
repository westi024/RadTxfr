# Imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import atmos

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Functions
def standardize(x):
    m = x.mean()
    s = x.std()
    return (x - m) / s, m, s

def istandardize(x,m,s):
    return x * s + m

def fit_pca(X, n_components=20):
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X)
    return pca

def fit_pdf(X, n_components=10):
    pdf = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=10000)
    pdf.fit(X)
    return pdf

def pca_gmm_gen_mdl(X, n_pca=15, n_gmm=10, scree=True):
    pca = fit_pca(X, n_components=n_pca)
    Xr = pca.transform(X)
    Xm = pca.inverse_transform(Xr)
    pdf = fit_pdf(Xr, n_components=n_gmm)
    if scree:
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Scree plot')
        plt.xlabel('PCA Components')
        plt.ylabel('Explained Variance')
    def gen_samples(n):
        Xr_n, _ = pdf.sample(n)
        Xr_n_ll = pdf.score_samples(Xr_n)
        return Xr_n, pca.inverse_transform(Xr_n), Xr_n_ll
    return gen_samples, Xr, Xm

def mf2rh(P, T, mf):
    mix2mass = (18 / (0.8 * 28 + 0.2 * 32))
    W = np.copy(mf)
    W[W < 0] = 0
    W[P < 101325*np.exp(-3)] = 0
    RH = atmos.calculate('RH', T=T, p=P, rt=W*mix2mass)
    RH[(RH < 0) | (W == 0)] = 0
    return RH

# Atmospheric state parameters
f = h5py.File("data/LWIR_TUD.h5", "r")
z, T, P, H2O, O3 = tuple(map(lambda x: f[x][...], ['z', 'T', 'P', 'H2O', 'O3']))
P = np.ones(T.shape) * P[None,:] # expand P to same size as T for convenience

# Fix weird profiles
H2O[H2O < 0] = 0
O3[O3 < 0] = 0

# Filter out super-saturated air
RH_max = 98
RH = mf2rh(P, T, H2O)
ixBad = np.any(RH > RH_max, axis=1)
print(f"Removed {np.sum(ixBad):d} profiles with one or more layers exceeding {RH_max:d}% RH")
ix = ~ixBad
T = T[ix,:]
H2O = H2O[ix,:]
O3 = O3[ix,:]

# Cumulative concentrations
cH2O = np.cumsum(H2O, axis=1)
cO3 = np.cumsum(O3, axis=1)

def atmos_filter(T_n, H2O_n, cH2O_n, O3_n, cO3_n):

    ixBad_C = np.any(cH2O_n < 0, axis=1) | np.any(cO3_n < 0, axis=1)
    ixBad_W = np.any(cH2O_n - cH2O.min(axis=0)[None,:] < 0, axis=1) | np.any(cH2O_n - cH2O.max(axis=0)[None,:] > 0, axis=1)
    ixBad_W = ixBad_W | np.any(np.diff(cH2O_n, axis=1) < 0, axis=1)
    ixBad_W = ixBad_W | np.any((np.abs(np.diff(H2O_n, axis=1)) - np.abs(np.diff(H2O, axis=1)).max(axis=0)[None,:] > 0)[:,z[:-1]<20], axis=1)
    ixBad_O = np.any(cO3_n - cO3.min(axis=0)[None,:] < 0, axis=1) | np.any(cO3_n - cO3.max(axis=0)[None,:] > 0, axis=1)
    ixBad_O = ixBad_O | np.any(np.diff(cO3_n, axis=1) < 0, axis=1)
    ixBad_O = ixBad_O | np.any((np.abs(np.diff(O3_n, axis=1)) - np.abs(np.diff(O3, axis=1)).max(axis=0)[None,:] > 0)[:,z[:-1]>10], axis=1)
    ixBad_T = np.any(T_n - T.min(axis=0)[None,:] < 0, axis=1) | np.any(T_n - T.max(axis=0)[None,:] > 0, axis=1)
    ixBad_T = ixBad_T | np.any(np.abs(np.diff(T_n, axis=1)) - np.abs(np.diff(T, axis=1)).max(axis=0)[None,:] > 0, axis=1)
    ixBad = ixBad_T | ixBad_C | ixBad_W | ixBad_O

    return ~ixBad

def atmos_generator(n_pca=15, n_gmm=10):

    # Standardize and concatenate into feature vector
    Tz, Tm, Ts = standardize(T)
    cH2Oz, cH2Om, cH2Os = standardize(cH2O)
    cO3z, cO3m, cO3s = standardize(cO3)

    # Input data
    X = np.concatenate((Tz, cH2Oz, cO3z), axis=1)

    # Generative model
    generator, Xr, Xm = pca_gmm_gen_mdl(X, n_pca=n_pca, n_gmm=n_gmm)

    def atm_gen(n):
        (Xr_n, X_n, ll) = generator(int(1.5 * n)) # generate more than required so we can throw some away
        N = int(X_n.shape[-1]/3)
        T_n = istandardize(X_n[:,0:N], Tm, Ts)
        cH2O_n = istandardize(X_n[:,N:N*2], cH2Om, cH2Os)
        cO3_n = istandardize(X_n[:,N*2:N*3], cO3m, cO3s)
        H2O_n = np.concatenate((cH2O_n[:, 0][:, None], np.diff(cH2O_n, axis=1)), axis=1)
        O3_n = np.concatenate((cO3_n[:, 0][:, None], np.diff(cO3_n, axis=1)), axis=1)

        # Filter out physically impossible or "unrealistic" atmospheric states
        ix = atmos_filter(T_n, H2O_n, cH2O_n, O3_n, cO3_n)
        T_n = T_n[ix,:]
        H2O_n = H2O_n[ix,:]
        O3_n = O3_n[ix,:]
        Xr_n = Xr_n[ix,:]
        ll = ll[ix]

        # Sort by likelihood
        ix = np.argsort(ll)[::-1]
        T_n = T_n[ix,:]
        H2O_n = H2O_n[ix,:]
        O3_n = O3_n[ix,:]
        Xr_n = Xr_n[ix,:]
        ll = ll[ix]

        # Return only the number of samples requested
        N = np.min([n, T_n.shape[0]])
        return T_n[:N,:], H2O_n[:N,:], O3_n[:N,:], ll, Xr_n
    return atm_gen, Xr, Xm

# Test it out
atm_gen, Xr, Xm = atmos_generator()
(Tn, H2On, O3n, ll, Xrn) = atm_gen(2000)
cH2On = np.cumsum(H2On, axis=1)
cO3n = np.cumsum(O3n,axis=1)

ii = 0
jj = 1
plt.figure()
plt.plot(Xr[:, ii], Xr[:, jj], '.', label='Original')
plt.plot(Xrn[:, ii], Xrn[:, jj], '.', label='New')
plt.legend()

plt.figure()
plt.plot(T.T, z, '-b', Tn.T, z, '-r')
plt.figure()
ix=z<20
plt.plot(H2O[:,ix].T, z[ix], '-b', H2On[:,ix].T, z[ix], '-r')
plt.figure()
plt.plot(O3.T, z, '-b', O3n.T, z, '-r')


# # ---------------
# from wpca import WPCA

# # Atmospheric state parameters
# wH2O = H2O.std(axis=0)
# wH2O /= wH2O.sum()
# wO3 = O3.std(axis=0)
# wO3 /= wO3.sum()
# wC = wH2O + wO3
# wC /= wC.sum()
# wT = np.std(T * wC, axis=0)
# wT /= wT.sum()

# # Standardize and concatenate into feature vector
# Tz, Tm, Ts = standardize(T)
# H2Oz, H2Om, H2Os = standardize(H2O)
# O3z, O3m, O3s = standardize(O3)

# # Input data
# X = np.concatenate((Tz, H2Oz, O3z), axis=1)

# # WCA-based low-dimensionality model
# def wpcaMdl(X_, w, n=2, xi=0, reg=0):
#     if len(w.shape) == 1:
#         w = np.tile(w, (len(X_),1))
#     wpca = WPCA(n_components=n, xi=xi, regularization=reg)
#     X_r = wpca.fit_transform(X_, weights=w)
#     X_m = wpca.inverse_transform(X_r)
#     X_e = np.std(X_ - X_m, axis=1)
#     return X_r, X_m, X_e



def trans_T(T):
    T_ = np.copy(T)
    Tg = T_[:, 0]
    T_ = T_ - Tg[:,None]
    Tm = T_.mean()
    Ts = T_.std()
    Tgm = Tg.mean()
    Tgs = Tg.std()
    Tg = (Tg - Tgm) / Tgs
    Tr = T_[:,1:]
    Tout = np.vstack((Tg,Tr))
    return Tout, Tg, Tgm, Tgs, Tm, Ts

def itrans_T(T, Tm, Ts):
    return T * Ts + Tm

def trans_C(x):
    c = np.cumsum(x)
    cm = c.mean()
    cs = c.std()
    c = (c - cm) / cs
    c_pk = c.max(c,axis=1)
    c = c_pk - c



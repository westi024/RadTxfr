# ---------------------------------------------------------------------
# Import required packages
# ---------------------------------------------------------------------
import h5py
import numpy as np
import matplotlib.pyplot as plt
import atmos

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------
# Functions for building PCA and GMM models
# ---------------------------------------------------------------------

def pca_mdl(X, n_components=20, w=None):
    pca = PCA(n_components=n_components, whiten=True)
    if w is None:
        w = np.ones(X.shape[-1])
    w[w==0] = w[w>0].min() / 100
    Xr = pca.fit_transform(X * w[None,:])
    Xm = pca.inverse_transform(Xr) / w[None,:]
    return pca, Xr, Xm

def pca_gmm_gen_mdl(X, n_pca=15, n_gmm=10, scree=True, w=None):
    pca, Xr, Xm = pca_mdl(X, n_components=n_pca, w=w)
    pdf = GaussianMixture(n_components=n_gmm, covariance_type='full', max_iter=10000)
    pdf.fit(Xr)
    if scree:
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Scree plot')
        plt.xlabel('PCA Components')
        plt.ylabel('Explained Variance')
    if w is not None:
        def gen_samples(n):
            Xr_n, _ = pdf.sample(n)
            Xr_n_ll = pdf.score_samples(Xr_n)
            return Xr_n, pca.inverse_transform(Xr_n) / w[None,:], Xr_n_ll
    else:
        def gen_samples(n):
            Xr_n, _ = pdf.sample(n)
            Xr_n_ll = pdf.score_samples(Xr_n)
            return Xr_n, pca.inverse_transform(Xr_n), Xr_n_ll
    return gen_samples, Xr, Xm

# ---------------------------------------------------------------------
# Atmospheric variable conversion functions
# ---------------------------------------------------------------------

def mf2rh(P, T, mf):
    mix2mass = (18 / (0.8 * 28 + 0.2 * 32))
    W = np.copy(mf)
    W[W < 0] = 0
    W[:, P < 101325 * np.exp(-3)] = 0
    RH = atmos.calculate('RH', T=T, p=P, rt=W*mix2mass)
    RH[(RH < 0) | (W == 0)] = 0
    return RH

def mf2mol_cum(x, P, T):
    R = 8.314 # [J/K/mol]
    rho = P[None,:] / T
    rho /= R
    c = np.cumsum(rho * x, axis=1)
    return c

def mol_cum2mf(c, P, T):
    c[c<0] = 0
    c_diff = np.diff(c, axis=1)
    c_diff[c_diff < 0] = 0
    x = np.concatenate((c[:, 0][:, None], c_diff), axis=1)
    R = 8.314 # [J/K/mol]
    rho = P[None,:] / T
    rho /= R
    x /= rho
    return x

# ---------------------------------------------------------------------
# Functions to map atmospheric variables to feature space
# ---------------------------------------------------------------------

def trans_T(T, P):
    T_ = np.copy(T)
    Tg = T_[:,0]
    T_ = T_ - Tg[:,None]
    Tr = T_[:,1:]
    Trm = Tr.mean()
    Trs = Tr.std()
    Tgm = Tg.mean()
    Tgs = Tg.std()
    Tg = (Tg - Tgm) / Tgs
    Tr = (Tr - Trm) / Trs
    w = (P[1:]*Tr).std(axis=0)
    w /= w.sum()
    w = np.append(w, 3*w.max())
    T_ = np.hstack((Tr, Tg[:, None]))
    trans_vars_T = (Tgm, Tgs, Trm, Trs)
    return T_, trans_vars_T, w

def itrans_T(T_, trans_vars_T, T=None, q=0.05):
    Tgm, Tgs, Trm, Trs = trans_vars_T
    Tg = T_[:,-1]
    Tg = Tg * Tgs + Tgm
    Tr = T_[:, :-1]
    Tr = Tr * Trs + Trm
    Tr = Tr + Tg[:, None]
    T_ = np.hstack((Tg[:, None], Tr))

    ix = np.ones(T_.shape[0]) == 1
    if T is not None:
        ixBad = np.any(T_ - (1-q)*(T.min(axis=0)[None,:]) < 0, axis=1) | np.any(T_ - (1+q)*(T.max(axis=0)[None,:]) > 0, axis=1)
        ixBad = ixBad | np.any(np.abs(np.diff(T_, axis=1)) - (1+q)*(np.abs(np.diff(T, axis=1)).max(axis=0)[None,:]) > 0, axis=1)
        ix = ~ixBad
    print(len(ix),sum(ix))
    return T_, ix

def trans_C(x, P, T):
    c = mf2mol_cum(x, P, T)
    cp = c[:, -1]
    cp[cp==0] = np.min(cp[cp>0])
    cr = c[:,:-1] / cp[:,None]
    crm = cr.mean()
    crs = cr.std()
    cr = (cr - crm) / crs
    cpm = cp.mean()
    cps = cp.std()
    cp = (cp - cpm) / cps
    w = cr.std(axis=0)
    w /= w.sum()
    w = np.append(w, 3*w.max())
    c_ = np.hstack((cr, cp[:, None]))
    trans_vars_C = (crm, crs, cpm, cps)
    return c_, trans_vars_C, w

def itrans_C(c_, trans_vars_C, P, T, c=None, q=0.1):
    crm, crs, cpm, cps = trans_vars_C
    cu = np.copy(c_)
    cp = cu[:, -1]
    cp = cp * cps + cpm
    cr = cu[:,:-1]
    cr = cr * crs + crm
    cu = np.hstack((cr * cp[:, None], cp[:, None]))
    x_ = mol_cum2mf(cu, P, T)

    c_diff = np.diff(cu, axis=1)
    c_sm = np.percentile(np.abs(cu[cu>0]), 5)
    c_diff_sm = np.percentile(np.abs(c_diff), 5)
    ixBad = np.any(cu < -c_sm, axis=1) | np.any(c_diff < -c_diff_sm, axis=1) | (cu[:,-1] == 0)
    ix = ~ixBad
    if c is not None:
        metric = (cu - (1 - q) * (c.min(axis=0)[None,:]) < 0) | (cu - (1 + q) * (c.max(axis=0)[None,:]) > 0)
        ixBad = ixBad | np.any(metric, axis=1)
        ix = ~ixBad
    return x_, ix

def atmos_to_features(P, T, H2O, O3, transform=False):
    ixT = np.arange(0, T.shape[1]).astype(np.int)
    ixH2O = 1 + ixT[-1] + np.arange(0, H2O.shape[1]).astype(np.int)
    ixO3 = 1 + ixH2O[-1] + np.arange(0, O3.shape[1]).astype(np.int)
    if transform:
        T_, vars_T, wT = trans_T(T, P)
        H2O_, vars_H2O, wH2O = trans_C(H2O, P, T)
        O3_, vars_O3, wO3 = trans_C(O3, P, T)
        trans_vars_atmos = (vars_T, ixT, vars_H2O, ixH2O, vars_O3, ixO3)
        wC = wH2O/wH2O.max() + wO3/wO3.max()
        wC /= wC.sum()
        wT = wT * wC
        wT /= wT[:-1].sum()
        wT[-1] = wT[:-1].max()
    else:
        T_, H2O_, O3_ = T, H2O, O3
        trans_vars_atmos = ((), ixT, (), ixH2O, (), ixO3)
        wT, wH2O, wO3 = tuple(map(lambda x: np.ones(x.shape[1]),[T_, H2O_, O3_]))
    X = np.concatenate((T_, H2O_, O3_), axis=1)
    wX = np.concatenate((wT / wT.max(), wH2O / wH2O.max(), wO3 / wO3.max()))
    wX /= wX.sum()
    return X, trans_vars_atmos, wX

def features_to_atmos(X, trans_vars_atmos, P, T=None, cH2O=None, cO3=None):
    vars_T, ixT, vars_H2O, ixH2O, vars_O3, ixO3 = trans_vars_atmos
    T_ = X[:, ixT]
    H2O_ = X[:, ixH2O]
    O3_ = X[:, ixO3]
    if len(vars_T) > 0:
        T_, ixT = itrans_T(T_, vars_T, T)
        print(f'T:   {sum(ixT):d} good out of {len(ixT):d} total')
    if len(vars_H2O) > 0:
        H2O_, ixH2O = itrans_C(H2O_, vars_H2O, P, T_, cH2O)
        print(f'W:   {sum(ixH2O):d} good out of {len(ixH2O):d} total')
    if len(vars_O3) > 0:
        O3_, ixO3 = itrans_C(O3_, vars_O3, P, T_, cO3)
        print(f'O:   {sum(ixO3):d} good out of {len(ixO3):d} total')
    ix = ~(~ixT | ~ixH2O | ~ixO3)
    print(f'Agg: {sum(ix):d} good out of {len(ix):d} total')
    return T_, H2O_, O3_, ix

# ---------------------------------------------------------------------
# Load and pre-process atmospheric data
# ---------------------------------------------------------------------

# Atmospheric state parameters
f = h5py.File("data/LWIR_TUD.h5", "r")
z, T, P, H2O, O3 = tuple(map(lambda x: f[x][...], ['z', 'T', 'P', 'H2O', 'O3']))

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
cH2O = mf2mol_cum(H2O, P, T)
cO3 = mf2mol_cum(O3, P, T)

# ---------------------------------------------------------------------
# Atmospheric generative model
# ---------------------------------------------------------------------

def atmos_generator(P, T, H2O, O3, n_pca=15, n_gmm=10, transform=True, weight=True, filt=True):

    # Build features
    X, trans_vars_atmos, wX = atmos_to_features(P, T, H2O, O3, transform=transform)
    cH2O = mf2mol_cum(H2O, P, T)
    cO3 = mf2mol_cum(O3, P, T)

    # Generative model
    if weight:
        generator, Xr, Xm = pca_gmm_gen_mdl(X, n_pca=n_pca, n_gmm=n_gmm, w=wX)
    else:
        generator, Xr, Xm = pca_gmm_gen_mdl(X, n_pca=n_pca, n_gmm=n_gmm, w=None)

    def atm_gen(n):
        (Xr_n, X_n, ll) = generator(int(1.25 * n))  # generate more than required so we can throw some away
        T_n, H2O_n, O3_n, ix = features_to_atmos(X_n, trans_vars_atmos, P, T=T, cH2O=cH2O, cO3=cO3)

        # Filter out physically impossible or "unrealistic" atmospheric states
        if filt:
            T_n = T_n[ix,:]
            H2O_n = H2O_n[ix,:]
            O3_n = O3_n[ix,:]
            Xr_n = Xr_n[ix,:]
            ll = ll[ix]

        # Return only the number of samples requested
        N = np.min([n, T_n.shape[0]])
        return T_n[:N,:], H2O_n[:N,:], O3_n[:N,:], ll, Xr_n, X_n, trans_vars_atmos
    return atm_gen, X, trans_vars_atmos, wX, Xr, Xm

# ---------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------

def plot_pca_components(Xr, Xrn, ii=0, jj=1): 
    fig = plt.figure() 
    plt.plot(Xr[:, ii], Xr[:, jj], '.', label='Original')
    plt.plot(Xrn[:, ii], Xrn[:, jj], '.', label='New')
    plt.legend()
    return fig

def plot_data_model(P, T, Tm, H2O, H2Om, O3, O3m, ii=0):

    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    l1 = plt.plot(T[ii,:], z, '-b', label='T (data)')
    l2 = plt.plot(Tm[ii,:], z, 'b--', label='T (model)')
    ax1.set_xlabel('T [K]', color='b')
    ax1.tick_params('x', colors='b')
    plt.ylabel('z [km]')
    ax2 = ax1.twiny()
    l3 = plt.plot(P/101325, z, '-r', label='P')
    ax2.set_xlabel('P [atm]', color='r')
    ax2.tick_params('x', colors='r')
    ls = l1 + l2 + l3
    labs = [l.get_label() for l in ls]
    ax1.legend(ls, labs)

    ax1 = plt.subplot(1, 3, 2)
    plt.plot(H2O[ii,:] * 100, z, '-b', H2Om[ii,:] * 100, z,'--b')
    ax1.set_xlabel('H2O [%]', color='b')
    ax1.tick_params('x', colors='b')
    ax2 = ax1.twiny()
    plt.plot(cH2O[ii,:], z, '-r', cH2Om[ii,:], z,'--r')
    ax2.set_xlabel('cH2O [mol]', color='r')
    ax2.tick_params('x', colors='r')

    ax1 = plt.subplot(1, 3, 3)
    plt.plot(O3[ii,:] * 1e6, z, '-b', O3m[ii,:] * 1e6, z,'--b')
    ax1.set_xlabel('O3 [ppm]', color='b')
    ax1.tick_params('x', colors='b')
    ax2 = ax1.twiny()
    plt.plot(cO3[ii,:]*1e6, z, '-r', cO3m[ii,:]*1e6, z,'--r')
    ax2.set_xlabel('cO3 [µmol]', color='r')
    ax2.tick_params('x', colors='r')

    fig.tight_layout()
    return fig

def plot_gen_data(P, T, Tm, H2O, H2Om, O3, O3m, N=10, q=[5,50,95]):

    def mk_plt(x, xn):
        plt.plot(x[::N,:].T, z, '-b', alpha=0.25)
        plt.plot(xn[::N,:].T, z, '-r', alpha=0.25)
        plt.plot(np.percentile(x, q, axis=0).T, z, '-', linewidth=2, c=(0, 0, 0.67, 0.75))
        plt.plot(np.percentile(xn, q, axis=0).T, z, '-', linewidth=2, c=(0.67, 0, 0, 0.75))

    fig = plt.figure()
    plt.subplot(1, 3, 1)
    mk_plt(T, Tn)
    plt.xlabel('T [K]')
    plt.ylabel('z [km]')
    plt.subplot(1, 3, 2)
    mk_plt(H2O, H2Om)
    plt.xlabel('cH2O [mol]')
    plt.subplot(1, 3, 3)
    mk_plt(1e6*O3, 1e6*O3m)
    plt.xlabel('cO3 [µmol]')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------
# Test the approach
# ---------------------------------------------------------------------

np.random.seed(42)
atm_gen, X, trans_vars_atmos, wX, Xr, Xm = atmos_generator(P, T, H2O, O3, n_pca=15, filt=True)
Tm, H2Om, O3m, ix = features_to_atmos(Xm, trans_vars_atmos, P, T=T, cH2O=cH2O, cO3=cO3)
(Tn, H2On, O3n, ll, Xrn, Xn, trans_vars_atmos) = atm_gen(2000)

ix_err = np.argsort(np.sqrt(np.mean((X - Xm)**2, axis=1)))
ix50 = ix_err[int(0.50*len(ix_err))]
ix95 = ix_err[int(0.95*len(ix_err))]
cH2Om = mf2mol_cum(H2Om, P, Tm)
cO3m = mf2mol_cum(O3m, P, Tm)
cH2On = mf2mol_cum(H2On, P, Tn)
cO3n = mf2mol_cum(O3n, P, Tn)

plot_pca_components(Xr, Xrn, ii=0, jj=1)
plot_data_model(P, T, Tm, H2O, H2Om, O3, O3m, ii=ix50)
plot_data_model(P, T, Tm, H2O, H2Om, O3, O3m, ii=ix95)

plot_gen_data(P, T, Tn, cH2O, cH2On, cO3, cO3n, N=10)

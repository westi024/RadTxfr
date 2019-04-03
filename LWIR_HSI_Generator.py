# %% [markdown]

# # This is a header
# This is some text

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interp

import radiative_transfer as rt

# %%
# Load TUD from TIGR atmospheric inputs
TUD = np.load("data/LWIR-TUD-4wn.npz")
X = TUD["X"]
OD = TUD["OD"]
tau = np.exp(-OD)
La = TUD["La"]
Ld = TUD["Ld"]
Ts = TUD["T"][:, 0]

# %%
# Load emissivities from ASTER 2.0 database
em = np.load("data/emissivity_ASTER_2.0_LWIR.npz")
mID = em["material_ID"]
emis = np.zeros((len(mID), len(X)))
X_ = em["X"]
e_ = em["emis"]

# Interpolate onto TUD spectral axis
f = lambda x, y:  interp.interp1d(x, y, kind="cubic", fill_value="extrapolate")
for ii, id in enumerate(mID):
    emis[ii, :] = f(X_, e_[ii, :])(X)


# %%

# Define a HSI data generator
np.random.seed(42)

N_emis = 6
N_mix = 2
dT = 3


def LWIR_HSI_gen(N=100, dT=3, N_emis=6, N_mix=2, N_atm=3):

    L = []
    emis_labels = []
    mix_frac = []
    Ts_pix = []
    atmos_labels = np.random.randint(0, len(tau), N_atm)

    for ix_atm in atmos_labels:
        # Emissivity sampling and linear mixing
        ix_em = np.random.randint(0, len(emis), N_emis)
        # TODO: incorporate ability to weight different materials
        # by their fractional abundance in a scene (see `p` option)
        ix_em = np.random.choice(ix_em, (N, N_mix), replace=True)
        mixFrac = np.random.rand(N, N_mix)
        mixFrac /= mixFrac.sum(axis=1)[:, None]
        em = np.array(tuple(np.dot(mixFrac[ii], emis[ix_em[ii]]) for ii in range(N)))
        T = Ts[ix_atm] + dT * np.random.standard_normal(N)
        B = rt.planckian(X, T).T
        Ls = em * B + (1 - em) * Ld[ix_atm, :][None, :]
        L_ = tau[ix_atm, :] * Ls + La[ix_atm, :][None, :]

        L.append(L_)
        Ts_pix.append(T)
        emis_labels.append(ix_em)
        mix_frac.append(mixFrac)

    (L, atmos_labels, Ts_pix, emis_labels, mix_frac) = tuple(
        map(np.array, [L, atmos_labels, Ts_pix, emis_labels, mix_frac])
    )
    return (L, atmos_labels, Ts_pix, emis_labels, mix_frac)

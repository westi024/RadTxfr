# -*- coding: utf-8 -*-
# ---
# title: LWIR HSI data generator
# author: Kevin Gross
# date: 06-Apr-2019
#
# jupyter:
#   jupytext:
#     cmds:
#       html: build_pandoc_html.sh LWIR_HSI_Generator.md
#       init: jupytext --set-formats py:percent,md,ipynb LWIR_HSI_Generator.ipynb
#       update: jupytext --sync --pipe black --to py:percent LWIR_HSI_Generator.ipynb
#     formats: py:percent,md,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.5
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Overview
#
# The following script builds a HSI data generator with spectral coverage between $6.75 \,\mathrm{\mu m} < \lambda < 14.5 \,\mathrm{\mu m}$. Realistic atmospheric TUDs were computed via LBLRTM using inputs from the TIGR database. Pixels are formed from linear mixtures of emissivities from the ASTER 2.0 database. Surface temperatures for each pixel are taken from a normal distribution centered about the appropriate TIGR surface temperature with a user-defined width. The spectral model assumes a space-borne nadir-looking sensor imaging flat, Lambertian pixels. Specifically, the at-sensor radiance $L_{i,j}$ corresponding to the $i^\mathrm{th}$ pixel as viewed through the $j^\mathrm{th}$ TIGR atmosphere can be expressed as:
#
# $$
# L_{i,j}(\nu) = \tau_j(\nu) \left[ \varepsilon_i(\nu) B(\nu, T_{s,j} + \delta T_i) + (1-\varepsilon_i(\nu)) L_{d,j}(\nu)\right] + L_{u,j}(\nu)
# $$
#
# where $\tau_j$, $L_{d,j}$, and $L_{u,j}$ are the atmospheric transmittance, downwelling, and upwelling terms, respectively, for the $j^\mathrm{th}$ atmosphere, and $T_{s,j}$ is the associated surface-level air temperature for the $j^\mathrm{th}$ atmosphere. $\delta T_i$ represents a random perturbation of the ground air temperature so that there is temperature variability across the surface. The user can specify the width of the normal distribution used for sampling these temperature perturbations. The effective emissivity $\varepsilon_i(\nu)$ is a linear mixture of material emissivities, and is given by:
#
# $$
# \varepsilon_i(\nu) = \sum_k f_k \epsilon_k(\nu)
# $$
#
# Here, $f_k$ is the mixing fraction of the $k^\mathrm{th}$ material, i.e. $0 \leq f_k \leq 1$ and $\sum_k f_k = 1$.
#
# ## Requirements
#
# 1. Emissivity database: `emissivity_ASTER_2.0_LWIR.npz`
# 2. TUD database: `LWIR-TUD-4wn.npz`
# 3. Radiative transfer package `radiative_transfer`
#
# ## Current limitations
#
# 1. No correlation between surface temperature and VNIR reflectance.
# 2. No non-linear mixing effects for surface emissivities.
# 3. No realistic distribution of materials being chosen from the ASTER database.
#
# # Code
#
# Import the required packages.

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import radiative_transfer as rt

# %% [markdown]
# Load the TIGR atmospheric TUD database at $4\,\mathrm{cm^{-1}}$ resolution. Pull out the surface temperature `Ts` for each TIGR entry. The spectral axis `X` will be used throughout.

# %%
# Load TUD from TIGR atmospheric inputs
TUD = np.load("data/LWIR-TUD-4wn.npz")
X = TUD["X"]
OD = TUD["OD"]
tau = np.exp(-OD)
La = TUD["La"]
Ld = TUD["Ld"]
Ts = TUD["T"][:, 0]

# %% [markdown]
# Load the ASTER 2.0 database and resample the emissivities onto the common spectral axis (`X`).

# %%
# Load emissivities from ASTER 2.0 database
em = np.load("data/emissivity_ASTER_2.0_LWIR.npz")
mID = em["material_ID"]
emis = np.zeros((len(mID), len(X)))
X_ = em["X"]
e_ = em["emis"]

# Interpolate onto TUD spectral axis
f = lambda x, y: interp.interp1d(x, y, kind="cubic", fill_value="extrapolate")
for ii, id in enumerate(mID):
    emis[ii, :] = f(X_, e_[ii, :])(X)

# %% [markdown]
# Define the HSI data generator.

# %%
# Define a HSI data generator
np.random.seed(42)

N_emis = 6
N_mix = 2
dT = 3


def LWIR_HSI_gen(N=100, dT=3, N_emis=6, N_mix=2, N_atm=3):
    """
    Randomly generate at-sensor spectral radiances for mixed pixels.

    Parameters
    __________
    N: int 
      number of simulated pixels to generate per atmospheric TUD
    dT: double
      standard deviation of the Gaussian surface temperature distribution
    N_emis: int
      number of pure emissivities (i.e., end members) to randomly select
      from the ASTER database
    N_mix: int
      number of materials to mix in each pixel
    N_atm: int
      number of distinct atmospheric TUDs to use

    Returns
    _______
    L: array_like (N_atm, N, nX)
      apparent spectral radiance
    atmos_labels: array_like (N_atm,)
      Numeric label for the atmospheric TUD
    Ts_pix: array_like (N,)
      surface temperature at each pixel
    emis_label: array_like (N, N_mix)
      pure emissivity labels for each pixel
    mix_frac: array_like (N, N_mix)
      mixing fraction associated with each pure emissivity
    """
    # Pre-allocate and randomly choose atmospheric TUDs
    L = []
    emis_labels = []
    mix_frac = []
    Ts_pix = []
    atmos_labels = np.random.randint(0, len(tau), N_atm)

    # Generate multiple pixels of HSI data for each atmospheric TUD
    for ix_atm in atmos_labels:

        # Randomly sample ASTER emissivity database
        ix_em = np.random.randint(0, len(emis), N_emis)
        # TODO: incorporate ability to weight different materials
        # by their fractional abundance in a scene (see `p` option)
        ix_em = np.random.choice(ix_em, (N, N_mix), replace=True)

        # Randomly chosen mixing fractions (fractional abundances)
        mixFrac = np.random.rand(N, N_mix)
        mixFrac /= mixFrac.sum(axis=1)[:, None]
        em = np.array(tuple(np.dot(mixFrac[ii], emis[ix_em[ii]]) for ii in range(N)))

        # Randomly chosen temp and planckian distribution
        T = Ts[ix_atm] + dT * np.random.standard_normal(N)
        B = rt.planckian(X, T).T

        # Surface-leaving and at-aperture spectral radiance
        Ls = em * B + (1 - em) * Ld[ix_atm, :][None, :]
        L_ = tau[ix_atm, :] * Ls + La[ix_atm, :][None, :]

        # Append to outputs components
        L.append(L_)
        Ts_pix.append(T)
        emis_labels.append(ix_em)
        mix_frac.append(mixFrac)

    # Convert to numpy arrays and return
    (L, atmos_labels, Ts_pix, emis_labels, mix_frac) = tuple(
        map(np.array, [L, atmos_labels, Ts_pix, emis_labels, mix_frac])
    )
    return (L, atmos_labels, Ts_pix, emis_labels, mix_frac)


# %% [markdown]
# Visualize results

# %%
# Generate spectra
(L, atmos_labels, Ts_pix, emis_labels, mix_frac) = LWIR_HSI_gen(
    N=25, dT=3, N_emis=5, N_mix=2, N_atm=2
)

# Visualize results
plt.figure()
plt.plot(X, L[0, :, :].T, "-r", label=f"TUD {atmos_labels[0]}")
plt.plot(X, L[1, :, :].T, "-b", label=f"TUD {atmos_labels[1]}")
plt.xlabel(r"Wavenumbers, $\nu$ [cm$^{-1}$]")
plt.ylabel(r"Radiance, $L(\nu)$ [µW/(cm$^2$ sr cm$^{-1}$)]")
plt.savefig("data/LWIR_HSI_generator.png")

# %% [markdown]
# ![LWIR radiance spectra. Each color represents a distinct atmospheric TUD.](data/LWIR_HSI_generator.png)

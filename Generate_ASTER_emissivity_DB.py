# ---
# title: ASTER emissivity database for LWIR spectral simulations
# author: Kevin Gross
# date: 04-Apr-2019
#
# jupyter:
#   jupytext:
#     cmds:
#       html: build_pandoc_html.sh Generate_ASTER_emissivity_DB.md
#       init: jupytext --set-formats py:percent,md,ipynb Generate_ASTER_emissivity_DB.ipynb
#       update: jupytext --sync --pipe black --to py:percent Generate_ASTER_emissivity_DB.ipynb
#     formats: py:percent,md,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.4
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Overview
# The following script extracts all of the materials in the ASTER 2.0 reflectivity database with spectral coverage between $\lambda_{min} < \lambda < \lambda_{max}$ and stores them in an HDF5 file. The reflectivities are all resampled onto a common spectral axis in wavenumbers.
#
# # Code
#
# Import the required packages.

# %%
# Imports
import csv
import numpy as np
from scipy.interpolate import interp1d
import spectral
import h5py
import matplotlib.pyplot as plt

# %% [markdown]
# Set the wavelength limits for the database. Only materials having spectral reflectances between these limits will be included. Also define the spacing of the common spectral axis.

# %%
# Constants
lambda_min = 6.75
lambda_max = 14.5
X_min = 10000.0 / lambda_max
X_max = 10000.0 / lambda_min
dX = 1.0

# %% [markdown]
# Use the spectral python package to load the ASTER database and perform a query to find only those materials having reflectance measurements in the spectral band of interest.

# %%
# Load ASTER spectral reflectance database
fname = "/Users/grosskc/Documents/Research/Code/m-files/HSI/ASTER/data"
db = spectral.AsterDatabase.create("data/emissivity_ASTER_2.0.db", fname)

# Once the database has been created, the following can be used
# db = spectral.AsterDatabase("data/emissivity_ASTER_2.0.db")

# Filter database to materials with in-band reflectance values
rows = db.query(
    "SELECT SpectrumID FROM Samples, Spectra "
    + "WHERE Samples.SampleID = Spectra.SampleID AND "
    + f"MinWavelength <= {lambda_min-0.25} AND "
    + f"MaxWavelength >= {lambda_max+0.25}"
)

# index of materials satisfying the wavelength requirements
emis_idx = [r[0] for r in rows]

# %% [markdown]
# Resample the reflectances onto a common spectral axis and convert to emissivity.

# %%
# Resample onto common LWIR spectral axis using cubic spline interpolation
X = np.linspace(X_min, X_max, int((X_max - X_min) / dX))

f = lambda x, y: interp1d(x, y, kind="cubic", fill_value="extrapolate")

# pre-allocate
emis = np.zeros((len(emis_idx), len(X)))
material_description = []

# loop over materials
for (ii, idx) in enumerate(emis_idx):

    # extract signature info from database
    s = db.get_signature(idx)
    material_description.append(s.sample_name)
    X_, R_ = s.x, s.y

    # X_ to wavenumbers, R_ to fractional, physical reflectance vals
    X_, R_ = 10000.0 / np.array(X_), np.array(R_) / 100
    R_[R_ < 0.0] = 0.0
    R_[R_ > 1.0] = 1.0

    # ensure wavenumber axis is increasing
    ix = np.argsort(X_)
    X_, R_ = X_[ix], R_[ix]

    # trim spectral axis and eliminate any duplicate entries
    ix = (X_ >= X.min()) & (X_ <= X.max())
    X_, R_ = X_[ix], R_[ix]
    _, ix = np.unique(X_, return_index=True)
    X_, R_ = X_[ix], R_[ix]

    # resample and convert to emissivity
    emis[ii, :] = 1.0 - f(X_, R_)(X)

# Clean up un-physical values caused by measurement noise, error, or interpolation
emis[emis < 0] = 0.0
emis[emis > 1] = 1.0

# %% [markdown]
# Export labels to a `csv` file and data to both `hdf5` and Numpy `npz` files.
# %%
# Create CSV mapping ASTER material IDs to material descriptions
with open("data/emissivity_ASTER_2.0_LWIR.csv", "w", newline="") as csvfile:
    fieldnames = ["IDX", "Description"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for (ii, idx) in enumerate(emis_idx):
        writer.writerow({"IDX": idx, "Description": material_description[ii]})

# Create Numpy NPZ database
np.savez(
    "data/emissivity_ASTER_2.0_LWIR.npz",
    X=X,
    emis=emis,
    material_ID=emis_idx,
    material_description=np.asarray(material_description),
)

# Create HDF5 database
hf = h5py.File("data/emissivity_ASTER_2.0_LWIR.h5", "w")

d = hf.create_dataset("X", data=X)
d.attrs["units"] = "cm^{-1}"
d.attrs["name"] = "Wavenumbers"
d.attrs["info"] = "Spectral axis for emissivity"
d.attrs["label"] = r"$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$"

d = hf.create_dataset("emis", data=emis)
d.attrs["units"] = "none"
d.attrs["name"] = "Emissivity"
d.attrs["info"] = "Hemispherically-averaged emissivity"
d.attrs["label"] = r"$\varepsilon(\tilde{\nu})$"

d = hf.create_dataset("material_ID", data=emis_idx)
d.attrs["units"] = "none"
d.attrs["name"] = "Material ID"
d.attrs["info"] = "Numerical index of the material in the ASTER 2.0 database"
d.attrs["label"] = "Material ID"

dt = h5py.special_dtype(vlen=str)
d = hf.create_dataset("vlen_str", (len(material_description),), dtype=dt)
for ii, descr in enumerate(material_description):
    d[ii] = descr
d.attrs["units"] = "none"
d.attrs["name"] = "Material description"
d.attrs["info"] = "Description of the material in the ASTER database"
d.attrs["label"] = "Material Description"

hf.close()

# %% [markdown]
# Visualize the emissivity database.

# %%
plt.figure()
plt.plot(X, emis.T)
plt.xlabel("Wavenumbers [cm$^{-1}$]")
plt.ylabel("Emissivity")
plt.title(f"ASTER 2.0 Database - LWIR - {len(emis_idx):d} materials")
plt.savefig("data/emissivity_ASTER_2.0_LWIR.png")

# %% [markdown]
# ![ASTER 2.0 emissivity values](data/emissivity_ASTER_2.0_LWIR.png)

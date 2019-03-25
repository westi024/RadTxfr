# %%

# Imports
import numpy as np
from scipy.interpolate import interp1d
import spectral
import csv
import h5py
import matplotlib.pyplot as plt

# %%

# Constants
lambda_min = 6.5
lambda_max = 13.5
X_min = 10000.0 / lambda_max
X_max = 10000.0 / lambda_min
dX = 1.0

# %%

# Load ASTER spectral reflectance database
fname = "/Users/grosskc/Documents/Research/Code/m-files/HSI/ASTER/data"
db = spectral.AsterDatabase.create("data/emissivity_ASTER_2.0.db", fname)
# db = spectral.AsterDatabase("data/emissivity_ASTER_2.0.db")

# Filter database to materials with in-band reflectance values
rows = db.query(
    "SELECT SpectrumID FROM Samples, Spectra "
    + "WHERE Samples.SampleID = Spectra.SampleID AND "
    + f"MinWavelength <= {lambda_min-0.25} AND "
    + f"MaxWavelength >= {lambda_max+0.25}"
)
emis_idx = [r[0] for r in rows]

# %%

# Resample onto common LWIR spectral axis using cubic spline interpolation
X = np.linspace(X_min, X_max, int((X_max - X_min) / dX))

f = lambda x, y: interp1d(x, y, kind="cubic", fill_value="extrapolate")

# pre-allocate
emis = np.zeros((len(emis_idx), len(X)))
material_description = []

# loop over materials
for (ii, idx) in enumerate(emis_idx):
    s = db.get_signature(idx)
    material_description.append(s.sample_name)
    X_, R_ = s.x, s.y
    # X_, R_ = db.get_spectrum(idx)
    X_, R_ = 10000.0 / np.array(X_), np.array(R_) / 100
    R_[R_ < 0] = 0.0
    R_[R_ > 1] = 1.0
    ix = np.argsort(X_)
    X_, R_ = X_[ix], R_[ix]
    ix = (X_ >= X.min()) & (X_ <= X.max())
    X_, R_ = X_[ix], R_[ix]
    _, ix = np.unique(X_, return_index=True)
    X_, R_ = X_[ix], R_[ix]
    emis[ii, :] = 1.0 - f(X_, R_)(X)

# Clean up unphysical values caused by measurement noise, error, or interpolation
emis[emis < 0] = 0.0
emis[emis > 1] = 1.0

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

# %%
plt.figure()
plt.plot(X, emis.T)
plt.xlabel("Wavenumbers [cm$^{-1}$]")
plt.ylabel("Emissivity")
plt.title("ASTER 2.0 Database - LWIR")
plt.savefig("data/emissivity_ASTER_2.0_LWIR.png")

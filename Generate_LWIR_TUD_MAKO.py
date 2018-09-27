import numpy as np
import matplotlib.pyplot as plt
import h5py
from ILS_MAKO import ILS_MAKO

f = h5py.File("LWIR_TUD.h5", "r")
print(list(f.keys()))

# Extract spectral axis, wavenumbers, (nX,)
X_HI = f["X"][...]

# example demonstrating that there is metadata in this HDF5 file
print(f"""The spectral axis, {f["X"].attrs['name']}, has units """ +
      f"""{f["X"].attrs['units']} and spans {X_HI.min():0.1f} ≤ X ≤ {X_HI.max():0.1f}""")

# atmospheric state variables, (nA, nZ)
StdAtmos = f["StdAtmos"][...] # Standard Atmosphere Table
z = StdAtmos["Z0_km"]         # altitude above sea level, [km]
P = StdAtmos["P_Pa"]          # pressure [Pa]
T = f["T"][...]               # temperature profile, [K]
Ts = T[:, 0]                  # surface temperature, [K]
H2O = f["H2O"][...]           # water vapor volume mixing fraction, [ppm]
O3 = f["O3"][...]             # ozone volume mixing fraction, [ppm]

# atmospheric radiative transfer terms, transposed so spectral axis first dimension
tau_HI = f["tau"][:, -1, :]  # transmittance, [no units]
La_HI = f["La"][:, -1, :]    # atmospheric path radiance, [µW/(cm^2 sr cm^{-1})]
Ld_HI = f["Ld"][...]         # downwelling radiance, [µW/(cm^2 sr cm^{-1})]

# close H5 file
f.close()

# Convolve atmospheric parameters with MAKO lineshape
X, tau = ILS_MAKO(X_HI, tau_HI)
_, La = ILS_MAKO(X_HI, La_HI)
_, Ld = ILS_MAKO(X_HI, Ld_HI)

# Expand the atmospheric database with linear mixtures of neighboring atmospheric states
ixSrt = np.argsort(np.mean(tau, axis=0))
tau = tau[:, ixSrt]
La = La[:, ixSrt]
Ld = Ld[:, ixSrt]
Ts = Ts[ixSrt]

ixSrt = np.argsort(X)
X = X[ixSrt]
tau = tau[ixSrt, :]
La = La[ixSrt, :]
Ld = Ld[ixSrt, :]

# mixFrac = np.arange(0, 1.1, 0.1)
# nX = X.size
# nA = tau.shape[1]
# nF = mixFrac.size
# tauMix, LaMix, LdMix = [np.zeros((nX, nA, nF)) for _ in range(3)]
# TsMix = np.zeros((nA,nF))
# for ii in range(tau.shape[1] - 1):
#     for jj, f in enumerate(mixFrac):
#         tauMix[:, ii, jj] = f * tau[:, ii] + (1 - f) * tau[:, ii + 1]
#         LaMix[:, ii, jj] = f * La[:, ii] + (1 - f) * La[:, ii + 1]
#         LdMix[:, ii, jj] = f * Ld[:, ii] + (1 - f) * Ld[:, ii + 1]
#         TsMix[ii, jj] = f * Ts[ii] + (1-f)*Ts[ii+1]
# tauMix = np.reshape(tauMix, (tauMix.shape[0], np.prod(tauMix.shape[1:])))
# LaMix = np.reshape(LaMix, (tauMix.shape[0], np.prod(tauMix.shape[1:])))
# LdMix = np.reshape(LdMix, (tauMix.shape[0], np.prod(tauMix.shape[1:])))
# TsMix = np.reshape(TsMix, np.prod(TsMix.shape))
# tau, ix = np.unique(tauMix, axis=1, return_index=True)
# tau = tau[:,1:]
# La = LaMix[:, ix[1:]]
# Ld = LdMix[:, ix[1:]]
# Ts = TsMix[ix[1:]]
# ix = np.argsort(np.mean(tau,axis=0))
# tau = tau[:, ix]
# La = La[:, ix]
# Ld = Ld[:, ix]

# tau_mean = np.mean(tau, axis=0)
# tau_vals = np.linspace(tau_mean.min(), tau_mean.max(), 208)
# ix = np.unique(np.argmin(np.abs(tau_mean[np.newaxis,:] - tau_vals[:, np.newaxis]), axis=1))

# tau = tau[:, ix]
# La = La[:, ix]
# Ld = Ld[:, ix]
# Ts = Ts[ix]

# Save as HDF5 file
hf = h5py.File('LWIR_TUD_MAKO.h5', 'w')
d = hf.create_dataset('X', data=X)
d.attrs['units'] = 'cm^{-1}'
d.attrs['name'] = 'Wavenumbers'
d.attrs['info'] = 'Spectral axis for tau, La, Ld'
d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

d = hf.create_dataset('tau', data=tau)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Transmissivity'
d.attrs['info'] = 'For nadir-viewing path'
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

d = hf.create_dataset('Ts', data=Ts)
d.attrs['units'] = 'K'
d.attrs['name'] = 'Surface temperature'
d.attrs['info'] = ''
d.attrs['label'] = r'$T_s \,\, \left[ \si{K} \right]$'

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
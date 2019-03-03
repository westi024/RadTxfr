import multiprocessing as mp
import ctypes
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py

import radiative_transfer as rt

PARALLEL = True

# Load standard atmosphere
fname = '/Users/grosskc/Documents/Research/Code/python/RadTxfr/StandardAtmosphere.csv'
# fname = '/Users/kevin/Documents/Research/Code/python/RadTxfr/StandardAtmosphere.csv'
StdAtmos = np.genfromtxt(fname, dtype=float, delimiter=',', names=True)
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

# z = []
# for z0, z1 in zip(StdAtmos['Z0_km'], StdAtmos['Z1_km']):
#     z.append(np.linspace(z0, z1, 3))
# z = np.unique(np.array(z).ravel())

def T_ISAC(T0=296, kT=6.49):
    T = T0 - kT * z
    ix = T < np.interp(z,Z_SA,T_SA)
    T[ix] = np.interp(z[ix], Z_SA, T_SA)
    return T

# def T_ISAC(T0):
#     T = T0 - 6.49 * z
#     T[T < 216.7] = 216.7
#     return T

def W_ISAC(C0):
    Dat_1976 = np.array([0, 7330, 1.007, 5072.974, 2.012, 3565.067, 3.030, 2176.595, 4.006, 1364.121, 5.006, 786.532, 6.007, 459.895, 6.994, 255.593, 7.998, 122.755, 8.997, 56.630, 9.995, 23.590, 10.996, 10.002, 11.997, 4.502, 13.001, 2.224, 14.006, 1.040, 14.981, 0.729, 15.980, 0.495, 16.988, 0.417, 17.970, 0.357, 18.988, 0.303, 19.984, 0.267])
    Z_1976 = Dat_1976[0::2]
    W_1976 = Dat_1976[1::2]
    W = np.interp(z, Z_1976, W_1976)
    ix = z <= 16
    W[ix] = W[ix] * (1 + (C0 / 7330 - 1) * (1 - np.exp(-0.8 * (16 - z[ix])))**6)
    return W

# T0_grid = np.linspace(285, 315, 10)
# C0_grid = 7330 * np.array([2**i for i in range(-5, 6)])

T0_grid = np.linspace(290, 310, 10)
C0_grid = 7330 * np.logspace(-5, 4, 40, base=2)

nAtm = np.prod(list(map(len,[T0_grid,C0_grid])))

T, H2O = [np.zeros((nAtm, z.size)) for _ in range(2)]
params = np.zeros((nAtm, 2))
ii=0
for T0 in T0_grid:
    for C0 in C0_grid:
        params[ii,:] = np.array([T0, C0])
        T[ii,:] = T_ISAC(T0)
        H2O[ii,:] = W_ISAC(C0)
        ii += 1
        # print(f'{ii:04d} of {nAtm:04d}')

values = np.zeros((T0_grid.size, C0_grid.size, 2))
for ii,T0 in enumerate(T0_grid):
    for jj,C0 in enumerate(C0_grid):
        values[ii,jj,:] = np.array([T0, C0])

# Determine mean profile
Tm, H2Om = tuple(map(lambda x: np.mean(x, axis=0), (T, H2O)))

# Altitudes at which to compute tau and La
Altitudes = np.array([12210]) * 0.3048 / 1e3

# LOS viewing angles at which to compute tau and La
Angles = np.linspace(0, 60 * np.pi / 180, 15)

# Compute TUD for StdAtmos
Xmin = 690
Xmax = 1410
DV_out = 0.25
DV = 0.002
runLBLRTM = lambda T, H2O: rt.compute_TUD(Xmin, Xmax, MFs_VAL=H2O[None,:].T, MFs_ID=[1],  Ts=T, theta_r=Angles, Altitudes=Altitudes, DVOUT=DV,  returnOD=True)
XX, OD_SA_HI, La_SA_HI, Ld_SA_HI = runLBLRTM(Tm, H2Om)
X, OD_SA = rt.reduceResolution(XX, OD_SA_HI, DV_out)
f = lambda X_in, Y_in: rt.reduceResolution(X_in, Y_in, DV_out, X_out=X)

OD_SA = f(XX, OD_SA_HI)
La_SA = f(XX, La_SA_HI)
Ld_SA = f(XX, Ld_SA_HI)

# Generate plot
plt.figure()
ax = plt.subplot(2, 1, 1)
plt.semilogy(X, OD_SA)
plt.subplot(2, 1, 2, sharex=ax)
plt.plot(X, La_SA[:,[0,-1]], label="La")
plt.plot(X, Ld_SA, label="Ld")
plt.legend()
plt.show()

# Compute TUD for scenario
OD, La = [np.zeros((len(X), len(Angles), nAtm)) for _ in range(2)]
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
fname = datetime.now().strftime('%Y%m%d-%H%M%S') + '-OISAC-MAKO-TUD.npz'

def parallel_function(jj):
    """Function that operates on shared memory."""

    nu, OD_, La_, Ld_ = runLBLRTM(T[jj,:], H2O[jj,:].T)

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
    N = nProc
    nBatch = np.ceil(nAtm / N).astype(np.int)
    vals = np.arange(N)
    if PARALLEL:
        with mp.Pool(processes=nProc) as pool:
            for ii in np.arange(nBatch):
                idx = ii * N + vals
                idx = (idx[idx < nAtm]).astype(np.int)
                print(f"Batch {ii+1:0d} of {nBatch:0d}, indices {idx[0]:d} through {idx[-1]:d}")
                print(idx)
                pool.map(parallel_function, idx)
                np.savez(fname, X=X, OD=OD, La=La, Ld=Ld, Altitudes=Altitudes, Angles=Angles, params=params)
    else:
        for ii in np.arange(nBatch):
            idx = ii * N + vals
            idx = (idx[idx < nAtm]).astype(np.int)
            print(f"Batch {ii+1:0d} of {nBatch:0d}, indices {idx[0]:d} through {idx[-1]:d}")
            print(idx)
            list(map(parallel_function, idx))

    # Save as HDF5 file
    fname_H5 = fname.split('.')[0] + '.h5'
    hf = h5py.File(fname_H5, 'w')
    d = hf.create_dataset('X', data=X)
    d.attrs['units'] = 'cm^{-1}'
    d.attrs['name'] = 'Wavenumbers'
    d.attrs['info'] = 'Spectral axis for tau, La, Ld'
    d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

    d = hf.create_dataset('OD', data=OD)
    d.attrs['units'] = 'none'
    d.attrs['name'] = 'Optical Depth'
    d.attrs['info'] = 'For nadir-viewing path. tau = np.exp(-OD)'
    d.attrs['label'] = r'$\mathrm{OD}(\tilde{\nu})$'

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

    d = hf.create_dataset('SensorLookAngle', data=Angles)
    d.attrs['units'] = 'rad'
    d.attrs['name'] = 'Sensor Look Angle'
    d.attrs['info'] = 'Off-nadir angle (nadir = 0 rad)'
    d.attrs['label'] = r'$\theta_r \,\, \left[ \si{rad} \right]$'

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

    hf.close()



# ---------------------------------------------------
# Improvements to the OISAC database
# ---------------------------------------------------

# def T_ISAC(T0=296, kT=6.5):
#     T = T0 - kT * z
#     ix = T < np.interp(z,Z_SA,T_SA)
#     T[ix] = np.interp(z[ix], Z_SA, T_SA)
#     return T

# def W_ISAC(C0, HW=1.5, H0=8.0, T=300):
#     P = np.exp(-(z / H0))
#     W = (P / T) * np.exp(-(z / HW))
#     W /= np.sum(W)
#     W *= C0 * 1e-6
#     mix2mass = (18/(0.8*28 + 0.2*32))
#     W_max = atmos.calculate('rv', T=T, p=P*101325.0, RH=99.9*np.ones(T.shape)) / mix2mass
#     W[W > W_max] = W_max[W > W_max]
#     W[W < 0] = 0
#     W[P < np.exp(-3)] = 0
#     W[W > 0.1] = 0
#     RH = atmos.calculate('RH', T=T, p=P*101325.0, rv=W*mix2mass)
#     RH[RH<0] = 0
#     return (W*1e6, RH)


# kT_grid = np.arange(5.75, 7.5, 0.5)
# kT_grid = np.array([5.5, 6.5, 7.5])
# T0_grid = np.arange(290, 315, 5)
# C0_grid = np.logspace(-5,-1,40) * 32 * 1e6
# HW_grid = np.arange(1, 4, 1)

# nAtm = np.prod(list(map(len,[kT_grid,T0_grid,C0_grid,HW_grid])))

# T, H2O, RH = [np.zeros((nAtm, z.size)) for _ in range(3)]
# params = np.zeros((nAtm, 4))
# ii=0
# for T0 in T0_grid:
#     for kT in kT_grid:
#         for C0 in C0_grid:
#             for HW in HW_grid:
#                 params[ii,:] = np.array([T0, kT, C0, HW])
#                 T[ii,:] = T_ISAC(T0, kT)
#                 (H2O[ii,:], RH[ii,:]) = W_ISAC(C0, HW, T=T[ii,:])
#                 ii += 1
#                 print(f'{ii:04d} of {nAtm:04d}')

# values = np.zeros((T0_grid.size, C0_grid.size, HW_grid.size,3))
# for ii,T0 in enumerate(T0_grid):
#     for jj,C0 in enumerate(C0_grid):
#         for kk, HW in enumerate(HW_grid):
#             values[ii,jj,kk,:] = np.array([T0, C0, HW])

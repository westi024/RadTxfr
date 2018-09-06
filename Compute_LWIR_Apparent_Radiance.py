import numpy as np
import matplotlib.pyplot as plt
import h5py
import radiative_transfer as rt

f = h5py.File("LWIR_TUD_MAKO.h5", "r")
print(list(f.keys()))

X = f["X"][...].astype(np.float32)
tau = f["tau"][...].astype(np.float32)
La = f["La"][...].astype(np.float32)
Ld = f["Ld"][...].astype(np.float32)
Ts = f["Ts"][...].astype(np.float32)

f.close()

f = h5py.File("LWIR_Emissivity_DB_MAKO.h5")
print(list(f.keys()))

emis = f["emis"][...].astype(np.float32)

f.close()

dT = np.arange(-10, 10.5, 0.5).astype(np.float32)
L = rt.compute_LWIR_apparent_radiance(X, emis, Ts, tau, La, Ld, dT)
T = Ts[:, np.newaxis] + dT[np.newaxis,:]

# Save as HDF5 file
hf = h5py.File('LWIR_HSI_MAKO.h5', 'w')
d = hf.create_dataset('X', data=X)
d.attrs['units'] = 'cm^{-1}'
d.attrs['name'] = 'Wavenumbers'
d.attrs['info'] = 'Spectral axis for L, emis, tau, La, Ld'
d.attrs['label'] = r'$\tilde{\nu} \,\, \left[\si{cm^{-1}} \right]$'

d = hf.create_dataset('L', data=L)
d.attrs['units'] = 'µW/(cm^2 sr cm^{-1})'
d.attrs['name'] = 'Apparent Spectral Radiance'
d.attrs['info'] = 'For spaceborn nadir-viewing sensor. Shape is (nX, nE, nA, nT) where nX is # spectral channels, nE is # materials, nA is # atmospheres, nT is # surface temperatures'
d.attrs['label'] = r'$L(\tilde{\nu})\,\,\left[\si{\micro W/(cm^2.sr.cm^{-1})}\right]$'

d = hf.create_dataset('emis', data=emis)
d.attrs['units'] = 'none'
d.attrs['name'] = 'Emissivity'
d.attrs['info'] = 'Effective, Hemispherically-averaged Emissivity'
d.attrs['label'] = r'$\varepsilon(\tilde{\nu})$'

d = hf.create_dataset('T', data=T)
d.attrs['units'] = 'K'
d.attrs['name'] = 'Surface temperature'
d.attrs['info'] = ''
d.attrs['label'] = r'$T_s \,\, \left[ \si{K} \right]$'

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

hf.close()

# Reshape and split into training, testing, and validation subsets
nX, nE, nA, nT = L.shape
idx=[]
for ixE in range(nE):
    for ixA in range(nA):
        for ixT in range(nT):
            idx.append([ixE, ixA, ixT])
idx = np.asarray(idx)
L = np.reshape(L, (L.shape[0], np.prod(L.shape[1:])))
L = L.T
emis = emis.T
tau = tau.T
La = La.T
Ld = Ld.T

ixP = np.random.permutation(np.arange(L.shape[0]))
L = L[ixP,:]
idx = idx[ixP,:]
ixE = idx[:, 0]
ixA = idx[:, 1]
ixT = idx[:, 2]

# Split into training, testing, and validation
f_tr = 0.75

def gen_indices(f, N):
    ix_tr = np.round(np.linspace(0, N-1, np.int(f * N))).astype(np.int)
    ix_diff = np.sort(np.asarray(list(set.difference(set(np.arange(N)), set(ix_tr)))))
    ix_te = ix_diff[0::2]
    ix_va = ix_diff[1::2]
    return ix_tr, ix_te, ix_va

ixTrain, ixTest, ixValidate = gen_indices(f_tr,L.shape[0])

np.savez('LWIR_HSI_MAKO.npz', X=X, L=L, ixE=ixE, ixA=ixA, ixT=ixT, emis=emis, T=T, tau=tau, La=La, Ld=Ld,
    ixTrain=ixTrain, ixTest=ixTest, ixValidate=ixValidate)

for _ in range(5):
    ii = np.random.randint(0,L.shape[0])
    ixE = idx[ii, 0]
    ixA = idx[ii, 1]
    ixT = idx[ii, 2]
    mdl = tau[ixA,:] * (emis[ixE,:] * rt.planckian(X, T[ixA,ixT]) + (1 - emis[ixE,:]) * Ld[ixA,:]) + La[ixA,:]
    plt.plot(X, L[ii,:])
    plt.plot(X,mdl,'.')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import h5py
import radiative_transfer as rt

f = h5py.File("../RadTxfr/LWIR_TUD.h5", "r")
print(list(f.keys()))

# Convert to float32
fl = lambda x: x.astype(np.float32)

# Atmospheric state parameters
z, T, P, H2O, O3 = tuple(map(lambda x: fl(f[x][...]), ['z', 'T', 'P', 'H2O', 'O3']))

# Atmospheric radiative transfer terms
X_HI, OD_HI, La_HI, Ld_HI = tuple(map(lambda x: fl(f[x][...]), ['X', 'OD', 'La', 'Ld']))
OD_HI = OD_HI[:, -1,:]
tau_HI = np.exp(-OD_HI)
La_HI = La_HI[:, -1, :]

f.close()

# # Reduce spectral resolution
# X_min, X_max = 700, 1400
# DV_out = 2.0
# X, tmp = rt.reduceResolution(X_HI, OD_HI[:, 0], DV_out)
# dX = np.diff(X).mean()
# ix = (X >= X_min-dX) & (X <= X_max+dX)
# X = X[ix]
# print(len(X))

# Loop over all atmospheric states
X, _ = rt.ILS_MAKO(X_HI, OD_HI[:, 0], returnX=True)
nX = X.size
nAtm = OD_HI.shape[1]
OD, tau, La, Ld = tuple(np.zeros(shape=(nAtm, nX)) for _ in range(4))
# ILS = lambda Y_in: rt.reduceResolution(X_HI, Y_in, DV_out, X_out=X)
ILS = lambda Y_in: rt.ILS_MAKO(X_HI, Y_in, returnX=False)

for ii in np.arange(nAtm):
    OD[ii,:], tau[ii,:], La[ii,:], Ld[ii,:] = tuple(map(ILS, [OD_HI[:, ii], tau_HI[:, ii], La_HI[:, ii], Ld_HI[:, ii]]))
    print(f'Iteration {ii} of {nAtm}')

# save results
np.savez('LWIR-TUD-MAKO.npz', X=X, OD=OD, tau=tau, La=La, Ld=Ld, z=z, P=P, T=T, H2O=H2O, O3=O3)

# Loop over all atmospheric states
X, _ = rt.ILS_MAKO(X_HI, OD_HI[:, 0], resFactor=4, returnX=True)
nX = X.size
nAtm = OD_HI.shape[1]
OD, tau, La, Ld = tuple(np.zeros(shape=(nAtm, nX)) for _ in range(4))
ILS = lambda Y_in: rt.ILS_MAKO(X_HI, Y_in, resFactor=4, returnX=False)

for ii in np.arange(nAtm):
    OD[ii,:], tau[ii,:], La[ii,:], Ld[ii,:] = tuple(map(ILS, [OD_HI[:, ii], tau_HI[:, ii], La_HI[:, ii], Ld_HI[:, ii]]))
    print(f'Iteration {ii} of {nAtm}')

# save results
np.savez('LWIR-TUD-MAKO-future.npz', X=X, OD=OD, tau=tau, La=La, Ld=Ld, z=z, P=P, T=T, H2O=H2O, O3=O3)


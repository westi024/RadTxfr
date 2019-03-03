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

# Loop over all atmospheric states
X, _ = rt.ILS_MAKO(X_HI, OD_HI[:, 0], returnX=True)
nX = X.size
nAtm = OD_HI.shape[1]
OD, tau, La, Ld = tuple(np.zeros(shape=(nAtm, nX)) for _ in range(4))
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

# Jacobian
f = h5py.File("../RadTxfr/LWIR_TUD_JACOBIAN.h5", "r")
print(list(f.keys()))

# Convert to float32
fl = lambda x: x.astype(np.float32)

# Atmospheric state parameters
z, T, P, H2O, O3 = tuple(map(lambda x: fl(f[x][...]), ['z', 'T', 'P', 'H2O', 'O3']))
dz = np.diff(np.append(z, 70.0))
H2O *= 1e6
O3 *= 1e9

# Atmospheric radiative transfer terms
X_HI, OD_HI, La_HI, Ld_HI = tuple(map(lambda x: fl(f[x][...]), ['X', 'OD', 'La', 'Ld']))
OD_HI = OD_HI[:, -1,:]
tau_HI = np.exp(-OD_HI)
La_HI = La_HI[:, -1,:]
f.close()

jac = lambda y,x: (y[:, 1:] - y[:, 0][:,None]) / (np.tile(dz,(3,))[None,:] * x)
param = np.diag(np.hstack((T[1:,:]-T[0,:],H2O[1:,:]-H2O[0,:],O3[1:,:]-O3[0,:])))
J_HI = np.hstack(tuple(map(lambda x: jac(x, param), [OD_HI, La_HI, Ld_HI])))

# Loop over all atmospheric states
# X, _ = rt.ILS_MAKO(X_HI, J_HI[:, 0], returnX=True)
X, _ = rt.ILS_MAKO(X_HI, J_HI[:, 0], resFactor=4, returnX=True)
nX = X.size
nAtm = J_HI.shape[1]
J = np.zeros((nX,nAtm))
# ILS = lambda Y_in: rt.ILS_MAKO(X_HI, Y_in, returnX=False)
ILS = lambda Y_in: rt.ILS_MAKO(X_HI, Y_in, resFactor=4, returnX=False)
for ii in np.arange(nAtm):
    J[:, ii] = ILS(J_HI[:, ii])
    if ii % 10 == 0:
        print(f'{ii} of {nAtm}')
J = np.reshape(J, (J.shape[0], 3, 3, 66))
J = np.transpose(J, axes=(1, 2, 3, 0))

# w = np.sqrt(np.mean(J**2, axis=0))
# w /= w.max(axis=2)[:,:, None]
# w = np.mean(w,axis=0)

np.savez('Jacobian-MAKO-future.npz', X=X, J=J, P=P, T=T, H2O=H2O, O3=O3)

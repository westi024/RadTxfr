import os
# import glob
import numpy as np
import scipy.io as sio
import pickle
import tempfile
# import json
import sys


def serializeInputs(**kwargs):
    s = []
    if kwargs is not None:
        s = pickle.dumps(kwargs)
    return np.array(s)


def serializeInputsMatlab(**kwargs):
    s = []
    if kwargs is not None:
        H = {}
        for key,val in kwargs.items():
            H[key] = val
        with tempfile.TemporaryFile() as f:
            sio.savemat(f,{"Header":H})
            f.seek(0)
            s = bytes(f.read())
    return s


def deserializeInput(s):
    return pickle.loads(s)


def mapDtypeToString(Data):
    """Extract string descriptor of Data dtype."""
    s = str(Data.dtype)
    if s == 'float64':
        s = 'double'
    if s == 'float32':
        s = 'single'
    return "{0:<6}".format(s)


def AFIT_MBI_export(fname, Data, Rows=[], Columns=[], Bands=[],
                    Sidecar=True, BandsFirstDim=True, Verbose=True,
                    **kwargs):
    """Export data in BIP or BSQ binary format with arbitrary Header metadata.
    
    Store multiband imagery Data in a standard binary format. The data wil be
    stored on disk in either a band-interleaved-by-pixel (BIP) or band-
    sequential (BSQ) arrangement. Arbitrary python variables can be stored in
    Header, and optionally as an additional sidecar file in YAML. Arrays
    describing Rows, Columns, and Bands can also be stored in the binary file.
    Note that the Header is stored as a binary representation of a MATLAB v5
    file. As such, the AFIT MBI format may be of limited utility for non-MATLAB
    users. However, Data, Rows, Columns, and Bands are a separate component of
    the Header.

    Parameters
    ----------
        fname : str
            Name of file the data is written to. The extension should be either
            '.bip' or '.bsq', and that will determine how the Data is stored
        Data : array
            3D array of shape `(nB, nR, nC)` where `nB` is the number of Bands, 
            `nR` is the number of Rows, and `nC` is the number of Columns.
        Rows : array, optional
            Row index values. Default is numeric index from `np.arange(nR)`
        Columns : array, optional
            Column index values. Default is numeric index from `np.arange(nC)`
        Bands : array, optional
            Band index values. Default is numeric index from `np.arange(nB)`
        Sidecar : bool, optional
            If `True`, save Header and MetaData information in a YAML sidecar 
            file with same basename as fname and extension '.yaml'
        BandsFirst : bool, optional
            If True, assumes `Data` is of shape `(nB,nR,nC)` and transposes the 
            dimensions so that BIP / BSQ storage conventions are followed.
        Verbose : bool
            If True, prints diagnostic information to the terminal.
        kwargs : keyword arguments, optional
            Python / numpy variables to save to the file. Variables will be saved 
            within `Header` with the corresponding keyword names.
        

    """
    # Ensure Data is Numpy array of requested dtype
    Data = np.array(Data)

    # Consistency with multiband format convention req's band dimension is last
    if BandsFirstDim:
        Data = Data.transpose([1, 2, 0]).copy(order='C')

    # Fill empty rcb values
    nR, nC, nB = Data.shape
    if len(Bands) == 0:
        Bands = np.arange(nB)
    if len(Rows) == 0:
        Rows = np.arange(nR)
    if len(Columns) == 0:
        Columns = np.arange(nC)
    if len(kwargs) == 0:
        kwargs = {'MetaData': 'None'}

    # Extract path, name, and extension
    fdir, fname = os.path.split(fname)
    _, fext = os.path.splitext(fname)

    # Define elements of header data
    dims = np.array(Data.shape)
    dtypeStr = mapDtypeToString(Data)
    rcb = np.concatenate((Rows, Columns, Bands)).astype('float64')
    # serialMD = np.array(serializeInputsMatlab(**kwargs), dtype=np.uint8)
    serialMD = serializeInputsMatlab(**kwargs)
    interleave = fext[1:].lower()

    descriptorBytes = 12+6+3+4
    offsetBytes = np.array(descriptorBytes + len(serialMD) + rcb.nbytes)

    # Force little-endian encoding
    t1 = Data.dtype.byteorder == '>'
    t2 = sys.byteorder == 'little'
    t3 = Data.dtype.byteorder == '='
    if t1 or not (t2 and t3):
        Data = Data.byteswap()

    # write header to file
    with open(fdir + fname, 'wb') as f:
        dims.astype('uint32').tofile(f)
        f.write(bytes(dtypeStr, 'utf-8'))
        f.write(bytes(interleave, 'utf-8'))
        offsetBytes.astype('uint32').tofile(f)
        rcb.astype('float64').tofile(f)
        # serialMD.tofile(f)
        f.write(serialMD)
        if interleave == 'bsq':  # Band-sequential
            Data = Data.copy(order='C')
            for ii in np.arange(nB):
                tmp = Data[:,:,ii].squeeze().copy(order='C')
                tmp.flatten().tofile(f)
        elif interleave == 'bip':  # Band-interleaved by pixel
            Data = Data.transpose((2, 0, 1))
            dims = Data.shape
            Data = Data.reshape([dims[0], np.prod(dims[1:])]).copy(order='C')
            for ii in np.arange(Data.shape[1]):
                Data[:,ii].flatten().tofile(f)
        else:
            print('Unknown file extension -- no data saved')

    if Verbose:
        infoString1 = "Wrote {0} bytes ({1} header, {2} data) to {3}"
        infoString1 = infoString1.format(Data.nbytes + offsetBytes, offsetBytes, Data.nbytes, fname)
        print(infoString1)
        infoString2 = "The data is stored as {0}'s and arranged as {1} in {2} format"
        infoString2 = infoString2.format(Data.dtype, Data.shape, interleave.upper())
        print(infoString2)

# Test the code
import matplotlib.pyplot as plt

# Generate data -- band is *first* dimension
nB = 3
nR = 320
nC = 256
img = np.zeros((nB,nR,nC), dtype=np.uint8, order='C')
for ii in np.arange(nR):
    for jj in np.arange(nC):
        img[0,ii,jj] = (ii**2 + jj**2) < 128**2
        img[1,ii,jj] = abs(ii-160) > 0 and abs(jj-128) > 0
        img[2,ii,jj] = ii>160 and jj<128

# for ii in np.arange(3):
#     plt.imshow(img[ii,:,:])
#     plt.show()

fname1 = 'test.bsq'
metaData = {'Q': np.array([1,2,3]), 'Meta1': 2, 'myString': 'hello'}
AFIT_MBI_export(fname1, img, **metaData)

fname2 = 'test.bip'
AFIT_MBI_export(fname2, img)

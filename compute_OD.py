import os, os.path
import inspect
import subprocess
import tempfile

import numpy as np

# Define default options
LBL_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LBLRTM = os.path.join(LBL_dir, 'lblrtm_v12.8_OS_X_gnu_sgl')
TAPE3 = os.path.join(LBL_dir, 'AER-v3.6-0500-6000.tp3')
options = {
    # options for write_tape5
    'V1': 2000.00,
    'V2': 3333.33,
    'T': 296.0,
    'P': 101325.0,
    'PL': 1.0,
    'mixing_fraction': np.zeros(38),
    'mf_ID': np.array([]),
    'mf_val': np.array([]),
    'continuum_factors': np.zeros(7),
    'continuum_override': False,
    'description': 'TAPE5 for single layer calculation by compute_OD.py',
    'DVOUT': 0.0025,
    # options for run_LBLRTM
    'debug': True,
    'LBL_dir': LBL_dir,
    'LBLRTM': LBLRTM,
    'TAPE3': TAPE3,
    }
print(options)


def compute_OD(Xmin_in, Xmax_in, opts=options, ** kwargs):
    opts.update(kwargs)
    DVOUT = opts.get("DVOUT",0.025)

    # Set up parameters for looping over spectral range in 2020/cm chunks
    myround = lambda x: float("{0:10.3f}".format(x))
    pad = 100
    olp = 5
    Xmin = np.max([myround(Xmin_in - pad), 0])
    Xmax = myround(Xmax_in + pad)
    maxBW = 2020 - olp
    nBand = int(np.ceil((Xmax - Xmin) / maxBW))
    nPts = int(np.floor(maxBW / DVOUT))

    # Compute OD for each spectral chunck    
    X = []
    OD = []
    for ii in range(nBand):
        if ii > 0:
            Xmin = myround(np.max(X[ii - 1]) + DVOUT - olp)
        Xmax1 = np.min([Xmax+pad, myround(Xmin + DVOUT * (nPts - 1) + olp)])
        nu, od = run_LBLRTM(Xmin, Xmax1, opts=opts)
        X.append(nu)
        OD.append(od)
    
    # Stitch chunks together into single output vector
    N = np.ceil((Xmax_in - Xmin_in) / DVOUT)
    X_out = np.linspace(Xmin_in, Xmax_in, N)
    OD_out = np.zeros((nBand, X_out.size))
    for ii in range(nBand):
        OD_out[ii, :] = np.interp(X_out, X[ii], OD[ii], left=0, right=0)
    OD_out = np.sum(OD_out, axis=0) / np.sum(OD_out > 0, axis=0)
    OD_out = OD_out.flatten()
    return X_out, OD_out


def run_LBLRTM(V1, V2, opts=options, **kwargs):
    
    opts.update(kwargs)
    opts["V1"] = V1
    opts["V2"] = V2
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        os.symlink(opts.get('TAPE3'), 'TAPE3')
        os.symlink(opts.get('LBLRTM'), 'lblrtm')
        write_tape5(fname="TAPE5", **opts)
        ex = subprocess.run('./lblrtm', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ex.stderr == b'STOP  LBLRTM EXIT \n':
            nu, od = read_tape12()
        os.chdir(cwd)
    return nu, od


def write_tape5(fname="TAPE5", opts=options, **kwargs):

    # Extract critical values with reasonable defaults
    opts.update(kwargs) # update opts dictionary with user-supplied keys/vals
    V1 = opts.get("V1", 2000.00)      # [cm^{-1}]
    V2 = opts.get("V2", 3333.33)      # [cm^{-1}]
    DVOUT = opts.get("DVOUT", 0.0025) # [cm^{-1}]
    T = opts.get("T", 296.0)          # [K]
    P = opts.get("P", 101325.0)       # [Pa]
    PL = opts.get("PL", 1.0)          # [km]
    CF = opts.get("continuum_factors", np.zeros(7))

    # Update mixing fraction
    C = opts.get("mixing_fraction", np.zeros(38))
    if "mf_ID" in opts.keys() and "mf_val" in opts.keys():
        idx = [i-1 for i in list(opts['mf_ID'])]
        C[idx] = opts['mf_val']

    # Update mixing fraction via molecule name specification
    hitranMolecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3',
                       'HNO3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN',
                       'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2',
                       'O+', 'ClONO2', 'NO+', 'HOBr', 'C2H4']
    mol_ix, mol_key = [], []
    for k in opts.keys():
        # index in hitranMolecule list that matches the molecule specified in opts
        loc = [i for i, j in enumerate(hitranMolecules) if j.upper() == k.upper()]
        if loc: # if loc is not empty
            mol_ix.append(loc) # add the molecule index
            mol_key.append(k)  # store the name so we can retrieve it later
    mol_ix = np.asarray(mol_ix).flatten()
    for i, k in enumerate(mol_key):
        C[mol_ix[i]] = opts[k]

    # Ensure only present species have continuum effects included
    if not opts.get("continuum_override", False):
        if C[0] > 0:
            CF[[0,1]] = 1
        if C[1] > 0:
            CF[2] = 1
        if C[2] > 0:
            CF[3] = 1
        if C[6] > 0:
            CF[4] = 1
        if C[21] > 0:
            CF[5] = 1

    # This will hold each individual record in the "punch card"
    CARD = []

    # RECORD 1.1 — Title
    RECORD = opts.get("description",'TAPE5 for single layer calculation by compute_OD.py')
    CARD.append(RECORD)
    CARD.append('         1         2         3         4         5         6         7         8         9         0')
    CARD.append('123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789')
    CARD.append('$ None')

    # RECORD 1.2 — General LBLRTM control — set up for single-layer OD calc
    IHIRAC = 1 # Voigt line profile
    ILBLF4 = 1 # Line-by-line function
    ICNTNM = 6 # User-supplied continuum scale factors
    IAERSL = 0 # No aerosols used in calculation
    IEMIT  = 0 # Optical depth only
    ISCAN  = 0 # No scanning / interpolation used
    IFILTR = 0 # No filter
    IPLOT  = 0 # No plot
    ITEST  = 0 # No test
    IATM   = 1 # Use LBLATM (RECORD 1.3)
    IMRG   = 0 # Normal merge
    ILAS   = 0 # Not for laser calculation
    IOD    = 1 # Normal calculation when layering multiple OD calculations
    IXSECT = 0 # No cross-sections included in calculation
    MPTS   = 0
    NPTS   = 0
    RECORD =  " HI={:1d} F4={:1d} CN={:1d} AE={:1d} EM={:1d} SC={:1d} FI={:1d} PL={:1d}"
    RECORD += " TS={:1d} AM={:1d} MG={:1d} LA={:1d} MS={:1d} XS={:1d}  {:2d}  {:2d}"
    RECORD = RECORD.format(IHIRAC, ILBLF4, ICNTNM, IAERSL, IEMIT, ISCAN, IFILTR, IPLOT,
                           ITEST, IATM, IMRG, ILAS, IOD, IXSECT, MPTS, NPTS)
    CARD.append(RECORD)

    # RECORD 1.2a — continuum scale factors
    RECORD = ((len(CF) * "{:8.6f} ").format(*CF)).rstrip()
    CARD.append(RECORD)

    # RECORD 1.3 — spectral range and related details
    SAMPLE = 4    # number of sample points per mean halfwidth (default)
    DVSET  = 0    # [cm^{-1}] selected DV for the final monochromatic calculation (default)
    ALFAL0 = 0.04 # [cm^{-1} / atm] average collision broadened halfwidth (default)
    AVMASS = 36   # [amu] average molecular mass (amu) for Doppler halfwidth (default)
    DPTMIN = 0    # minimum molecular optical depth below which lines will be rejected (0, no rejection)
    DPTFAC = 0    # factor for continuum optical depth for rejecting lines (0, no rejection)
    ILNFLG = 0    # flag for binary record of line rejection information (default)
    NMOL_SCAL = 0 # number of molecular profiles to scale (default)
    RECORD = 8 * "{:10.3f}" + "    {:1d}     {:10.3E}   {:2d}"
    RECORD = RECORD.format(V1, V2, SAMPLE, DVSET, ALFAL0, AVMASS, DPTMIN, DPTFAC,
                           ILNFLG, DVOUT, NMOL_SCAL)
    CARD.append(RECORD)

    # RECORD 3.1 — LBLATM - atmospheric and pathlength description
    MODEL  = 0  # User-supplied model
    ITYPE  = 1  # Horizonatal path
    IBMAX  = 0  # Number of layer boundaries (default)
    ZERO   = 0  # Do not zero out absorbers contributing less than 0.1%
    NOPRNT = 0  # Full print out
    NMOL   = C.size # Number of molecules in the HITRAN database
    RECORD = (5 * "{:5d}").format(MODEL, ITYPE, IBMAX, ZERO, NOPRNT, NMOL)
    CARD.append(RECORD)

    # RECORD 3.2 — Slant path geometry
    H1 = 0
    RANGEF = PL
    RECORD = "{:10.3E}                    {:10.3E}".format(H1, RANGEF)
    CARD.append(RECORD)

    # RECORD 3.4 — User-defined atmospheric profile set-up
    RECORD = "    1 (1 homogeneous layer)"
    CARD.append(RECORD)

    # RECORD 3.5 — User-defined atmospheric profile thermodynamic data
    ZM = 0             # [km]
    PM = P / 101325.0  # [atm]
    TM = T - 273.15    # [C]
    RECORD = '{0:10.3E}{1:10.3E}{2:10.3E}     BB L AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    RECORD = RECORD.format(ZM, PM, TM)
    CARD.append(RECORD)

    # RECORD 3.6 — User-defined atmospheric profile species data
    ix0 = 0
    ix1 = 8
    ix1 = min(ix1, NMOL)
    for _ in range(round(NMOL / 8) - 1):
        CARD.append((8*"{:15.8E}").format(*C[ix0:ix1]))
        ix0 += 8
        ix1 += 8
    ix1 = min(ix1, NMOL)
    CARD.append(((ix1-ix0)*"{:15.8E}").format(*C[ix0:ix1]))

    # TERMINATE TAPE5
    CARD.append(r'%%')

    # Write TAPE5 to file
    with open(fname, mode='w') as f:
        f.write('\n'.join(CARD))


def read_tape12(fname="TAPE12"):
    with open(fname, 'rb') as fid:
        _ = np.fromfile(fid, np.dtype('<i4'), count=266)
        test_val = np.fromfile(fid, np.dtype('<i4'), count=1)
        if test_val != 24:
            print('Cannot currently read big-endian OD files.')

    v1, v2 = np.array([], dtype=np.dtype('float64')), np.array([], dtype=np.dtype('float64'))
    dv = np.array([], dtype=np.dtype('float32'))
    N = np.array([], np.dtype('i4'))
    od = np.array([], np.dtype('float32'))

    with open(fname, 'rb') as fid:
        _ = np.fromfile(fid, np.dtype('i4'), count=266)
        nBytes = os.path.getsize(fname)
        while True:
            _ = np.fromfile(fid, np.dtype('i4'), count=1)
            v1 = np.append(v1, np.fromfile(fid, np.dtype('float64'), count=1))
            v2 = np.append(v2, np.fromfile(fid, np.dtype('float64'), count=1))
            dv = np.append(dv, np.fromfile(fid, np.dtype('float32'), count=1))
            N = np.append(N, np.fromfile(fid, np.dtype('i4'), count=1))
            _ = np.fromfile(fid, np.dtype('i4'), count=1)
            L1 = np.fromfile(fid, np.dtype('i4'), count=1)
            if L1 != N[-1] * 4:
                print(f"Internal inconsistency in file {fname}")
                break
            od = np.append(od, np.fromfile(fid, np.dtype('float32'), count=N[-1]))
            L2 = np.fromfile(fid, np.dtype('i4'), count=1)
            if L1 != L2:
                print(f"Internal inconsistency in file {fname}")
                break
            f_loc = fid.tell()
            if f_loc == nBytes:
                break

    nu = np.array([], np.dtype('float64'))
    for V1, V2, n in zip(v1, v2, N):
        nu = np.append(nu, np.linspace(V1, V2, n))

    return nu, od

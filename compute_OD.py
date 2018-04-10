import os.path

import numpy as np

def write_tape5(fname="TAPE5", V1=2000, V2=3333, P=101325, T=296, PL=1,
                mixing_fraction=np.zeros(38), **kwargs):

    # Default options dictionary
    opts = {
        'description': 'TAPE5 for single layer calculation by compute_OD.py',
        'continuum_factors': [1, 1, 1, 1, 1, 1, 1],
        'DVOUT': 0.0025,
        'debug': True,
        'mf_idx': np.zeros(mixing_fraction.shape),
        'mf_val': np.zeros(mixing_fraction.shape),
        }

    opts.update(kwargs) # update options dictionary with user-supplied kwargs
    C = mixing_fraction
    C[opts['mf_idx']] = opts['mf_val']
    print(*C)

    # This will hold each individual record in the "punch card"
    CARD = []

    # RECORD 1.1 — Title
    RECORD = opts["description"]
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
    CF = opts["continuum_factors"]
    RECORD = (len(CF) * "{:8.6f} ".format(*CF)).rstrip()
    CARD.append(RECORD)

    # RECORD 1.3 — spectral range and related details
    SAMPLE = 4    # number of sample points per mean halfwidth (default)
    DVSET  = 0    # [cm^{-1}] selected DV for the final monochromatic calculation (default)
    ALFAL0 = 0.04 # [cm^{-1} / atm] average collision broadened halfwidth (default)
    AVMASS = 36   # [amu] average molecular mass (amu) for Doppler halfwidth (default)
    DPTMIN = 0    # minimum molecular optical depth below which lines will be rejected (0, no rejection)
    DPTFAC = 0    # factor for continuum optical depth for rejecting lines (0, no rejection)
    ILNFLG = 0    # flag for binary record of line rejection information (default)
    DVOUT  = opts["DVOUT"] # [cm^{-1}] selected DV grid for the OD "monochromatic" output spacing
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
    NMOL   = mixing_fraction.size # Number of molecules in the HITRAN database
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
    OD = np.array([], np.dtype('float32'))
    
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
            OD = np.append(OD, np.fromfile(fid, np.dtype('float32'), count=N[-1]))
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
    
    return nu, OD
    


"""
AFIT/CTISR radiative transfer module.

Code for generating the Planckian distribution, brightness temperature, and
radiative transfer calculations using LBLRTM.

Contact information
-------------------
Kevin C. Gross (AFIT/ENP)
Air Force Institute of Technology
Wright-Patterson AFB, Ohio
Kevin.Gross@afit.edu
grosskc.afit@gmail.com
(937) 255-3636 x4558
Version 0.5.5 -- 19-Oct-2018

Version History
---------------
V 0.1.0 05-Jun-2017: Initial code.
V 0.2.0 13-Feb-2018: Added comments.
V 0.3.0 28-Feb-2018: Convert python numeric inputs to numpy arrays. Fixed
  comments to pass muster with the linter. Ensured 1D vectors are really 1D.
  Simplified array reshaping. Changed convention to spectral axis as the *first*
  dimension. Calculations now performed in SI units. Simplified units conversion.
V 0.4.0 28-Feb-2018: Fixed regression for scalar temperature inputs. Added
  make_array convenience function.
V 0.4.1 02-Mar-2018: Updated comments. Removed brackets around T when ensuring T
  a NumPy array. Made plotting function "private".
V 0.4.2 19-Mar-2018: Minor updates to comments and print-out formatting.
V 0.5.0 25-Apr-2018: Major update - added ability to compute transmittance and
  upwelling & downwelling radiance (TUD) using LBLRTM.
V 0.5.1 07-Jun-2018: Added compute_LWIR_apparent_radiance and added option to
  specify output value for brightnessTemperature when an error is encountered.
  Added Altitude option for compute_TUD so that T & U can be computed at multiple
  sensor altitudes for a single atmospheric state. Added option to return surface-
  leaving radiance in compute_LWIR_apparent_radiance. Updated some docstrings.
V 0.5.2 06-Sep-2018: Added BT2L (brightness temperature to radiance) and added
  option to return OD instead of T in computeTUD
V 0.5.3 27-Sep-2018: Added smooth, reduce resolution, and ILS_MAKO functions
V 0.5.4 15-Oct-2018: Removed for-loop in ILS_MAKO; added ability to increase
  resolution for MAKO-like instrument; improved documentation of ILS_MAKO
V 0.5.5 19-Oct-2018: Added ability for ILS_MAKO to return only Y_out
V 0.5.6 04-Dec-2018: Fixed bug with brightness temperature computation when only
  a single spectral location is given. Fixed ILS_MAKO FWHM estimation.

TODO
____
* Add absorption cross-section database handling
* Improve testing
* Make a package with subfolders
"""

# Imports
from io import StringIO
import os, os.path
import inspect
import subprocess
import tempfile

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


# Module constants
# h  = 6.6260689633e-34 # [J s]       - Planck's constant
# c  = 299792458        # [m/s]       - speed of light
# k  = 1.380650424e-23  # [J/K]       - Boltzmann constant
c1 = 1.19104295315e-16  # [J m^2 / s] - 1st radiation constant, c1 = 2*h*c^2
c2 = 1.43877736830e-02  # [m K]       - 2nd radiation constant, c2 = h * c / k

# Define standard atmosphere
StdAtmosCSV = StringIO(
    """
#, Z0 [km], Z1 [km], PL [km], P [Pa], T [K], H2O, CO2, O3, N2O, CO, CH4, O2, N2, Ar
 1,  0.00,  0.10, 0.10, 100697.30, 287.87, 7.657E-03, 3.800E-04, 2.674E-08, 3.202E-07, 1.499E-07, 1.701E-06, 2.092E-01, 7.736E-01, 9.253E-03
 2,  0.10,  0.20, 0.10,  99500.31, 287.23, 7.473E-03, 3.800E-04, 2.701E-08, 3.202E-07, 1.494E-07, 1.701E-06, 2.092E-01, 7.737E-01, 9.255E-03
 3,  0.20,  0.30, 0.10,  98317.27, 286.58, 7.293E-03, 3.801E-04, 2.727E-08, 3.202E-07, 1.488E-07, 1.701E-06, 2.092E-01, 7.739E-01, 9.257E-03
 4,  0.30,  0.40, 0.10,  97148.21, 285.93, 7.118E-03, 3.801E-04, 2.753E-08, 3.202E-07, 1.483E-07, 1.701E-06, 2.092E-01, 7.741E-01, 9.259E-03
 5,  0.40,  0.50, 0.10,  95993.00, 285.27, 6.947E-03, 3.801E-04, 2.781E-08, 3.202E-07, 1.478E-07, 1.701E-06, 2.092E-01, 7.742E-01, 9.261E-03
 6,  0.50,  0.60, 0.10,  94851.97, 284.63, 6.780E-03, 3.801E-04, 2.808E-08, 3.203E-07, 1.473E-07, 1.701E-06, 2.092E-01, 7.744E-01, 9.263E-03
 7,  0.60,  0.70, 0.10,  93723.86, 283.98, 6.617E-03, 3.801E-04, 2.836E-08, 3.203E-07, 1.468E-07, 1.701E-06, 2.092E-01, 7.746E-01, 9.265E-03
 8,  0.70,  0.80, 0.10,  92610.18, 283.33, 6.458E-03, 3.801E-04, 2.863E-08, 3.203E-07, 1.464E-07, 1.701E-06, 2.092E-01, 7.747E-01, 9.267E-03
 9,  0.80,  0.90, 0.10,  91508.80, 282.68, 6.302E-03, 3.801E-04, 2.890E-08, 3.203E-07, 1.459E-07, 1.701E-06, 2.092E-01, 7.749E-01, 9.269E-03
10,  0.90,  1.00, 0.10,  90420.13, 282.02, 6.151E-03, 3.801E-04, 2.919E-08, 3.203E-07, 1.454E-07, 1.701E-06, 2.092E-01, 7.750E-01, 9.270E-03
11,  1.00,  1.25, 0.25,  88520.92, 280.89, 5.876E-03, 3.801E-04, 2.970E-08, 3.203E-07, 1.445E-07, 1.701E-06, 2.092E-01, 7.753E-01, 9.274E-03
12,  1.25,  1.50, 0.25,  85846.37, 279.26, 5.491E-03, 3.801E-04, 3.045E-08, 3.203E-07, 1.432E-07, 1.701E-06, 2.092E-01, 7.757E-01, 9.278E-03
13,  1.50,  1.75, 0.25,  83252.58, 277.64, 5.132E-03, 3.801E-04, 3.121E-08, 3.202E-07, 1.419E-07, 1.701E-06, 2.092E-01, 7.760E-01, 9.283E-03
14,  1.75,  2.00, 0.25,  80737.30, 276.02, 4.796E-03, 3.800E-04, 3.199E-08, 3.202E-07, 1.406E-07, 1.701E-06, 2.092E-01, 7.764E-01, 9.287E-03
15,  2.00,  2.25, 0.25,  78270.48, 274.39, 4.424E-03, 3.800E-04, 3.249E-08, 3.202E-07, 1.394E-07, 1.701E-06, 2.091E-01, 7.768E-01, 9.291E-03
16,  2.25,  2.50, 0.25,  75851.93, 272.77, 4.028E-03, 3.800E-04, 3.269E-08, 3.202E-07, 1.381E-07, 1.701E-06, 2.091E-01, 7.772E-01, 9.296E-03
17,  2.50,  2.75, 0.25,  73508.10, 271.14, 3.667E-03, 3.800E-04, 3.289E-08, 3.202E-07, 1.368E-07, 1.701E-06, 2.091E-01, 7.775E-01, 9.301E-03
18,  2.75,  3.00, 0.25,  71236.71, 269.51, 3.338E-03, 3.799E-04, 3.309E-08, 3.201E-07, 1.356E-07, 1.701E-06, 2.091E-01, 7.779E-01, 9.305E-03
19,  3.00,  3.25, 0.25,  69009.92, 267.89, 3.034E-03, 3.799E-04, 3.328E-08, 3.201E-07, 1.345E-07, 1.701E-06, 2.091E-01, 7.782E-01, 9.308E-03
20,  3.25,  3.50, 0.25,  66826.91, 266.27, 2.754E-03, 3.799E-04, 3.345E-08, 3.201E-07, 1.336E-07, 1.701E-06, 2.091E-01, 7.785E-01, 9.312E-03
21,  3.50,  4.00, 0.50,  63702.84, 263.84, 2.385E-03, 3.799E-04, 3.371E-08, 3.201E-07, 1.322E-07, 1.701E-06, 2.091E-01, 7.788E-01, 9.316E-03
22,  4.00,  4.50, 0.50,  59690.78, 260.59, 1.942E-03, 3.799E-04, 3.480E-08, 3.201E-07, 1.310E-07, 1.701E-06, 2.091E-01, 7.793E-01, 9.321E-03
23,  4.50,  5.00, 0.50,  55886.00, 257.34, 1.563E-03, 3.800E-04, 3.670E-08, 3.202E-07, 1.306E-07, 1.701E-06, 2.091E-01, 7.796E-01, 9.325E-03
24,  5.00,  5.50, 0.50,  52281.07, 254.09, 1.264E-03, 3.800E-04, 3.853E-08, 3.202E-07, 1.300E-07, 1.701E-06, 2.091E-01, 7.799E-01, 9.329E-03
25,  5.50,  6.00, 0.50,  48866.29, 250.84, 1.029E-03, 3.799E-04, 4.024E-08, 3.201E-07, 1.292E-07, 1.701E-06, 2.091E-01, 7.802E-01, 9.332E-03
26,  6.00,  6.50, 0.50,  45636.13, 247.59, 8.239E-04, 3.800E-04, 4.322E-08, 3.202E-07, 1.278E-07, 1.701E-06, 2.091E-01, 7.804E-01, 9.334E-03
27,  6.50,  7.00, 0.50,  42581.36, 244.34, 6.479E-04, 3.801E-04, 4.771E-08, 3.202E-07, 1.258E-07, 1.701E-06, 2.092E-01, 7.805E-01, 9.336E-03
28,  7.00,  7.50, 0.50,  39693.21, 241.09, 5.139E-04, 3.801E-04, 5.237E-08, 3.203E-07, 1.232E-07, 1.700E-06, 2.092E-01, 7.806E-01, 9.337E-03
29,  7.50,  8.00, 0.50,  36963.39, 237.84, 4.114E-04, 3.801E-04, 5.715E-08, 3.202E-07, 1.201E-07, 1.699E-06, 2.092E-01, 7.807E-01, 9.338E-03
30,  8.00,  8.50, 0.50,  34390.29, 234.59, 3.003E-04, 3.800E-04, 6.653E-08, 3.201E-07, 1.163E-07, 1.697E-06, 2.092E-01, 7.808E-01, 9.340E-03
31,  8.50,  9.00, 0.50,  31965.46, 231.34, 1.973E-04, 3.800E-04, 8.247E-08, 3.199E-07, 1.117E-07, 1.695E-06, 2.091E-01, 7.809E-01, 9.341E-03
32,  9.00,  9.50, 0.50,  29682.01, 228.11, 1.303E-04, 3.800E-04, 1.004E-07, 3.193E-07, 1.070E-07, 1.692E-06, 2.092E-01, 7.810E-01, 9.342E-03
33,  9.50, 10.00, 0.50,  27532.14, 224.91, 8.664E-05, 3.800E-04, 1.202E-07, 3.185E-07, 1.021E-07, 1.688E-06, 2.092E-01, 7.810E-01, 9.342E-03
34, 10.00, 10.50, 0.50,  25510.85, 221.69, 5.972E-05, 3.800E-04, 1.488E-07, 3.172E-07, 9.714E-08, 1.684E-06, 2.092E-01, 7.811E-01, 9.343E-03
35, 10.50, 11.00, 0.50,  23610.99, 218.44, 4.292E-05, 3.800E-04, 1.904E-07, 3.152E-07, 9.214E-08, 1.679E-06, 2.091E-01, 7.811E-01, 9.343E-03
36, 11.00, 11.50, 0.50,  21842.58, 216.78, 3.101E-05, 3.800E-04, 2.356E-07, 3.131E-07, 8.673E-08, 1.673E-06, 2.091E-01, 7.811E-01, 9.343E-03
37, 11.50, 12.00, 0.50,  20192.60, 216.73, 2.252E-05, 3.800E-04, 2.828E-07, 3.108E-07, 8.098E-08, 1.666E-06, 2.091E-01, 7.811E-01, 9.343E-03
38, 12.00, 12.50, 0.50,  18667.33, 216.70, 1.665E-05, 3.800E-04, 3.269E-07, 3.085E-07, 7.439E-08, 1.659E-06, 2.091E-01, 7.811E-01, 9.343E-03
39, 12.50, 13.00, 0.50,  17257.33, 216.70, 1.256E-05, 3.800E-04, 3.645E-07, 3.062E-07, 6.719E-08, 1.650E-06, 2.091E-01, 7.811E-01, 9.343E-03
40, 13.00, 13.50, 0.50,  15953.85, 216.70, 9.388E-06, 3.800E-04, 4.115E-07, 3.038E-07, 6.018E-08, 1.641E-06, 2.091E-01, 7.811E-01, 9.343E-03
41, 13.50, 14.00, 0.50,  14748.85, 216.70, 6.938E-06, 3.800E-04, 4.706E-07, 3.013E-07, 5.343E-08, 1.632E-06, 2.091E-01, 7.811E-01, 9.344E-03
42, 14.00, 14.50, 0.50,  13634.79, 216.70, 5.688E-06, 3.800E-04, 5.366E-07, 2.987E-07, 4.738E-08, 1.622E-06, 2.091E-01, 7.812E-01, 9.344E-03
43, 14.50, 15.00, 0.50,  12604.79, 216.70, 5.224E-06, 3.800E-04, 6.102E-07, 2.959E-07, 4.196E-08, 1.611E-06, 2.091E-01, 7.812E-01, 9.344E-03
44, 15.00, 16.00, 1.00,  11230.00, 216.70, 4.471E-06, 3.800E-04, 7.526E-07, 2.913E-07, 3.500E-08, 1.595E-06, 2.091E-01, 7.811E-01, 9.344E-03
45, 16.00, 17.00, 1.00,   9600.00, 216.70, 3.904E-06, 3.800E-04, 1.017E-06, 2.833E-07, 2.778E-08, 1.569E-06, 2.091E-01, 7.811E-01, 9.343E-03
46, 17.00, 18.00, 1.00,   8207.50, 216.70, 3.840E-06, 3.800E-04, 1.373E-06, 2.730E-07, 2.225E-08, 1.538E-06, 2.091E-01, 7.811E-01, 9.344E-03
47, 18.00, 19.00, 1.00,   7016.00, 216.70, 3.839E-06, 3.800E-04, 1.795E-06, 2.602E-07, 1.756E-08, 1.502E-06, 2.091E-01, 7.811E-01, 9.344E-03
48, 19.00, 20.00, 1.00,   5998.00, 216.70, 3.876E-06, 3.800E-04, 2.288E-06, 2.449E-07, 1.441E-08, 1.453E-06, 2.091E-01, 7.812E-01, 9.344E-03
49, 20.00, 22.00, 2.00,   4788.42, 217.57, 3.976E-06, 3.800E-04, 3.036E-06, 2.210E-07, 1.260E-08, 1.356E-06, 2.091E-01, 7.811E-01, 9.344E-03
50, 22.00, 24.00, 2.00,   3509.77, 219.55, 4.187E-06, 3.800E-04, 4.124E-06, 1.970E-07, 1.307E-08, 1.197E-06, 2.091E-01, 7.812E-01, 9.344E-03
51, 24.00, 26.00, 2.00,   2580.36, 221.54, 4.406E-06, 3.800E-04, 5.042E-06, 1.774E-07, 1.480E-08, 1.067E-06, 2.091E-01, 7.811E-01, 9.343E-03
52, 26.00, 28.00, 2.00,   1902.99, 223.47, 4.544E-06, 3.800E-04, 5.650E-06, 1.625E-07, 1.577E-08, 1.002E-06, 2.091E-01, 7.811E-01, 9.343E-03
53, 28.00, 30.00, 2.00,   1407.21, 225.45, 4.664E-06, 3.800E-04, 6.233E-06, 1.487E-07, 1.663E-08, 9.445E-07, 2.091E-01, 7.811E-01, 9.344E-03
54, 30.00, 32.00, 2.00,   1043.50, 227.67, 4.707E-06, 3.754E-04, 6.775E-06, 1.301E-07, 1.741E-08, 8.711E-07, 2.066E-01, 7.837E-01, 9.374E-03
55, 32.00, 34.00, 2.00,    776.75, 230.94, 4.723E-06, 3.708E-04, 7.267E-06, 1.092E-07, 1.834E-08, 7.949E-07, 2.040E-01, 7.862E-01, 9.404E-03
56, 34.00, 36.00, 2.00,    581.43, 236.36, 4.895E-06, 3.797E-04, 7.779E-06, 9.249E-08, 2.009E-08, 7.467E-07, 2.089E-01, 7.813E-01, 9.346E-03
57, 36.00, 38.00, 2.00,    438.19, 241.88, 5.047E-06, 3.881E-04, 7.968E-06, 7.357E-08, 2.221E-08, 6.939E-07, 2.136E-01, 7.767E-01, 9.291E-03
58, 38.00, 40.00, 2.00,    332.42, 247.45, 5.057E-06, 3.846E-04, 7.603E-06, 5.418E-08, 2.407E-08, 6.112E-07, 2.116E-01, 7.786E-01, 9.314E-03
59, 40.00, 42.00, 2.00,    253.74, 253.02, 5.076E-06, 3.800E-04, 6.869E-06, 3.764E-08, 2.620E-08, 5.233E-07, 2.091E-01, 7.811E-01, 9.343E-03
60, 42.00, 46.00, 4.00,    175.77, 260.90, 5.190E-06, 3.800E-04, 5.685E-06, 2.127E-08, 3.045E-08, 4.088E-07, 2.091E-01, 7.811E-01, 9.344E-03
61, 46.00, 50.00, 4.00,    105.70, 269.76, 5.244E-06, 3.800E-04, 3.967E-06, 8.751E-09, 3.879E-08, 2.693E-07, 2.092E-01, 7.811E-01, 9.343E-03
62, 50.00, 54.00, 4.00,     63.91, 267.03, 5.182E-06, 3.800E-04, 2.559E-06, 4.036E-09, 5.286E-08, 1.926E-07, 2.091E-01, 7.811E-01, 9.344E-03
63, 54.00, 58.00, 4.00,     38.42, 258.35, 5.036E-06, 3.800E-04, 1.672E-06, 2.838E-09, 7.266E-08, 1.633E-07, 2.091E-01, 7.811E-01, 9.343E-03
64, 58.00, 62.00, 4.00,     22.59, 247.46, 4.745E-06, 3.801E-04, 1.129E-06, 2.109E-09, 1.071E-07, 1.518E-07, 2.092E-01, 7.811E-01, 9.343E-03
65, 62.00, 66.00, 4.00,     12.99, 236.49, 4.323E-06, 3.800E-04, 7.779E-07, 1.629E-09, 1.649E-07, 1.501E-07, 2.092E-01, 7.811E-01, 9.343E-03
66, 66.00, 70.00, 4.00,      7.30, 225.53, 3.796E-06, 3.801E-04, 4.425E-07, 1.297E-09, 2.482E-07, 1.501E-07, 2.092E-01, 7.811E-01, 9.343E-03
"""
)
StdAtmos = np.loadtxt(StdAtmosCSV, delimiter=",", skiprows=1)

# Define default options dictionary
LBL_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LBLRTM = os.path.join(LBL_dir, "lblrtm_v12.8_OS_X_gnu_sgl")
TAPE3 = os.path.join(LBL_dir, "AER-v3.6-0500-6000.tp3")
options = {
    # options for write_tape5
    "V1": 2000.00,  # [cm^{-1}]
    "V2": 3333.33,  # [cm^{-1}]
    "T": 296.0,  # [K]
    "P": 101325.0,  # [Pa]
    "PL": 1.0,  # [km]
    "MF": np.zeros(38),  # [ppmv]
    "MF_ID": np.array([]),
    "MF_VAL": np.array([]),
    "continuum_factors": np.zeros(7),
    "continuum_override": False,
    "description": "TAPE5 for single layer calculation by compute_OD.py",
    "DVOUT": 0.0005,  # [cm^{-1}]
    # options for run_LBLRTM
    "debug": True,
    "LBL_dir": LBL_dir,
    "LBLRTM": LBLRTM,
    "TAPE3": TAPE3,
    # options for compute_TUD
    "Zs": StdAtmos[:, 1],  # [km]
    "Ts": StdAtmos[:, 5],  # [K]
    "Ps": StdAtmos[:, 4],  # [Pa]
    "PLs": StdAtmos[:, 3],  # [km]
    "MFs_VAL": StdAtmos[:, 6:14] * 1e6,  # [ppmv]
    "MFs_ID": np.array([1, 2, 3, 4, 5, 6, 7, 22]),
    "theta_r": 0,
    "N_angle": 30,
    "Altitudes": np.asarray([500]),
    "save": False,
    "returnOD": False,
}


def rs1D(y):
    """
    Reshape ND-array into a 1D vector.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 1D version of the array

    """
    y = np.array(y)
    dims = y.shape
    return y.flatten(), dims


def rs2D(y):
    """
    Convert ND-array to 2D by flattening 2nd thru Nth dimensions.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array

    Returns
    _______
    y : numpy array
      A reshaped, 2D version of the array

    """
    y = np.array(y)
    if y.ndim < 2:
        y = np.array([y]).flatten()
        y = y[np.newaxis, :]  # per convention, return as row vector
        return y, y.shape
    else:
        dims = y.shape
        return y.reshape((dims[0], np.prod(dims[1:]))), dims


def rsND(y, dims):
    """
    Reshape a 1D- or 2D-array back into an ND-array.

    Parameters
    __________
    y : numpy array
      A multi-dimensional array
    dims : tuple
      The shape of the ND-array

    Returns
    _______
    y : numpy array
      A reshaped, ND version of the array

    """
    return y.reshape(dims)


def make_spectral_axis(Xmin, Xmax, DVOUT):
    """
    Generate a spectral axis between `Xmin` and `Xmax` with spacing `DVOUT`.

    Parameters
    __________
    Xmin: float
      lower spectral bound, wavenumbers [cm^{-1}]
    Xmax: float
      upper spectral bound, wavenumbers [cm^{-1}]
    DVOUT: float
      spectral axis spacing, wavenumbers [cm^{-1}]

    Returns
    _______
    X: array-like
      spectral axis, wavenumbers [cm^{-1}]
    """
    nX = np.ceil((Xmax - Xmin) / DVOUT)
    X_ = np.linspace(Xmin, Xmax, nX)
    return X_


def compute_TUD(Xmin, Xmax, opts=options, **kwargs):
    """
    Compute monochromatic transmittance and upwelling/downwelling radiance (TUD).

    Parameters
    __________
    Xmin: float
      lower spectral bound, wavenumbers [cm^{-1}]
    Xmax: float
      upper spectral bound, wavenumbers [cm^{-1}]
    opts: dictionary, optional
      options dictionary with keys/vals defining various aspects of the calc;
        see code for defaults
    kwargs: named parameters, optional
      can use named parameters to over-ride defaults in options dictionary

    Returns
    _______
    X: array_like
      spectral axis, wavenumbers [cm^{-1}]
    tau: array_like
      transmittance, unitless
    Lu: array_like
      upwelling (path) radiance, [µW/(cm^2·sr·cm^{-1})]
    Ld: array_like
      downwelling radiance, [µW/(cm^2·sr·cm^{-1})]
    """

    # Update opts dictionary and extract atmospheric variables
    opts.update(kwargs)
    Z = opts["Zs"]
    T = opts["Ts"]
    P = opts["Ps"]
    PL = opts["PLs"]
    MF = opts["MFs_VAL"]
    ID = opts["MFs_ID"]
    nL = T.size
    nA = opts["N_angle"]
    Z_s = opts["Altitudes"]  # [km] sensor altitude
    mu_s = 1.0 / np.cos(opts["theta_r"])
    returnOD = opts["returnOD"]

    # Ensure Z_s and mu are numpy arrays
    f = lambda x: np.array([x]).ravel()
    Z_s, mu_s = tuple(map(f, [Z_s, mu_s]))

    # Preallocate arrays
    X_ = make_spectral_axis(Xmin, Xmax, opts["DVOUT"])
    OD = np.zeros((X_.size, nL))
    Lu_ = np.zeros((X_.size, Z_s.size, mu_s.size))
    Ld_ = np.zeros((X_.size, nA))
    tau_ = Lu_.copy()

    # Compute OD's and Planckian distribution for each layer
    for ii in np.arange(nL):
        _, OD[:, ii] = compute_OD(
            Xmin,
            Xmax,
            opts=options,
            T=T[ii],
            P=P[ii],
            PL=PL[ii],
            MF_VAL=MF[ii, :],
            MF_ID=ID,
        )
        print(f"Computing layer {ii+1:3d} of {nL:3d}")
    B = planckian(X_, T)

    # transmittance and upwelling
    print("Computing transmittance and upwelling")
    if returnOD:
        print("Returning optical depth in place of transmittance")
    for ii, zs in enumerate(Z_s):
        for jj, mu in enumerate(mu_s):
            ix = Z <= zs
            if returnOD:
                tau_[:, ii, jj] = np.sum(OD[:, ix] * mu, axis=1)
            else:
                tau_[:, ii, jj] = np.exp(-1.0 * np.sum(OD[:, ix] * mu, axis=1))
            nL = np.sum(ix)
            for kk in np.arange(nL):
                t = np.exp(-OD[:, kk] * mu)
                Lu_[:, ii, jj] = t * Lu_[:, ii, jj] + (1 - t) * B[:, kk]
    if (len(Z_s) == 1) and (len(mu_s) == 1):
        tau_ = tau_[:, 0, 0]
        Lu_ = Lu_[:, 0, 0]
    if (len(Z_s) == 1) and not (len(mu_s) == 1):
        tau_ = tau_[:, 0, :]
        Lu_ = Lu_[:, 0, :]
    if not (len(Z_s) == 1) and (len(mu_s) == 1):
        tau_ = tau_[:, :, 0]
        Lu_ = Lu_[:, :, 0]

    print("Computing downwelling")
    angles = np.linspace(0, np.pi / 2.0, nA, endpoint=False)
    for ii, th in enumerate(angles):
        for jj in np.arange(nL)[::-1]:
            t = np.exp(-OD[:, jj] / np.cos(th))
            Ld_[:, ii] = t * Ld_[:, ii] + (1 - t) * B[:, jj]
        print(f"Computing angle {ii+1:3d} of {nA:3d}")
    if opts["save"]:
        np.savez(
            "ComputeTUD.npz",
            OD=OD,
            B=B,
            tau=tau_,
            Ld=Ld_,
            Lu=Lu_,
            X=X_,
            angles=angles,
            Z_s=Z_s,
            mu_s=mu_s,
        )
    cos_dOmega = np.cos(angles) * np.sin(angles)
    Ld_ = np.sum(Ld_ * cos_dOmega, axis=1) / np.sum(cos_dOmega)
    Ld_ = Ld_.flatten()

    # Return XTUD
    return X_, tau_, Lu_, Ld_


def compute_OD(Xmin_in, Xmax_in, opts=options, **kwargs):
    """
    Computes the high-resolution ("monochromatic") optical depth of a single,
    homogeneous, non-scattering gaseous layer using LBLRTM. The transmittance T
    of the layer is given by T = exp(-OD).

    Parameters
    __________
    Xmin: float
      lower spectral bound for OD computation, wavenumbers [cm^{-1}]
    Xmax: float
      upper spectral bound for OD computation, wavenumbers [cm^{-1}]
    opts: dictionary, optional
      options dictionary with keys/vals defining various aspects of the calc;
        see code for defaults
    kwargs: named parameters, optional
      can use named parameters to over-ride defaults in options dictionary

    Returns
    _______
    X: array_like
      spectral axis, wavenumbers [cm^{-1}]
    OD: array_like
      optical depth, unitless
    """
    # Update opts dictionary and pull out required values
    opts.update(kwargs)
    DVOUT = opts.get("DVOUT", 0.025)

    # Set up parameters for looping over spectral range in 2020/cm chunks
    myround = lambda x: float("{0:10.3f}".format(x))
    pad = 25  # padding around each spectral bin that is trimmed from every run
    olp = 5  # overlap between spectral bins for averaging OD
    Xmin = np.max([myround(Xmin_in - pad - olp), 0])
    Xmax = myround(Xmax_in + pad + olp)
    maxBW = 2020 - olp - 2 * pad
    nBand = int(np.ceil((Xmax - Xmin) / maxBW))
    nPts = int(np.floor(maxBW / DVOUT))

    # Compute OD for each spectral chunk
    X = []
    OD = []
    for ii in range(nBand):
        if ii > 0:
            Xmin = myround(np.max(X[ii - 1]) - olp - pad)
        Xmax1 = np.min([Xmax + pad, myround(Xmin + DVOUT * (nPts - 1) + olp + pad)])
        nu, od = run_LBLRTM(Xmin, Xmax1, opts=opts)
        XX = make_spectral_axis(Xmin + pad, Xmax1 - pad, DVOUT)
        X.append(XX)
        OD.append(np.interp(XX, nu, od))

    # Stitch chunks together into single output vector
    N = np.ceil((Xmax_in - Xmin_in) / DVOUT)
    X_out = np.linspace(Xmin_in, Xmax_in, N)
    OD_out = np.zeros((nBand, X_out.size))
    for ii in range(nBand):
        OD_out[ii, :] = np.interp(X_out, X[ii], OD[ii], left=0, right=0)
    nrm = np.sum(OD_out > 0, axis=0)
    nrm[nrm < 1] = 1
    OD_out = np.sum(OD_out, axis=0) / nrm
    OD_out = OD_out.flatten()
    return X_out, OD_out


def run_LBLRTM(V1, V2, opts=options, **kwargs):
    """
    Runs LBLRTM OD calc based on settings passed in `opts` and by `kwargs`.

    Parameters
    __________
    V1: float
      lower spectral bound for OD computation, wavenumbers [cm^{-1}]
    V2: float
      upper spectral bound for OD computation, wavenumbers [cm^{-1}]
    opts: dictionary, optional
      options dictionary with keys/vals defining various aspects of the calc;
        see code for defaults
    kwargs: named parameters, optional
      can use named parameters to over-ride defaults in options dictionary

    Returns
    _______
    nu: array-like
      spectral axis, wavenumbers [cm^{-1}]
    od: array-like
      optical depth, unitless
    """
    # Update opts dictionary and override spectral limits
    opts.update(kwargs)
    opts["V1"] = V1
    opts["V2"] = V2

    # perform LBLRTM calculation in temporary directory
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        os.symlink(opts.get("TAPE3"), "TAPE3")
        os.symlink(opts.get("LBLRTM"), "lblrtm")
        write_tape5(fname="TAPE5", **opts)
        ex = subprocess.run("./lblrtm", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ex.stderr == b"STOP  LBLRTM EXIT \n":
            nu, od = read_tape12()
        else:
            print(ex.stderr)
            nu, od = [], []
        os.chdir(cwd)
    return nu, od


def write_tape5(fname="TAPE5", opts=options, **kwargs):
    """
    Write an LBLRTM TAPE5 file for an optical depth calculation.

    Parameters
    __________
    fname: string, optional, {"TAPE5"}
      name of file to write to
    opts: dictionary, optional
      options dictionary with keys/vals defining various aspects of the calc;
        see code for defaults
    kwargs: named parameters, optional
      can use named parameters to over-ride defaults in options dictionary

    Returns
    _______
    None
    """
    # Extract required values with reasonable defaults
    opts.update(kwargs)  # update opts dictionary with user-supplied keys/vals
    V1 = opts.get("V1", 2000.00)  # [cm^{-1}]
    V2 = opts.get("V2", 3333.33)  # [cm^{-1}]
    DVOUT = opts.get("DVOUT", 0.0025)  # [cm^{-1}]
    T = opts.get("T", 296.0)  # [K]
    P = opts.get("P", 101325.0)  # [Pa]
    PL = opts.get("PL", 1.0)  # [km]
    CF = opts.get("continuum_factors", np.zeros(7))

    # Update mixing fraction
    C = opts.get("MF", np.zeros(38))
    if "MF_ID" in opts.keys() and "MF_VAL" in opts.keys():
        idx = [i - 1 for i in list(opts["MF_ID"])]
        C[idx] = opts["MF_VAL"]  # [ppmv]

    # Update mixing fraction via molecule name specification
    hitranMolecules = [
        "H2O",
        "CO2",
        "O3",
        "N2O",
        "CO",
        "CH4",
        "O2",
        "NO",
        "SO2",
        "NO2",
        "NH3",
        "HNO3",
        "OH",
        "HF",
        "HCl",
        "HBr",
        "HI",
        "ClO",
        "OCS",
        "H2CO",
        "HOCl",
        "N2",
        "HCN",
        "CH3Cl",
        "H2O2",
        "C2H2",
        "C2H6",
        "PH3",
        "COF2",
        "SF6",
        "H2S",
        "HCOOH",
        "HO2",
        "O+",
        "ClONO2",
        "NO+",
        "HOBr",
        "C2H4",
    ]
    mol_ix, mol_key = [], []
    for k in opts.keys():
        # index in hitranMolecule list that matches the molecule specified in opts
        loc = [i for i, j in enumerate(hitranMolecules) if j.upper() == k.upper()]
        if loc:  # if loc is not empty
            mol_ix.append(loc)  # add the molecule index
            mol_key.append(k)  # store the name so we can retrieve it later
    mol_ix = np.asarray(mol_ix).flatten()
    for i, k in enumerate(mol_key):
        C[mol_ix[i]] = opts[k]

    # Ensure only present species have continuum effects included
    if not opts.get("continuum_override", False):
        if C[0] > 0:
            CF[[0, 1]] = 1
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
    RECORD = opts.get(
        "description", "TAPE5 for single layer calculation by compute_OD.py"
    )
    CARD.append(RECORD)
    CARD.append(
        "         1         2         3         4         5         6         7         8         9         0"
    )
    CARD.append(
        "123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789"
    )
    CARD.append("$ None")

    # RECORD 1.2 — General LBLRTM control — set up for single-layer OD calc
    IHIRAC = 1  # Voigt line profile
    ILBLF4 = 1  # Line-by-line function
    ICNTNM = 6  # User-supplied continuum scale factors
    IAERSL = 0  # No aerosols used in calculation
    IEMIT = 0  # Optical depth only
    ISCAN = 0  # No scanning / interpolation used
    IFILTR = 0  # No filter
    IPLOT = 0  # No plot
    ITEST = 0  # No test
    IATM = 1  # Use LBLATM (RECORD 1.3)
    IMRG = 0  # Normal merge
    ILAS = 0  # Not for laser calculation
    IOD = 1  # Normal calculation when layering multiple OD calculations
    IXSECT = 0  # No cross-sections included in calculation
    MPTS = 0
    NPTS = 0
    RECORD = " HI={:1d} F4={:1d} CN={:1d} AE={:1d} EM={:1d} SC={:1d} FI={:1d} PL={:1d}"
    RECORD += " TS={:1d} AM={:1d} MG={:1d} LA={:1d} MS={:1d} XS={:1d}  {:2d}  {:2d}"
    RECORD = RECORD.format(
        IHIRAC,
        ILBLF4,
        ICNTNM,
        IAERSL,
        IEMIT,
        ISCAN,
        IFILTR,
        IPLOT,
        ITEST,
        IATM,
        IMRG,
        ILAS,
        IOD,
        IXSECT,
        MPTS,
        NPTS,
    )
    CARD.append(RECORD)

    # RECORD 1.2a — continuum scale factors
    RECORD = ((len(CF) * "{:8.6f} ").format(*CF)).rstrip()
    CARD.append(RECORD)

    # RECORD 1.3 — spectral range and related details
    SAMPLE = 4  # number of sample points per mean halfwidth (default)
    DVSET = 0  # [cm^{-1}] selected DV for the final monochromatic calculation (default)
    ALFAL0 = 0.04  # [cm^{-1} / atm] average collision broadened halfwidth (default)
    AVMASS = 36  # [amu] average molecular mass (amu) for Doppler halfwidth (default)
    DPTMIN = (
        0
    )  # minimum molecular optical depth below which lines will be rejected (0, no rejection)
    DPTFAC = (
        0
    )  # factor for continuum optical depth for rejecting lines (0, no rejection)
    ILNFLG = 0  # flag for binary record of line rejection information (default)
    NMOL_SCAL = 0  # number of molecular profiles to scale (default)
    RECORD = 8 * "{:10.3f}" + "    {:1d}     {:10.3E}   {:2d}"
    RECORD = RECORD.format(
        V1, V2, SAMPLE, DVSET, ALFAL0, AVMASS, DPTMIN, DPTFAC, ILNFLG, DVOUT, NMOL_SCAL
    )
    CARD.append(RECORD)

    # RECORD 3.1 — LBLATM - atmospheric and pathlength description
    MODEL = 0  # User-supplied model
    ITYPE = 1  # Horizonatal path
    IBMAX = 0  # Number of layer boundaries (default)
    ZERO = 0  # Do not zero out absorbers contributing less than 0.1%
    NOPRNT = 0  # Full print out
    NMOL = C.size  # Number of molecules in the HITRAN database
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
    ZM = 0  # [km]
    PM = P / 101325.0  # [atm]
    TM = T - 273.15  # [C]
    RECORD = (
        "{0:10.3E}{1:10.3E}{2:10.3E}     BB L AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    RECORD = RECORD.format(ZM, PM, TM)
    CARD.append(RECORD)

    # RECORD 3.6 — User-defined atmospheric profile species data
    ix0 = 0
    ix1 = 8
    ix1 = min(ix1, NMOL)
    for _ in range(round(NMOL / 8) - 1):
        CARD.append((8 * "{:15.8E}").format(*C[ix0:ix1]))
        ix0 += 8
        ix1 += 8
    ix1 = min(ix1, NMOL)
    CARD.append(((ix1 - ix0) * "{:15.8E}").format(*C[ix0:ix1]))

    # TERMINATE TAPE5
    CARD.append(r"%%")

    # Write TAPE5 to file
    with open(fname, mode="w") as f:
        f.write("\n".join(CARD))


def read_tape12(fname="TAPE12"):
    """
    Reads single-precision, little-endian LBLRTM OD TAPE12 file.

    Parameters
    __________
    fname: string, optional, {"TAPE12"}
      filename of the TAPE12-formatted file to read in

    Returns
    _______
    nu, array-like
      spectral axis, wavenumbers [cm^{-1}]
    od, array-like
      optical depth, unitless
    """
    # This is a python port of the MATLAB code by Xianglei Huang provided on the
    # file exchange: <http://www.mathworks.com/matlabcentral/fileexchange/8467>
    with open(fname, "rb") as fid:
        _ = np.fromfile(fid, np.dtype("<i4"), count=266)
        test_val = np.fromfile(fid, np.dtype("<i4"), count=1)
        if test_val != 24:
            print("Cannot currently read big-endian OD files.")

    v1, v2 = (
        np.array([], dtype=np.dtype("float64")),
        np.array([], dtype=np.dtype("float64")),
    )
    dv = np.array([], dtype=np.dtype("float32"))
    N = np.array([], np.dtype("i4"))
    od = np.array([], np.dtype("float32"))

    with open(fname, "rb") as fid:
        _ = np.fromfile(fid, np.dtype("i4"), count=266)
        nBytes = os.path.getsize(fname)
        while True:
            _ = np.fromfile(fid, np.dtype("i4"), count=1)
            v1 = np.append(v1, np.fromfile(fid, np.dtype("float64"), count=1))
            v2 = np.append(v2, np.fromfile(fid, np.dtype("float64"), count=1))
            dv = np.append(dv, np.fromfile(fid, np.dtype("float32"), count=1))
            N = np.append(N, np.fromfile(fid, np.dtype("i4"), count=1))
            _ = np.fromfile(fid, np.dtype("i4"), count=1)
            L1 = np.fromfile(fid, np.dtype("i4"), count=1)
            if L1 != N[-1] * 4:
                print(f"Internal inconsistency in file {fname}")
                break
            od = np.append(od, np.fromfile(fid, np.dtype("float32"), count=N[-1]))
            L2 = np.fromfile(fid, np.dtype("i4"), count=1)
            if L1 != L2:
                print(f"Internal inconsistency in file {fname}")
                break
            f_loc = fid.tell()
            if f_loc == nBytes:
                break

    nu = np.array([], np.dtype("float64"))
    for V1, V2, n in zip(v1, v2, N):
        nu = np.append(nu, np.linspace(V1, V2, n))

    return nu, od


def planckian(X_in, T_in, wavelength=False):
    """
    Compute the Planckian spectral radiance distribution.

    Computes the spectral radiance `L` at wavenumber(s) `X` for a system at
    temperature(s) `T` using Planck's distribution function. `X` must be a scalar
    or a vector. `T` can be of arbitrary dimensions. The shape of output `L` will
    be `(X.size, *T.shape)`.

    Parameters
    __________
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    T : array_like
      temperature array, Kelvin [K], arbitrary dimensions
    wavelength : logical
      if true, interprets spectral input `X` as wavelength [micron, µm]

    Returns
    _______
    L : array_like
      spectral radiance in [µW/(cm^2·sr·cm^-1)], or if wavelength=True,
      spectral radiance in [µW/(cm^2·sr·µm)] (microflick, µF)

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import radiative_transfer as rt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = rt.planckian(X,T)
    >>> plt.plot(X,L)
    """
    # Ensure inputs are NumPy arrays
    X = np.asarray(np.copy(X_in)).flatten()  # X must be 1D array
    T = np.asarray(np.copy(T_in))

    # Make X a column vector and T a row vector for broadcasting into 2D arrays
    X = X[:, np.newaxis]
    dimsT = T.shape  # keep shape info for later reshaping into ND array
    T = T.flatten()[np.newaxis, :]

    # Compute Planck's spectral radiance distribution
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not wavelength:
            print("Assumes X given in µm; returning L in µF")
        X *= 1e-6  # convert to m from µm
        L = c1 / (X ** 5 * (np.exp(c2 / (X * T)) - 1))  # [W/(m^2 sr m)] SI
        L *= 1e-4  # convert to [µW/(cm^2 sr µm^{-1})]
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L = c1 * X ** 3 / (np.exp(c2 * X / T) - 1)  # [W/(m^2 sr m^{-1})]
        L *= 1e4  # convert to [µW/(cm^2 sr cm^{-1})] (1e6 / 1e2)

    # Reshape L if necessary and return
    return np.reshape(L, (X.size, *dimsT))


def brightnessTemperature(
    X_in, L_in, wavelength=False, bad_value=np.nan, spectral_dim=0
):
    """
    Compute brightness temperature at given spectral radiance.

    The brightness temperature is the temperature at which a perfect blackbody
    would need to be to produce the same spectral radiance L at each specified
    wavenumber X. The shape of output T will be ``(X.size, *L.shape)``.

    Parameters
    __________
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    L : array_like
      spectral radiance in [µW/(cm^2·sr·cm^-1)], arbitrary dimensions with
      spectral dimension first
    wavelength : logical
      if true, interprets spectral input `X` in wavelength [micron, µm]
      and spectral radiance `L` in [µW/(cm^2·sr·µm)] (microflick, µF)
    bad_value : value to use when an unphysical brightness temperature is
      computed. Default is np.nan.

    Returns
    _______
    T : numpy array
      brightness temperature in [K]

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import radiative_transfer as rt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = rt.planckian(X,T)
    >>> T = rt.brightnessTemperature(X,L)
    >>> plt.plot(X,T)
    """
    # Ensure inputs are NumPy arrays
    X = np.asarray(np.copy(X_in)).flatten()  # X must be 1D array
    L = np.asarray(np.copy(L_in))

    # Swap axes, if necessary
    if spectral_dim != 0:
        L = np.swapaxes(L, 0, spectral_dim)

    # Ensure X is row vector for outer products
    X = X[:, np.newaxis]

    # Make L a column vector or 2D array w/ spectral axis as 1st dimension
    if L.ndim == 1:  # if it is a vector, must be same shape as X
        L = L[:, np.newaxis]
        dimsL = L.shape
    else:  # otherwise collapse / reshape with 1st dimension corresponding to X
        dimsL = L.shape
        L = L.reshape((dimsL[0], np.prod(dimsL[1:])))

    # Evaluate brightness temperature
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not wavelength:
            print("Assumes X given in µm and L given in µF")
        X *= 1e-6  # convert to m from µm
        L *= 1e4  # convert to SI units, [W/(m^2 sr m)] from [µW/(cm^2 sr µm)]
        T = c2 / (X * np.log(1 + c1 / (X ** 5 * L)))
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L *= 1e-4  # convert to [W/(m^2 sr m^{-1})] from [µW/(cm^2 sr cm^{-1})]
        T = c2 * X / np.log(c1 * X ** 3 / L + 1)

    # NaN-ify garbage results
    ixBad = ~np.isfinite(L) | (np.real(L) <= 0) | (np.abs(np.imag(T)) > 0)
    T[ixBad] = bad_value

    # Reshape T if necessary
    if [*dimsL[1:]] != [1]:
        T = np.reshape(T, (X.size, *dimsL[1:]))

    # Swap axes, if necessary
    if spectral_dim != 0:
        T = np.swapaxes(T, 0, spectral_dim)

    return T


def BT2L(X_in, T_in, wavelength=False, bad_value=np.nan, spectral_dim=0):
    """
    Compute brightness temperature at given spectral radiance.

    The brightness temperature is the temperature at which a perfect blackbody
    would need to be to produce the same spectral radiance L at each specified
    wavenumber X. The shape of output T will be ``(X.size, *L.shape)``.

    Parameters
    __________
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    T : array_like
      spectral brightness temperature in [K], arbitrary dimensions with spectral
      dimension first
    wavelength : logical
      if true, interprets spectral input `X` in wavelength [micron, µm]
    bad_value : value to use when an unphysical brightness temperature is
      computed. Default is np.nan.

    Returns
    _______
    L : numpy array
      spectral radiance in [µW/(cm^2 sr cm^{-1})] or [µW/(cm^2 sr µm)]

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import radiative_transfer as rt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = rt.planckian(X,T)
    >>> T = rt.brightnessTemperature(X,L)
    >>> plt.plot(X,T)
    """
    # Ensure inputs are NumPy arrays
    X = np.asarray(np.copy(X_in)).flatten()  # X must be 1D array
    T = np.asarray(np.copy(T_in))

    # Swap axes, if necessary
    if spectral_dim != 0:
        T = np.swapaxes(T, 0, spectral_dim)

    # Ensure X is row vector for outer products
    X = X[:, np.newaxis]

    # Make T a column vector or 2D array w/ spectral axis as 1st dimension
    if T.ndim == 1:  # if it is a vector, must be same shape as X
        T = T[:, np.newaxis]
        dimsT = T.shape
    else:  # otherwise collapse / reshape with 1st dimension corresponding to X
        dimsT = T.shape
        T = T.reshape((dimsT[0], np.prod(dimsT[1:])))

    # Evaluate brightness temperature
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not wavelength:
            print("Assumes X given in µm and L given in µF")
        X *= 1e-6  # convert to m from µm
        L = c1 / (X ** 5 * (np.exp(c2 / (X * T)) - 1))  # [W/(m^2 sr m)] SI
        L *= 1e-4  # convert to [µW/(cm^2 sr µm^{-1})]
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L = c1 * X ** 3 / (np.exp(c2 * X / T) - 1)  # [W/(m^2 sr m^{-1})]
        L *= 1e4  # convert to [µW/(cm^2 sr cm^{-1})] (1e6 / 1e2)

    # NaN-ify garbage results
    ixBad = ~np.isfinite(L) | (np.real(T) <= 0) | (np.abs(np.imag(T)) > 0)
    L[ixBad] = bad_value

    # Reshape output
    L = np.reshape(L, (X.size, *dimsT[1:]))

    # Swap axes, if necessary
    if spectral_dim != 0:
        L = np.swapaxes(L, 0, spectral_dim)

    return L


def compute_LWIR_apparent_radiance(X, emis, Ts, tau, La, Ld, dT=None, return_Ls=False):
    r"""
    Compute LWIR spectral radiance for given emissivities and atmospheric states.

    Efficienetly computes (via broadcasting) every combination of spectral radiance
    for a set of emissivity profiles, a set of atmospheric radiative terms, and an
    optional range of surface temperatures. Assumes pure, flat, lambertian material.

    :math: L_i(\nu) = \tau(\nu) \left[ \varepsilon_i(\nu) B(\nu,T_i) + (1-\varepsilon(\nu)) L_d(\nu) \right] + L_a(\nu)

    Parameters
    __________
    X: array_like (nX,)
      spectral axis in wavenumbers [1/cm], 1D array of length `nX`
    emis: array_like (nX, nE)
      emissivity array – `nE` is the number of materials
    Ts: array_like (nA,)
      surface temperature [K], 1D array of length `nA`
    tau: array_like (nX, nA)
      atmospheric transmittance between source and sensor [0 ≤ tau ≤ 1]
    La: array_like (nX, nA)
      upwelling atmospheric path radiance [µW/(cm^2 sr cm^{-1})]
    Ld: array_like (nX, nA)
      hemispherically-averaged atmospheric downwelling radiance [µW/(cm^2 sr cm^{-1})]
    dT: array_like (nT,), optional {None}
      surface temperature deltas, relative to `Ts` [K]

    Returns
    _______
    L: array_like (nX, nE, nA) or (nX, nE, nA, nT)
      apparent spectral radiance
    """
    if dT is not None:
        T_ = Ts.flatten()[:, np.newaxis] + np.asarray(dT).flatten()[np.newaxis, :]
        B_ = planckian(X, T_)[:, np.newaxis, :, :]
        tau_ = tau[:, np.newaxis, :, np.newaxis]
        La_ = La[:, np.newaxis, :, np.newaxis]
        Ld_ = Ld[:, np.newaxis, :, np.newaxis]
        em_ = emis[:, :, np.newaxis, np.newaxis]
    else:
        T_ = Ts.flatten()
        B_ = planckian(X, T_)[:, np.newaxis, :]
        tau_ = tau[:, np.newaxis, :]
        La_ = La[:, np.newaxis, :]
        Ld_ = Ld[:, np.newaxis, :]
        em_ = emis[:, :, np.newaxis]
    if return_Ls:
        Ls = em_ * B_ + (1 - em_) * Ld_
        L = tau_ * Ls + La_
        return L, Ls
    else:
        L = tau_ * (em_ * B_ + (1 - em_) * Ld_) + La_
        return L


def ILS_MAKO(X, Y, resFactor=None, returnX=True, fwhm_sf=1.0, shift=0.0, scale=1.0):
    """
    Apply MAKO instrument line shape (ILS) to high-resolution spectrum.

    Parameters
    __________
    X : array_like (nX,)
        spectral axis in wavenumbers [1/cm], 1D array of length `nX`
    Y : array_like (nX,) or (nX, nS)
        high-resolution spectrum or spectral array to convolve with ILS
    
    Returns
    _______
    X_out : array_like (128,) or (128*resFactor, )
        output spectral axis in wavenumbers
    Y_out : array_like (128, nS) or (128*resFactor, nS)
        convolved spectrum or spectral array
    """

    # MAKO spectral axis in µm
    X_out = np.array(
        [
            7.5711,
            7.6158,
            7.6606,
            7.7053,
            7.7500,
            7.7947,
            7.8394,
            7.8841,
            7.9288,
            7.9734,
            8.0181,
            8.0627,
            8.1073,
            8.1519,
            8.1965,
            8.2411,
            8.2857,
            8.3303,
            8.3748,
            8.4194,
            8.4639,
            8.5084,
            8.5529,
            8.5974,
            8.6419,
            8.6863,
            8.7308,
            8.7752,
            8.8197,
            8.8641,
            8.9085,
            8.9529,
            8.9973,
            9.0417,
            9.0860,
            9.1304,
            9.1747,
            9.2190,
            9.2633,
            9.3076,
            9.3519,
            9.3962,
            9.4405,
            9.4847,
            9.5290,
            9.5732,
            9.6174,
            9.6616,
            9.7058,
            9.7500,
            9.7942,
            9.8383,
            9.8825,
            9.9266,
            9.9707,
            10.0148,
            10.0589,
            10.1030,
            10.1471,
            10.1912,
            10.2352,
            10.2792,
            10.3233,
            10.3673,
            10.4113,
            10.4553,
            10.4993,
            10.5432,
            10.5872,
            10.6311,
            10.6751,
            10.7190,
            10.7629,
            10.8068,
            10.8507,
            10.8945,
            10.9384,
            10.9822,
            11.0261,
            11.0699,
            11.1137,
            11.1575,
            11.2013,
            11.2451,
            11.2888,
            11.3326,
            11.3763,
            11.4201,
            11.4638,
            11.5075,
            11.5512,
            11.5948,
            11.6385,
            11.6822,
            11.7258,
            11.7694,
            11.8131,
            11.8567,
            11.9003,
            11.9439,
            11.9874,
            12.0310,
            12.0745,
            12.1181,
            12.1616,
            12.2051,
            12.2486,
            12.2921,
            12.3356,
            12.3791,
            12.4225,
            12.4660,
            12.5094,
            12.5528,
            12.5962,
            12.6396,
            12.6830,
            12.7264,
            12.7697,
            12.8131,
            12.8564,
            12.8997,
            12.9430,
            12.9863,
            13.0296,
            13.0729,
            13.1162,
            13.1594,
        ]
    )

    # Increase spectral resolution by resFactor for a MAKO-like sensor
    if resFactor is not None:
        _x0 = np.linspace(0, 1, len(X_out))
        _x1 = np.linspace(0, 1, int(len(X_out) * resFactor))
        X_out = np.interp(_x1, _x0, X_out)

    # Convert to wavenumbers
    X_out = np.sort(10000.0 / X_out)  # [1/cm] from [µm]
    X_out = X_out[(X_out > X.min()) & (X_out < X.max())]

    # Triangle lineshape
    def tri(x, x0, s):
        w = 1.0 - np.abs(x - x0) / s
        w[w < 0] = 0
        return w

    sigma_out = fwhm_sf * np.abs(np.gradient(X_out)) * 1.6
    ILS = tri(X[:, None], scale * X_out[None, :] + shift, sigma_out[None, :])
    N = np.sum(ILS, axis=0)

    # # Gaussian lineshape
    # g = lambda x, x0, s: np.exp(-0.5 * ((x - x0) / s)** 2) / (s * np.sqrt(2.0 * np.pi))
    # ILS = g(X[:, np.newaxis], scale * X_out[np.newaxis, :] + shift, sigma_out[np.newaxis, :])
    # N = np.sum(ILS, axis=0)

    # Convolve with input spectrum / spectra
    if len(Y.shape) == 1:
        Y_out = np.sum(ILS * Y[:, np.newaxis], axis=0) / N
    else:
        Y_out = np.transpose(
            np.sum(ILS[None, :, :] * Y.T[:, :, None], axis=1) / N[None, :]
        )
        # Broadcasting eliminates this equivalent for-loop
        # Y_out = np.zeros((X_out.size, Y.shape[-1]))
        # for ii in range(Y.shape[-1]):
        #     Y_out[:,ii] = np.sum(ILS*Y[:,ii][:,np.newaxis], axis=0)/N
    if returnX:
        return X_out, Y_out
    return Y_out


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)]
          instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        return x

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        return x

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        return x

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    ix0 = int(np.ceil(window_len / 2 - 1))
    ix1 = -int(np.floor(window_len / 2))
    return y[ix0:ix1]


def reduceResolution(X, Y, dX, N=4, window="hanning", X_out=None):
    dX_in = np.mean(np.diff(X))
    smFactor = np.int(np.round(dX / dX_in))
    smFcn1 = lambda y: smooth(y, window_len=smFactor, window=window)
    smFcn = lambda y: 0.5 * (smFcn1(y) + smFcn1(y[::-1])[::-1])
    interpFcn = lambda x, y, x0: scipy.interpolate.interp1d(
        x, y, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )(x0)
    X_ = smFcn(X)
    nPts = np.int(np.ceil(N * (X_[-smFactor - 1] - X_[smFactor]) / dX)) + 1
    returnX_out = False
    if X_out is None:
        X_out = np.linspace(X_[smFactor], X_[-smFactor - 1], nPts)
        returnX_out = True
    if len(Y.shape) > 1:
        Y_out = np.zeros((X_out.size, Y.shape[-1]))
        for ii in range(Y.shape[-1]):
            Y_out[:, ii] = interpFcn(X_, smFcn(Y[:, ii]), X_out)
    else:
        Y_out = interpFcn(X_, smFcn(Y), X_out)
    if returnX_out:
        return X_out, Y_out
    else:
        return Y_out


# if __name__ == "__main__":
#     # Simple test script
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt

#     # Set plotting defaults
#     mpl.rcParams['text.usetex'] = True
#     mpl.rcParams['font.family'] = 'serif'
#     mpl.rcParams['text.latex.preamble'] = r'\usepackage[adobe-utopia]{mathdesign}, \usepackage{siunitx}'

#     # Define strings containing LaTeX formatted stuff for plots
#     s_rad_wn = r'Spectral Radiance, $L(\tilde{\nu})$ $\left[ \si{ {\micro}W/(cm^2.sr.cm^{-1}) } \right]$'
#     s_rad_wl = r'Spectral Radiance, $L(\lambda)$ $\left[ \si{ {\micro}W/(cm^2.sr.\um) } \right]$'
#     s_Tb_wn = r'Brightness Temperature, $T_B(\tilde{\nu})$ $\left[\si{K}\right]$'
#     s_Tb_wl = r'Brightness Temperature, $T_B(\lambda)$ $\left[\si{K}\right]$'
#     s_wn = r'Wavenumbers, $\tilde{\nu}$ $\left[\si{cm^{-1}}\right]$'
#     s_wl = r'Wavelength, $\lambda$ $\left[\si{{\micro}m}\right]$'

#     # Test at known temperatures and wavenumbers / wavelengths -- print results
#     T = 296
#     wn = 500  # wavenumber
#     wl = 10000 / wn  # equivalent wavelength
#     d_wn = 1  # differential wavenumber
#     d_wl = (d_wn / wn) * wl  # equivalent differential wavelength
#     L_wn = planckian(wn, T)
#     L_wl = planckian(wl, T, wavelength=True)
#     s1 = "L(X = {0}/cm, T = {1}K) = {2:0.6e} µW/(cm^2·sr·cm^{{-1}})\n".format(wn, T, float(L_wn))
#     s2 = "L(X = {0}µm, T = {1}K) = {2:0.6e} µW/(cm^2·sr·µm)\n".format(wl, T, float(L_wl))
#     sa = "L(X = {0}/cm, T = {1}K) * (ΔX = {2:0.2e}/cm) = {3:0.6e} µW/(cm^2·sr)\n".format(
#         wn, T, d_wn, float(L_wn * d_wn))
#     sb = "L(X = {0}µm, T = {1}K) * (ΔX = {2:0.2e}µm) = {3:0.6e} µW/(cm^2·sr)\n".format(
#         wl, T, d_wl, float(L_wl * d_wl))
#     print(s1 + s2 + sa + sb)

#     # plotting function (private)
#     def _plot_rad_Tb(X, L, Tb, T, xl=None, yl_L=None, yl_T=None):
#         """Plot Planckian and brightness temp distribution for V&V."""
#         def my_legend(T):
#             if T is not None:
#                 return ["$T = {0}$ K".format(TT) for TT in np.array(T).flatten()]
#             else:
#                 return None
#         plt.figure(figsize=(7.5, 10.5))
#         plt.subplot(2, 1, 1)
#         plt.plot(X, L)
#         plt.xlabel(xl)
#         plt.ylabel(yl_L)
#         plt.legend(my_legend(T))
#         plt.title('Planckian Spectral Radiance Distribution')
#         plt.subplot(2, 1, 2)
#         plt.plot(X, Tb)
#         plt.title('Spectral Brightness Temperature Distribution')
#         try:
#             if len(T) > 3:
#                 plt.yticks(T)
#         except:
#             None
#         plt.xlabel(xl)
#         plt.ylabel(yl_T)
#         plt.show()

#     # Common spectral axis for visualizations
#     X1 = np.linspace(100, 2500, 500)  # [1/cm] wavenumbers
#     X2 = 10000 / X1  # [µm] wavelength

#     # Compute and visualize radiance and brightness temperature -- scalar T
#     T = 296
#     L1 = planckian(X1, T)
#     L2 = planckian(X2, T, wavelength=True)
#     Tb1 = brightnessTemperature(X1, L1)
#     Tb2 = brightnessTemperature(X2, L2, wavelength=True)
#     # _plot_rad_Tb(X1, L1, Tb1, T, xl=s_wn, yl_L=s_rad_wn, yl_T=s_Tb_wn)
#     # _plot_rad_Tb(X2, L2, Tb2, T, xl=s_wl, yl_L=s_rad_wl, yl_T=s_Tb_wl)

#     # Compute and visualize radiance and brightness temperature -- vector T
#     T = np.arange(250, 375, 25)
#     L1 = planckian(X1, T)
#     L2 = planckian(X2, T, wavelength=True)

#     Tb1 = brightnessTemperature(X1, L1)
#     Tb2 = brightnessTemperature(X2, L2, wavelength=True)
#     # _plot_rad_Tb(X1, L1, Tb1, T, xl=s_wn, yl_L=s_rad_wn, yl_T=s_Tb_wn)
#     # _plot_rad_Tb(X2, L2, Tb2, T, xl=s_wl, yl_L=s_rad_wl, yl_T=s_Tb_wl)

#     Xmin = 650
#     Xmax = 1550
#     DV = 0.001
#     X, tau_SA, La_SA, Ld_SA = compute_TUD(Xmin, Xmax, DVOUT=DV)

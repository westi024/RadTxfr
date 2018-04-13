from io import StringIO
import os, os.path
import inspect
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
# h  = 6.6260689633e-34 # [J s]       - Planck's constant
# c  = 299792458        # [m/s]       - speed of light
# k  = 1.380650424e-23  # [J/K]       - Boltzman constant
c1 = 1.19104295315e-16  # [J m^2 / s] - 1st radiation constant, c1 = 2 * h * c**2
c2 = 1.43877736830e-02  # [m K]       - 2nd radiation constant, c2 = h * c / k

# Define standard atmosphere
StdAtmosCSV = StringIO("""
#,Z0 [km],Z1 [km],PL [km],P [Pa],T [K],H2O,CO2,O3,N2O,CO,CH4,O2,N2,Ar
1,0.00,0.10,0.10000,100697.30225,287.87,7.6573221E-03,3.8004431E-04,2.6744971E-08,3.2023897E-07,1.4985868E-07,1.7012743E-06,2.0915587E-01,7.7355174E-01,9.2528217E-03
2,0.10,0.20,0.10000,99500.30518,287.23,7.4731829E-03,3.8004757E-04,2.7011305E-08,3.2024224E-07,1.4935317E-07,1.7012859E-06,2.0915878E-01,7.7373083E-01,9.2549638E-03
3,0.20,0.30,0.10000,98317.27295,286.58,7.2934777E-03,3.8005254E-04,2.7272709E-08,3.2024707E-07,1.4884883E-07,1.7013097E-06,2.0916124E-01,7.7390597E-01,9.2570588E-03
4,0.30,0.40,0.10000,97148.21167,285.93,7.1180305E-03,3.8005490E-04,2.7533348E-08,3.2024772E-07,1.4834590E-07,1.7013143E-06,2.0916177E-01,7.7407882E-01,9.2591263E-03
5,0.40,0.50,0.10000,95992.99927,285.27,6.9468380E-03,3.8005615E-04,2.7807420E-08,3.2024988E-07,1.4784467E-07,1.7013228E-06,2.0916334E-01,7.7424644E-01,9.2611312E-03
6,0.50,0.60,0.10000,94851.96533,284.63,6.7797713E-03,3.8006171E-04,2.8076512E-08,3.2025335E-07,1.4734614E-07,1.7013465E-06,2.0916519E-01,7.7440970E-01,9.2630841E-03
7,0.60,0.70,0.10000,93723.85864,283.98,6.6166990E-03,3.8006323E-04,2.8359112E-08,3.2025477E-07,1.4684780E-07,1.7013551E-06,2.0916586E-01,7.7457018E-01,9.2650037E-03
8,0.70,0.80,0.10000,92610.18066,283.33,6.4575248E-03,3.8006328E-04,2.8626875E-08,3.2025568E-07,1.4635113E-07,1.7013542E-06,2.0916755E-01,7.7472580E-01,9.2668651E-03
9,0.80,0.90,0.10000,91508.79517,282.68,6.3021979E-03,3.8006456E-04,2.8903722E-08,3.2025673E-07,1.4585650E-07,1.7013672E-06,2.0916763E-01,7.7487921E-01,9.2687002E-03
10,0.90,1.00,0.10000,90420.12939,282.02,6.1505721E-03,3.8006506E-04,2.9194448E-08,3.2025693E-07,1.4536292E-07,1.7013666E-06,2.0916760E-01,7.7502907E-01,9.2704927E-03
11,1.00,1.25,0.25000,88520.92285,280.89,5.8755800E-03,3.8006250E-04,2.9700320E-08,3.2025457E-07,1.4447069E-07,1.7013558E-06,2.0916663E-01,7.7530178E-01,9.2737547E-03
12,1.25,1.50,0.25000,85846.36841,279.26,5.4909759E-03,3.8005889E-04,3.0446010E-08,3.2025147E-07,1.4318157E-07,1.7013346E-06,2.0916434E-01,7.7568410E-01,9.2783278E-03
13,1.50,1.75,0.25000,83252.58179,277.64,5.1315110E-03,3.8005147E-04,3.1212348E-08,3.2024522E-07,1.4190299E-07,1.7013047E-06,2.0916015E-01,7.7604347E-01,9.2826264E-03
14,1.75,2.00,0.25000,80737.29858,276.02,4.7955187E-03,3.8003986E-04,3.1993196E-08,3.2023584E-07,1.4063411E-07,1.7012502E-06,2.0915397E-01,7.7638161E-01,9.2866710E-03
15,2.00,2.25,0.25000,78270.48340,274.39,4.4242670E-03,3.8001745E-04,3.2491752E-08,3.2021686E-07,1.3936234E-07,1.7011517E-06,2.0914139E-01,7.7676092E-01,9.2912082E-03
16,2.25,2.50,0.25000,75851.93481,272.77,4.0277741E-03,3.7998860E-04,3.2690732E-08,3.2019284E-07,1.3808986E-07,1.7010235E-06,2.0912592E-01,7.7716805E-01,9.2960780E-03
17,2.50,2.75,0.25000,73508.09937,271.14,3.6668028E-03,3.7995877E-04,3.2890703E-08,3.2016789E-07,1.3682843E-07,1.7008889E-06,2.0910963E-01,7.7754088E-01,9.3005377E-03
18,2.75,3.00,0.25000,71236.71265,269.51,3.3381749E-03,3.7992848E-04,3.3091947E-08,3.2014157E-07,1.3557833E-07,1.7007527E-06,2.0909248E-01,7.7788260E-01,9.3046251E-03
19,3.00,3.25,0.25000,69009.91821,267.89,3.0342883E-03,3.7991515E-04,3.3278667E-08,3.2013017E-07,1.3448891E-07,1.7006920E-06,2.0908479E-01,7.7819051E-01,9.3083082E-03
20,3.25,3.50,0.25000,66826.91040,266.27,2.7535658E-03,3.7991517E-04,3.3450437E-08,3.2013048E-07,1.3355726E-07,1.7006946E-06,2.0908572E-01,7.7846700E-01,9.3116154E-03
21,3.50,4.00,0.50000,63702.83813,263.84,2.3846428E-03,3.7990557E-04,3.3707664E-08,3.2012204E-07,1.3217480E-07,1.7006492E-06,2.0907986E-01,7.7883736E-01,9.3160455E-03
22,4.00,4.50,0.50000,59690.77759,260.59,1.9422086E-03,3.7992481E-04,3.4795157E-08,3.2013850E-07,1.3103309E-07,1.7007364E-06,2.0909050E-01,7.7926403E-01,9.3211491E-03
23,4.50,5.00,0.50000,55886.00464,257.34,1.5628497E-03,3.7996727E-04,3.6704041E-08,3.2017431E-07,1.3059758E-07,1.7009270E-06,2.0911388E-01,7.7961576E-01,9.3253563E-03
24,5.00,5.50,0.50000,52281.07300,254.09,1.2644272E-03,3.7997120E-04,3.8528793E-08,3.2017758E-07,1.2999899E-07,1.7009434E-06,2.0911607E-01,7.7990848E-01,9.3288577E-03
25,5.50,6.00,0.50000,48866.29333,250.84,1.0289896E-03,3.7992757E-04,4.0244537E-08,3.2014094E-07,1.2923383E-07,1.7007494E-06,2.0909223E-01,7.8016474E-01,9.3319229E-03
26,6.00,6.50,0.50000,45636.12976,247.59,8.2385325E-04,3.7995592E-04,4.3219430E-08,3.2016479E-07,1.2783930E-07,1.7006266E-06,2.0910759E-01,7.8035224E-01,9.3341657E-03
27,6.50,7.00,0.50000,42581.36292,244.34,6.4788532E-04,3.8005636E-04,4.7713680E-08,3.2024974E-07,1.2582142E-07,1.7005776E-06,2.0916297E-01,7.8047130E-01,9.3355899E-03
28,7.00,7.50,0.50000,39693.20984,241.09,5.1387883E-04,3.8009160E-04,5.2368396E-08,3.2027927E-07,1.2324612E-07,1.6999863E-06,2.0918229E-01,7.8058460E-01,9.3369450E-03
29,7.50,8.00,0.50000,36963.39417,237.84,4.1141221E-04,3.8005578E-04,5.7147069E-08,3.2024894E-07,1.2013186E-07,1.6988258E-06,2.0916265E-01,7.8070530E-01,9.3383888E-03
30,8.00,8.50,0.50000,34390.29236,234.59,3.0026093E-04,3.8003389E-04,6.6530347E-08,3.2010675E-07,1.1627073E-07,1.6972315E-06,2.0915066E-01,7.8082700E-01,9.3398445E-03
31,8.50,9.00,0.50000,31965.45715,231.34,1.9727937E-04,3.8003217E-04,8.2473292E-08,3.1985491E-07,1.1171657E-07,1.6952224E-06,2.0914972E-01,7.8092969E-01,9.3410728E-03
32,9.00,9.50,0.50000,29682.00989,228.11,1.3033135E-04,3.8003601E-04,1.0040944E-07,3.1933499E-07,1.0698130E-07,1.6922456E-06,2.0915176E-01,7.8099381E-01,9.3418398E-03
33,9.50,10.00,0.50000,27532.14417,224.91,8.6644133E-05,3.8004096E-04,1.2016432E-07,3.1853878E-07,1.0208884E-07,1.6882650E-06,2.0915465E-01,7.8103411E-01,9.3423219E-03
34,10.00,10.50,0.50000,25510.84900,221.69,5.9723599E-05,3.8004090E-04,1.4880519E-07,3.1716539E-07,9.7135150E-08,1.6837640E-06,2.0915459E-01,7.8106076E-01,9.3426406E-03
35,10.50,11.00,0.50000,23610.99091,218.44,4.2918404E-05,3.8003110E-04,1.9036740E-07,3.1520574E-07,9.2138784E-08,1.6787164E-06,2.0914885E-01,7.8108302E-01,9.3429068E-03
36,11.00,11.50,0.50000,21842.58118,216.78,3.1007603E-05,3.8000883E-04,2.3561967E-07,3.1308883E-07,8.6727809E-08,1.6728778E-06,2.0913664E-01,7.8110684E-01,9.3431918E-03
37,11.50,12.00,0.50000,20192.59796,216.73,2.2521850E-05,3.8001611E-04,2.8276929E-07,3.1084335E-07,8.0975298E-08,1.6664052E-06,2.0914076E-01,7.8111112E-01,9.3432430E-03
38,12.00,12.50,0.50000,18667.33093,216.7,1.6652726E-05,3.8002420E-04,3.2692944E-07,3.0854861E-07,7.4393796E-08,1.6589435E-06,2.0914517E-01,7.8111252E-01,9.3432598E-03
39,12.50,13.00,0.50000,17257.33032,216.7,1.2564588E-05,3.8003232E-04,3.6445050E-07,3.0620373E-07,6.7191678E-08,1.6504725E-06,2.0914973E-01,7.8111203E-01,9.3432539E-03
40,13.00,13.50,0.50000,15953.85284,216.7,9.3883436E-06,3.8002004E-04,4.1151947E-07,3.0379221E-07,6.0184412E-08,1.6414161E-06,2.0914303E-01,7.8112177E-01,9.3433704E-03
41,13.50,14.00,0.50000,14748.85101,216.7,6.9382863E-06,3.7998561E-04,4.7057611E-07,3.0131307E-07,5.3432629E-08,1.6317608E-06,2.0912392E-01,7.8114307E-01,9.3436252E-03
42,14.00,14.50,0.50000,13634.78546,216.7,5.6882345E-06,3.7996686E-04,5.3660847E-07,2.9869690E-07,4.7380766E-08,1.6216783E-06,2.0911379E-01,7.8115429E-01,9.3437594E-03
43,14.50,15.00,0.50000,12604.78745,216.7,5.2244709E-06,3.7996451E-04,6.1023087E-07,2.9594358E-07,4.1959911E-08,1.6111613E-06,2.0911241E-01,7.8115606E-01,9.3437806E-03
44,15.00,16.00,1.00000,11230.00183,216.7,4.4708750E-06,3.7999541E-04,7.5258697E-07,2.9130481E-07,3.5004010E-08,1.5947592E-06,2.0912933E-01,7.8113994E-01,9.3435878E-03
45,16.00,17.00,1.00000,9599.99771,216.7,3.9036895E-06,3.8001325E-04,1.0169126E-06,2.8328515E-07,2.7782994E-08,1.5688780E-06,2.0913923E-01,7.8113048E-01,9.3434746E-03
46,17.00,18.00,1.00000,8207.49893,216.7,3.8400353E-06,3.7998025E-04,1.3729086E-06,2.7296613E-07,2.2253532E-08,1.5382528E-06,2.0912108E-01,7.8114821E-01,9.3436866E-03
47,18.00,19.00,1.00000,7015.99960,216.7,3.8393960E-06,3.7998176E-04,1.7946930E-06,2.6017312E-07,1.7556889E-08,1.5018171E-06,2.0912179E-01,7.8114714E-01,9.3436739E-03
48,19.00,20.00,1.00000,5997.99995,216.7,3.8764683E-06,3.7997364E-04,2.2876818E-06,2.4485973E-07,1.4409030E-08,1.4533676E-06,2.0911746E-01,7.8115097E-01,9.3437197E-03
49,20.00,22.00,2.00000,4788.41553,217.57,3.9764941E-06,3.7997778E-04,3.0358181E-06,2.2097053E-07,1.2598199E-08,1.3559661E-06,2.0911972E-01,7.8114802E-01,9.3436843E-03
50,22.00,24.00,2.00000,3509.76715,219.55,4.1870412E-06,3.7997050E-04,4.1238559E-06,1.9704146E-07,1.3073440E-08,1.1973241E-06,2.0911567E-01,7.8115092E-01,9.3437191E-03
51,24.00,26.00,2.00000,2580.36308,221.54,4.4063904E-06,3.8000545E-04,5.0418298E-06,1.7740817E-07,1.4803017E-08,1.0667032E-06,2.0913491E-01,7.8113090E-01,9.3434796E-03
52,26.00,28.00,2.00000,1902.99072,223.47,4.5444967E-06,3.8001285E-04,5.6498752E-06,1.6246848E-07,1.5766954E-08,1.0021820E-06,2.0913896E-01,7.8112623E-01,9.3434237E-03
53,28.00,30.00,2.00000,1407.21016,225.45,4.6642508E-06,3.7998866E-04,6.2325935E-06,1.4873616E-07,1.6631679E-08,9.4450036E-07,2.0912573E-01,7.8113870E-01,9.3435729E-03
54,30.00,32.00,2.00000,1043.50023,227.67,4.7074454E-06,3.7536325E-04,6.7746596E-06,1.3014554E-07,1.7414909E-08,8.7108037E-07,2.0658007E-01,7.8365835E-01,9.3737117E-03
55,32.00,34.00,2.00000,776.75004,230.94,4.7230560E-06,3.7075937E-04,7.2671910E-06,1.0919968E-07,1.8337271E-08,7.9487688E-07,2.0404629E-01,7.8616633E-01,9.4037107E-03
56,34.00,36.00,2.00000,581.42772,236.36,4.8948600E-06,3.7965289E-04,7.7794029E-06,9.2485571E-08,2.0089036E-08,7.4666866E-07,2.0894088E-01,7.8132019E-01,9.3457438E-03
57,36.00,38.00,2.00000,438.18765,241.88,5.0474359E-06,3.8805505E-04,7.9678821E-06,7.3569971E-08,2.2208699E-08,6.9390808E-07,2.1356495E-01,7.7674221E-01,9.2909843E-03
58,38.00,40.00,2.00000,332.41796,247.45,5.0566086E-06,3.8457511E-04,7.6026390E-06,5.4176880E-08,2.4073520E-08,6.1123779E-07,2.1164984E-01,7.7863857E-01,9.3136676E-03
59,40.00,42.00,2.00000,253.74177,253.02,5.0756607E-06,3.8000967E-04,6.8690915E-06,3.7642952E-08,2.6195995E-08,5.2326124E-07,2.0913728E-01,7.8112675E-01,9.3434299E-03
60,42.00,46.00,4.00000,175.76751,260.9,5.1898683E-06,3.7999873E-04,5.6845497E-06,2.1269509E-08,3.0447396E-08,4.0876645E-07,2.0913120E-01,7.8113395E-01,9.3435161E-03
61,46.00,50.00,4.00000,105.69584,269.76,5.2437290E-06,3.8003578E-04,3.9669685E-06,8.7507939E-09,3.8790922E-08,2.6933361E-07,2.0915161E-01,7.8111553E-01,9.3432958E-03
62,50.00,54.00,4.00000,63.91487,267.03,5.1818847E-06,3.7999649E-04,2.5585023E-06,4.0358978E-09,5.2859484E-08,1.9255586E-07,2.0913003E-01,7.8113841E-01,9.3435695E-03
63,54.00,58.00,4.00000,38.41681,258.35,5.0363133E-06,3.8001817E-04,1.6718860E-06,2.8382956E-09,7.2663589E-08,1.6327704E-07,2.0914195E-01,7.8112764E-01,9.3434406E-03
64,58.00,62.00,4.00000,22.59020,247.46,4.7452750E-06,3.8005048E-04,1.1293572E-06,2.1092150E-09,1.0706705E-07,1.5181011E-07,2.0915969E-01,7.8111088E-01,9.3432402E-03
65,62.00,66.00,4.00000,12.99117,236.49,4.3229702E-06,3.8004681E-04,7.7787752E-07,1.6291317E-09,1.6487030E-07,1.5011317E-07,2.0915766E-01,7.8111360E-01,9.3432727E-03
66,66.00,70.00,4.00000,7.29694,225.53,3.7958089E-06,3.8006320E-04,4.4250618E-07,1.2967116E-09,2.4819016E-07,1.5011966E-07,2.0916671E-01,7.8110541E-01,9.3431747E-03
""")
StdAtmos = np.loadtxt(StdAtmosCSV, delimiter=',', skiprows=1)

# Define default options
LBL_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LBLRTM = os.path.join(LBL_dir, 'lblrtm_v12.8_OS_X_gnu_sgl')
TAPE3 = os.path.join(LBL_dir, 'AER-v3.6-0500-6000.tp3')
options = {
    # options for write_tape5
    'V1': 2000.00, # [cm^{-1}]
    'V2': 3333.33, # [cm^{-1}]
    'T': 296.0, # [K]
    'P': 101325.0, # [Pa]
    'PL': 1.0, # [km]
    'MF': np.zeros(38), # [ppmv]
    'MF_ID': np.array([]),
    'MF_VAL': np.array([]),
    'continuum_factors': np.zeros(7),
    'continuum_override': False,
    'description': 'TAPE5 for single layer calculation by compute_OD.py',
    'DVOUT': 0.0025, # [cm^{-1}]
    # options for run_LBLRTM
    'debug': True,
    'LBL_dir': LBL_dir,
    'LBLRTM': LBLRTM,
    'TAPE3': TAPE3,
    # options for compute_TUD
    'Ts': StdAtmos[:,5], # [K]
    'Ps': StdAtmos[:,4], # [Pa]
    'PLs': StdAtmos[:,3], # [km]
    'MFs_VAL': StdAtmos[:, 6:14]*1e6, # [ppmv]
    'MFs_ID': np.array([1, 2, 3, 4, 5, 6, 7, 22]),
    'theta_r': 0,
    'N_angle': 30,
    }


def make_spectral_axis(Xmin, Xmax, DVOUT):
    nX = np.ceil((Xmax - Xmin) / DVOUT)
    X = np.linspace(Xmin, Xmax, nX)
    return X


def compute_TUD(Xmin, Xmax, opts=options, **kwargs):
    tau, Lu, Ld=[], [], []
    opts.update(kwargs)
    # Extract atmospheric variables
    T = opts["Ts"]
    P = opts["Ps"]
    PL = opts["PLs"]
    MF = opts["MFs_VAL"]
    ID = opts["MFs_ID"]
    nL = T.size
    nA = opts["N_angle"]

    # Preallocate arrays
    X = make_spectral_axis(Xmin, Xmax, opts["DVOUT"])
    OD = np.zeros((X.size, nL))
    Lu = np.zeros(X.size)
    Ld = np.zeros((X.size, nA))
    

    # Compute OD's and Planckian distribution for each layer
    for ii in np.arange(nL):
        _, OD[:,ii] = compute_OD(Xmin, Xmax, opts=options, T=T[ii], P=P[ii],
                                 PL = PL[ii], MF_VAL=MF[ii, :], MF_ID=ID)
        print(f"Computing layer {ii+1:3d} of {nL:3d}")
    B = planckian(X, T)

    # transmittance
    print("Computing transmittance")
    mu = 1.0/np.cos(opts["theta_r"])
    tau = np.exp(-1.0*np.sum(OD * mu, axis=1))
    
    # upwelling
    print("Computing upwelling")
    for ii in np.arange(nL):
        t = np.exp( - OD[:, ii] * mu)
        Lu = t * Lu + (1 - t) * B[:, ii]
    
    # downwelling
    print("Computing downwelling")
    angles = np.linspace(0, np.pi / 2.0, nA, endpoint=False)
    for jj, th in enumerate(angles):
        for ii in np.arange(nL)[::-1]:
            t = np.exp( - OD[:, ii] / np.cos(th))
            Ld[:, jj] = t * Ld[:, jj] + (1 - t) * B[:, ii]
        print(f"Computing angle {jj+1:3d} of {nA:3d}")
    np.savez('ComputeTUD.npz', OD=OD, B=B, tau=tau, Ld=Ld, Lu=Lu, X=X, angles=angles)
    cos_dOmega = np.cos(angles) * np.sin(angles)
    Ld = np.sum(Ld * cos_dOmega, axis=1) / np.sum(cos_dOmega)
    Ld = Ld.flatten()

    return X, tau, Lu, Ld


def compute_OD(Xmin_in, Xmax_in, opts=options, **kwargs):
    opts.update(kwargs)
    DVOUT = opts.get("DVOUT", 0.025)

    # Set up parameters for looping over spectral range in 2020/cm chunks
    myround = lambda x: float("{0:10.3f}".format(x))
    pad = 25 # padding around each spectral bin that is trimmed from every run
    olp = 5 # overlap between spectral bins for averaging OD
    Xmin = np.max([myround(Xmin_in - pad - olp), 0])
    Xmax = myround(Xmax_in + pad + olp)
    maxBW = 2020 - olp - 2 * pad
    nBand = int(np.ceil((Xmax - Xmin) / maxBW))
    nPts = int(np.floor(maxBW / DVOUT))

    # Compute OD for each spectral chunck
    X = []
    OD = []
    for ii in range(nBand):
        if ii > 0:
            Xmin = myround(np.max(X[ii - 1]) - olp - pad)
        Xmax1 = np.min([Xmax+pad, myround(Xmin + DVOUT * (nPts - 1) + olp + pad)])
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
    nrm[nrm<1] = 1
    OD_out = np.sum(OD_out, axis=0) / nrm
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
        else:
            print(ex.stderr)
            nu, od = [], []
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
    C = opts.get("MF", np.zeros(38))
    if "MF_ID" in opts.keys() and "MF_VAL" in opts.keys():
        idx = [i-1 for i in list(opts['MF_ID'])]
        C[idx] = opts['MF_VAL']

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


def planckian(X_in, T_in, wavelength=False):
    """
    Compute the Planckian spectral radiance distribution.

    Computes the spectral radiance `L` at wavenumber(s) `X` for a system at
    temperature(s) `T` using Planck's distribution function. `X` must be a scalar
    or a vector. `T` can be of arbitrary dimensions. The shape of output `L` will
    be `(X.size, *T.shape)`.

    Parameters
    ----------
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    T : array_like
      temperature array, Kelvin [K], arbitrary dimensions
    wavelength : logical
      if true, interprets spectral input `X` as wavelength [micron, µm]

    Returns
    -------
    L : array_like
      spectral radiance in [µW/(cm^2·sr·cm^-1)], or if wavelength=True,
      spectral radiance in [µW/(cm^2·sr·µm)] (microflick, µF)

    Example
    _______
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> X = np.linspace(2000,5000,100)
    >>> T = np.linspace(273,373,10)
    >>> L = planckian(X,T)
    >>> plt.plot(X,L)
    """
    # Ensure inputs are NumPy arrays
    X = np.asarray(X_in).flatten()  # X must be 1D array
    T = np.asarray(T_in)

    # Make X a column vector and T a row vector for broadcasting into 2D arrays
    X = X[:, np.newaxis]
    dimsT = T.shape  # keep shape info for later reshaping into ND array
    T = T.flatten()[np.newaxis, :]

    # Compute Planck's spectral radiance distribution
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not wavelength:
            print('Assumes X given in µm; returning L in µF')
        X *= 1e-6  # convert to m from µm
        L = c1 / (X**5 * (np.exp(c2 / (X * T)) - 1))  # [W/(m^2 sr m)] SI
        L *= 1e-4  # convert to [µW/(cm^2 sr µm^{-1})]
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L = c1 * X**3 / (np.exp(c2 * X / T) - 1)  # [W/(m^2 sr m^{-1})]
        L *= 1e4  # convert to [µW/(cm^2 sr cm^{-1})] (1e6 / 1e2)

    # Reshape L if necessary and return
    return np.reshape(L, (X.size, *dimsT))
    

def brightnessTemperature(X_in, L_in, wavelength=False):
    """
    Compute brightness temperature at given spectral radiance.

    The brightness temperature is the temperature at which a perfect blackbody
    would need to be to produce the same spectral radiance L at each specified
    wavenumber X. The shape of output T will be ``(X.size, *L.shape)``.

    Parameters
    ----------
    X : array_like (N,)
      spectral axis, wavenumbers [1/cm], 1D array
    L : array_like
      spectral radiance in [µW/(cm^2·sr·cm^-1)], arbitrary dimensions with
      spectral dimension first
    wavelength : logical
      if true, interprets spectral input `X` in wavelength [micron, µm]
      and spectral radiance `L` in [µW/(cm^2·sr·µm)] (microflick, µF)

    Parameters
    ----------
    X: numpy array (must be a vector)
      spectral input in wavenumbers [cm^{-1}]
    L: numpy array
      spectral radiance in [µW/(cm^2·sr·cm^-1)]
    f: logical
      if true, spectral input X is given in wavelength [µm] and L is given
      in [µW/(cm^2·sr·µm)] (microflick, µF)

    Returns
    -------
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
    X = np.asarray(X_in).flatten()  # X must be 1D array
    L = np.asarray(L_in)

    # Ensure X is row vector for outer products
    X = X[:, np.newaxis]

    # Make L a column vector or 2D array w/ spectral axis as 1st dimension
    if L.ndim == 1:  # if it is a vector, must be same shape as X
        L = L[:, np.newaxis]
        dimsL = L.shape
    else: # otherwise collapse / reshape with 1st dimension corresponding to X
        dimsL = L.shape
        L = L.reshape((dimsL[0], np.prod(dimsL[1:])))

    # Evaluate brightness temperature
    if wavelength or np.mean(X) < 50:  # compute using wavelength (with hueristics)
        if not wavelength:
            print('Assumes X given in µm and L given in µF')
        X *= 1e-6  # convert to m from µm
        L *= 1e+4  # convert to SI units, [W/(m^2 sr m)] from [µW/(cm^2 sr µm)]
        T = c2 / (X * np.log(1 + c1 / (X**5 * L)))
    else:  # compute using wavenumbers
        X *= 100  # convert to 1/m from 1/cm
        L *= 1e-4 # convert to [W/(m^2 sr m^{-1})] from [µW/(cm^2 sr cm^{-1})]
        T = c2 * X / np.log(c1 * X**3 / L + 1)

    # NaN-ify garbage results
    ixBad = np.logical_or(np.real(L) <= 0, np.abs(np.imag(T)) > 0)
    T[ixBad] = np.nan

    # Reshape T if necessary
    return np.reshape(T, (X.size, *dimsL[1:]))

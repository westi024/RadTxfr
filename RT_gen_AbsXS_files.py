#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:54:59 2017

@author: grosskc
"""
import numpy as np
import hapi as ht
# import pylab as plt

ht.db_begin('data')

# Set up calculation parameters
nu1 = 400
nu2 = 7100
dnu = 0.0025
wnW = 350
nX = round((nu2 - nu1) / dnu) + 1
X = np.linspace(nu1, nu2, nX)
T1 = 275
T2 = 320
dT = 5.0
nT = round((T2 - T1) / dT) + 1
T = np.linspace(T1, T2, nT)
P1 = 0.85
P2 = 1.05
dP = 0.05
nP = round((P2 - P1) / dP) + 1
P = np.linspace(P1, P2, nP)
molec = [1, 2]
molecID = ["H2O", "CO2"]
descr = "HITRAN2016 - HAPI - SDVoigt"

# Trim HITRAN database to spectral range and molecule
for mID, mName in zip(molec, molecID):
    cond1 = ('between', 'nu', nu1 - 100, nu2 + 100)
    cond2 = ('==', 'molec_id', mID)
    Cond = ('and', cond1, cond2)
    ht.select('HITRAN2016', Conditions=Cond, DestinationTableName=mName,
              File=mName)

# %%
# Function to write cross-section as binary file
def AFIT_XS_write(X,Y,T,P,ID,fnDB,File=[]):
    """
    Function to save molecular absorption cross-section in the AFIT_XS binary
    format. See code for description of binary format.

    Parameters
    ----------
    X:     spectral axis [1/cm]
    Y:     absorption cross-section [cm^2]
    T:     temperature [K]
    P:     pressure [Pa]
    ID:    HITRAN numerical ID for molecule (1-H2O, 2-CO2, etc.)
    fnDB:  filename of the spectral database used for computation of XS
    File:  filename to which XS is saved {XS-ID-TTTTK-ppppppPa.bin}. Here ID
           is the numerical HITRAN ID, TTTT is the temperature in Kelvin, and
           pppppp is the pressure in Pa.

    Returns
    -------
    File: filename to which XS was saved
    """

    # Define elements to be written to the binary file
    version = np.array('v1','<S2')                                # 4 bytes
    params = np.array([X.min(), X.max(), X.size, ID, T, P],'<f8') # 6 8-byte doubles = 48 bytes
    HITRAN_DB = np.array(fnDB,'<S128')                            # 256 bytes

    # Define default filename if none specified
    if len(File) == 0:
        File = "XS-{0:02d}-{1:04d}K-{2:06d}Pa.bin".format(int(ID),int(T),int(P))

    # Save binary file
    f = open(File,'wb')
    version.tofile(f)
    params.tofile(f)
    HITRAN_DB.tofile(f)
    Y.astype('<f8').tofile(f)
    f.close()
    return File

# %% Compute cross-sections - loop over molecules, pressures, and temperatures
## Loop over molecules, pressures, and temperatures
for mID,mName in zip(molec,molecID):
    for t in T:
        for p in P:
            nu,xs = ht.absorptionCoefficient_SDVoigt(SourceTables=mName,HITRAN_units=True,Environment={'T':t,'p':p},WavenumberStep=dnu,WavenumberRange=[nu1,nu2],IntensityThreshold=0,WavenumberWingHW=wnW)
            fname = AFIT_XS_write(nu,xs,t,101325*p,mID,descr)
            print(fname)

#%%
import os, fnmatch
tmp=fnmatch.filter(os.listdir('.'),'*.bin');
mID = list(set([int(ii[3:5]) for ii in tmp]))

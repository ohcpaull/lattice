# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:00:40 2018

In this version I graph the output as H K L parameters

@author: oliver
"""

import os #this deals with file directories
import xrayutilities as xu #this is the main package that is used
import numpy as np #useful mathematics package
import tkinter as tk #user interface for file IO
import xrdtools #easy way for reading XRDML
from tkinter import filedialog #file dialogue from TKinter
import matplotlib.pyplot as plt #plotting package
import matplotlib as mpl #plotting
#%matplotlib notebook
#%matplotlib inline

__all__ = ["lattice", "find_nearest"]

class lattice(object):
    
    def __init__(self):
        # omega 2theta and data arrays. Omega and 2theta are 1D arrays while data is a 2D array with dimensions of omega and 2theta
        self.omega = []
        self.tt = []
        self.data = []
        self.meta = []
        #test files for dev
        self.xrdmlTestFile = 'REAL1203_RSM_220_perp.xrdml'
        self.rasTestFile = 'N0044dOP_RSM_310.ras'
        #theoretical omega and 2theta positions for a given substrate peak
        self.theoOm, self.theoTT = [], []
        self.qx, self.qy, self.qz = [], [], []
        self.omCorr, self.ttCorr = [], []
        self.gridder = []
        self.rsmRef = []
        self.iHKL, self.oHKL = [], []
        self.h, self.k, self.l = 0, 0, 0
        self.maxInt = 0#maximum intensity of RSM peak
        self.filepath = self.openDialogue() # open GUI for user to select file
        #self.filepath = self.xrdmlTestFile
        self.p = []
        
        b = self.filepath.index('.') # find the period that separates filename and extension
        ext = self.filepath[(b+1):] # get extension and determine whether it is RAS or XRDML
        
        a = self.filepath.rfind('/')  # find the period that separates the directory and filename
        self.filename = self.filepath[(a+1):(b)]
        
        if ext == 'xrdml':
            (self.omega, self.tt, self.data) = self.xrdml_file(self, self.filepath)
            print('omega = ' + str(self.omega.shape))
            print('2theta = ' + str(self.tt.shape))
            print('data = ' + str(self.data.shape))
        elif ext == 'ras':
            (self.omega, self.tt, self.data) = self.ras_file(self, self.filepath)
        else:
            print('filetype not supported.')
        
        
        print(self.data.shape)
        [self.h, self.k, self.l] = input('Substrate reflection?').split(' ')
        print(str(self.h) + ' ' + str(self.k) + ' ' + str(self.l))
        self.rsmRef = (int(self.h), int(self.k), int(self.l))
        print('rsmRef = ' + str(self.rsmRef))
            
        [self.substrate, self.hxrd] = self.initSubstrate()
        
        self.p, [self.qx, self.qy, self.qz] = self.alignSubstratePeak()
        #[self.omCorr, self.ttCorr] = self.hxrd.Q2Ang([self.qx, self.qy, self.qz])
        
        
        
    @staticmethod
    def xrdml_file(self, file):
        #file = self.openDialogue()
        #print(file)
        data = xrdtools.read_xrdml(file)
        #data['Omega'] = data['Omega'][:,0]
        #data['2Theta'] = data['2Theta'][0]
        #om, tt = np.meshgrid(data['Omega'], data['2Theta'])
        om = data['Omega']
        tt = data['2Theta']
        
        return (om,
                tt,
                data['data'])
    @staticmethod
    def ras_file(self, file):
        # Read RAS data to object
        rasFile = xu.io.rigaku_ras.RASFile(file)
        
        self.scanaxis = rasFile.scans[1].scan_axis
        self.stepSize = rasFile.scans[1].meas_step
        self.measureSpeed= rasFile.scans[1].meas_speed
        self.dataCount = rasFile.scans[1].length
        # Read raw motor position and intensity data to large 1D arrays
        
        '''
            Can i get more data out from data['variable']? Can I adjust things to deal with 
            2theta-omega scans better?
        '''
        #print(rasFile.scans[1].scan_axis)
        omttRaw, data = xu.io.getras_scan(rasFile.filename+'%s', '', self.scanaxis)
        #print('omttRaw shape is ' + str(omttRaw.shape))
        npinte = np.array(data['int'])
        intensities = npinte.reshape(len(rasFile.scans), rasFile.scans[0].length)
        # Read omega data from motor positions at the start of each 2theta-Omega scan
        om = [rasFile.scans[i].init_mopo['Omega'] for i in range(0, len(rasFile.scans))]
        # Convert 2theta-omega data to 1D array
        tt = [omttRaw.data[i] for i in range(0, rasFile.scans[0].length)]
        
        if self.scanaxis == 'TwoThetaOmega': # If RSM is 2theta/omega vs omega scan, adjust omega values in 2D matrix
            omga = [[om[i] + (n * self.stepSize/2) for n in range(0,len(tt))] for i in range(0,len(om))]
            omga = np.array(omga)
            #print('length of omega = ' + str(len(om)))
            ttheta = np.array(tt)
            #print(len(ttheta))
            tt = [[ttheta[i] for i in range(0,len(ttheta))] for j in range(0, len(omga))]
            tt = np.array(tt)
            #print('ras_file adjusted tt shape = ' + str(tt.shape))
            #print('ras_file adjusted omega shape = ' + str(omga.shape))
            #print('ras_file adjusted int shape = ' + str(intensities.shape))

        #self.motor = rasFile.scans[1].moname
        
        return (np.transpose(omga), np.transpose(tt), np.transpose(intensities))
    
    def openDialogue(self):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename()
        filename = os.path.basename(filepath)
        return filepath
    
    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        #plt.figure(figsize=[12,10])
        
        #listData= self.data.tolist()
        
        #ax.imshow(self.omega,self.tt, np.transpose(np.log10(self.data)).tolist(), cmap='jet', origin='lower', **kwargs)
        om = np.array(self.omega)
        tt = np.array(self.tt)
        if(om.ndim == 1 and tt.ndim == 1):
            om, tt = np.meshgrid(self.omega, self.tt)
        elif (om.ndim != tt.ndim):
            print('Error. omega and twotheta arrays must have same dimension. dim(omega) = ' \
                  + str(om.shape) + 'and dim(tt) = ' + str(tt.shape))
        '''    
        if self.scanaxis == "TwoThetaOmega":
            om1D = np.ravel(om)
            tt1D = np.ravel(tt)
            dat1D = np.ravel(self.data)
            ax.scatter(om1D,tt1D,c=dat1D,cmap='jet')
        '''    
        #
        #a = ax.contour(om,tt, np.log10(self.data).tolist(), cmap='jet', origin='lower', **kwargs)
        #a = ax.contourf(om,tt, np.log10(self.data), cmap='jet')
        a = ax.contourf(np.transpose(om[0,:]),np.transpose(tt[:,0]), np.log10(self.data), cmap='jet')
        #ax.clabel(a, inline=True, fontsize=10)
        #ax.colorbar(label='Data')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'2$\theta$-$\omega$')
        return ax
    
    def plotQ(self, xGrid, yGrid, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=[7,5])
        ax.set_title( self.filename )
        
        
        self.gridder = xu.Gridder2D(xGrid,yGrid)
        self.gridder(self.qy, self.qz, self.data)
        intHigh = np.argmax(self.data)
        dynhigh = np.rint(np.log10(intHigh))
        #INT = xu.maplog(self.gridder.data.transpose(), 3, -1)
        ax.contourf(self.gridder.xaxis, self.gridder.yaxis, np.transpose(np.log10(self.gridder.data)), cmap='jet', **kwargs)
        #ax.colorbar(label='Data')
        ax.set_xlabel(r'$Q_{[001]}$')
        ax.set_ylabel(r'$Q_{[110]}$')
        
        return ax

    def to_csv(self, fname=None):
        if fname is None:
            fname = 'some_placeholder.csv'
        np.savetxt(fname, list(zip(self.omege, self.tt, self.tt)), delimiter=',')
        
    ''' Peakfitting-related functions (maybe make another class??)'''
    
    def findSubstratePeak(self, axis1, axis2, data):
        ''' Returns indices of axis1 and axis2 where the substrate peak is positioned (based off maximum counts)'''
        maxIndex = np.argmax(data)
        return np.unravel_index( maxIndex, (len(data[:,0]), len(data[0,:])))

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def initSubstrate(self):
        La = xu.materials.elements.La
        Al = xu.materials.elements.Al
        O = xu.materials.elements.O2m
        Sr = xu.materials.elements.Sr
        Ti = xu.materials.elements.Ti
        Ta = xu.materials.elements.Ta
        energy = 1240/0.154
        
        substrateMat = []
        while (substrateMat != 'LAO' and substrateMat != 'STO' and substrateMat != 'LSAT'):
            substrateMat = input('Sample substrate (LAO, STO or LSAT)?')
            print(substrateMat)
            print('input valid substrate material')
            
            
        [ipH, ipK, ipL] = input("Input in-plane direction of substrate without spaces (i.e. 100, 110, etc): ").split(' ')
        ipHKL = [ipH, ipK, ipL]
        [opH, opK, opL] = input("Input out-of-plane direction of substrate without spaces (i.e. 001, 110, 210 etc): ").split(' ')
        opHKL = [opH, opK, opL]
        """
        substrateMat = "LAO"
        ipHKL = (1, 0, 0)
        opHKL = (0, 0, 1)
        """
        self.iHKL = ipHKL
        self.oHKL = opHKL
        if substrateMat == "LAO":
            substrate = xu.materials.Crystal("LaAlO3", xu.materials.SGLattice(221, 3.784, \
                            atoms=[La, Al, O], pos=['1a', '1b', '3c']))
            hxrd = xu.HXRD(substrate.Q(int(ipHKL[0]), int(ipHKL[1]), int(ipHKL[2])), \
                           substrate.Q(int(opHKL[0]), int(opHKL[1]), int(opHKL[2])), en=energy, geometry = 'lo_hi')
        elif substrateMat == "STO":
            substrate = xu.materials.SrTiO3
            #substrate = xu.materials.Crystal("SrTiO3", xu.materials.SGLattice(221, 3.905, atoms=[Sr, Ti, O], \
            #pos=['1a', '1b', '3c']))
            hxrd = xu.HXRD(substrate.Q(int(ipHKL[0]), int(ipHKL[1]), int(ipHKL[2])), \
                           substrate.Q(int(opHKL[0]), int(opHKL[1]), int(opHKL[2])), en=energy)
        elif substrateMat == "LSAT": # need to make an alloy of LaAlO3 and Sr2AlTaO6
            mat1 = xu.materials.Crystal("LaAlO3", xu.materials.SGLattice(221, 3.79, \
                           atoms=[La, Al, O], pos=['1a', '1b', '3c']))
            mat2 = xu.materials.Crystal("Sr2AlTaO6", xu.materials.SGLattice(221, 3.898,\
                           atoms=[Sr, Al, Ta, O], pos=['8c', '4a', '4b', '24c']))
            substrate = xu.materials.CubicAlloy(mat1, mat2, 0.71)
            hxrd = xu.HXRD(substrate.Q(int(ipHKL[0]), int(ipHKL[1]), int(ipHKL[2])), \
                           substrate.Q(int(opHKL[0]), int(opHKL[1]), int(opHKL[2])), en=energy)
           
        return [substrate, hxrd]
    
    def alignSubstratePeak(self):
        nchannel = 255
        chpdeg = nchannel/2.511 #2.511 degrees is angular acceptance of xrays to detector
        center_ch = 128
        energy = 1240/0.154


        indexMax = np.argmax(self.data)
        self.maxInt = indexMax
        print('max index is ' + str(indexMax))
        data = np.array(self.data)
        #print('data shape = ' + str(data.shape))
        tupleIndex = np.unravel_index( indexMax, (len(data[:,0]), len(data[0,:])) )
        #print('tupleIndex = ' + str(tupleIndex))
        
        
        [self.theoOm, dummy, dummy, self.theoTT] = self.hxrd.Q2Ang(self.substrate.Q(self.rsmRef))
        expTT = self.tt[tupleIndex[0],tupleIndex[1]]
        expOm = self.omega[tupleIndex[0],tupleIndex[1]]
        print('experimental omega = ' + str(expOm))
        print('experimental tt = ' + str(expTT))
        print('theoretical omega = ' + str(self.theoOm))
        print('theoretical 2theta = ' + str(self.theoTT))
        print('omega shape = ' + str(self.omega.shape))
        #print('expt omega = ' + str(self.omega[tupleIndex[1], tupleIndex[0]]))

        expOm, expTT, p, cov = xu.analysis.fit_bragg_peak( self.omega, self.tt, self.data, expOm, expTT, self.hxrd, plot=True)
        
        offset = (self.theoTT/2) + self.theoOm
        omnominal = (self.theoTT/2) + offset
        self.hxrd.Ang2Q.init_linear('y+', center_ch, nchannel, chpdeg=chpdeg) 

        #[omega, tt] = np.meshgrid(self.omega, self.tt)
        #print('shape of omega is' + str(np.array(omega).shape))
        #print((self.omega[tupleIndex[0]]+ tupleIndex[1] * self.stepSize/2) )
        delta = (self.omega[tupleIndex[0], tupleIndex[1]] - self.theoOm, self.tt[tupleIndex[0], tupleIndex[1]] - self.theoTT)
        print('delta = ' + str(delta))
        
        [qx, qy, qz] = self.hxrd.Ang2Q(self.omega, self.tt, delta=[expOm - self.theoOm, expTT - self.theoTT] )
        
        #print('delta = ' + str(delta))
        #[qx, qy, qz] = self.hxrd.Ang2Q(self.omega, self.tt, delta = [ -0.031223667727843463, -0.11234498129975634 ])
        #[qx, qy, qz] = self.hxrd.Ang2Q(self.omega, self.tt, delta = [ 0,0 ])
        return p, [qx, qy, qz]

    def fit_zoom_peak( self, angPlot, *kwargs ):

        yminInd =  ( np.abs(self.tt[:,0] - angPlot.get_ylim()[0]) ).argmin()
        ymaxInd =  ( np.abs(self.tt[:,0] - angPlot.get_ylim()[1]) ).argmin()
        xminInd =  ( np.abs(self.omega[0, :] - angPlot.get_xlim()[0]) ).argmin()
        xmaxInd =  ( np.abs(self.omega[0, :] - angPlot.get_xlim()[1]) ).argmin() 

        fitRange = [self.data[0, xminInd], self.data[0, xmaxInd], self.data[yminInd, 0], self.data[ymaxInd, 0]]

        tupleIndex = np.unravel_index(np.argmax(self.data[yminInd:ymaxInd, xminInd:xmaxInd].flatten()), \
                     (len(self.data[yminInd:ymaxInd, 0]), len(self.data[0, xminInd:xmaxInd])))

        cropOm = self.omega[yminInd:ymaxInd, xminInd:xmaxInd]
        cropTT = self.tt[yminInd:ymaxInd, xminInd:xmaxInd]
        cropData = self.data[yminInd:ymaxInd, xminInd:xmaxInd]
        #cropX, cropY = np.meshgrid(cropX, cropY)
        xC = cropOm[tupleIndex]
        yC = cropTT[tupleIndex]
        amp = self.gridder.data[tupleIndex]
        xSigma = 0.1
        ySigma = 0.1
        angle = 0
        background = 1
        self.p = [xC, yC, xSigma, ySigma, amp, background, angle]
        #params = [xC, yC, 0.001, 0.001, cropData.max, 0, 0.]
        #gaussFit = xu.fitpeak2d(cropX.flatten(), cropY.flatten(), cropData.flatten(), p, drange=[cropX[0], cropX[-1], cropY[0], cropY[-1]], xu.math.functions.Gauss2d(cropX, cropY,))

        fitRange = [self.omega[0,xminInd], self.omega[0,xmaxInd], self.tt[yminInd, 0], self.tt[ymaxInd, 0] ]

        fitParams, cov = xu.math.fit.fit_peak2d(self.omega, self.tt, self.data, self.p, fitRange, xu.math.Gauss2d)

        cl = angPlot.contour(self.omega[0,xminInd:xmaxInd], self.tt[yminInd:ymaxInd,0], \
                 np.log10(xu.math.Gauss2d(self.omega[yminInd:ymaxInd, xminInd:xmaxInd], \
                 self.tt[yminInd:ymaxInd,xminInd:xmaxInd], *fitParams)), 8, colors='k', linestyles='solid')

        return cl, fitParams, cov
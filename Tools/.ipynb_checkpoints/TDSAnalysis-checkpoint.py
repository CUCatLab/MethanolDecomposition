import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import xarray as xr
import re
import yaml
import struct

sys.path.append(os.getcwd() + '/Tools/')

# Plotly settings
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.9,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)'
))

class DataTools :
    
    def __init__(self) :
        
        pass
    
    def FileList(self,FolderPath,Filter) :
        
        FileList = [f for f in listdir(FolderPath) if isfile(join(FolderPath, f))]
        for i in range(len(Filter)):
            FileList = [k for k in FileList if Filter[i] in k]
        for i in range(len(FileList)):
            FileList[i] = FileList[i].replace('.yaml','')
        
        return FileList
    
    def LoadData(self,ParameterFile) :
        
        with open(ParameterFile[0]+'/'+ParameterFile[1]+'.yaml', 'r') as stream:
            Parameters = yaml.safe_load(stream)
        
        FolderPath = Parameters['FolderPath']
        FileName = Parameters['FileName']
        Masses = Parameters['Masses']

        with open(FolderPath + '/' + FileName, mode='rb') as file:
            fileContent = file.read()

        NumChan = len(Masses) + 1
        DataLength = int((len(fileContent)-5)/(46*NumChan))
        Data = np.zeros((int(1+NumChan),DataLength))

        for i in range(len(Data)) :
            for j in range(len(Data[0])) :
                if i == 0 :
                    index = int(31+j*46*NumChan)
                    Data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]/1000
                else :
                    index = int(43+j*46*NumChan + (i-1)*46)
                    Data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]
        
        Header = list()
        Header.append('Time (s)')
        for idx in range(NumChan) :
            if idx == 0 :
                Header.append('Temperature (K)')
            else :
                Header.append('Mass '+str(Masses[idx-1]))
        Data = df(np.transpose(Data),columns=Header)
        Data = Data.set_index('Temperature (K)')
        
        Parameters['HeatingRate'] = np.mean(np.diff(Data.index)/np.diff(Data['Time (s)']))
            
        return Data, Parameters
    
    def TrimData(self,Data,Min,Max) :
        
        Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
        Data = Data[Mask]
        
        return Data

class TDS :
    
    def __init__(self,ParameterFile) :
        
        dt = DataTools()
        
        Data, Parameters = dt.LoadData(ParameterFile)
        Assignments = Assignments = df(Parameters['Assignments'],index=Parameters['Masses'],columns=['Assignments'])
        
        self.Assignments = Assignments
        self.ParameterFile = ParameterFile
        self.Parameters = Parameters
        self.Data = Data
    
    def ReloadData(self) :
        
        dt = DataTools()
        
        ParameterFile = self.ParameterFile
        
        Data, Parameters = dt.LoadData(ParameterFile)
        
        self.Data = Data
        self.Parameters = Parameters
    
    def SimulateData(self,Rate) :
        
        Data = self.Data
        Parameters = self.Parameters
        
        if 'Simulations' in Parameters :

            # Initial parameters
            kB = 8.617e-5                 # eV/K
            T0 = 100                      # K
            
            Temperature, deltaT = np.linspace(min(Data.index),max(Data.index),1001,retstep =True)
            Time = Temperature / Rate
            deltat = deltaT / Rate
            Size = len(Temperature)
            
            Traces = df(index=Temperature)
            Coverages = df(index=Temperature)
            
            # Calculate traces
            for Mass in Parameters['Simulations'] :
                Trace = np.zeros((Size))
                Coverage = np.zeros((Size))
                for idx, Peak in enumerate(Parameters['Simulations'][Mass]) :
                    PeakParameters = Parameters['Simulations'][Mass][Peak]
                    Offset = PeakParameters['Offset']
                    Scaling = PeakParameters['Scaling']
                    Ni = PeakParameters['Coverage']
                    Ea = PeakParameters['Barrier']
                    nu = PeakParameters['Prefactor']
                    n = PeakParameters['Order']
                    
                    PeakTrace = np.zeros((Size))
                    PeakCoverage = np.zeros((Size))
                    IntRate = 0
                    for idx, T in enumerate(Temperature) :
                        PeakTrace[idx] = nu*(Ni - IntRate)**n * np.exp(-Ea/(kB*T))
                        IntRate += PeakTrace[idx] * deltat
                        PeakCoverage[idx] = Ni - IntRate
                        if IntRate >= Ni :
                            IntRate = Ni
                            PeakCoverage[idx] = 0
                        if PeakCoverage[idx] < 0 or PeakCoverage[idx] > Ni :
                            PeakCoverage[idx] = 0
                            PeakTrace[idx] = 0
                    Trace += PeakTrace * Scaling + Offset
                    Coverage += PeakCoverage
                
                Traces[Mass] = Trace
                Coverages[Mass] = Coverage
                
        self.SimulatedData = Traces
        self.SimulatedCoverages = Coverages
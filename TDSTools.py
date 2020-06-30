from os import listdir
from os.path import isfile, join
import numpy as np
from pandas import DataFrame as df
import xarray as xr
import re
import yaml
import struct

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
        Header = 31
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
        Data = Data[1:]
        
        Header = list()
        for idx in range(len(Data)) :
            if idx == 0 :
                Header.append('Temperature (K)')
            else :
                Header.append('Mass '+str(idx))
        Data = df(np.transpose(Data),columns=Header)
            
        return Data, Parameters
    
    def TrimData(self,Data,Min,Max) :
        
        iMin = (np.abs(Data['Temperature (K)'].values - Min)).argmin()
        iMax = (np.abs(Data['Temperature (K)'].values - Max)).argmin() + 1
        Data = df.drop(Data,index=range(iMax,len(Data)))
        Data = df.drop(Data,index=range(iMin))
        
        return Data
    
class TDS :
    
    def __init__(self,ParameterFile) :
        
        dt = DataTools()
        
        Data, Parameters = dt.LoadData(ParameterFile)
        
        self.ParameterFile = ParameterFile
        self.Parameters = Parameters
        self.Data = Data
    
    def ReloadData(self) :
        
        dt = DataTools()
        
        ParameterFile = self.ParameterFile
        
        Data, Parameters = dt.LoadData(ParameterFile)
        
        self.Data = Data
        self.Parameters = Parameters
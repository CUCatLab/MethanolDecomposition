import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import cmath
import igor.igorpy as igor
import re
import yaml
import struct
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel

# SIF reader
sys.path.append(os.getcwd() + '/Tools/sif_reader/')
sys.path.append(os.getcwd() + '/Tools/')
import sif_reader

##### Data Tools #####

class DataTools :
    
    def __init__(self) :
        
        self.ParametersFolder = os.getcwd()+'/Parameters'
        self.FitsFolder = os.getcwd()+'/Fits'
        self.FiguresFolder = os.getcwd()+'/Figures'
    
    def FileList(self,FolderPath,Filter) :
        
        FileList = [f for f in listdir(FolderPath) if isfile(join(FolderPath, f))]
        for i in range(len(Filter)):
            FileList = [k for k in FileList if Filter[i] in k]
        for i in range(len(FileList)):
            FileList[i] = FileList[i].replace('.yaml','')
        
        return FileList
    
    def Load_SFG(self,ParameterFile) :
        
        with open(ParameterFile[0]+'/'+ParameterFile[1]+'.yaml', 'r') as stream:
            Parameters = yaml.safe_load(stream)
        
        FolderPath = Parameters['FolderPath']
        FileName = Parameters['FileName']

        if FileName.endswith('.ibw') :
            d = binarywave.load(FolderPath + '/' + FileName)
            y = np.transpose(d['wave']['wData'])
            Start = d['wave']['wave_header']['sfB']
            Delta = d['wave']['wave_header']['sfA']
            x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
            z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
            print('Igor binary data loaded')
        elif FileName.endswith('.itx') :
            y = np.loadtxt(FolderPath + '/' + FileName,comments =list(string.ascii_uppercase))
            y = np.transpose(y)
            with open(FolderPath + '/' + FileName) as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if 'SetScale/P' in row[0]:
                        SetScale = row[0]
            xScale = re.findall(r' x (.*?),"",', SetScale)
            xScale = xScale[0].split(',')
            zScale = re.findall(r' y (.*?),"",', SetScale)
            zScale = zScale[0].split(',')
            Start = [float(xScale[0]),float(zScale[0])]
            Delta = [float(xScale[1]),float(zScale[1])]
            x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
            z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
            print('Igor text data loaded')
        elif FileName.endswith('.pxp') :
            DataName = Parameters['DataName']
            igor.ENCODING = 'UTF-8'
            d = igor.load(FolderPath + '/' + FileName)
            for i in range(len(d.children)) :
                if 'data' in str(d[i]) and len(d[i].data) < 10000 :
                    globals()[d[i].name] = np.array(d[i].data)
                    if len(d[i].axis[0]) > 0 :
                        Name = d[i].name+'_x'
                        globals()[Name] = np.array([])
                        for j in range(len(d[i].axis[0])) :
                            globals()[Name] = np.append(globals()[Name], d[i].axis[0][-1] + d[i].axis[0][0] * j)
                    if len(d[i].axis[1]) > 0 :
                        globals()[d[i].name] = np.transpose(globals()[d[i].name])
                        Name = d[i].name+'_y'
                        globals()[Name] = np.array([])
                        for j in range(len(d[i].axis[1])) :
                            globals()[Name] = np.append(globals()[Name], d[i].axis[1][-1] + d[i].axis[1][0] * j)
            x = eval(DataName+'_x')
            y = eval(DataName)
            z = eval(DataName+'_y')
            z = np.round(z,decimals=1)

        elif FileName.endswith('sif') :
            FileData = sif_reader.xr_open(FolderPath + '/' + FileName)
            y = FileData.values[:,0,:]
            x = [i for i in range(len(np.transpose(y)))]
            z = [i+1 for i in range(len(y))]
            
            try :
                FileData.attrs['WavelengthCalibration0']
                FileData.attrs['WavelengthCalibration1']
                FileData.attrs['WavelengthCalibration2']
                FileData.attrs['WavelengthCalibration3']
            except :
                print('Warning: Wavelength calibration not found')
            else :
                c0 = FileData.attrs['WavelengthCalibration0']
                c1 = FileData.attrs['WavelengthCalibration1']
                c2 = FileData.attrs['WavelengthCalibration2']
                c3 = FileData.attrs['WavelengthCalibration3']
                for i in x :
                    x[i] = c0 + c1*i + c2*1**2 + c3*i**3
                x = np.array(x)
                x = 1e7 / x - 12500
            
            try :
                Frame = Parameters['Heating']['Frame']
                Temperature = Parameters['Heating']['Temperature']
            except :
                print('Warning: Temperature data not found')
            else :
                FitModel = QuadraticModel()
                ModelParameters = FitModel.make_params()
                FitResults = FitModel.fit(Temperature, ModelParameters,x=Frame)
                idx = np.array(z)
                z = FitResults.eval(x=idx)
                z = np.round(z,1)
        
        Data = df(np.transpose(y),index=x,columns=z)
            
        return Data, Parameters
    
    def Load_TDS(self,ParameterFile) :
        
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
    
    def RemoveEmptyDataSets(self,Data,Threshold) :
        
        Index = list()
        for i in Data.columns :
            if np.mean(Data[i]) < Threshold :
                Index.append(i)
        for i in Index :
            del Data[i]
        
        return Data
    
    def TrimData(self,Data,Min,Max) :
        
        Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
        Data = Data[Mask]
        
        return Data
    
    def ReduceResolution(self,Data,Resolution=1) :
        
        Counter = 0
        ReducedData = df()
        for i in range(int(len(Data.columns.values)/Resolution)) :
            Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
            ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
            Counter = Counter + Resolution
        
        return ReducedData

##### Fit Tools #####

class FitTools :
    
    def __init__(self,Data,FitInfo,Name='') :
        
        self.Data = Data
        self.FitInfo = FitInfo
        self.Name = Name
        
        try :
            FitInfo['ModelType']
            FitInfo['Models']
        except:
            ModelType = 'None'
            ModelString = ''
        else :
            if FitInfo['ModelType'] == 'BuiltIn' :
                self.BuiltInModels()
            if FitInfo['ModelType'] == 'SFG' :
                self.SFGModel()
    
    def BuiltInModels(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append((key,FitInfo['Models'][key]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        self.FitModel = FitModel
        self.ModelParameters = FitModel.make_params()
        
    def SFGModel(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append([key])
        
        if len(ModelString) == 2 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 3 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 4 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 5 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 6 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 7 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 8 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak7_amp*(cmath.exp(Peak7_phi*1j)/(x-Peak7_omega+Peak7_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 9 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma,
                            Peak8_amp,Peak8_phi,Peak8_omega,Peak8_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak8_amp*(cmath.exp(Peak8_phi*1j)/(x-Peak8_omega+Peak8_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        
        FitModel = Model(SFGFunction)
        ModelParameters = FitModel.make_params()
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
    
    def SetParameters(self, Value = None) :
        
        FitInfo = self.FitInfo
        ModelParameters = self.ModelParameters
        
        ParameterList = ['amp','phi','omega','gamma','center','sigma','c']
        Parameters = {'Standard': FitInfo['Models']}

        if 'Cases' in FitInfo and Value != None:
            for Case in FitInfo['Cases'] :
                if Value >= min(FitInfo['Cases'][Case]['zRange']) and Value <= max(FitInfo['Cases'][Case]['zRange']) :
                    Parameters[Case] = FitInfo['Cases'][Case]
        
        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
        
        self.ModelParameters = ModelParameters
    
    def Fit(self,**kwargs) :
        
        for kwarg in kwargs :
            if kwarg == 'fit_x':
                fit_x = kwargs[kwarg]
        
        dt = DataTools()
        
        Data = self.Data
        Name = self.Name
        FitModel = self.FitModel
        ModelParameters = self.ModelParameters
        FitInfo = self.FitInfo
        
        if 'xRange' in FitInfo :
            Data = dt.TrimData(Data,FitInfo['xRange'][0],FitInfo['xRange'][1])
        x = Data.index.values
        try:
            fit_x
        except :
            try :
                NumberPoints
            except :
                fit_x = x
            else :
                for i in NumberPoints :
                    fit_x[i] = min(x) + i * (max(x) - min(x)) / (Numberpoints - 1)
        
        Fits = df(index=fit_x,columns=Data.columns.values)
        FitsParameters = df(index=ModelParameters.keys(),columns=Data.columns.values)
        FitsResults = list()
        FitsComponents = list()
        
        for idx,Column in enumerate(Data) :
            
            self.SetParameters(Column)
            
            y = Data[Column].values
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
            fit_y = FitResults.eval(x=fit_x)
            ParameterNames = [i for i in FitResults.params.keys()]
            for Parameter in (ParameterNames) :
                FitsParameters[Column][Parameter] = FitResults.params[Parameter].value
            Fits[Column] = fit_y
            FitsResults.append(FitResults)
            FitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(Data.shape[1])) % (idx+1))
            sys.stdout.flush()
        
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
    
    def ShowFits(self,xLabel='',yLabel='') :
        
        Data = self.Data
        Fits = self.Fits
        FitInfo = self.FitInfo
        
        FitsParameters = self.FitsParameters
        FitsComponents = self.FitsComponents
        
        for idx,Column in enumerate(Data) :
            
            plt.figure(figsize = [6,4])
            plt.plot(Data.index, Data[Column],'k.', label='Data')
            plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
            for Component in FitsComponents[idx] :
                if not isinstance(FitsComponents[idx][Component],float) :
                    plt.fill(Fits.index, FitsComponents[idx][Component], '--', label=Component, alpha=0.5)
            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
            plt.xlabel(xLabel), plt.ylabel(yLabel)
            if 'xRange' in FitInfo :
                plt.xlim(FitInfo['xRange'][0],FitInfo['xRange'][1])
            plt.title(str(Column))
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)

            string = ''
            for Peak in Peaks :
                string = string + Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string = string + Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                        string = string + ', '
                string = string[:-2] + '\n'
            print(string)
            print(75*'_')
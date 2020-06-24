import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import cmath
import xarray as xr
from PIL import Image
import igor.igorpy as igor
import re
import yaml
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
import ipywidgets as widgets

This_Dir = os.getcwd()
sys.path.append(This_Dir + '/sif_reader/')
import sif_reader

class FitTools :
    
    def __init__(self,x,y,Name='') :
        
        self.x = x
        self.y = y
        self.Name = Name

    def BuiltInModels(self,ModelString) :
        
        for i in ModelString :
            
            try :
                FitModel
            except :
                if i[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=i[0]+'_')
                if i[1] == 'Linear' :
                    FitModel = LinearModel(prefix=i[0]+'_')
                if i[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=i[0]+'_')
                if i[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=i[0]+'_')
                if i[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=i[0]+'_')
            else :
                if i[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=i[0]+'_')
                if i[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=i[0]+'_')
                if i[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=i[0]+'_')
                if i[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=i[0]+'_')
                if i[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=i[0]+'_')
        
        self.ModelType = 'BuiltIn'
        self.ModelString = ModelString
        self.FitModel = FitModel
        self.ModelParameters = FitModel.make_params()
        
    def SFGModel(self,ModelString) :
        
        Parameters = ModelString
        
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
        
        self.ModelType = 'SFG'
        self.ModelString = ModelString
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
    
    def SetParameters(self,Parameters) :
        
        for Model in Parameters :
            for Parameter in Parameters[Model] :
                if Parameter != 'model' and Parameter != 'assignment' :
                    for Key in Parameters[Model][Parameter] :
                        exec('self.ModelParameters["'+Model+'_'+Parameter+'"].'+Key+'='+str(Parameters[Model][Parameter][Key]))
                        
        self.Parameters = Parameters
    
    def Fit(self,**kwargs) :
        
        x = self.x
        y = self.y
        Name = self.Name
        FitModel = self.FitModel
        Parameters = self.Parameters
        ModelParameters = self.ModelParameters
        FitResults = FitModel.fit(y, ModelParameters, x=x)
        
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
                
        fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
        fit_y = FitResults.eval(x=fit_x)
        
        ParameterNames = [i for i in FitResults.params.keys()]
        FitParameters = np.zeros((1,len(ParameterNames)))
        for i in range(len(ParameterNames)) :
            FitParameters[0,i] = FitResults.params[ParameterNames[i]].value
        if Name == '' :
            FitParameters = df(data=FitParameters,columns=ParameterNames)
        else :
            FitParameters = df(data=FitParameters,columns=ParameterNames,index=[Name])
            
        for idx,name in enumerate(FitParameters.columns):
            Model = name.split('_')
            try :
                Parameters[Model[0]]['assignment']
            except :
                pass
            else :
                OldName = Model[0] + '_' + Model[1]
                NewName = Parameters[Model[0]]['assignment'] + '_' + Model[1]
                FitParameters.rename(columns={OldName:NewName}, inplace=True)
        
        Fit = np.array((fit_x,fit_y))
        Fit = df(np.transpose(Fit),columns=('x','y'))
        
        self.x = x
        self.y = y
        self.FitParameters = FitParameters
        self.Fit = Fit
        self.FitResults = FitResults
        self.fit_comps = fit_comps
    
    def Plot(self,Title='',xLabel='',yLabel='') :
        
        x = self.x
        y = self.y
        Name = self.Name
        ModelString = self.ModelString
        ModelType = self.ModelType
        Fit = self.Fit
        fit_comps = self.fit_comps
        
        if Title == '' and not Name == '' :
            Title = str(Name)
        
        plt.figure(figsize = [6,4])
        plt.plot(x, y,'k.', label='Data')
        if ModelType == 'BuiltIn' :
            plt.plot(Fit['x'], Fit['y'], 'r--', label='Fit')
            for i in ModelString :
                if i[1] == 'Linear' :
                    plt.plot(Fit['x'], fit_comps[i[0]+'_'], 'k--', label=i[0])
                if i[1] == 'Gaussian' :
                    plt.fill(Fit['x'], fit_comps[i[0]+'_'], '--', label=i[0], alpha=0.5)
                if i[1] == 'SkewedGaussian' :
                    plt.fill(Fit['x'], fit_comps[i[0]+'_'], '--', label=i[0], alpha=0.5)
                if i[1] == 'Voigt' :
                    plt.fill(Fit['x'], fit_comps[i[0]+'_'], '--', label=i[0], alpha=0.5)
        elif ModelType == 'SFG' :
            plt.plot(Fit['x'], Fit['y'], 'r-', label='Fit')
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.xlabel(xLabel), plt.ylabel(yLabel)
        plt.xlim(min(x),max(x))
        plt.title(Title)
        plt.show()
    
    def PrintParameters(self) :
        
        FitParameters = self.FitParameters
        Parameters = self.Parameters

        ParameterNames = list()
        ParameterAttributes = list()

        for Parameter in FitParameters.columns :
            Name = Parameter.split('_')[0]
            if Name not in ParameterNames :
                ParameterNames.append(Name)

        ParameterAttributes = {}
        for i in ParameterNames :
            Attributes = {}
            for j in FitParameters :
                if j.startswith(i) :
                    Attributes.update({j.split('_')[1]: FitParameters[j].values[0]})
            ParameterAttributes.update({i:Attributes})
        
        FitParameters = ParameterAttributes
        string = ''
        for i in FitParameters :
            string = string + i + ' | '
            for idx, j in enumerate(FitParameters[i]) :
                string = string + j + ': ' + str(round(FitParameters[i][j],2))
                if idx < len(FitParameters[i]) - 1 :
                    string = string  + ', '
            string = string + '\n'
        print(string)

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
        
        with open(ParameterFile+'.yaml', 'r') as stream:
            Parameters = yaml.safe_load(stream)
        
        FolderPath = Parameters['FolderPath']
        FileName = Parameters['FileName']
        DataName = Parameters['DataName']

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
    
    def RemoveEmptyDataSets(self,Data,Threshold) :
        
        Index = list()
        for i in Data.columns :
            if np.mean(Data[i]) < Threshold :
                Index.append(i)
        for i in Index :
            del Data[i]
        
        return Data
    
    def TrimData(self,x,y,xMin,xMax) :
        
        Min_Index = (np.abs(x - xMin)).argmin()
        Max_Index = (np.abs(x - xMax)).argmin()
        x = x[Min_Index:Max_Index]
        y = y[Min_Index:Max_Index]
        
        return x, y
    
    def ReduceData(self,Data,CutoffTemp,Resolution=1) :
        
        Cutoff = (np.abs(Data.columns.values - CutoffTemp)).argmin() + 1
        Data = Data.drop(Data.columns[Cutoff:], 1)
        
        Counter = 0
        ReducedData = df()
        for i in range(int(len(Data.columns.values)/Resolution)) :
            Temperature = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
            ReducedData[Temperature] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
            Counter = Counter + Resolution
        
        return ReducedData
    
    def Plot(self,x,y,Title='',xLabel='',yLabel='') :

        plt.figure(figsize = [6,4])
        plt.plot(x, y,'k.')
        plt.xlabel(xLabel), plt.ylabel(yLabel)
        plt.title(Title)
        plt.show()
    
class SFG :
    
    def __init__(self,ParameterFile) :
        
        dt = DataTools()
        
        Data, Parameters = dt.LoadData(ParameterFile)
        
        Threshold = Parameters['Background']['Threshold']
        
        Data = dt.RemoveEmptyDataSets(Data,Threshold)
        
        self.ParameterFile = ParameterFile
        self.Parameters = Parameters
        self.Data = Data
    
    def ReloadData(self) :
        
        dt = DataTools()
        
        ParameterFile = self.ParameterFile
        
        Data, Parameters = dt.LoadData(ParameterFile)
        
        self.Data = Data
        self.Parameters = Parameters
    
    def GetBackground(self) :
        
        Parameters = self.Parameters
        Data = self.Data
        
        TMin = Parameters['Background']['TempRange'][0]
        TMax = Parameters['Background']['TempRange'][1]
        
        DataNames = list()
        for i in Data.columns :
            if i > TMin and i < TMax :
                DataNames.append(i)
        Background = df(Data[DataNames].mean(axis=1),columns=['Data'])
        
        try :
            Parameters['Background']['Models']
        except :
            dt = DataTools()
            dt.Plot(Data.index.values,Background['Data'].values,Title='Background',xLabel='Wavenumber',yLabel='Intensity')
        else :
            x = Background.index.values
            y = Background['Data'].values
            fit = FitTools(x,y,'Background')
            Models = Parameters['Background']['Models']
            ModelString = list()
            for key in Models :
                ModelString.append([key,Models[key]['model']])
            fit.BuiltInModels(ModelString)
            fit.SetParameters(Parameters['Background']['Models'])
            fit.Fit()
            fit.Plot(Title='Background',xLabel='Wavenumber',yLabel='Intensity')
            fit.PrintParameters()
            
            Background['Fit'] = fit.Fit['y'].values
            self.BackgroundFit = {'FitModel': fit.FitModel, 'FitParameters': fit.FitParameters, 'FitResults': fit.FitResults, 'ModelParameters': fit.ModelParameters}
            
        print("_"*100)
        
        self.Background = Background
    
    def FitData(self) :
        
        dt = DataTools()
        
        Parameters = self.Parameters
        Data = self.Data
        Background = self.Background
        
        xMin = Parameters['ROI'][0]
        xMax = Parameters['ROI'][1]
        TempCutoff = Parameters['Background']['TempRange'][0]
        Resolution = Parameters['Resolution']
        
        Data = dt.ReduceData(Data,TempCutoff,Resolution)
        
        try :
            Background['Fit']
        except :
            Backgroundy = Background['Data'].values
        else :
            Backgroundy = Background['Fit'].values
        x = Data.index.values
        FitParameters = df()
        Models = Parameters['Data']['Models']
        ModelString = list()
        for key in Models :
            ModelString.append([key])
        for i in Data.columns :
            
            y = Data[i].values
            y1 = y / Backgroundy
            x1,y1 = dt.TrimData(x,y1,xMin,xMax)
            fit = FitTools(x1,y1,i)
            fit.SFGModel(ModelString)
            fit.SetParameters(Parameters['Data']['Models'])
            fit.Fit(fit_x=x)
            fit.Plot(Title=('Temperature: ' + str(i) + ' K'),xLabel='Wavenumber ($cm^{-1}$)',yLabel='Intensity (au)')
            fit.PrintParameters()
            
            fit_y = fit.FitResults.eval(x=x)
            fit_y = fit_y * Backgroundy
            fig = plt.figure(figsize=(5,5))
            plt.plot(x, y,'k.', label=str(i)+' K')
            plt.plot(x,fit_y, 'r-', label='fit')
            plt.xlabel('Wavenumber ($cm^{-1}$)')
            plt.ylabel('Temperature (K)')
            plt.title('Data: '+Parameters['DataName'])
            plt.legend(), plt.xlabel('Energy (cm-1)'), plt.ylabel('Signal (au)')
            plt.show()
            
            try :
                Fits
            except :
                Fits = df(fit_y,index=x,columns=[i])
            else :
                Fits[i] = fit_y
            FitParameters = FitParameters.append(fit.FitParameters)
            
            print("_"*100)
        
        x = Data.index.values
        y = Data.columns.values
        z = np.transpose(Data.values)
        
        fig = plt.figure(figsize=(8,5))
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        pcm = plt.pcolor(x, y, z, cmap='jet')
        plt.show()
        
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        
        fig = plt.figure(figsize=(8,5))
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        pcm = plt.pcolor(x, y, z, cmap='jet')
        plt.show()
        
        self.ModelParameters = fit.ModelParameters
        self.Data2Fit = Data
        self.Fits = Fits
        self.FitParameters = FitParameters
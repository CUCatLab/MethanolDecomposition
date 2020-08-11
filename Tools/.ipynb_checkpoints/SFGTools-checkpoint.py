import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import cmath
import xarray as xr
import igor.igorpy as igor
import re
import yaml
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
import ipywidgets as widgets
from IPython.display import clear_output

# SIF reader
sys.path.append(os.getcwd() + '/Tools/sif_reader/')
sys.path.append(os.getcwd() + '/Tools/')
import sif_reader

# Plotly settings
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.95,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)',
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=50, #top margin
        )
))

SmallPlotLayout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
        ),
    width=300,
    height=300,
    hoverdistance=100, # Distance to show hover label of data point
    spikedistance=1000, # Distance to show spike
    xaxis=dict(
        showspikes=True, # Show spike line for X-axis
        spikethickness=2,
        spikedash="dot",
        spikecolor="#999999",
        spikemode="across",
        showgrid=False
        ),
    yaxis=dict(
        showgrid=False
        ),
    legend=dict(
        itemclick="toggleothers",
        itemdoubleclick="toggle",
        ),
    )

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
    
    def LoadData(self,InfoFile) :
        
        with open(InfoFile[0]+'/'+InfoFile[1]+'.yaml', 'r') as stream:
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
    
    def ReduceData(self,Data,yCutoff=2000,Resolution=1) :
        
        Cutoff = (np.abs(Data.columns.values - yCutoff)).argmin() + 1
        Data = Data.drop(Data.columns[Cutoff:], 1)
        
        Counter = 0
        ReducedData = df()
        for i in range(int(len(Data.columns.values)/Resolution)) :
            Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
            ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
            Counter = Counter + Resolution
        
        return ReducedData

class FitTools :
    
    def __init__(self,Data,FitInfo,Name='') :
        
        try :
            FitInfo['ModelType']
            FitInfo['Models']
        except:
            ModelType = 'None'
            ModelString = ''
        else :
            if FitInfo['ModelType'] == 'BuiltIn' :
                self.BuiltInModels(FitInfo)
            if FitInfo['ModelType'] == 'SFG' :
                self.SFGModel(FitInfo)
            for Model in FitInfo['Models'] :
                for Parameter in FitInfo['Models'][Model] :
                    if Parameter != 'model' and Parameter != 'assignment' :
                        for Key in FitInfo['Models'][Model][Parameter] :
                            exec('self.ModelParameters["'+Model+'_'+Parameter+'"].'+Key+'='+str(FitInfo['Models'][Model][Parameter][Key]))
        
        self.Data = Data
        self.FitInfo = FitInfo
        self.Name = Name

    def BuiltInModels(self,FitInfo) :
        
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
        
    def SFGModel(self,FitInfo) :
        
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
    
    def Fit(self,**kwargs) :
        
        for kwarg in kwargs :
            if kwarg == 'fit_x':
                fit_x = kwargs[kwarg]
        
        data = DataTools()
        
        Data = self.Data
        Name = self.Name
        FitModel = self.FitModel
        ModelParameters = self.ModelParameters
        FitInfo = self.FitInfo
        
        if 'Range' in FitInfo :
            Data = data.TrimData(Data,FitInfo['Range'][0],FitInfo['Range'][1])
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
            y = Data[Column].values
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
            fit_y = FitResults.eval(x=fit_x)
            ParameterNames = [i for i in FitResults.params.keys()]
            FitParameters = np.zeros((1,len(ParameterNames)))
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
            if 'Range' in FitInfo :
                plt.xlim(FitInfo['Range'][0],FitInfo['Range'][1])
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

class SFG :
    
    def __init__(self,InfoFile) :
        
        data = DataTools()
        
        Data, Info = data.LoadData(InfoFile)
        Threshold = Info['Background']['Threshold']
        Data = data.RemoveEmptyDataSets(Data,Threshold)
        
        self.InfoFile = InfoFile
        self.Info = Info
        self.Data = Data
    
    def ReloadData(self) :
        
        data = DataTools()
        
        InfoFile = self.InfoFile
        
        Data, Info = data.LoadData(InfoFile)
        
        self.Data = Data
        self.Info = Info
    
    def FitData(self) :
        
        Data = self.Data
        Info = self.Info
        
        ##### Fit Data #####
        
        data = DataTools()

        TBackground = Info['Background']['TempRange']
        DataNames = list()
        for i in Data.columns :
            if i >= min(TBackground) and i <= max(TBackground) :
                DataNames.append(i)
        Background = df(Data[DataNames].mean(axis=1),columns=['Data'])
        
        Resolution = Info['Resolution']
        yCutOff = max(TBackground)
        Data = data.ReduceData(Data,yCutOff,Resolution)
        
        try :
            Info['Background']['Models']
        except :
            Data_BC = Data.divide(Background['Data'],axis=0)
        else :
            print('Fitting Background')
            fit = FitTools(Background,Info['Background'],'Background')
            fit.Fit()
            Background['Fit'] = fit.Fits['Data']
            Data_BC = Data.divide(Background['Fit'],axis=0)
            if 'Range' in Info['Fit'] :
                Data_BC = data.TrimData(Data_BC,Info['Fit']['Range'][0],Info['Fit']['Range'][1])
        
        print('\nFitting Data')

        fit = FitTools(Data_BC,Info['Fit'])
        fit.Fit(fit_x=Data.index.values)

        Fits_BC = fit.Fits
        FitsParameters = fit.FitsParameters
        
        if 'Fit' in Background :
            Fits = Fits_BC.multiply(Background['Fit'],axis=0)
        else :
            Fits = Fits_BC.multiply(Background['Data'],axis=0)
        
        print('\nDone fitting data')
        print('\n'+100*'_')
        
        ##### Show Fits & Data #####
        
        plt.figure(figsize = [6,4])
        plt.plot(Background.index, Background['Data'],'k.', label='Data')
        if 'Fit' in Background :
            plt.plot(Background.index, Background['Fit'], 'r-', label='Fit')
        plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
        plt.title('Background')
        plt.show()
        
        print(100*'_')
        
        for Column in Data :
    
            plt.figure(figsize = [12,4])

            plt.subplot(1, 2, 1)
            plt.plot(Data.index, Data[Column],'k.', label='Data')
            plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
            plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
            plt.title('Temperature: '+str(Column)+' K')

            plt.subplot(1, 2, 2)
            plt.plot(Data_BC.index, Data_BC[Column],'k.', label='Data')
            plt.plot(Fits_BC.index, Fits_BC[Column], 'r-', label='Fit')
            plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
            if 'Range' in Info['Fit'] :
                plt.xlim(Info['Fit']['Range'][0],Info['Fit']['Range'][1])

            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
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
            print(100*'_')
        
        plt.figure(figsize = [8,12])
        
        plt.subplot(2, 1, 1)
        x = Data.index.values
        y = Data.columns.values
        z = np.transpose(Data.values)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Data', fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet')
        
        plt.subplot(2, 1, 2)
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Fits', fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet')
        
        plt.show()
        
        plt.figure(figsize = [8,16])
        
        plt.subplot(4, 1, 1)
        for Parameter in FitsParameters.T :
            if 'amp' in Parameter :
                plt.plot(FitsParameters.T[Parameter],'--.', label=Parameter)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Amplitude (au)', fontsize=12)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        
        plt.subplot(4, 1, 2)
        for Parameter in FitsParameters.T :
            if 'omega' in Parameter :
                plt.plot(FitsParameters.T[Parameter],'--.', label=Parameter)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Omega (cm$^{-1}$)', fontsize=12)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        
        plt.subplot(4, 1, 3)
        for Parameter in FitsParameters.T :
            if 'phi' in Parameter :
                plt.plot(FitsParameters.T[Parameter],'--.', label=Parameter)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Phi (Radians)', fontsize=12)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        
        plt.subplot(4, 1, 4)
        for Parameter in FitsParameters.T :
            if 'gamma' in Parameter :
                plt.plot(FitsParameters.T[Parameter],'--.', label=Parameter)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Gamma (cm$^{-1}$)', fontsize=12)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.show()
        
        self.Background = Background
        self.Data = Data
        self.Fits = Fits
        self.FitsParameters = FitsParameters
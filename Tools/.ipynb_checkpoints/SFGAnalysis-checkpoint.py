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
from multiprocessing import Pool

# SIF reader
sys.path.append(os.getcwd() + '/Tools/sif_reader/')
sys.path.append(os.getcwd() + '/Tools/')
import sif_reader
import AnalysisTools

##### Plotly settings ######

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


class SFG :
    
    def __init__(self,ParameterFile) :
        
        dt = AnalysisTools.DataTools()
        
        Data, Info = dt.Load_SFG(ParameterFile)
        Threshold = Info['Background']['Threshold']
        Data = dt.RemoveEmptyDataSets(Data,Threshold)
        
        self.ParameterFile = ParameterFile
        self.Info = Info
        self.Data = Data
    
    def ReLoad_SFG(self) :
        
        data = AnalysisTools.DataTools()
        
        ParameterFile = self.ParameterFile
        
        Data, Info = dt.Load_SFG(ParameterFile)
        
        self.Data = Data
        self.Info = Info
    
    def FitData(self) :
        
        Data = self.Data
        Info = self.Info
        
        if 'DataName' in Info :
            DataName = Info['DataName']
        else :
            DataName = Info['FileName']
        DataName = str.replace(DataName,'.sif','')
        
        print('Data: '+DataName)
        print('Description: '+Info['Description'])
        
        ##### Fit Data #####
        
        dt = AnalysisTools.DataTools()

        TBackground = Info['Background']['zRange']
        DataNames = list()
        for i in Data.columns :
            if i >= min(TBackground) and i <= max(TBackground) :
                DataNames.append(i)
        Background = df(Data[DataNames].mean(axis=1),columns=['Data'])
        
        Resolution = Info['Resolution']
        Data = dt.ReduceResolution(Data,Resolution)
        
        try :
            Info['Background']['Models']
        except :
            Data_BC = Data.divide(Background['Data'],axis=0)
        else :
            print('Fitting Background')
            fit = AnalysisTools.FitTools(Background,Info['Background'],'Background')
            fit.Fit()
            Background['Fit'] = fit.Fits['Data']
            Data_BC = Data.divide(Background['Fit'],axis=0)
        
        if 'xRange' in Info['Fit'] :
            Data_BC = dt.TrimData(Data_BC,Info['Fit']['xRange'][0],Info['Fit']['xRange'][1])
        
        if 'zRange' in Info['Fit'] :
            T_mask = []
            T_mask.append(Data.columns<=max(Info['Fit']['zRange']))
            T_mask.append(Data.columns>=min(Info['Fit']['zRange']))
            T_mask = np.all(T_mask, axis=0)
            Data_BC = Data_BC.T[T_mask].T
        
        fit = AnalysisTools.FitTools(Data_BC,Info['Fit'])
        fit.Fit(fit_x=Data.index.values)

        Fits_BC = fit.Fits
        FitsParameters = fit.FitsParameters
        
        if 'Fit' in Background :
            Fits = Fits_BC.multiply(Background['Fit'],axis=0)
        else :
            Fits = Fits_BC.multiply(Background['Data'],axis=0)
        
        print('\n'+100*'_')
        
        ##### Peak Assignments #####
        
        PeakList = list()
        AssignmentList = list()
        for Peak in Info['Fit']['Models'] :
            PeakList.append(Peak)
            if 'assignment' in Info['Fit']['Models'][Peak] :
                AssignmentList.append(Info['Fit']['Models'][Peak]['assignment'])
            else :
                AssignmentList.append(Peak)
        FitsAssignments = df(AssignmentList,index=PeakList,columns=['Assignment'])
        
        ##### Show Fits & Data #####
        
        if 'ShowFits' in Info['Fit'] :
            ShowFits = Info['Fit']['ShowFits']
        else :
            ShowFits = True

        if ShowFits :
            
            plt.figure(figsize = [6,4])
            plt.plot(Background.index, Background['Data'],'k.', label='Data')
            if 'Fit' in Background :
                plt.plot(Background.index, Background['Fit'], 'r-', label='Fit')
            plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
            plt.title('Background')
            plt.show()

            print(100*'_')
        
            for Column in Data_BC :

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
                if 'xRange' in Info['Fit'] :
                    plt.xlim(Info['Fit']['xRange'][0],Info['Fit']['xRange'][1])

                plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
                plt.show()

                Peaks = list()
                for Parameter in FitsParameters.index :
                    Name = Parameter.split('_')[0]
                    if Name not in Peaks :
                        Peaks.append(Name)

                string = ''
                for Peak in Peaks :
                    if 'assignment' in Info['Fit']['Models'][Peak] :
                        string += Info['Fit']['Models'][Peak]['assignment'] + ' | '
                    else :
                        string += Peak + ' | '
                    for Parameter in FitsParameters.index :
                        if Peak == Parameter.split('_')[0] : 
                            string += Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                            string += ', '
                    string = string[:-2] + '\n'
                print(string)
                print(100*'_')
        FitsParameters = FitsParameters.T
        FitsParameters = FitsParameters[np.concatenate((FitsParameters.columns.values[1:],FitsParameters.columns.values[0:1]))]
        
        # Plot 2D Data & Fits
        
        plt.figure(figsize = [8,12])
        
        plt.subplot(2, 1, 1)
        x = Data.index.values
        y = Data.columns.values
        z = np.transpose(Data.values)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Data: '+DataName, fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.subplot(2, 1, 2)
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Fits: '+DataName, fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.show()
        
        # Plot Trends
        
        UniqueParameters = []
        [UniqueParameters.append(x.split('_')[1]) for x in FitsParameters.columns if x.split('_')[1] not in UniqueParameters][0]
        for uniqueParameter in UniqueParameters :
            fig = go.Figure()
            for parameter in FitsParameters :
                if uniqueParameter in parameter :
                    Name = parameter.split('_')[0]
                    if 'assignment' in Info['Fit']['Models'][Name] :
                        Name = Info['Fit']['Models'][Name]['assignment']
                    fig.add_trace(go.Scatter(x=FitsParameters.index,y=FitsParameters[parameter],name=Name,mode='lines+markers'))
            fig.update_layout(xaxis_title='Temperature (K)',yaxis_title=uniqueParameter,title=DataName,legend_title='',width=800,height=400)
            fig.show()
        
        self.Background = Background
        self.Data = Data
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsAssignments = FitsAssignments
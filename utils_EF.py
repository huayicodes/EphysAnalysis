
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os, errno
import pandas as pd
from scipy.stats import ttest_ind_from_stats
import scipy.signal as signal
import scipy.interpolate as interp

def FiltV(V, N, Wn):
    # design the Buterworth filter & then apply the filer. 
    B, A = signal.butter(N, Wn, output='ba'); fV = signal.filtfilt(B,A,V)  
    return(fV)

def makedir(newdir):
    try:
        os.makedirs(newdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def meanAP(bins,rawdf,ls):    
    Mean = np.empty([len(ls),len(bins)-1])
    RawBin =np.empty(len(bins)-1)
    for n,i in enumerate(ls):
        AP = rawdf['Apdf'][rawdf['Apdf']['Angles'] == [i]].values[:,2:]          
        AP_bin= np.asarray([np.sum(AP[:,bins[i]:bins[i+1]],axis =1) for i in np.arange(np.size(bins)-1)]).transpose()
        RawBin = np.vstack((RawBin,AP_bin))
        Mean[n,:] = np.nanmean(AP_bin,axis =0)         
    meanbindf = pd.DataFrame(Mean) ; meanbindf.insert(0, 'Angles', ls)
    rawbindf = pd.DataFrame(RawBin[1:,:]) ; rawbindf.insert(0, 'Angles', np.repeat(ls,3))
    return(meanbindf, rawbindf)

def meanVm(bins,rootdf,ls,bin_size):    
    Mean = np.empty([len(ls),len(bins)-1])
    RawBin =np.empty(len(bins)-1)
    bl_size = int(0.25/bin_size) # use the first 1/4 second to compute the baseline
    for n,i in enumerate(ls):
        nanbool = np.isnan(rootdf['Vdf'][rootdf['Vdf']['Angles'] == [i]].values[:,2:])
        Vm = rootdf['Vdf'][rootdf['Vdf']['Angles'] == [i]].fillna(np.nanmean(np.nanmean(rootdf['Vdf'].values[:,2:]))).values[:,2:] 
        Vm = FiltV(Vm, rootdf['filter']['N'],rootdf['filter']['Wn']) 
        Vm[nanbool]= np.nan
        Vm_bin= np.asarray([np.mean(Vm[:,bins[i]:bins[i+1]],axis =1) for i in np.arange(np.size(bins)-1)]).transpose()
        Bls = np.mean(Vm_bin[:,:bl_size],axis =1)# compute baseline for each cell
        Vm_bin_r = np.zeros(np.shape(Vm_bin))
        for m in range(len(Bls)):
            Vm_bin_r[m,:] = Vm_bin[m,:] - Bls[m]            
        RawBin = np.vstack((RawBin,Vm_bin))
        Mean[n,:] = np.nanmean(Vm_bin_r,axis =0)         
    meandf = pd.DataFrame(Mean) ; meandf.insert(0, 'Angles', ls)
    rawdf = pd.DataFrame(RawBin[1:,:]) ; rawdf.insert(0, 'Angles', np.repeat(ls,3))
#     Mean = np.empty([len(ls),np.shape(rootdf['Vdf'])[1]-2])
#     Indi =np.empty(np.shape(rootdf['Vdf'])[1]-2)
#     for n,i in enumerate(ls):
#         Vm = rootdf['Vdf'][rootdf['Vdf']['Angles'] == [i]].fillna(0).values[:,2:] 
#         Vm = FiltV(Vm, rootdf['filter']['N'],rootdf['filter']['Wn']) 
#         Indi = np.vstack((Indi,Vm))
#         Mean[n,:] = np.nanmean(Vm,axis =0)         
#     meandf = pd.DataFrame(Mean) ; meandf.insert(0, 'Angles', ls)
#     rawdf = pd.DataFrame(Indi[1:,:]) ; rawdf.insert(0, 'Angles', np.repeat(ls,3))
    return(meandf, rawdf)

def MergeBincells(df,csvroot,stimulus,ls, celltype): 
    recN = 0
    if celltype[:2] == 'HS':
        spiking = 0
        MeanBindf = pd.DataFrame(); RawBindf = pd.DataFrame(); cellBindf = pd.DataFrame(); 
    Meandf = pd.DataFrame(); Rawdf = pd.DataFrame(); celldf = pd.DataFrame();
    for i in np.arange(len(df['cells'][:-1])): #[:-1] is for getting rid of 'totalN'
        cell = df['cells'][i]
        if str(df[stimulus][i]) !='nan':
            if isinstance(df[stimulus][i], str):
                recs = [int(x.strip()) for x in str(df[stimulus][i]).split(',')]
            else:           
                recs = [int(x.strip()) for x in str(df[stimulus][i].astype('int')).split(',')]
            for n in recs:                 
                dfname = csvroot+'Analysis/%s/%s/'%(celltype,cell)+'rec%dRaw.pkl'%n
                rootdf = pd.read_pickle(dfname)                               
                bin_size = 0.05
                bins = np.arange(0, np.shape(rootdf['Vdf'].values[:,2:])[1],int(round(bin_size/(rootdf['t'][1]-rootdf['t'][0]))))  
                if spiking:
                    meanbindf, rawbindf = meanAP(bins,rootdf,ls)               
                    MeanBindf = MeanBindf.add(meanbindf,fill_value=0); RawBindf = RawBindf.append(rawbindf)
                    cellBindf = cellBindf.append(meanbindf)
                meandf, rawdf = meanVm(bins,rootdf,ls,bin_size)   
                Meandf = Meandf.add(meandf,fill_value=0); Rawdf = Rawdf.append(rawdf)
                celldf = celldf.append(meandf)
                recN +=1
    
    dirs = csvroot+'Analysis/'+ stimulus +'/'+ celltype ; makedir(dirs)
    if spiking:    
        MeanBindf = MeanBindf/recN; 
        MeanBindf.to_csv(dirs+'/MeanBindf.csv'); RawBindf.to_csv(dirs+'/RawBindf.csv'); cellBindf.to_csv(dirs+'/cellBindf.csv')
    Meandf = Meandf/recN        
    Meandf.to_csv(dirs+'/Meandf.csv'); Rawdf.to_csv(dirs+'/Rawdf.csv'); celldf.to_csv(dirs+'/celldf.csv')
    if spiking: 
        return(MeanBindf,RawBindf,cellBindf,Meandf,Rawdf,celldf)
    else:
        return(Meandf,Rawdf,celldf)

def OnData(MeanName, CellName, csvroot,celltype):
    #L
    stimulus = 'flashOffOn_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    FlashOnMeanL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    FlashOnCellL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    FlashOnCellL = FlashOnCellL.sort_values(by=['Angles'])
    FlashOnlenCellL = int(np.shape(FlashOnCellL)[0]/7)
    #FlashOnRawL = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    #FlashOnRawL = FlashOnRawL.sort_values(by=['Angles'])
    #FlashOnlenRawL = int(np.shape(FlashOnRawL)[0]/2)

    stimulus = 'Edges_light_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMeanL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    EdgeOnCellL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    EdgeOnCellL = EdgeOnCellL.sort_values(by=['Angles'])
    EdgeOnlenCellL = int(np.shape(EdgeOnCellL)[0]/14)
    #EdgeOnRawL = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    #EdgeOnRawL = EdgeOnRawL.sort_values(by=['Angles'])
    #EdgeOnlenRawL = int(np.shape(EdgeOnRawL)[0]/4)

    stimulus = 'Edges_light_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMeanLv = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    EdgeOnCellLv = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    EdgeOnCellLv = EdgeOnCellLv.sort_values(by=['Angles'])
    EdgeOnlenCellLv = int(np.shape(EdgeOnCellLv)[0]/14)
    # EdgeOnRawLv = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    # EdgeOnRawLv = EdgeOnRawLv.sort_values(by=['Angles'])
    # EdgeOnlenRawLv = int(np.shape(EdgeOnRawLv)[0]/4)

    # R
    stimulus = 'flashOffon_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    FlashOnMeanR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    FlashOnCellR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    FlashOnCellR = FlashOnCellR.sort_values(by=['Angles'])
    FlashOnlenCellR = int(np.shape(FlashOnCellR)[0]/7)
    # FlashOnRawR = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # FlashOnRawR = FlashOnRawR.sort_values(by=['Angles'])
    # FlashOnlenRawR = int(np.shape(FlashOnRawR)[0]/2)

    stimulus = 'Edges_light_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMeanR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    EdgeOnCellR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    EdgeOnCellR = EdgeOnCellR.sort_values(by=['Angles'])
    EdgeOnlenCellR = int(np.shape(EdgeOnCellR)[0]/14)
    # EdgeOnRawR = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # EdgeOnRawR = EdgeOnRawR.sort_values(by=['Angles'])
    # EdgeOnlenRawR = int(np.shape(EdgeOnRawR)[0]/4)

    stimulus = 'Edges_light_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMeanRv = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    EdgeOnCellRv = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    EdgeOnCellRv = EdgeOnCellRv.sort_values(by=['Angles'])
    EdgeOnlenCellRv = int(np.shape(EdgeOnCellRv)[0]/14)
    # EdgeOnRawRv = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # EdgeOnRawRv = EdgeOnRawRv.sort_values(by=['Angles'])
    # EdgeOnlenRawRv = int(np.shape(EdgeOnRawRv)[0]/4)

    # both eyes
    stimulus = 'flashOffOn_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    FlashOnMean = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    FlashOnCell = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    FlashOnCell = FlashOnCell.sort_values(by=['Angles'])
    FlashOnlenCell = int(np.shape(FlashOnCell)[0]/7)
    # FlashOnRaw = pd.read_csv(dirs + '/H2/RawBindf.csv')
    # FlashOnRaw = FlashOnRaw.sort_values(by=['Angles'])
    # FlashOnlenRaw = int(np.shape(FlashOnRaw)[0]/2)

    stimulus = 'Edges_light_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMean = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    EdgeOnCell = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    EdgeOnCell = EdgeOnCell.sort_values(by=['Angles'])
    EdgeOnlenCell = int(np.shape(EdgeOnCell)[0]/14)
    # EdgeOnRaw = pd.read_csv(dirs + '/H2/RawBindf.csv')
    # EdgeOnRaw = EdgeOnRaw.sort_values(by=['Angles'])
    # EdgeOnlenRaw = int(np.shape(EdgeOnRaw)[0]/4)
    
    stimulus = 'Edges_light_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOnMeanv = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    EdgeOnCellv = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    EdgeOnCellv = EdgeOnCellv.sort_values(by=['Angles'])
    EdgeOnlenCellv = int(np.shape(EdgeOnCellv)[0]/14)
    
    return(FlashOnMeanL, FlashOnCellL, FlashOnlenCellL, EdgeOnMeanL, EdgeOnCellL, EdgeOnlenCellL, 
           EdgeOnMeanLv, EdgeOnCellLv, EdgeOnlenCellLv,
           FlashOnMeanR, FlashOnCellR, FlashOnlenCellR, EdgeOnMeanR, EdgeOnCellR, EdgeOnlenCellR,
           EdgeOnMeanRv, EdgeOnCellRv, EdgeOnlenCellRv,
           FlashOnMean, FlashOnCell, FlashOnlenCell, EdgeOnMean, EdgeOnCell, EdgeOnlenCell,
           EdgeOnMeanv, EdgeOnCellv, EdgeOnlenCellv)

## Off
def OffData(MeanName, CellName, csvroot, celltype):
    #L
    stimulus = 'flashOnOff_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    FlashOffMeanL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    FlashOffCellL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    FlashOffCellL = FlashOffCellL.sort_values(by=['Angles'])
    FlashOfflenCellL = int(np.shape(FlashOffCellL)[0]/7)
    #FlashOnRawL = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    #FlashOnRawL = FlashOnRawL.sort_values(by=['Angles'])
    #FlashOnlenRawL = int(np.shape(FlashOnRawL)[0]/2)

    stimulus = 'Edges_dark_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    EdgeOffMeanL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    EdgeOffCellL = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    EdgeOffCellL = EdgeOffCellL.sort_values(by=['Angles'])
    EdgeOfflenCellL = int(np.shape(EdgeOffCellL)[0]/14)
    #EdgeOnRawL = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    #EdgeOnRawL = EdgeOnRawL.sort_values(by=['Angles'])
    #EdgeOnlenRawL = int(np.shape(EdgeOnRawL)[0]/4)

    stimulus = 'Edges_dark_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOffMeanLv = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %MeanName)
    EdgeOffCellLv = pd.read_csv(dirs + '/'+celltype[:2]+'_L/%s.csv' %CellName)
    EdgeOffCellLv = EdgeOffCellLv.sort_values(by=['Angles'])
    EdgeOfflenCellLv = int(np.shape(EdgeOffCellLv)[0]/14)
    # # EdgeOnRawLv = pd.read_csv(dirs + '/H2_L/RawBindf.csv')
    # # EdgeOnRawLv = EdgeOnRawLv.sort_values(by=['Angles'])
    # # EdgeOnlenRawLv = int(np.shape(EdgeOnRawLv)[0]/4)

    # R
    stimulus = 'flashOnOff_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    FlashOffMeanR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    FlashOffCellR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    FlashOffCellR = FlashOffCellR.sort_values(by=['Angles'])
    FlashOfflenCellR = int(np.shape(FlashOffCellR)[0]/7)
    # FlashOnRawR = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # FlashOnRawR = FlashOnRawR.sort_values(by=['Angles'])
    # FlashOnlenRawR = int(np.shape(FlashOnRawR)[0]/2)

    stimulus = 'Edges_Dark_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    EdgeOffMeanR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    EdgeOffCellR = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    EdgeOffCellR = EdgeOffCellR.sort_values(by=['Angles'])
    EdgeOfflenCellR = int(np.shape(EdgeOffCellR)[0]/14)
    # EdgeOnRawR = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # EdgeOnRawR = EdgeOnRawR.sort_values(by=['Angles'])
    # EdgeOnlenRawR = int(np.shape(EdgeOnRawR)[0]/4)

    stimulus = 'Edges_Dark_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOffMeanRv = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %MeanName)
    EdgeOffCellRv = pd.read_csv(dirs + '/'+celltype[:2]+'_R/%s.csv' %CellName)
    EdgeOffCellRv = EdgeOffCellRv.sort_values(by=['Angles'])
    EdgeOfflenCellRv = int(np.shape(EdgeOffCellRv)[0]/14)
    # EdgeOnRawRv = pd.read_csv(dirs + '/H2_R/RawBindf.csv')
    # EdgeOnRawRv = EdgeOnRawRv.sort_values(by=['Angles'])
    # EdgeOnlenRawRv = int(np.shape(EdgeOnRawRv)[0]/4)

    # both eyes
    stimulus = 'flashOnOff_mHz_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    FlashOffMean = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    FlashOffCell = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    FlashOffCell = FlashOffCell.sort_values(by=['Angles'])
    FlashOfflenCell = int(np.shape(FlashOffCell)[0]/7)
    # FlashOnRaw = pd.read_csv(dirs + '/H2/RawBindf.csv')
    # FlashOnRaw = FlashOnRaw.sort_values(by=['Angles'])
    # FlashOnlenRaw = int(np.shape(FlashOnRaw)[0]/2)

    stimulus = 'Edges_Dark_vel_grey'
    dirs = csvroot+'Analysis/'+ stimulus 
    EdgeOffMean = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    EdgeOffCell = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    EdgeOffCell = EdgeOffCell.sort_values(by=['Angles'])
    EdgeOfflenCell = int(np.shape(EdgeOffCell)[0]/14)
    # EdgeOnRaw = pd.read_csv(dirs + '/H2/RawBindf.csv')
    # EdgeOnRaw = EdgeOnRaw.sort_values(by=['Angles'])
    # EdgeOnlenRaw = int(np.shape(EdgeOnRaw)[0]/4)
    
    stimulus = 'Edges_dark_vel_vert_grey'
    dirs = csvroot+'Analysis/'+ stimulus +'/'
    EdgeOffMeanv = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %MeanName)
    EdgeOffCellv = pd.read_csv(dirs + '/'+celltype[:2]+'/%s.csv' %CellName)
    EdgeOffCellv = EdgeOffCellv.sort_values(by=['Angles'])
    EdgeOfflenCellv = int(np.shape(EdgeOffCellv)[0]/14)
    
    return(FlashOffMeanL, FlashOffCellL, FlashOfflenCellL, EdgeOffMeanL, EdgeOffCellL, EdgeOfflenCellL,
           EdgeOffMeanLv, EdgeOffCellLv, EdgeOfflenCellLv,
           FlashOffMeanR, FlashOffCellR, FlashOfflenCellR, EdgeOffMeanR, EdgeOffCellR, EdgeOfflenCellR,
           EdgeOffMeanRv, EdgeOffCellRv, EdgeOfflenCellRv,
           FlashOffMean, FlashOffCell, FlashOfflenCell, EdgeOffMean, EdgeOffCell, EdgeOfflenCell,
           EdgeOffMeanv, EdgeOffCellv, EdgeOfflenCellv)

def FlashMeanPeak(csvroot, datatype, FlashOnCell,FlashOnCellL,FlashOnCellR,FlashOffCell,FlashOffCellL,FlashOffCellR):
    FlashOnCut = FlashOnCell.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    FlashOnCutL = FlashOnCellL.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    FlashOnCutR = FlashOnCellR.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    blOn= np.mean(FlashOnCell.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOnL= np.mean(FlashOnCellL.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOnR= np.mean(FlashOnCellR.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    FlashOnCut = [FlashOnCut[n,:] - i for n,i in enumerate(blOn)]
    FlashOnCutL = [FlashOnCutL[n,:] - i for n,i in enumerate(blOnL)]
    FlashOnCutR = [FlashOnCutR[n,:] - i for n,i in enumerate(blOnR)]
    FlashOnmean = np.mean(FlashOnCut,axis =0); FlashOnpeak = np.max(FlashOnCut,axis =1)
    FlashOnmeanL = np.mean(FlashOnCutL,axis =0);FlashOnpeakL = np.max(FlashOnCutL,axis =1) 
    FlashOnmeanR = np.mean(FlashOnCutR,axis =0);FlashOnpeakR = np.max(FlashOnCutR,axis =1)
    
    FlashOnstd = np.std(FlashOnCut,axis =0);FlashOnstdL = np.std(FlashOnCutL,axis =0);FlashOnstdR = np.std(FlashOnCutR,axis =0)

    FlashOffCut = FlashOffCell.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    FlashOffCutL = FlashOffCellL.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    FlashOffCutR = FlashOffCellR.values[:,2:][:,int(0.5/0.05):int(1.8/0.05)]
    blOff= np.mean(FlashOffCell.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOffL= np.mean(FlashOffCellL.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOffR= np.mean(FlashOffCellR.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    FlashOffCut = [FlashOffCut[n,:] - i for n,i in enumerate(blOn)]
    FlashOffCutL = [FlashOffCutL[n,:] - i for n,i in enumerate(blOnL)]
    FlashOffCutR = [FlashOffCutR[n,:] - i for n,i in enumerate(blOnR)]
    FlashOffmean = np.mean(FlashOffCut,axis =0); FlashOffpeak = np.max(FlashOffCut,axis =1)
    FlashOffmeanL = np.mean(FlashOffCutL,axis =0);FlashOffpeakL = np.max(FlashOffCutL,axis =1) 
    FlashOffmeanR = np.mean(FlashOffCutR,axis =0);FlashOffpeakR = np.max(FlashOffCutR,axis =1)
    
    FlashOffstd = np.std(FlashOffCut,axis =0);FlashOffstdL = np.std(FlashOffCutL,axis =0);FlashOffstdR = np.std(FlashOffCutR,axis =0)


    FlashOndf = pd.DataFrame({'FlashOnmean':FlashOnmean,'FlashOnmeanL':FlashOnmeanL,'FlashOnmeanR':FlashOnmeanR})
    FlashOffdf = pd.DataFrame({'FlashOffmean':FlashOffmean,'FlashOffmeanL':FlashOffmeanL,'FlashOffmeanR':FlashOffmeanR})
    FlashOnstddf = pd.DataFrame({'FlashOnstd':FlashOnstd,'FlashOnstdL':FlashOnstdL,'FlashOnstdR':FlashOnstdR})
    FlashOffstddf = pd.DataFrame({'FlashOffstd':FlashOffstd,'FlashOffstdL':FlashOffstdL,'FlashOffstdR':FlashOffstdR})
    #FlashOnpeakdf = pd.DataFrame({'FlashOnpeak':FlashOnpeak,'FlashOnpeakL':FlashOnpeakL,'FlashOnpeakR':FlashOnpeakR})
    #FlashOffpeakdf = pd.DataFrame({'FlashOffpeak':FlashOffpeak,'FlashOffpeakL':FlashOffpeakL,'FlashOffpeakR':FlashOffpeakR})

    FlashOndf.to_csv(csvroot+'Analysis/flash_grey/FlashOndf_%s.csv' %datatype)
    FlashOffdf.to_csv(csvroot+'Analysis/flash_grey/FlashOffdf_%s.csv' %datatype)
    FlashOnstddf.to_csv(csvroot+'Analysis/flash_grey/FlashOnstddf_%s.csv' %datatype)
    FlashOffstddf.to_csv(csvroot+'Analysis/flash_grey/FlashOffstddf_%s.csv' %datatype)
    #FlashOnpeakdf.to_csv(csvroot+'Analysis/flash_grey/FlashOnpeakdf_%s.csv' %datatype)
    #FlashOffpeakdf.to_csv(csvroot+'Analysis/flash_grey/FlashOffpeakdf_%s.csv' %datatype)
    
    xlocs = np.arange(len(FlashOnmean))*0.05+0.5
    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(6,8))
    axes.plot(xlocs,FlashOnmean,'k'); axes.plot(xlocs,FlashOnmeanL,'b'); axes.plot(xlocs,FlashOnmeanR,'g')
    axes.fill_between(xlocs[:-1], np.array(FlashOnmean[:-1])-np.array(FlashOnstd[:-1]),np.array(FlashOnmean[:-1])+np.array(FlashOnstd[:-1]), alpha = 0.1, color = 'k')
    axes.fill_between(xlocs[:-1],np.array(FlashOnmeanL[:-1])-np.array(FlashOnstdL[:-1]),np.array(FlashOnmeanL[:-1])+np.array(FlashOnstdL[:-1]), alpha = 0.1, color = 'b')
    axes.fill_between(xlocs[:-1],np.array(FlashOnmeanR[:-1])-np.array(FlashOnstdR[:-1]),np.array(FlashOnmeanR[:-1])+np.array(FlashOnstdR[:-1]), alpha = 0.1, color = 'g')    
    plt.legend(['Both','Ipsi','Contra']); plt.title('MeanOnFlash_'+datatype)
    plt.savefig(csvroot+'Pics/MeanOnFlash_%s.png' %datatype)

    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(6,8))
    axes.plot(xlocs,FlashOffmean,'k'); axes.plot(xlocs,FlashOffmeanL,'b'); axes.plot(xlocs,FlashOffmeanR,'g')
    axes.fill_between(xlocs[:-1], np.array(FlashOffmean[:-1])-np.array(FlashOffstd[:-1]),np.array(FlashOffmean[:-1])+np.array(FlashOffstd[:-1]), alpha = 0.1, color = 'k')
    axes.fill_between(xlocs[:-1],np.array(FlashOffmeanL[:-1])-np.array(FlashOffstdL[:-1]),np.array(FlashOffmeanL[:-1])+np.array(FlashOffstdL[:-1]), alpha = 0.1, color = 'b')
    axes.fill_between(xlocs[:-1],np.array(FlashOffmeanR[:-1])-np.array(FlashOffstdR[:-1]),np.array(FlashOffmeanR[:-1])+np.array(FlashOffstdR[:-1]), alpha = 0.1, color = 'g')    
    plt.legend(['Both','Ipsi','Contra']); plt.title('MeanOffFlash_'+datatype)
    plt.savefig(csvroot+'Pics/MeanOffFlash_%s.png' %datatype)

def FlashDurIdx(csvroot, datatype, FlashOnCell,FlashOnCellL,FlashOnCellR,FlashOffCell,FlashOffCellL,FlashOffCellR):
    sti = np.asarray([60,120,180,240,300,360,420])
    FlashOnCut = FlashOnCell.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    FlashOnCutL = FlashOnCellL.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    FlashOnCutR = FlashOnCellR.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    blOn= np.mean(FlashOnCell.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOnL= np.mean(FlashOnCellL.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOnR= np.mean(FlashOnCellR.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    FlashOnCut = [FlashOnCut[n,:] - i for n,i in enumerate(blOn)]
    FlashOnCutL = [FlashOnCutL[n,:] - i for n,i in enumerate(blOnL)]
    FlashOnCutR = [FlashOnCutR[n,:] - i for n,i in enumerate(blOnR)]
    FlashOffpeak = np.nanmax(FlashOnCut,axis =1)
    FlashOffpeakL = np.nanmax(FlashOnCutL,axis =1) 
    FlashOffpeakR = np.nanmax(FlashOnCutR,axis =1)

    FlashOffavg = [np.mean(FlashOffpeak[np.argwhere(FlashOnCell['Angles'].values == i)]) for i in sti]
    FlashOffavgL = [np.mean(FlashOffpeakL[np.argwhere(FlashOnCellL['Angles'].values == i)]) for i in sti]
    FlashOffavgR = [np.mean(FlashOffpeakR[np.argwhere(FlashOnCellR['Angles'].values == i)]) for i in sti]

    FlashOffstd = [np.std(FlashOffpeak[np.argwhere(FlashOnCell['Angles'].values == i)]) for i in sti]
    FlashOffstdL = [np.std(FlashOffpeakL[np.argwhere(FlashOnCellL['Angles'].values == i)]) for i in sti]
    FlashOffstdR = [np.std(FlashOffpeakR[np.argwhere(FlashOnCellR['Angles'].values == i)]) for i in sti]

    FlashOffCut = FlashOffCell.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    FlashOffCutL = FlashOffCellL.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    FlashOffCutR = FlashOffCellR.values[:,2:][:,int(1.8/0.05):-int(0.9/0.05)]
    blOff= np.mean(FlashOffCell.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOffL= np.mean(FlashOffCellL.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    blOffR= np.mean(FlashOffCellR.values[:,2:][:,int(0.5/0.05):int(0.9/0.05)],axis=1)
    FlashOffCut = [FlashOffCut[n,:] - i for n,i in enumerate(blOn)]
    FlashOffCutL = [FlashOffCutL[n,:] - i for n,i in enumerate(blOnL)]
    FlashOffCutR = [FlashOffCutR[n,:] - i for n,i in enumerate(blOnR)]
    FlashOnpeak = np.nanmax(FlashOffCut,axis =1)
    FlashOnpeakL = np.nanmax(FlashOffCutL,axis =1) 
    FlashOnpeakR = np.nanmax(FlashOffCutR,axis =1)

    FlashOnavg = [np.mean(FlashOnpeak[np.argwhere(FlashOffCell['Angles'].values == i)]) for i in sti]
    FlashOnavgL = [np.mean(FlashOnpeakL[np.argwhere(FlashOffCellL['Angles'].values == i)]) for i in sti]
    FlashOnavgR = [np.mean(FlashOnpeakR[np.argwhere(FlashOffCellR['Angles'].values == i)]) for i in sti]

    FlashOnstd = [np.std(FlashOnpeak[np.argwhere(FlashOffCell['Angles'].values == i)]) for i in sti]
    FlashOnstdL = [np.std(FlashOnpeakL[np.argwhere(FlashOffCellL['Angles'].values == i)]) for i in sti]
    FlashOnstdR = [np.std(FlashOnpeakR[np.argwhere(FlashOffCellR['Angles'].values == i)]) for i in sti]

    FlashOnavgdf = pd.DataFrame({'Angles':sti,'FlashOnavg':FlashOnavg,'FlashOnavgL':FlashOnavgL,'FlashOnavgR':FlashOnavgR})
    FlashOffavgdf = pd.DataFrame({'Angles':sti,'FlashOffavg':FlashOffavg,'FlashOffavgL':FlashOffavgL,'FlashOffavgR':FlashOffavgR})
    FlashOnstddf = pd.DataFrame({'Angles':sti,'FlashOnstd':FlashOnstd,'FlashOnstdL':FlashOnstdL,'FlashOnstdR':FlashOnstdR})
    FlashOffstddf = pd.DataFrame({'Angles':sti,'FlashOffstd':FlashOffstd,'FlashOffstdL':FlashOffstdL,'FlashOffstdR':FlashOffstdR})

    FlashOnavgdf.to_csv(csvroot+'Analysis/flash_grey/FlashOnavgdf_%s.csv' %datatype)
    FlashOffavgdf.to_csv(csvroot+'Analysis/flash_grey/FlashOffavgdf_%s.csv' %datatype)
    FlashOnstddf.to_csv(csvroot+'Analysis/flash_grey/FlashOnstddf_%s.csv' %datatype)
    FlashOffstddf.to_csv(csvroot+'Analysis/flash_grey/FlashOffstddf_%s.csv' %datatype)

    # index. Preference for On. 
    Pidx = (np.array(FlashOnavg)-np.array(FlashOffavg))/np.max(np.array(FlashOnavg)+abs(np.array(FlashOffavg)))
    PidxL = (np.array(FlashOnavgL)-np.array(FlashOffavgL))/np.max(np.array(FlashOnavgL)+abs(np.array(FlashOffavgL)))
    PidxR = (np.array(FlashOnavgR)-np.array(FlashOffavgR))/np.max(np.array(FlashOnavgR)+abs(np.array(FlashOffavgR)))
    # normalize
    nPidx = Pidx/np.max(np.abs([Pidx,PidxL,PidxR])); nPidxL = PidxL/np.max(np.abs([Pidx,PidxL,PidxR]))
    nPidxR = PidxR/np.max(np.abs([Pidx,PidxL,PidxR]))
    Pidxdf = pd.DataFrame({'Angles':sti,'Pidx':nPidx,'PidxL':nPidxL,'PidxR':nPidxR})
    Pidxdf.to_csv(csvroot+'Analysis/flash_grey/Pidxdf_%s.csv' %datatype)   
    
    # Pidx for individual cell 
    FlashOffpeak_sort = FlashOffpeak[FlashOnCell.index.argsort()]
    FlashOffpeakL_sort = FlashOffpeakL[FlashOnCellL.index.argsort()] 
    FlashOffpeakR_sort = FlashOffpeakR[FlashOnCellR.index.argsort()]

    FlashOnpeak_sort = FlashOnpeak[FlashOffCell.index.argsort()]
    FlashOnpeakL_sort = FlashOnpeakL[FlashOffCellL.index.argsort()] 
    FlashOnpeakR_sort = FlashOnpeakR[FlashOffCellR.index.argsort()]

    PidxCell = np.zeros([len(sti),int(len(FlashOffpeak)/len(sti))])
    PidxCellL = np.zeros([len(sti),int(len(FlashOffpeakL)/len(sti))])
    PidxCellR = np.zeros([len(sti),int(len(FlashOffpeakR)/len(sti))])
    for i in range(int(len(FlashOffpeak)/len(sti))):
        onindi = FlashOnpeak_sort[i*7:(i+1)*7]; offindi = FlashOffpeak_sort[i*7:(i+1)*7]
        PidxCell[:,i] = (onindi-offindi)/np.max(onindi+abs(offindi))
    for i in range(int(len(FlashOffpeakL)/len(sti))):   
        onindiL = FlashOnpeakL_sort[i*7:(i+1)*7]; offindiL = FlashOffpeakL_sort[i*7:(i+1)*7]
        PidxCellL[:,i] = (onindiL-offindiL)/np.max(onindiL+abs(offindiL))
    for i in range(int(len(FlashOffpeakR)/len(sti))):     
        onindiR = FlashOnpeakR_sort[i*7:(i+1)*7]; offindiR = FlashOffpeakR_sort[i*7:(i+1)*7]
        PidxCellR[:,i] = (onindiR-offindiR)/np.max(onindiR+abs(offindiR))
        # normalize
    nPidxCell = PidxCell/np.max(np.abs([Pidx,PidxL,PidxR]))
    nPidxCellL = PidxCellL/np.max(np.abs([Pidx,PidxL,PidxR]))
    nPidxCellR = PidxCellR/np.max(np.abs([Pidx,PidxL,PidxR]))
    
    PidxCelldf = pd.concat([pd.DataFrame({'Angles':sti}), pd.DataFrame(nPidxCell)],axis=1,ignore_index=True)
    PidxCellLdf = pd.concat([pd.DataFrame({'Angles':sti}), pd.DataFrame(nPidxCellL)],axis=1,ignore_index=True)
    PidxCellRdf = pd.concat([pd.DataFrame({'Angles':sti}), pd.DataFrame(nPidxCellR)],axis=1,ignore_index=True)
    PidxCelldf.to_csv(csvroot+'Analysis/flash_grey/PidxCelldf_%s.csv' %datatype) 
    PidxCellLdf.to_csv(csvroot+'Analysis/flash_grey/PidxCellLdf_%s.csv' %datatype) 
    PidxCellRdf.to_csv(csvroot+'Analysis/flash_grey/PidxCellRdf_%s.csv' %datatype) 
    
    # plotting
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    axes[0].set_title('OnFlash_'+datatype); axes[1].set_title('OffFlash_'+datatype);
    axes[0].plot(sti[:-1],FlashOnavg[:-1],'k'); axes[0].plot(sti[:-1],FlashOnavgL[:-1],'b'); axes[0].plot(sti[:-1],FlashOnavgR[:-1],'g')
    axes[0].fill_between(sti[:-1],np.array(FlashOnavg[:-1])-np.array(FlashOnstd[:-1]),np.array(FlashOnavg[:-1])+np.array(FlashOnstd[:-1]), alpha = 0.1, color = 'k')
    axes[0].fill_between(sti[:-1],np.array(FlashOnavg[:-1])-np.array(FlashOnstd[:-1]),np.array(FlashOnavg[:-1])+np.array(FlashOnstd[:-1]), alpha = 0.1, color = 'k')
    axes[0].fill_between(sti[:-1],np.array(FlashOnavgL[:-1])-np.array(FlashOnstdL[:-1]),np.array(FlashOnavgL[:-1])+np.array(FlashOnstdL[:-1]), alpha = 0.1, color = 'b')
    axes[0].fill_between(sti[:-1],np.array(FlashOnavgR[:-1])-np.array(FlashOnstdR[:-1]),np.array(FlashOnavgR[:-1])+np.array(FlashOnstdR[:-1]), alpha = 0.1, color = 'g')
    axes[0].set_xlabel('Duration(s)')

    axes[1].plot(sti[:-1],FlashOffavg[:-1],'k'); axes[1].plot(sti[:-1],FlashOffavgL[:-1],'b'); axes[1].plot(sti[:-1],FlashOffavgR[:-1],'g')
    axes[1].fill_between(sti[:-1],np.array(FlashOffavg[:-1])-np.array(FlashOffstd[:-1]),np.array(FlashOffavg[:-1])+np.array(FlashOffstd[:-1]), alpha = 0.1, color = 'k')
    axes[1].fill_between(sti[:-1],np.array(FlashOffavgL[:-1])-np.array(FlashOffstdL[:-1]),np.array(FlashOffavgL[:-1])+np.array(FlashOffstdL[:-1]), alpha = 0.1, color = 'b')
    axes[1].fill_between(sti[:-1],np.array(FlashOffavgR[:-1])-np.array(FlashOffstdR[:-1]),np.array(FlashOffavgR[:-1])+np.array(FlashOffstdR[:-1]), alpha = 0.1, color = 'g')
    axes[1].legend(['Both','Ipsi','Contra']); axes[1].set_xticks(sti[:-1]); axes[1].set_xlabel('Duration(s)')
    plt.savefig(csvroot+'Pics/FlashDur_%s.png' %datatype)

    # index. Preference for On
    plt.figure()
    plt.plot(sti[:-1],nPidx[:-1],'k'); plt.plot(sti[:-1],nPidxL[:-1],'b'); plt.plot(sti[:-1],nPidxR[:-1],'g')
    plt.legend(['Both','Ipsi','Contra']); 
    [plt.plot(sti[:-1],nPidxCell[:-1,i], 'k',alpha =0.1) for i in range(int(np.shape(nPidxCell)[1]))]
    [plt.plot(sti[:-1],nPidxCellL[:-1,i], 'b',alpha =0.1) for i in range(int(np.shape(nPidxCellL)[1]))]
    [plt.plot(sti[:-1],nPidxCellR[:-1,i], 'g',alpha =0.1) for i in range(int(np.shape(nPidxCellR)[1]))]
    plt.title('Preference_for_On_Flash_%s' %datatype); plt.xlabel('Duration(s)'); plt.xticks(sti[:-1],sti[:-1])
    plt.savefig(csvroot+'Pics/Preference_for_On_Flash_%s.png' %datatype)
    
def EdgeOnPeak(csvroot, datatype,EdgeOnCell,EdgeOnCellL,EdgeOnCellR):
    sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]
    padnan = np.zeros((np.shape(EdgeOnCell.values[:,2:])[0],np.shape(EdgeOnCell.values[:,2:])[1]+1))
    padnan.fill(np.nan); padnan[:,:-1] = EdgeOnCell.values[:,2:]
    padnanL = np.zeros((np.shape(EdgeOnCellL.values[:,2:])[0],np.shape(EdgeOnCellL.values[:,2:])[1]+1))
    padnanL.fill(np.nan); padnanL[:,:-1] = EdgeOnCellL.values[:,2:]
    padnanR = np.zeros((np.shape(EdgeOnCellR.values[:,2:])[0],np.shape(EdgeOnCellR.values[:,2:])[1]+1))
    padnanR.fill(np.nan); padnanR[:,:-1] = EdgeOnCellR.values[:,2:]
    EdgeOndur = np.argwhere(np.diff(np.isnan(padnan), axis =1) == 1)[:,1]
    EdgeOndurL = np.argwhere(np.diff(np.isnan(padnanL), axis =1) == 1)[:,1]
    EdgeOndurR = np.argwhere(np.diff(np.isnan(padnanR), axis =1) == 1)[:,1]
    Ondur = [int(np.nanmean(EdgeOndur[np.argwhere(EdgeOnCell['Angles'].values == i)])) for i in sti]
    OndurL = [int(np.nanmean(EdgeOndurL[np.argwhere(EdgeOnCellL['Angles'].values == i)])) for i in sti]
    OndurR = [int(np.nanmean(EdgeOndurR[np.argwhere(EdgeOnCellR['Angles'].values == i)])) for i in sti]
    Ondurls = np.repeat(Ondur, int(np.size(EdgeOndur)/np.size(Ondur)))
    OndurlsL = np.repeat(OndurL, int(np.size(EdgeOndurL)/np.size(OndurL)))
    OndurlsR = np.repeat(OndurR, int(np.size(EdgeOndurR)/np.size(OndurR)))
    blOn= np.mean(EdgeOnCell.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    blOnL= np.mean(EdgeOnCellL.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    blOnR= np.mean(EdgeOnCellR.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    EdgeOnPeak = np.array([np.max(EdgeOnCell.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOn[n]) for n,i in enumerate(Ondurls)])
    EdgeOnPeakL = np.array([np.max(EdgeOnCellL.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOnL[n]) for n,i in enumerate(OndurlsL)])
    EdgeOnPeakR = np.array([np.max(EdgeOnCellR.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOnR[n]) for n,i in enumerate(OndurlsR)])
    EdgeOnPeakMean = np.array([np.mean(EdgeOnPeak[np.argwhere(EdgeOnCell['Angles'].values == i)]) for i in sti])
    EdgeOnPeakMeanL = np.array([np.mean(EdgeOnPeakL[np.argwhere(EdgeOnCellL['Angles'].values == i)]) for i in sti])
    EdgeOnPeakMeanR = np.array([np.mean(EdgeOnPeakR[np.argwhere(EdgeOnCellR['Angles'].values == i)]) for i in sti])

    EdgeOnPeakStd = np.array([np.std(EdgeOnPeak[np.argwhere(EdgeOnCell['Angles'].values == i)]) for i in sti])
    EdgeOnPeakStdL = np.array([np.std(EdgeOnPeakL[np.argwhere(EdgeOnCellL['Angles'].values == i)]) for i in sti])
    EdgeOnPeakStdR = np.array([np.std(EdgeOnPeakR[np.argwhere(EdgeOnCellR['Angles'].values == i)]) for i in sti])

    EdgeOnMin = np.array([np.min(EdgeOnCell.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOn[n]) for n,i in enumerate(Ondurls)])
    EdgeOnMinL = np.array([np.min(EdgeOnCellL.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOnL[n]) for n,i in enumerate(OndurlsL)])
    EdgeOnMinR = np.array([np.min(EdgeOnCellR.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOnR[n]) for n,i in enumerate(OndurlsR)])
    EdgeOnMinMean = np.array([np.mean(EdgeOnMin[np.argwhere(EdgeOnCell['Angles'].values == i)]) for i in sti])
    EdgeOnMinMeanL = np.array([np.mean(EdgeOnMinL[np.argwhere(EdgeOnCellL['Angles'].values == i)]) for i in sti])
    EdgeOnMinMeanR = np.array([np.mean(EdgeOnMinR[np.argwhere(EdgeOnCellR['Angles'].values == i)]) for i in sti])

    EdgeOnMinStd = np.array([np.std(EdgeOnMin[np.argwhere(EdgeOnCell['Angles'].values == i)]) for i in sti])
    EdgeOnMinStdL = np.array([np.std(EdgeOnMinL[np.argwhere(EdgeOnCellL['Angles'].values == i)]) for i in sti])
    EdgeOnMinStdR = np.array([np.std(EdgeOnMinR[np.argwhere(EdgeOnCellR['Angles'].values == i)]) for i in sti])

    EdgeOnPeakdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMean':EdgeOnPeakMean})
    EdgeOnPeakLdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMeanL':EdgeOnPeakMeanL})
    EdgeOnPeakRdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMeanR':EdgeOnPeakMeanR})
    EdgeOnPeakStddf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakStd':EdgeOnPeakStd})
    EdgeOnPeakStdLdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakStdL':EdgeOnPeakStdL})
    EdgeOnPeakStdRdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakStdR':EdgeOnPeakStdR})
    EdgeOnPeakCelldf = pd.DataFrame({'Angles':EdgeOnCell['Angles'].values, 'EdgeOnPeak':EdgeOnPeak})
    EdgeOnPeakCellLdf = pd.DataFrame({'Angles':EdgeOnCellL['Angles'].values, 'EdgeOnPeakL':EdgeOnPeakL})
    EdgeOnPeakCellRdf = pd.DataFrame({'Angles':EdgeOnCellR['Angles'].values, 'EdgeOnPeakR':EdgeOnPeakR})
    EdgeOnMindf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMean':EdgeOnMinMean})
    EdgeOnMinLdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMeanL':EdgeOnMinMeanL})
    EdgeOnMinRdf = pd.DataFrame({'Angles':sti, 'EdgeOnPeakMeanR':EdgeOnMinMeanR})
    EdgeOnMinStddf = pd.DataFrame({'Angles':sti, 'EdgeOnMinStd':EdgeOnMinStd})
    EdgeOnMinStdLdf = pd.DataFrame({'Angles':sti, 'EdgeOnMinStdL':EdgeOnMinStdL})
    EdgeOnMinStdRdf = pd.DataFrame({'Angles':sti, 'EdgeOnMinStdR':EdgeOnMinStdR})
    EdgeOnMinCelldf = pd.DataFrame({'Angles':EdgeOnCell['Angles'].values, 'EdgeOnPeak':EdgeOnMin })
    EdgeOnMinCellLdf = pd.DataFrame({'Angles':EdgeOnCellL['Angles'].values, 'EdgeOnPeakL':EdgeOnMinL })
    EdgeOnMinCellRdf = pd.DataFrame({'Angles':EdgeOnCellR['Angles'].values, 'EdgeOnPeakR':EdgeOnMinR })

    EdgeOnPeakdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakdf%s.csv' %datatype)
    EdgeOnPeakLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakLdf%s.csv' %datatype)
    EdgeOnPeakRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakRdf%s.csv' %datatype)
    EdgeOnPeakStddf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakStddf%s.csv' %datatype)
    EdgeOnPeakStdLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakStdLdf%s.csv' %datatype)
    EdgeOnPeakStdRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakStdRdf%s.csv' %datatype)
    EdgeOnPeakCelldf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakCelldf%s.csv' %datatype)
    EdgeOnPeakCellLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakCellLdf%s.csv' %datatype)
    EdgeOnPeakCellRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnPeakCellRdf%s.csv' %datatype)

    EdgeOnMindf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMindf%s.csv' %datatype)
    EdgeOnMinLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinLdf%s.csv' %datatype)
    EdgeOnMinRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinRdf%s.csv' %datatype)
    EdgeOnMinStddf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinStddf%s.csv' %datatype)
    EdgeOnMinStdLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinStdLdf%s.csv' %datatype)
    EdgeOnMinStdRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinStdRdf%s.csv' %datatype)
    EdgeOnMinCelldf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinCelldf%s.csv' %datatype)
    EdgeOnMinCellLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinCellLdf%s.csv' %datatype)
    EdgeOnMinCellRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOnMinCellRdf%s.csv' %datatype)

    plt.figure()
    plt.plot(sti,EdgeOnPeakMean, 'k'); plt.plot(sti,EdgeOnPeakMeanL, 'b'); plt.plot(sti,EdgeOnPeakMeanR, 'g')
    plt.fill_between(sti,EdgeOnPeakMean - EdgeOnPeakStd, EdgeOnPeakMean + EdgeOnPeakStd, color = 'k', alpha = 0.1)
    plt.fill_between(sti,EdgeOnPeakMeanL - EdgeOnPeakStdL, EdgeOnPeakMeanL + EdgeOnPeakStdL, color = 'b', alpha = 0.1)
    plt.fill_between(sti,EdgeOnPeakMeanR - EdgeOnPeakStdR, EdgeOnPeakMeanR + EdgeOnPeakStdR, color = 'g', alpha = 0.1)
    plt.xticks(sti); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)')#plt.xscale('symlog')
    plt.title('MeanOnPeak_'+datatype); plt.savefig(csvroot+'Pics/MeanOnPeak_'+datatype)

    plt.figure()
    plt.plot(sti,EdgeOnMinMean, 'k'); plt.plot(sti,EdgeOnMinMeanL, 'b'); plt.plot(sti,EdgeOnMinMeanR, 'g')
    plt.fill_between(sti,EdgeOnMinMean - EdgeOnMinStd, EdgeOnMinMean + EdgeOnMinStd, color = 'k', alpha = 0.1)
    plt.fill_between(sti,EdgeOnMinMeanL - EdgeOnMinStdL, EdgeOnMinMeanL + EdgeOnMinStdL, color = 'b', alpha = 0.1)
    plt.fill_between(sti,EdgeOnMinMeanR - EdgeOnMinStdR, EdgeOnMinMeanR + EdgeOnMinStdR, color = 'g', alpha = 0.1)
    plt.xticks(sti); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)')#plt.xscale('symlog')
    plt.title('MeanOnMin_'+datatype); plt.savefig(csvroot+'Pics/MeanOnMin_'+datatype)
    return(EdgeOnPeakdf, EdgeOnPeakLdf, EdgeOnPeakRdf, EdgeOnMindf, EdgeOnMinLdf, EdgeOnMinRdf)

def EdgeOffPeak(csvroot, datatype,EdgeOffCell,EdgeOffCellL,EdgeOffCellR):
    sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]
    padnan = np.zeros((np.shape(EdgeOffCell.values[:,2:])[0],np.shape(EdgeOffCell.values[:,2:])[1]+1))
    padnan.fill(np.nan); padnan[:,:-1] = EdgeOffCell.values[:,2:]
    padnanL = np.zeros((np.shape(EdgeOffCellL.values[:,2:])[0],np.shape(EdgeOffCellL.values[:,2:])[1]+1))
    padnanL.fill(np.nan); padnanL[:,:-1] = EdgeOffCellL.values[:,2:]
    padnanR = np.zeros((np.shape(EdgeOffCellR.values[:,2:])[0],np.shape(EdgeOffCellR.values[:,2:])[1]+1))
    padnanR.fill(np.nan); padnanR[:,:-1] = EdgeOffCellR.values[:,2:]
    EdgeOffdur = np.argwhere(np.diff(np.isnan(padnan), axis =1) == 1)[:,1]
    EdgeOffdurL = np.argwhere(np.diff(np.isnan(padnanL), axis =1) == 1)[:,1]
    EdgeOffdurR = np.argwhere(np.diff(np.isnan(padnanR), axis =1) == 1)[:,1]
    Offdur = [int(np.nanmean(EdgeOffdur[np.argwhere(EdgeOffCell['Angles'].values == i)])) for i in sti]
    OffdurL = [int(np.nanmean(EdgeOffdurL[np.argwhere(EdgeOffCellL['Angles'].values == i)])) for i in sti]
    OffdurR = [int(np.nanmean(EdgeOffdurR[np.argwhere(EdgeOffCellR['Angles'].values == i)])) for i in sti]
    Offdurls = np.repeat(Offdur, int(np.size(EdgeOffdur)/np.size(Offdur)))
    OffdurlsL = np.repeat(OffdurL, int(np.size(EdgeOffdurL)/np.size(OffdurL)))
    OffdurlsR = np.repeat(OffdurR, int(np.size(EdgeOffdurR)/np.size(OffdurR)))
    blOff= np.mean(EdgeOffCell.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    blOffL= np.mean(EdgeOffCellL.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    blOffR= np.mean(EdgeOffCellR.values[:,2:][:,int(0.2/0.05):int(0.5/0.05)],axis=1)
    EdgeOffPeak = np.array([np.max(EdgeOffCell.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOff[n]) for n,i in enumerate(Offdurls)])
    EdgeOffPeakL = np.array([np.max(EdgeOffCellL.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOffL[n]) for n,i in enumerate(OffdurlsL)])
    EdgeOffPeakR = np.array([np.max(EdgeOffCellR.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOffR[n]) for n,i in enumerate(OffdurlsR)])
    EdgeOffPeakMean = np.array([np.mean(EdgeOffPeak[np.argwhere(EdgeOffCell['Angles'].values == i)]) for i in sti])
    EdgeOffPeakMeanL = np.array([np.mean(EdgeOffPeakL[np.argwhere(EdgeOffCellL['Angles'].values == i)]) for i in sti])
    EdgeOffPeakMeanR = np.array([np.mean(EdgeOffPeakR[np.argwhere(EdgeOffCellR['Angles'].values == i)]) for i in sti])

    EdgeOffPeakStd = np.array([np.std(EdgeOffPeak[np.argwhere(EdgeOffCell['Angles'].values == i)]) for i in sti])
    EdgeOffPeakStdL = np.array([np.std(EdgeOffPeakL[np.argwhere(EdgeOffCellL['Angles'].values == i)]) for i in sti])
    EdgeOffPeakStdR = np.array([np.std(EdgeOffPeakR[np.argwhere(EdgeOffCellR['Angles'].values == i)]) for i in sti])

    EdgeOffMin = np.array([np.min(EdgeOffCell.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOff[n]) for n,i in enumerate(Offdurls)])
    EdgeOffMinL = np.array([np.min(EdgeOffCellL.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOffL[n]) for n,i in enumerate(OffdurlsL)])
    EdgeOffMinR = np.array([np.min(EdgeOffCellR.values[:,2:][n,int(0.5/0.05):int(i-(0.55/0.05))]-blOffR[n]) for n,i in enumerate(OffdurlsR)])
    EdgeOffMinMean = np.array([np.mean(EdgeOffMin[np.argwhere(EdgeOffCell['Angles'].values == i)]) for i in sti])
    EdgeOffMinMeanL = np.array([np.mean(EdgeOffMinL[np.argwhere(EdgeOffCellL['Angles'].values == i)]) for i in sti])
    EdgeOffMinMeanR = np.array([np.mean(EdgeOffMinR[np.argwhere(EdgeOffCellR['Angles'].values == i)]) for i in sti])

    EdgeOffMinStd = np.array([np.std(EdgeOffMin[np.argwhere(EdgeOffCell['Angles'].values == i)]) for i in sti])
    EdgeOffMinStdL = np.array([np.std(EdgeOffMinL[np.argwhere(EdgeOffCellL['Angles'].values == i)]) for i in sti])
    EdgeOffMinStdR = np.array([np.std(EdgeOffMinR[np.argwhere(EdgeOffCellR['Angles'].values == i)]) for i in sti])

    EdgeOffPeakdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMean':EdgeOffPeakMean})
    EdgeOffPeakLdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMeanL':EdgeOffPeakMeanL})
    EdgeOffPeakRdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMeanR':EdgeOffPeakMeanR})
    EdgeOffPeakStddf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakStd':EdgeOffPeakStd})
    EdgeOffPeakStdLdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakStdL':EdgeOffPeakStdL})
    EdgeOffPeakStdRdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakStdR':EdgeOffPeakStdR})
    EdgeOffPeakCelldf = pd.DataFrame({'Angles':EdgeOffCell['Angles'].values, 'EdgeOffPeak':EdgeOffPeak})
    EdgeOffPeakCellLdf = pd.DataFrame({'Angles':EdgeOffCellL['Angles'].values, 'EdgeOffPeakL':EdgeOffPeakL})
    EdgeOffPeakCellRdf = pd.DataFrame({'Angles':EdgeOffCellR['Angles'].values, 'EdgeOffPeakR':EdgeOffPeakR})
    EdgeOffMindf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMean':EdgeOffMinMean})
    EdgeOffMinLdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMeanL':EdgeOffMinMeanL})
    EdgeOffMinRdf = pd.DataFrame({'Angles':sti, 'EdgeOffPeakMeanR':EdgeOffMinMeanR})
    EdgeOffMinStddf = pd.DataFrame({'Angles':sti, 'EdgeOffMinStd':EdgeOffMinStd})
    EdgeOffMinStdLdf = pd.DataFrame({'Angles':sti, 'EdgeOffMinStdL':EdgeOffMinStdL})
    EdgeOffMinStdRdf = pd.DataFrame({'Angles':sti, 'EdgeOffMinStdR':EdgeOffMinStdR})
    EdgeOffMinCelldf = pd.DataFrame({'Angles':EdgeOffCell['Angles'].values, 'EdgeOffPeak':EdgeOffMin })
    EdgeOffMinCellLdf = pd.DataFrame({'Angles':EdgeOffCellL['Angles'].values, 'EdgeOffPeakL':EdgeOffMinL })
    EdgeOffMinCellRdf = pd.DataFrame({'Angles':EdgeOffCellR['Angles'].values, 'EdgeOffPeakR':EdgeOffMinR })

    EdgeOffPeakdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakdf%s.csv' %datatype)
    EdgeOffPeakLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakLdf%s.csv' %datatype)
    EdgeOffPeakRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakRdf%s.csv' %datatype)
    EdgeOffPeakStddf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakStddf%s.csv' %datatype)
    EdgeOffPeakStdLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakStdLdf%s.csv' %datatype)
    EdgeOffPeakStdRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakStdRdf%s.csv' %datatype)
    EdgeOffPeakCelldf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakCelldf%s.csv' %datatype)
    EdgeOffPeakCellLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakCellLdf%s.csv' %datatype)
    EdgeOffPeakCellRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffPeakCellRdf%s.csv' %datatype)

    EdgeOffMindf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMindf%s.csv' %datatype)
    EdgeOffMinLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinLdf%s.csv' %datatype)
    EdgeOffMinRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinRdf%s.csv' %datatype)
    EdgeOffMinStddf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinStddf%s.csv' %datatype)
    EdgeOffMinStdLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinStdLdf%s.csv' %datatype)
    EdgeOffMinStdRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinStdRdf%s.csv' %datatype)
    EdgeOffMinCelldf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinCelldf%s.csv' %datatype)
    EdgeOffMinCellLdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinCellLdf%s.csv' %datatype)
    EdgeOffMinCellRdf.to_csv(csvroot+'Analysis/Edge_peak/EdgeOffMinCellRdf%s.csv' %datatype)

    plt.figure()
    plt.plot(sti,EdgeOffPeakMean, 'k'); plt.plot(sti,EdgeOffPeakMeanL, 'b'); plt.plot(sti,EdgeOffPeakMeanR, 'g')
    plt.fill_between(sti,EdgeOffPeakMean - EdgeOffPeakStd, EdgeOffPeakMean + EdgeOffPeakStd, color = 'k', alpha = 0.1)
    plt.fill_between(sti,EdgeOffPeakMeanL - EdgeOffPeakStdL, EdgeOffPeakMeanL + EdgeOffPeakStdL, color = 'b', alpha = 0.1)
    plt.fill_between(sti,EdgeOffPeakMeanR - EdgeOffPeakStdR, EdgeOffPeakMeanR + EdgeOffPeakStdR, color = 'g', alpha = 0.1)
    plt.xticks(sti); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)')#plt.xscale('symlog')
    plt.title('MeanOffPeak_'+datatype); plt.savefig(csvroot+'Pics/MeanOffPeak_'+datatype)

    plt.figure()
    plt.plot(sti,EdgeOffMinMean, 'k'); plt.plot(sti,EdgeOffMinMeanL, 'b'); plt.plot(sti,EdgeOffMinMeanR, 'g')
    plt.fill_between(sti,EdgeOffMinMean - EdgeOffMinStd, EdgeOffMinMean + EdgeOffMinStd, color = 'k', alpha = 0.1)
    plt.fill_between(sti,EdgeOffMinMeanL - EdgeOffMinStdL, EdgeOffMinMeanL + EdgeOffMinStdL, color = 'b', alpha = 0.1)
    plt.fill_between(sti,EdgeOffMinMeanR - EdgeOffMinStdR, EdgeOffMinMeanR + EdgeOffMinStdR, color = 'g', alpha = 0.1)
    plt.xticks(sti); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)')#plt.xscale('symlog')
    plt.title('MeanOffMin_'+datatype); plt.savefig(csvroot+'Pics/MeanOffMin_'+datatype)
    return(EdgeOffPeakdf, EdgeOffPeakLdf, EdgeOffPeakRdf, EdgeOffMindf, EdgeOffMinLdf, EdgeOffMinRdf)

def allEFs(celltype,csvroot,pageName):
    for i in [celltype, celltype+'_L', celltype+'_R']:
        csvpath = csvroot+'%s/%s.xlsx' %(i,i)
        xls = pd.ExcelFile(csvpath)
        df = pd.read_excel(xls, 'Set1dupe')
        df = df.copy()
        for j in ['Edges_light_vel_grey', 'Edges_dark_vel_grey','Edges_light_vel_vert_grey','Edges_dark_vel_vert_grey']:
            ls = [12.5,50,100,200,300,500,900,-12.5,-50,-100,-200,-300,-500,-900]
            if celltype[:2] == 'HS':
                Meandf,Rawdf,celldf= MergeBincells(df,csvroot,j,ls,i)
            else:
                MeanBindf,RawBindf,cellBindf,Meandf,Rawdf,celldf = MergeBincells(df,csvroot,j,ls,i)
        for j in ['flashOnOff_mHz_grey', 'flashOffOn_mHz_grey']:
            ls = [60,120,180,240,300,360,420]
            if celltype[:2] == 'HS':
                Meandf,Rawdf,celldf= MergeBincells(df,csvroot,j,ls,i)
            else:
                MeanBindf,RawBindf,cellBindf,Meandf,Rawdf,celldf = MergeBincells(df,csvroot,j,ls,i)
                
def pltEF(csvroot, celltype):
    sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]
    celltype = celltype[:2]
    if celltype[:2] != 'HS':
        nameset =[['Meanbindf', 'cellbindf','AP'],['Meandf', 'celldf','Vm']]
    else:
        nameset = [['Meandf', 'celldf','Vm']]
    for i in nameset: 
        MeanName = i[0]; CellName = i[1]; datatype = i[2]
        (FlashOnMeanL, FlashOnCellL, FlashOnlenCellL, EdgeOnMeanL, EdgeOnCellL, EdgeOnlenCellL, 
        EdgeOnMeanLv, EdgeOnCellLv, EdgeOnlenCellLv,
        FlashOnMeanR, FlashOnCellR, FlashOnlenCellR, EdgeOnMeanR, EdgeOnCellR, EdgeOnlenCellR,
        EdgeOnMeanRv, EdgeOnCellRv, EdgeOnlenCellRv,
        FlashOnMean, FlashOnCell, FlashOnlenCell, EdgeOnMean, EdgeOnCell, EdgeOnlenCell,
        EdgeOnMeanv, EdgeOnCellv, EdgeOnlenCellv) = OnData(MeanName, CellName, csvroot,celltype)

        (FlashOffMeanL, FlashOffCellL, FlashOfflenCellL, EdgeOffMeanL, EdgeOffCellL, EdgeOfflenCellL,
        EdgeOffMeanLv, EdgeOffCellLv, EdgeOfflenCellLv,
        FlashOffMeanR, FlashOffCellR, FlashOfflenCellR, EdgeOffMeanR, EdgeOffCellR, EdgeOfflenCellR,
        EdgeOffMeanRv, EdgeOffCellRv, EdgeOfflenCellRv,
        FlashOffMean, FlashOffCell, FlashOfflenCell, EdgeOffMean, EdgeOffCell, EdgeOfflenCell,
        EdgeOffMeanv, EdgeOffCellv, EdgeOfflenCellv) = OffData(MeanName, CellName, csvroot,celltype)

        FlashMeanPeak(csvroot, datatype, FlashOnCell,FlashOnCellL,FlashOnCellR,FlashOffCell,FlashOffCellL,FlashOffCellR)
        FlashDurIdx(csvroot, datatype, FlashOnCell,FlashOnCellL,FlashOnCellR,FlashOffCell,FlashOffCellL,FlashOffCellR)

        EdgeOnPeakdf, EdgeOnPeakLdf, EdgeOnPeakRdf, EdgeOnMindf, EdgeOnMinLdf, EdgeOnMinRdf = EdgeOnPeak(csvroot, datatype,EdgeOnCell,EdgeOnCellL,EdgeOnCellR)
        EdgeOffPeakdf, EdgeOffPeakLdf, EdgeOffPeakRdf, EdgeOffMindf, EdgeOffMinLdf, EdgeOffMinRdf = EdgeOffPeak(csvroot, datatype,EdgeOffCell,EdgeOffCellL,EdgeOffCellR)
        
        sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]
        # for HS. The R & L are compared to 'both'
        pidxOn = (EdgeOnPeakdf.values[:,1][:7][::-1] - EdgeOnMindf.values[:,1][7:])/np.max(EdgeOnPeakdf.values[:,1][:7][::-1] + abs(EdgeOnMindf.values[:,1][7:]))
        pidxOff = (EdgeOffPeakdf.values[:,1][:7][::-1] - EdgeOffMindf.values[:,1][7:])/np.max(EdgeOffPeakdf.values[:,1][:7][::-1] + abs(EdgeOffMindf.values[:,1][7:]))

        pidxOnL = (EdgeOnPeakLdf.values[:,1][:7][::-1] - EdgeOnMinLdf.values[:,1][7:])/np.max(EdgeOnPeakdf.values[:,1][:7][::-1] + abs(EdgeOnMindf.values[:,1][7:]))
        pidxOffL = (EdgeOffPeakLdf.values[:,1][:7][::-1] - EdgeOffMinLdf.values[:,1][7:])/np.max(EdgeOffPeakdf.values[:,1][:7][::-1] + abs(EdgeOffMindf.values[:,1][7:]))

        pidxOnR = (EdgeOnPeakRdf.values[:,1][:7][::-1] - EdgeOnMinRdf.values[:,1][7:])/np.max(EdgeOnPeakdf.values[:,1][:7][::-1] + abs(EdgeOnMindf.values[:,1][7:]))
        pidxOffR = (EdgeOffPeakRdf.values[:,1][:7][::-1] - EdgeOffMinRdf.values[:,1][7:])/np.max(EdgeOffPeakdf.values[:,1][:7][::-1] + abs(EdgeOffMindf.values[:,1][7:]))
        
        plt.figure()
        plt.plot(sti[7:],pidxOn, 'k'); plt.plot(sti[7:],pidxOnL, 'b'); plt.plot(sti[7:],pidxOnR, 'g')
        plt.xticks(sti[7:]); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)'); #plt.xscale('symlog')
        plt.title('OnPreferenceIdx_'+datatype); plt.savefig(csvroot+'Pics/On_Preference_Idx_'+datatype)

        plt.figure()
        plt.plot(sti[7:],pidxOff, 'k'); plt.plot(sti[7:],pidxOffL, 'b'); plt.plot(sti[7:],pidxOffR, 'g')
        plt.xticks(sti[7:]); plt.legend(['Both','Ipsi','Contra']); plt.xlabel('Speed(degree/s)'); #plt.xscale('symlog')
        plt.title('OffPreferenceIdx_'+datatype); plt.savefig(csvroot+'Pics/Off_Preference_Idx_'+datatype)

        pidxOndf = pd.DataFrame({'Angles':sti[7:], 'pidxOn':pidxOn, 'pidxOnL':pidxOnL, 'pidxOnR':pidxOnR})
        pidxOffdf = pd.DataFrame({'Angles':sti[7:], 'pidxOff':pidxOff, 'pidxOffL':pidxOffL, 'pidxOffR':pidxOffR})

        pidxOndf.to_csv(csvroot+'Analysis/Edge_peak/pidxOndf_%s.csv' %datatype)
        pidxOffdf.to_csv(csvroot+'Analysis/Edge_peak/pidxOffdf_%s.csv' %datatype)

def EdgeStretch(arr,ref_len):
    arr_interp = interp.interp1d(np.arange(arr.size),arr)
    arr_stretch = arr_interp(np.linspace(0,arr.size-1,ref_len))
    return(arr_stretch)
        
def EdgePref(Edgetype,csvroot,celltype,EdgeOnMean,EdgeOnMeanL,EdgeOnMeanR,EdgeOnMeanv,EdgeOnMeanLv,EdgeOnMeanRv,nanidx_min, nanidxv_min):
    fig1, Onaxes = plt.subplots(7,3, sharey=True, sharex=True,figsize=(12,21))
    if celltype[:2] == 'HS':
        Prefsign = -1
    else:
        Prefsign = 0        
    ls = [12.5,50,100,200,300,500,900]
    for n in range(7):
        i = nanidx_min[n]; j = nanidxv_min[n]   
        EdgeOn = EdgeOnMean.values[n,2:i]-EdgeOnMean.values[n+7,2:i][::-1]
        EdgeOnL =EdgeOnMeanL.values[n,2:i]-EdgeOnMeanL.values[n+7,2:i][::-1]
        EdgeOnR =EdgeOnMeanR.values[n,2:i]-EdgeOnMeanR.values[n+7,2:i][::-1]    

        EdgeOnv = EdgeOnMeanv.values[n,2:j][::-1]-EdgeOnMeanv.values[n+7,2:j]
        EdgeOnLv =EdgeOnMeanLv.values[n,2:j][::-1]-EdgeOnMeanLv.values[n+7,2:j]
        EdgeOnRv =EdgeOnMeanRv.values[n,2:j][::-1]-EdgeOnMeanRv.values[n+7,2:j]

        if n !=0: # interpolate shorter sequences to make them longer
            EdgeOn = EdgeStretch(EdgeOn,int(nanidx_min[0]-2))
            EdgeOnL = EdgeStretch(EdgeOnL,int(nanidx_min[0]-2))
            EdgeOnR = EdgeStretch(EdgeOnR,int(nanidx_min[0]-2))

            EdgeOnv = EdgeStretch(EdgeOnv,int(nanidxv_min[0]-2))
            EdgeOnLv = EdgeStretch(EdgeOnLv,int(nanidxv_min[0]-2))
            EdgeOnRv = EdgeStretch(EdgeOnRv,int(nanidxv_min[0]-2))     

        EdgeOn2D =Prefsign*np.tile(EdgeOn,[np.size(EdgeOnv),1])
        EdgeOnL2D =Prefsign*np.tile(EdgeOnL,[np.size(EdgeOnLv),1])
        EdgeOnR2D =Prefsign*np.tile(EdgeOnR,[np.size(EdgeOnRv),1])

        EdgeOnV2D =Prefsign*np.tile(EdgeOnv,[np.size(EdgeOn),1]).transpose()  
        EdgeOnLV2D =Prefsign*np.tile(EdgeOnLv,[np.size(EdgeOnL),1]).transpose()  
        EdgeOnRV2D =Prefsign*np.tile(EdgeOnRv,[np.size(EdgeOnR),1]).transpose()

        imOn=Onaxes[n,0].imshow(EdgeOnL2D + EdgeOnLV2D,vmin = -5, vmax = 15) 
        Onaxes[n,1].imshow(EdgeOn2D + EdgeOnV2D,vmin = -5, vmax = 15)
        Onaxes[n,2].imshow(EdgeOnR2D + EdgeOnRV2D,vmin = -5, vmax = 15)
        Onaxes[n,0].set_title('Ipsi.\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnL)))); 
        Onaxes[n,1].set_title('Both.\n Speed ='+str(ls[n])+'\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOn)))); 
        Onaxes[n,2].set_title('Right\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnR))))    
        xmax = max(len(EdgeOnL),len(EdgeOn),len(EdgeOnR)); 
        Onaxes[n,0].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,0].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,1].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,1].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,2].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,2].set_xticklabels([-120,-60,0,60,120]);
        ymax = max(len(EdgeOnL2D),len(EdgeOn2D),len(EdgeOnR2D)); 
        Onaxes[n,0].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,0].set_yticklabels([66,33,0,-33]);
        Onaxes[n,1].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,1].set_yticklabels([66,33,0,-33]);
        Onaxes[n,2].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,2].set_yticklabels([66,33,0,-33]);
        plt.tight_layout(); 

    fig1.colorbar(imOn,ax=Onaxes)
    fig1.savefig(csvroot+'Pics/'+Edgetype+'SpeedsPref.png')  

def EdgePD(Edgetype,csvroot, celltype,EdgeOnMean,EdgeOnMeanL,EdgeOnMeanR,EdgeOnMeanv,EdgeOnMeanLv,EdgeOnMeanRv,nanidx_min, nanidxv_min):
    # PD for H2, ND for HS
    if celltype[:2] == 'HS':
        PDtitle = 'ND'
    else:
        PDtitle = 'PD'     
    fig1, Onaxes = plt.subplots(7,3, sharey=True, sharex=True,figsize=(12,21))
    ls = [12.5,50,100,200,300,500,900]
    for n in range(7):
        i = nanidx_min[n]; j = nanidxv_min[n]   
        EdgeOn = EdgeOnMean.values[n,2:i]#-EdgeOnMean.values[n+7,2:i][::-1]
        EdgeOnL =EdgeOnMeanL.values[n,2:i]#-EdgeOnMeanL.values[n+7,2:i][::-1]
        EdgeOnR =EdgeOnMeanR.values[n,2:i]#-EdgeOnMeanR.values[n+7,2:i][::-1]    

        EdgeOnv = EdgeOnMeanv.values[n,2:j][::-1]#-EdgeOnMeanv.values[n+7,2:j]
        EdgeOnLv =EdgeOnMeanLv.values[n,2:j][::-1]#-EdgeOnMeanLv.values[n+7,2:j]
        EdgeOnRv =EdgeOnMeanRv.values[n,2:j][::-1]#-EdgeOnMeanRv.values[n+7,2:j]

        if n !=0: # interpolate shorter sequences to make them longer
            EdgeOn = EdgeStretch(EdgeOn,int(nanidx_min[0]-2))
            EdgeOnL = EdgeStretch(EdgeOnL,int(nanidx_min[0]-2))
            EdgeOnR = EdgeStretch(EdgeOnR,int(nanidx_min[0]-2))

            EdgeOnv = EdgeStretch(EdgeOnv,int(nanidxv_min[0]-2))
            EdgeOnLv = EdgeStretch(EdgeOnLv,int(nanidxv_min[0]-2))
            EdgeOnRv = EdgeStretch(EdgeOnRv,int(nanidxv_min[0]-2))     

        EdgeOn2D =np.tile(EdgeOn,[np.size(EdgeOnv),1])
        EdgeOnL2D =np.tile(EdgeOnL,[np.size(EdgeOnLv),1])
        EdgeOnR2D =np.tile(EdgeOnR,[np.size(EdgeOnRv),1])

        EdgeOnV2D =np.tile(EdgeOnv,[np.size(EdgeOn),1]).transpose()  
        EdgeOnLV2D =np.tile(EdgeOnLv,[np.size(EdgeOnL),1]).transpose()  
        EdgeOnRV2D =np.tile(EdgeOnRv,[np.size(EdgeOnR),1]).transpose()

        imOn=Onaxes[n,0].imshow(EdgeOnL2D + EdgeOnLV2D,vmin = -5, vmax = 15) 
        Onaxes[n,1].imshow(EdgeOn2D + EdgeOnV2D,vmin = -5, vmax = 15)
        Onaxes[n,2].imshow(EdgeOnR2D + EdgeOnRV2D,vmin = -5, vmax = 15)
        Onaxes[n,0].set_title('Ipsi.\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnL)))); 
        Onaxes[n,1].set_title('Both.\n Speed ='+str(ls[n])+'\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOn)))); 
        Onaxes[n,2].set_title('Right\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnR))))    
        xmax = max(len(EdgeOnL),len(EdgeOn),len(EdgeOnR)); 
        Onaxes[n,0].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,0].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,1].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,1].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,2].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,2].set_xticklabels([-120,-60,0,60,120]);
        ymax = max(len(EdgeOnL2D),len(EdgeOn2D),len(EdgeOnR2D)); 
        Onaxes[n,0].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,0].set_yticklabels([66,33,0,-33]);
        Onaxes[n,1].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,1].set_yticklabels([66,33,0,-33]);
        Onaxes[n,2].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,2].set_yticklabels([66,33,0,-33]);
        plt.tight_layout(); 

    fig1.colorbar(imOn,ax=Onaxes)
    fig1.savefig(csvroot+'Pics/'+Edgetype+'_'+celltype[:2]+'_Speeds_'+PDtitle+'.png')  

def EdgeND(Edgetype,csvroot,celltype, EdgeOnMean,EdgeOnMeanL,EdgeOnMeanR,EdgeOnMeanv,EdgeOnMeanLv,EdgeOnMeanRv,nanidx_min, nanidxv_min):
    # ND for H2, PD for HS
    if celltype[:2] == 'HS':
        NDtitle = 'PD'
    else:
        NDtitle = 'ND'
    fig1, Onaxes = plt.subplots(7,3, sharey=True, sharex=True,figsize=(12,21))
    ls = [12.5,50,100,200,300,500,900]
    for n in range(7):
        i = nanidx_min[n]; j = nanidxv_min[n]   
        EdgeOn = EdgeOnMean.values[n+7,2:i][::-1]
        EdgeOnL =EdgeOnMeanL.values[n+7,2:i][::-1]
        EdgeOnR =EdgeOnMeanR.values[n+7,2:i][::-1]    

        EdgeOnv = EdgeOnMeanv.values[n+7,2:j]
        EdgeOnLv =EdgeOnMeanLv.values[n+7,2:j]
        EdgeOnRv =EdgeOnMeanRv.values[n+7,2:j]

        if n !=0: # interpolate shorter sequences to make them longer
            EdgeOn = EdgeStretch(EdgeOn,int(nanidx_min[0]-2))
            EdgeOnL = EdgeStretch(EdgeOnL,int(nanidx_min[0]-2))
            EdgeOnR = EdgeStretch(EdgeOnR,int(nanidx_min[0]-2))

            EdgeOnv = EdgeStretch(EdgeOnv,int(nanidxv_min[0]-2))
            EdgeOnLv = EdgeStretch(EdgeOnLv,int(nanidxv_min[0]-2))
            EdgeOnRv = EdgeStretch(EdgeOnRv,int(nanidxv_min[0]-2))     

        EdgeOn2D =np.tile(EdgeOn,[np.size(EdgeOnv),1])
        EdgeOnL2D =np.tile(EdgeOnL,[np.size(EdgeOnLv),1])
        EdgeOnR2D =np.tile(EdgeOnR,[np.size(EdgeOnRv),1])

        EdgeOnV2D =np.tile(EdgeOnv,[np.size(EdgeOn),1]).transpose()  
        EdgeOnLV2D =np.tile(EdgeOnLv,[np.size(EdgeOnL),1]).transpose()  
        EdgeOnRV2D =np.tile(EdgeOnRv,[np.size(EdgeOnR),1]).transpose()

        imOn=Onaxes[n,0].imshow(EdgeOnL2D + EdgeOnLV2D,vmin = -5, vmax = 15) 
        Onaxes[n,1].imshow(EdgeOn2D + EdgeOnV2D,vmin = -5, vmax = 15)
        Onaxes[n,2].imshow(EdgeOnR2D + EdgeOnRV2D,vmin = -5, vmax = 15)
        Onaxes[n,0].set_title('Ipsi.\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnL)))); 
        Onaxes[n,1].set_title('Both.\n Speed ='+str(ls[n])+'\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOn)))); 
        Onaxes[n,2].set_title('Right\n Avg=' + str("{:.2f}".format(-np.mean(EdgeOnR))))    
        xmax = max(len(EdgeOnL),len(EdgeOn),len(EdgeOnR)); 
        Onaxes[n,0].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,0].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,1].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,1].set_xticklabels([-120,-60,0,60,120]);
        Onaxes[n,2].set_xticks([0,xmax/4,xmax/2,xmax/4*3,xmax]); Onaxes[n,2].set_xticklabels([-120,-60,0,60,120]);
        ymax = max(len(EdgeOnL2D),len(EdgeOn2D),len(EdgeOnR2D)); 
        Onaxes[n,0].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,0].set_yticklabels([66,33,0,-33]);
        Onaxes[n,1].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,1].set_yticklabels([66,33,0,-33]);
        Onaxes[n,2].set_yticks([0,ymax/3,ymax/3*2,ymax]); Onaxes[n,2].set_yticklabels([66,33,0,-33]);
        plt.tight_layout(); 

    fig1.colorbar(imOn,ax=Onaxes)
    fig1.savefig(csvroot+'Pics/'+Edgetype+'_'+celltype[:2]+'_Speeds_'+NDtitle+'.png') 

def pltEdgeTracesSpace(csvroot, celltype, EdgeOnMean,EdgeOnMeanL,EdgeOnMeanR,EdgeOffMean,EdgeOffMeanL,EdgeOffMeanR):
    ## plot raw traces of Edge response 
    ## uncomment to plot Ipsi & Contra. Or Vertical (need to add input)
    if celltype[:2]=='HS':
        PDtitle = 'ND';  NDtitle = 'PD'; Prefsign = -1 # reverse the direction for PD & ND.
    else:
        PDtitle = 'PD';  NDtitle ='ND';Prefsign = 1 # stay the same direction for PD & ND.
    
    nanidx = np.asarray([np.shape(i)[1] - i.isna().sum(axis=1).values for i in [EdgeOnMean,EdgeOnMeanL,EdgeOnMeanR,EdgeOffMean,EdgeOffMeanL,EdgeOffMeanR]]).reshape((12,7))
    nanidx_min = nanidx.min(axis=0)
#     nanidxv = np.asarray([np.shape(i)[1] - i.isna().sum(axis=1).values for i in [EdgeOnMeanv,EdgeOnMeanLv,EdgeOnMeanRv,EdgeOffMeanv,EdgeOffMeanLv,EdgeOffMeanRv]]).reshape((12,7))
#     nanidxv_min = nanidxv.min(axis=0)

     # plot PD (or ND for HS)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,3))
    for n, i in enumerate(nanidx_min):    
        EdgeOn = EdgeOnMean.values[n,2:i];# EdgeOnL =EdgeOnMeanL.values[n,2:i]; EdgeOnR =EdgeOnMeanL.values[n,2:i]
        x = 240/(len(EdgeOn)-1)*np.arange(len(EdgeOn))-120
        axe1.plot(x,EdgeOn,'k',alpha = 0.1*np.sqrt(n)+0.15); axe1.axvline(x=0,c='r',ls=':')
        #axe1.plot(x,EdgeOnL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe1.plot(x,EdgeOnR,'g',alpha = 0.1*np.sqrt(n)+0.15);             

        EdgeOff = EdgeOffMean.values[n,2:i]; # EdgeOffL =EdgeOffMeanL.values[n,2:i]; EdgeOffR =EdgeOffMeanR.values[n,2:i]
        axe2.plot(x,EdgeOff,'k',alpha = 0.1*np.sqrt(n)+0.15);  axe2.axvline(x=0,c='r',ls=':')
        #axe2.plot(x,EdgeOffL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe2.plot(x,EdgeOffR,'g',alpha = 0.1*np.sqrt(n)+0.15);        
    fig.suptitle(PDtitle)
    fig.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeTracesSpace_'+PDtitle)
    
    # plot pref
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,3))
    for n, i in enumerate(nanidx_min):    
        EdgeOn = EdgeOnMean.values[n,2:i]-EdgeOnMean.values[n+7,2:i][::-1]
        #EdgeOnL =EdgeOnMeanL.values[n,2:i]-EdgeOnMeanL.values[n+7,2:i][::-1]
        #EdgeOnR =EdgeOnMeanR.values[n,2:i]-EdgeOnMeanR.values[n+7,2:i][::-1]
        x = 240/(len(EdgeOn)-1)*np.arange(len(EdgeOn))-120
        axe1.plot(x,Prefsign*EdgeOn,'k',alpha = 0.1*np.sqrt(n)+0.15); axe1.axvline(x=0,c='r',ls=':')
        #axe1.plot(x,Prefsign*EdgeOnL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe1.plot(x,Prefsign*EdgeOnR,'g',alpha = 0.1*np.sqrt(n)+0.15);          
        EdgeOff = EdgeOffMean.values[n,2:i]-EdgeOffMean.values[n+7,2:i][::-1]
        #EdgeOffL =EdgeOffMeanL.values[n,2:i]-EdgeOffMeanL.values[n+7,2:i][::-1]
        #EdgeOffR =EdgeOffMeanR.values[n,2:i]-EdgeOffMeanR.values[n+7,2:i][::-1]    
        axe2.plot(x,Prefsign*EdgeOff,'k',alpha = 0.1*np.sqrt(n)+0.15); axe2.axvline(x=0,c='r',ls=':')
        #axe2.plot(x,Prefsign*EdgeOffL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe2.plot(x,Prefsign*EdgeOffR,'g',alpha = 0.1*np.sqrt(n)+0.15);  
    fig.suptitle('Pref')
    fig.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeTracesSpace_Pref')

    # plot ND (or PD for HS)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,3))
    for n, i in enumerate(nanidx_min):    
        EdgeOn = EdgeOnMean.values[n+7,2:i][::-1]
        #EdgeOnL = EdgeOnMeanL.values[n+7,2:i][::-1]
        #EdgeOnR = EdgeOnMeanR.values[n+7, 2:i][::-1]
        x = 240/(len(EdgeOn)-1)*np.arange(len(EdgeOn))-120
        axe1.plot(x,EdgeOn,'k',alpha = 0.1*np.sqrt(n)+0.15); axe1.axvline(x=0,c='r',ls=':')
        #axe1.plot(x,EdgeOnL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe1.plot(x,EdgeOnR,'g',alpha = 0.1*np.sqrt(n)+0.15);             

        EdgeOff = EdgeOffMean.values[n+7,2:i][::-1]
        #EdgeOffL = EdgeOffMeanL.values[n+7,2:i][::-1]
        #EdgeOffR = EdgeOffMeanR.values[n+7,2:i][::-1]    
        axe2.plot(x,EdgeOff,'k',alpha = 0.1*np.sqrt(n)+0.15); axe2.axvline(x=0,c='r',ls=':')
        #axe2.plot(x,EdgeOffL,'b',alpha = 0.1*np.sqrt(n)+0.15); axe2.plot(x,EdgeOffR,'g',alpha = 0.1*np.sqrt(n)+0.15);        
    fig.suptitle(NDtitle)    
    fig.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeTracesSpace_'+NDtitle)

#     # plot vertical
#     fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,3))   
#     for n, j in enumerate(nanidxv_min):
#         EdgeOnv = EdgeOnMeanv.values[n,2:j][::-1]-EdgeOnMeanv.values[n+7,2:j]
#         EdgeOnLv =EdgeOnMeanLv.values[n,2:j][::-1]-EdgeOnMeanLv.values[n+7,2:j]
#         EdgeOnRv =EdgeOnMeanRv.values[n,2:j][::-1]-EdgeOnMeanRv.values[n+7,2:j]
#         y = 99/(len(EdgeOnv)-1)*np.arange(len(EdgeOnv))-33
#         axe1.plot(EdgeOnv,y,'k',alpha = 0.15*np.sqrt(n)); axe1.plot(EdgeOnLv,y,'b',alpha = 0.15*np.sqrt(n))
#         axe1.plot(EdgeOnRv,y,'g',alpha = 0.15*np.sqrt(n)); axe1.axhline(y=0,c='r',ls=':')    

#         EdgeOffv = EdgeOffMeanv.values[n,2:j][::-1]-EdgeOffMeanv.values[n+7,2:j]
#         EdgeOffLv =EdgeOffMeanLv.values[n,2:j][::-1]-EdgeOffMeanLv.values[n+7,2:j]
#         EdgeOffRv =EdgeOffMeanRv.values[n,2:j][::-1]-EdgeOffMeanRv.values[n+7,2:j]   
#         axe2.plot(EdgeOffv,y,'k',alpha =  0.1*np.sqrt(n)+0.15); axe2.plot(EdgeOffLv,y,'b',alpha =  0.1*np.sqrt(n)+0.15)
#         axe2.plot(EdgeOffRv,y,'g',alpha =  0.1*np.sqrt(n)+0.15); axe2.axhline(y=0,c='r',ls=':')    
        
def checksize(Lsetls):
    if np.size(Lsetls[Lsetls>0]) != 0:
        return(Lsetls[Lsetls>0].min())
    else:
        return(np.nan)

def EdgePeaknIdx(csvroot,celltype,EdgeOnCell): 
    # the percentage of receptive field Onset & Off for On & OFF. Based on averaged onsetidx & offsetidx
    onsetP = 0.15; offsetP = 0.75
    # Onset & Offset Latency
    # http://www.jneurosci.org/content/22/8/3189.long
    #Response onset latency was determined automatically by finding the maximum point on the response curve (or on the response difference curve) and searching backwards in time from the maximum for the first point that was above 5% of the maximum response. A similar procedure was used to find response offset latency, except the minimum point and the 5% drop to minimum were used.
    sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]
    cellN = int(np.shape(EdgeOnCell)[0]/len(sti))
    sizedic= np.asarray([np.shape(i)[1] - i.isna().sum(axis=1).values for i in [EdgeOnCell]])-2
    sizels = sizedic.reshape((14,cellN)).min(axis=1).astype(int)

    Highs = np.zeros((len(sti),cellN)); Lows = np.zeros((len(sti),cellN))
    Highidx = np.zeros((len(sti),cellN)); Lowidx = np.zeros((len(sti),cellN))
    Honsetidx = np.zeros((len(sti),cellN)); Hoffsetidx= np.zeros((len(sti),cellN))
    Lonsetidx = np.zeros((len(sti),cellN)); Loffsetidx= np.zeros((len(sti),cellN))
    Avgs = np.zeros((len(sti),cellN))
    
    # idx position is after the first 2 columns of loc & angle
    for n in range(len(sti)):
        Highs[n,:] = np.max(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1)
        Highidx[n,:] = int(sizels[n]*1/5)+ np.argmax(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1)
        Lows[n,:] = np.min(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1)
        Lowidx[n,:] = int(sizels[n]*1/5)+ np.argmin(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1)
        Avgs[n,:]= np.mean(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*onsetP):int(2+sizels[n]*offsetP)],axis=1)        
        valset = EdgeOnCell.values[n*7:(cellN+n*7),2:int(2+sizels[n])]
        for m in range(cellN):
            Hsetls = np.where(valset[m]<Highs[n,:][m]*0.05)-Highidx[n,:][m]        
            Honsetidx[n,m] = Hsetls[Hsetls<0].max()+Highidx[n,:][m]-int(onsetP*sizels[n])
            Hoffsetidx[n,m] =  checksize(Hsetls)+Highidx[n,:][m]-int(onsetP*sizels[n])
            Lsetls = np.where(valset[m]>Lows[n,:][m]*0.05)-Lowidx[n,:][m]
            Lonsetidx[n,m] = Lsetls[Lsetls<0].max()+Lowidx[n,:][m]-int(onsetP*sizels[n])
            Loffsetidx[n,m] = checksize(Lsetls)+Lowidx[n,:][m]-int(onsetP*sizels[n])
        Lowidx[n,:]  = Lowidx[n,:] -int(onsetP*sizels[n])
        Highidx[n,:] = Highidx[n,:] -int(onsetP*sizels[n])
    Highnorm = np.zeros((len(sti),cellN)); Lownorm = np.zeros((len(sti),cellN)); Avgnorm = np.zeros((len(sti),cellN))
    for i in range(cellN):
        Highnorm[:,i] = Highs[:,i]/np.max(Highs[:,i])*np.mean(np.max(Highs)) 
        Lownorm[:,i] =  Lows[:,i]/np.max(Highs[:,i])*np.mean(np.max(Highs))
        Avgnorm[:,i] = Avgs[:,i]/np.max(Avgs[:,i])*np.mean(np.max(Avgs))        
        
    dirs = csvroot+'Analysis/Edges/'+ celltype; makedir(dirs)
    Highdf = pd.DataFrame(Highs) ; Highdf.insert(0, 'Angles', sti); Highdf.to_csv(dirs+'/Highdf.csv')
    Lowdf = pd.DataFrame(Lows) ; Lowdf.insert(0, 'Angles', sti); Lowdf.to_csv(dirs+'/Lowdf.csv')
    Highnormdf = pd.DataFrame(Highnorm) ; Highnormdf.insert(0, 'Angles', sti); Highnormdf.to_csv(dirs+'/Highnormdf.csv')
    Lownormdf = pd.DataFrame(Lownorm) ; Lownormdf.insert(0, 'Angles', sti); Lownormdf.to_csv(dirs+'/Lownormdf.csv')    
    Avgdf = pd.DataFrame(Avgs); Avgdf.insert(0, 'Angles', sti); Avgdf.to_csv(dirs+'/Avgsdf.csv')
    Avgnormdf = pd.DataFrame(Avgnorm); Avgnormdf.insert(0, 'Angles', sti); Avgnormdf.to_csv(dirs+'/Avgnormdf.csv')
    
    Highidxdf = pd.DataFrame(Highidx) ; Highidxdf.insert(0, 'Angles', sti); Highidxdf.to_csv(dirs+'/Highidxdf.csv')
    Lowidxdf = pd.DataFrame(Lowidx) ; Lowidxdf.insert(0, 'Angles', sti); Lowidxdf.to_csv(dirs+'/Lowidxdf.csv')
    Honsetidxdf = pd.DataFrame(Honsetidx) ; Honsetidxdf.insert(0, 'Angles', sti); Honsetidxdf.to_csv(dirs+'/Honsetidxdf.csv')
    Hoffsetidxdf = pd.DataFrame(Hoffsetidx) ; Hoffsetidxdf.insert(0, 'Angles', sti); Hoffsetidxdf.to_csv(dirs+'/Hoffsetidxdf.csv')
    Lonsetidxdf = pd.DataFrame(Lonsetidx) ; Lonsetidxdf.insert(0, 'Angles', sti); Lonsetidxdf.to_csv(dirs+'/Lonsetidxdf.csv')
    Loffsetidxdf = pd.DataFrame(Loffsetidx) ; Loffsetidxdf.insert(0, 'Angles', sti); Loffsetidxdf.to_csv(dirs+'/Loffsetidxdf.csv')
    sizelsdf = pd.DataFrame(sizels); sizelsdf.insert(0, 'Angles', sti); sizelsdf.to_csv(dirs+'/sizelsdf.csv')
    
#     #plot to check
#     n=6; m =5

#     print(np.argmax(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1))
#     plt.plot(EdgeOnCell.values[n*7:(cellN+n*7),2:(2+sizels[n])][m])
#     #plt.plot(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)][m])
#     plt.axvline(x=int(sizels[n]*1/5),c='r'); plt.axvline(x=int(sizels[n]*4/5),c='r')
#     plt.axvline(x=int(sizels[n]*1/5)+np.argmax(EdgeOnCell.values[n*7:(cellN+n*7),int(2+sizels[n]*1/5):int(2+sizels[n]*4/5)],axis=1)[m],c='g')
#     plt.axvline(x=Honsetidxdf.values[n][1+m],c='c')
#     plt.axvline(x=Hoffsetidxdf.values[n][1+m],c='y')
#     plt.show()
    
    return(sizelsdf,Highdf, Lowdf, Highnormdf, Lownormdf, Avgdf, Avgnormdf, Highidxdf, Lowidxdf, Honsetidxdf, Hoffsetidxdf, Lonsetidxdf, Loffsetidxdf)

def pltEdgeLatResp(csvroot,celltype,OnAvgnormdf,OffAvgnormdf,OnLownormdf,
                   OffLownormdf,OnLonsetidxdf,OffLonsetidxdf, OnLoffsetidxdf,OffLoffsetidxdf,
                   OnHighnormdf,OffHighnormdf,OnHonsetidxdf,OffHonsetidxdf,
                   OnHoffsetidxdf,OffHoffsetidxdf):
    if celltype[:2] == 'HS':
        NDtitle = 'ND'; PDtitle = 'PD'
    else:
        NDtitle = 'PD'; PDtitle = 'ND'
    sti = [-900, -500, -300,-200,-100,-50, -12.5, 12.5,50,100,200,300,500,900]        
    # plot response for ND (or PD for H2)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,5))
    plt.xscale('symlog');# plt.yscale('symlog')
    axe1.plot(sti[7:],np.nanmean(OnAvgnormdf.values[:,1:][7:],axis=1),'b',marker='o')
    axe1.plot(sti[7:],np.nanmean(OffAvgnormdf.values[:,1:][7:],axis=1),'r',marker='o')
    [axe1.scatter(sti[7:], OnAvgnormdf.values[:,1:][7:][:,itpt], color='b', alpha=0.2) for itpt in range(7)]
    [axe1.scatter(sti[7:], OffAvgnormdf.values[:,1:][7:][:,itpt], color='r', alpha=0.2) for itpt in range(7)]
    axe1.set_title('Average'); axe1.set_ylabel('Response (mV)'); axe1.set_xlabel('Edge Velocity (degree/s)')

    axe2.plot(sti[7:],np.nanmean(OnLownormdf.values[:,1:][7:],axis=1),'b',marker='o')
    axe2.plot(sti[7:],np.nanmean(OffLownormdf.values[:,1:][7:],axis=1),'r',marker='o')
    [axe2.scatter(sti[7:], OnLownormdf.values[:,1:][7:][:,itpt], color='b', alpha=0.2) for itpt in range(7)]
    [axe2.scatter(sti[7:], OffLownormdf.values[:,1:][7:][:,itpt], color='r', alpha=0.2) for itpt in range(7)]
    axe2.set_title('Minimum'); axe2.set_xlabel('Edge Velocity (degree/s)')
    plt.legend(['On','Off'])
    fig.suptitle('Response '+ NDtitle)
    plt.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeResponse_'+ NDtitle)
    plt.show()

    # plot latency for ND (or PD for H2)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,5))
    plt.xscale('symlog'); plt.yscale('symlog')
    axe1.plot(sti[7:],np.nanmean(OnLonsetidxdf.values[:,1:][7:]*0.05,axis=1),'b',marker='o')
    axe1.plot(sti[7:],np.nanmean(OffLonsetidxdf.values[:,1:][7:]*0.05,axis=1),'r',marker='o')
    [axe1.scatter(sti[7:], OnLonsetidxdf.values[:,1:][7:][:,itpt]*0.05, color='b', alpha=0.2) for itpt in range(7)]
    [axe1.scatter(sti[7:], OffLonsetidxdf.values[:,1:][7:][:,itpt]*0.05, color='r', alpha=0.2) for itpt in range(7)]
    axe1.set_title('Onset'); axe1.set_ylabel('Latency (s)'); axe1.set_xlabel('Edge Velocity (degree/s)')
    axe2.plot(sti[7:],np.nanmean(OnLoffsetidxdf.values[:,1:][7:]*0.05,axis=1),'b',marker='o')
    axe2.plot(sti[7:],np.nanmean(OffLoffsetidxdf.values[:,1:][7:]*0.05,axis=1),'r',marker='o')
    [axe2.scatter(sti[7:], OnLoffsetidxdf.values[:,1:][7:][:,itpt]*0.05, color='b', alpha=0.2) for itpt in range(7)]
    [axe2.scatter(sti[7:], OffLoffsetidxdf.values[:,1:][7:][:,itpt]*0.05, color='r', alpha=0.2) for itpt in range(7)]
    axe2.set_title('Offset'); axe2.set_xlabel('Edge Velocity (degree/s)')
    plt.legend(['On','Off'])
    fig.suptitle('Latency '+ NDtitle)
    plt.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeLatency_'+ NDtitle)
    plt.show()

    # plot response for PD (or ND for H2)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,5))
    plt.xscale('symlog');# plt.yscale('symlog')
    axe1.plot(sti[7:][::-1],np.nanmean(OnAvgnormdf.values[:,1:][:7],axis=1),'b',marker='o')
    axe1.plot(sti[7:][::-1],np.nanmean(OffAvgnormdf.values[:,1:][:7],axis=1),'r',marker='o')
    [axe1.scatter(sti[7:][::-1], OnAvgnormdf.values[:,1:][:7][:,itpt], color='b', alpha=0.2) for itpt in range(7)]
    [axe1.scatter(sti[7:][::-1], OffAvgnormdf.values[:,1:][:7][:,itpt], color='r', alpha=0.2) for itpt in range(7)]
    axe1.set_title('Average'); axe1.set_ylabel('Response (mV)'); axe1.set_xlabel('Edge Velocity (degree/s)')

    axe2.plot(sti[7:][::-1],np.nanmean(OnHighnormdf.values[:,1:][:7],axis=1),'b',marker='o')
    axe2.plot(sti[7:][::-1],np.nanmean(OffHighnormdf.values[:,1:][:7],axis=1),'r',marker='o')
    [axe2.scatter(sti[7:][::-1], OnHighnormdf.values[:,1:][:7][:,itpt], color='b', alpha=0.2) for itpt in range(7)]
    [axe2.scatter(sti[7:][::-1], OffHighnormdf.values[:,1:][:7][:,itpt], color='r', alpha=0.2) for itpt in range(7)]
    axe2.set_title('Maximum'); axe2.set_xlabel('Edge Velocity (degree/s)')
    plt.legend(['On','Off'])
    fig.suptitle('Response '+ PDtitle)
    plt.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeResponse_'+ PDtitle)
    plt.show()

    # plot latency for PD (or ND for H2)
    fig, [axe1,axe2] = plt.subplots(1,2, sharey=True, sharex=True,figsize=(12,5))
    plt.xscale('symlog'); plt.yscale('symlog')
    axe1.plot(sti[7:][::-1],np.nanmean(OnHonsetidxdf.values[:,1:][:7]*0.05,axis=1),'b',marker='o')
    axe1.plot(sti[7:][::-1],np.nanmean(OffHonsetidxdf.values[:,1:][:7]*0.05,axis=1),'r',marker='o')
    [axe1.scatter(sti[7:][::-1], OnHonsetidxdf.values[:,1:][:7][:,itpt]*0.05, color='b', alpha=0.2) for itpt in range(7)]
    [axe1.scatter(sti[7:][::-1], OffHonsetidxdf.values[:,1:][:7][:,itpt]*0.05, color='r', alpha=0.2) for itpt in range(7)]
    axe1.set_title('Onset'); axe1.set_ylabel('Latency (s)'); axe1.set_xlabel('Edge Velocity (degree/s)')
    axe2.plot(sti[7:][::-1],np.nanmean(OnHoffsetidxdf.values[:,1:][:7]*0.05,axis=1),'b',marker='o')
    axe2.plot(sti[7:][::-1],np.nanmean(OffHoffsetidxdf.values[:,1:][:7]*0.05,axis=1),'r',marker='o')
    [axe2.scatter(sti[7:][::-1], OnHoffsetidxdf.values[:,1:][:7][:,itpt]*0.05, color='b', alpha=0.2) for itpt in range(7)]
    [axe2.scatter(sti[7:][::-1], OffHoffsetidxdf.values[:,1:][:7][:,itpt]*0.05, color='r', alpha=0.2) for itpt in range(7)]
    axe2.set_title('Offset'); axe2.set_xlabel('Edge Velocity (degree/s)')
    plt.legend(['On','Off'])
    fig.suptitle('Latency '+ PDtitle)
    plt.savefig(csvroot+'Pics/'+celltype[:2]+'_EdgeLatency_'+ PDtitle)
    plt.show()
 
    


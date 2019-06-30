
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import os, errno
import pandas as pd

def makedir(newdir):
    try:
        os.makedirs(newdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def MergecellsMap(df,csvroot,celltype,mapN, Hz): 
    if celltype[:2] == 'HS':
        spiking = 0
    else:
        spiking = 1
        Meandf = pd.DataFrame(); celldf = pd.DataFrame(); 
        Meannormdf = pd.DataFrame(); cellnormdf = pd.DataFrame() 
        Apmax = []    
    MeanVdf = pd.DataFrame(); cellVdf = pd.DataFrame();
    MeanVnormdf = pd.DataFrame(); cellVnormdf = pd.DataFrame();
    Vmax = []; recN = 0  
    if mapN == 'Map1':
        stimulus = 'Sine_patches_PDND_%dHZ_0.4c'%Hz; namest = "Vec_MT1_%d"%Hz
    else: 
        stimulus = 'Sine_patches2_PDND_%dHZ_0.4c'%Hz; namest = "Vec_MT2_%d"%Hz
    for i in np.arange(len(df['cells'][:-1])): #[:-1] is for getting rid of 'totalN'
        cell = df['cells'][i]
        if str(df[stimulus][i]) !='nan':       
            mapdir = csvroot+'Analysis/%s/%s/'%(celltype,cell)     
            Vdfname = [filename for filename in os.listdir(mapdir) if filename.startswith(namest) &  filename.endswith("Vmov.csv")] 
            Vdf = pd.read_csv(mapdir+Vdfname[0]);vmax = max(max(abs(Vdf['Vx'])),max(abs(Vdf['Vy']))); Vmax.append(vmax)
            normVavg = Vdf.copy(); normVavg[['Vx','Vy']] = Vdf[['Vx','Vy']]/vmax; 
            cellVdf = cellVdf.append(Vdf); cellVnormdf = cellVnormdf.append(normVavg)
            MeanVdf = MeanVdf.add(Vdf,fill_value=0);MeanVnormdf = MeanVnormdf.add(normVavg,fill_value=0)
            if spiking: 
                Apdfname = [filename for filename in os.listdir(mapdir) if filename.startswith(namest) &  filename.endswith("Apmov.csv")]
                Apdf = pd.read_csv(mapdir+Apdfname[0]); apmax = max(max(abs(Apdf['Vx'])),max(abs(Apdf['Vy']))); Apmax.append(apmax)
                normApavg = Apdf.copy(); normApavg[['Vx','Vy']] = Apdf[['Vx','Vy']]/apmax;
                celldf = celldf.append(Apdf);cellnormdf = cellnormdf.append(normApavg)
                Meandf = Meandf.add(Apdf,fill_value=0); Meannormdf = Meannormdf.add(normApavg,fill_value=0);                 
            recN +=1        
    MeanVdf = MeanVdf/recN;  MeanVnormdf = MeanVnormdf/recN 
    dirs = csvroot+'Analysis/'+ mapN +'/'+ celltype + '/' + str(Hz) +'Hz'; makedir(dirs)
    MeanVdf.to_csv(dirs+'/MeanVdf.csv'); cellVdf.to_csv(dirs+'/cellVdf.csv'); 
    MeanVnormdf.to_csv(dirs+'/MeanVnormdf.csv'); cellVnormdf.to_csv(dirs+'/cellVnormdf.csv'); 
    if spiking: 
        Meandf = Meandf/recN; Meannormdf = Meannormdf/recN
        Meandf.to_csv(dirs+'/Meandf.csv'); celldf.to_csv(dirs+'/celldf.csv'); 
        Meannormdf.to_csv(dirs+'/Meannormdf.csv'); cellnormdf.to_csv(dirs+'/cellnormdf.csv');
        return(Meandf,celldf,MeanVdf,cellVdf,Meannormdf,cellnormdf,MeanVnormdf,cellVnormdf)
    else:
        return(MeanVdf,cellVdf,MeanVnormdf,cellVnormdf)

def pltMap(Meandf,celldf,dtype,Hz):
    XVMov = Meandf['Vx']; YVMov = Meandf['Vy']
    xcenters = Meandf['x']; ycenters = Meandf['y']
    beta = 10/np.max([XVMov,YVMov]) # scale factor to see the point well
    fig = plt.figure(figsize=(8*2, 3.3*2));plt.axvline(x=0, c='g'); plt.grid(True)
    ax = fig.add_subplot(111); ax.set_xticks(np.arange(-120,122,40)); ax.set_yticks(np.arange(-33,67,33))
    for i in ycenters: plt.axhline(y = i, color = 'g', ls = ':',alpha =0.4)
    for i in xcenters: plt.axvline(x = i, color = 'g', ls = ':',alpha =0.4)

    for i in np.arange(int(len(celldf.index)/len(xcenters))):
        celldata = celldf[int(i*len(xcenters)):int((i+1)*len(xcenters))]
        xVMov = celldata['Vx']; yVMov = celldata['Vy']
        for n,i in enumerate(xcenters):   
            ax.arrow(i, ycenters[n], xVMov[n]*beta, yVMov[n]*beta, lw= 0.5,fc='m', ec='m', head_width=0.8, head_length =1, alpha = 0.1 ) 

    for n,i in enumerate(xcenters):   
        ax.arrow(i, ycenters[n], XVMov[n]*beta, YVMov[n]*beta, lw= 1,fc='r', ec='r', head_width=1.5, head_length =2) 

    plt.axes().set_aspect('equal')
    plt.xlim([-120,120]); plt.ylim([-33,66]);  plt.title(dtype+', %dHz, Scale = x %f'% (Hz,beta))
    plt.tight_layout();
    
def pltFF(meanN,cellN,dtype,Hz,csvroot,celltype):
    map1dir = csvroot + 'Analysis/Map1/'+celltype + '/' + str(Hz) +'Hz/'
    map2dir = csvroot + 'Analysis/Map2/'+celltype + '/' + str(Hz) +'Hz/'
    
    Mean1df = pd.read_csv(map1dir+meanN+'.csv'); cell1df = pd.read_csv(map1dir+cellN+'.csv');
    Mean2df = pd.read_csv(map2dir+meanN+'.csv'); cell2df = pd.read_csv(map2dir+cellN+'.csv');

    XVMov1 = Mean1df['Vx']; YVMov1 = Mean1df['Vy']; xcenters1 = Mean1df['x']; ycenters1 = Mean1df['y']
    XVMov2 = Mean2df['Vx']; YVMov2 = Mean2df['Vy']; xcenters2 = Mean2df['x']; ycenters2 = Mean2df['y']
    beta = 10/np.max(XVMov1.tolist()+YVMov1.tolist()+XVMov2.tolist()+YVMov2.tolist()) # scale factor to see the point well
    fig = plt.figure(figsize=(8*2, 3.3*2));plt.axvline(x=0, c='g'); plt.grid(True)
    ax = fig.add_subplot(111); ax.set_xticks(np.arange(-120,122,40)); ax.set_yticks(np.arange(-33,67,33))
    for i in ycenters1: plt.axhline(y = i, color = 'g', ls = ':',alpha =0.4)
    for i in xcenters1: plt.axvline(x = i, color = 'g', ls = ':',alpha =0.4)
    for i in ycenters2: plt.axhline(y = i, color = 'g', ls = ':',alpha =0.4)
    for i in xcenters2: plt.axvline(x = i, color = 'g', ls = ':',alpha =0.4)

    for i in np.arange(int(len(cell1df.index)/len(xcenters1))):
        celldata = cell1df[int(i*len(xcenters1)):int((i+1)*len(xcenters1))]
        xVMov = celldata['Vx'].values; yVMov = celldata['Vy'].values
        for n,i in enumerate(xcenters1):   
            ax.arrow(i, ycenters1[n], xVMov[n]*beta, yVMov[n]*beta, lw= 0.5,fc='m', ec='m', head_width=0.8, head_length =1, alpha = 0.2 ) 

    for i in np.arange(int(len(cell2df.index)/len(xcenters2))):
        celldata = cell2df[int(i*len(xcenters2)):int((i+1)*len(xcenters2))]
        xVMov = celldata['Vx'].values; yVMov = celldata['Vy'].values
        for n,i in enumerate(xcenters2):   
            ax.arrow(i, ycenters2[n], xVMov[n]*beta, yVMov[n]*beta, lw= 0.5,fc='m', ec='m', head_width=0.8, head_length =1, alpha = 0.2 ) 

    for n,i in enumerate(xcenters1):   
        ax.arrow(i, ycenters1[n], XVMov1[n]*beta, YVMov1[n]*beta, lw= 1,fc='r', ec='r', head_width=1.5, head_length =2) 
    for n,i in enumerate(xcenters2):   
        ax.arrow(i, ycenters2[n], XVMov2[n]*beta, YVMov2[n]*beta, lw= 1,fc='r', ec='r', head_width=1.5, head_length =2) 
    
    plt.axes().set_aspect('equal')
    plt.xlim([-120,120]); plt.ylim([-33,66]); plt.title(dtype+', %dHz, Scale = x %f'% (Hz,beta))
    plt.tight_layout();


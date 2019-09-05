
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os, errno
import pandas as pd
from scipy.stats import ttest_ind_from_stats
import scipy.signal as signal
from scipy import stats
import matplotlib as mpl
from matplotlib.lines import Line2D

def makedir(newdir):
    try:
        os.makedirs(newdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def Mergecells(df,csvroot,stimulus,celltype, spiking): 
    recN = 0 
    if spiking:
        Meandf = pd.DataFrame(); celldf = pd.DataFrame();    
        Meannormdf = pd.DataFrame(); cellnormdf = pd.DataFrame();
        Apmax = []
    MeanVdf = pd.DataFrame(); cellVdf = pd.DataFrame();
    MeanVnormdf = pd.DataFrame(); cellVnormdf = pd.DataFrame();
    Vmax = []
    for i in np.arange(len(df['cells'][:-1])): #[:-1] is for getting rid of 'totalN'
        cell = df['cells'][i]
        if str(df[stimulus][i]) !='nan':
            if isinstance(df[stimulus][i], str):
                recs = [int(x.strip()) for x in str(df[stimulus][i]).split(',')]
            else:           
                recs = [int(x.strip()) for x in str(df[stimulus][i].astype('int')).split(',')]
            for n in recs:                 
                if spiking:
                    Apdfname = csvroot+'Analysis/%s/%s/'%(celltype,cell)+'rec%dApavg_merged.pkl'%n
                    Apdf = pd.read_pickle(Apdfname); apmax = max(abs(Apdf['dmeandf']['APmovmean']))
                    Apmax.append(apmax);  normApavg = Apdf['dmeandf']/apmax
                    Apavgdf = pd.concat([pd.DataFrame(Apdf['ep_names'], index=Apdf['dmeandf'].index, columns = [ 
                        'Angles','Locs']),Apdf['dmeandf']],axis =1)
                    Apavgnormdf = pd.concat([pd.DataFrame(Apdf['ep_names'], index=Apdf['dmeandf'].index, columns = [ 
                        'Angles','Locs']),normApavg],axis =1)
                    celldf = celldf.append(Apavgdf);cellnormdf = cellnormdf.append(Apavgnormdf); 
                    Meandf = Meandf.add(Apavgdf,fill_value=0); Meannormdf = Meannormdf.add(Apavgnormdf,fill_value=0)
                    
                Vdfname = csvroot+'Analysis/%s/%s/'%(celltype,cell)+'rec%dVavg_merged.pkl'%n
                Vdf = pd.read_pickle(Vdfname); vmax = max(abs(Vdf['dmeandf']['Vmovmean']))
                Vmax.append(vmax); normVavg = Vdf['dmeandf']/vmax;
                Vavgdf = pd.concat([pd.DataFrame(Vdf['ep_names'], index=Vdf['dmeandf'].index, columns = [ 
                    'Angles','Locs']),Vdf['dmeandf']],axis =1)
                Vavgnormdf = pd.concat([pd.DataFrame(Vdf['ep_names'], index=Vdf['dmeandf'].index, columns = [ 
                    'Angles','Locs']),normVavg],axis =1)
                cellVdf = cellVdf.append(Vavgdf);cellVnormdf = cellVnormdf.append(Vavgnormdf)                
                MeanVdf = MeanVdf.add(Vavgdf,fill_value=0);MeanVnormdf = MeanVnormdf.add(Vavgnormdf,fill_value=0)                 
                recN +=1        
    if spiking:
        Meandf = Meandf/recN; Meannormdf = Meannormdf/recN
    MeanVdf = MeanVdf/recN; MeanVnormdf = MeanVnormdf/recN 
    
    if spiking:
        Stddf = pd.DataFrame(); Stdnormdf = pd.DataFrame()  
    StdVdf = pd.DataFrame(); StdVnormdf = pd.DataFrame()
    for i in Vdf['dmeandf'].index: 
        if spiking:
            std = celldf.loc[[i]].std(axis =0); 
            std['Angles'] = celldf.loc[[i]].iloc[0][0]; std['Locs'] = celldf.loc[[i]].iloc[0][1]
            Stddf = Stddf.append(std,ignore_index=True)
            
            stdnorm = cellnormdf.loc[[i]].std(axis =0); 
            stdnorm['Angles'] = cellnormdf.loc[[i]].iloc[0][0]; stdnorm['Locs'] = cellnormdf.loc[[i]].iloc[0][1]
            Stdnormdf = Stdnormdf.append(stdnorm,ignore_index=True)

        stdV = cellVdf.loc[[i]].std(axis =0); 
        stdV['Angles'] = cellVdf.loc[[i]].iloc[0][0]; stdV['Locs'] = cellVdf.loc[[i]].iloc[0][1]
        StdVdf = StdVdf.append(stdV,ignore_index=True)        

        stdVnorm = cellVnormdf.loc[[i]].std(axis =0); 
        stdVnorm['Angles'] = cellVnormdf.loc[[i]].iloc[0][0]; stdVnorm['Locs'] = cellVnormdf.loc[[i]].iloc[0][1]
        StdVnormdf = StdVnormdf.append(stdVnorm,ignore_index=True)
    
    if spiking:
        Stddf= Stddf[Meandf.columns.tolist()]; Stdnormdf= Stdnormdf[Meannormdf.columns.tolist()]
    StdVdf = StdVdf[MeanVdf.columns.tolist()]; StdVnormdf= StdVnormdf[MeanVnormdf.columns.tolist()]
    dirs = csvroot+'Analysis/'+ stimulus +'/'+ celltype ; makedir(dirs)
    if spiking:
        Meandf.to_csv(dirs+'/Meandf.csv'); celldf.to_csv(dirs+'/celldf.csv'); Stddf.to_csv(dirs+'/Stddf.csv')
        Meannormdf.to_csv(dirs+'/Meannormdf.csv'); cellnormdf.to_csv(dirs+'/cellnormdf.csv'); Stdnormdf.to_csv(dirs+'/Stdnormdf.csv')
    MeanVdf.to_csv(dirs+'/MeanVdf.csv'); cellVdf.to_csv(dirs+'/cellVdf.csv'); StdVdf.to_csv(dirs+'/StdVdf.csv')    
    MeanVnormdf.to_csv(dirs+'/MeanVnormdf.csv'); cellVnormdf.to_csv(dirs+'/cellVnormdf.csv'); StdVnormdf.to_csv(dirs+'/StdVnormdf.csv')
    if spiking: 
        return(Meandf,celldf,MeanVdf,cellVdf,Meannormdf,cellnormdf,MeanVnormdf,cellVnormdf,Stddf,StdVdf,Stdnormdf,StdVnormdf)
    else:
        return(MeanVdf,cellVdf,MeanVnormdf,cellVnormdf,StdVdf,StdVnormdf)

def pltVval(MeanVdf,StdVdf):
    Mov = MeanVdf['Vmovmean'].values; Movyerr = StdVdf['Vmovmean'].values 
    Mov0 = MeanVdf['Vmov0mean'].values; Mov0yerr = StdVdf['Vmov0mean'].values 
    Mov1 = MeanVdf['Vmov1mean'].values; Mov1yerr = StdVdf['Vmov1mean'].values
    Mov2 = MeanVdf['Vmov2mean'].values; Mov2yerr = StdVdf['Vmov2mean'].values
    return(Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr)

def pltAPval(Meandf,Stddf):
    Mov = Meandf['APmovmean'].values; Movyerr = Stddf['APmovmean'].values 
    Mov0 = Meandf['APmov0mean'].values; Mov0yerr = Stddf['APmov0mean'].values 
    Mov1 = Meandf['APmov1mean'].values; Mov1yerr = Stddf['APmov1mean'].values
    Mov2 = Meandf['APmov2mean'].values; Mov2yerr = Stddf['APmov2mean'].values
    return(Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr)

def pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):
    hlf = int(len(stim_dir)/2); fig = plt.figure(figsize=(8,13))  
    plt.subplot(211); plt.xscale('symlog',subsx = [2, 3, 4, 5, 6, 7, 8, 9]) 
    plt.plot(stim_dir[:hlf],Mov[:hlf],'rebeccapurple', alpha = 0.8); plt.plot(stim_dir[:hlf],Mov[hlf:],'coral', alpha = 0.8) 
    plt.legend(['mov_F','mov_M'])
    plt.fill_between(stim_dir[:hlf], Mov[:hlf]-Movyerr[:hlf],Mov[:hlf]+Movyerr[:hlf], color = 'rebeccapurple', alpha = 0.1)
    plt.fill_between(stim_dir[:hlf], Mov[hlf:]-Movyerr[hlf:],Mov[hlf:]+Movyerr[hlf:], color = 'coral', alpha = 0.1)
    #plt.errorbar(stim_dir[:hlf],Mov[:hlf],yerr = Movyerr[:hlf],  color = 'k', alpha = 0.8)
    #plt.errorbar(stim_dir[:hlf],Mov[hlf:],yerr = Movyerr[hlf:],  ls = '--', color = 'r', alpha = 0.8)
    #plt.errorbar(stim_dir[:hlf],Mov0[:hlf],yerr = Mov0yerr[:hlf], color = 'c', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf],Mov1[:hlf],yerr = Mov1yerr[:hlf], color = 'm', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf],Mov2[:hlf],yerr = Mov2yerr[:hlf], color = 'g', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf],Mov0[hlf:],yerr = Mov0yerr[hlf:], ls = '--', color = 'c', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf],Mov1[hlf:],yerr = Mov1yerr[hlf:], ls = '--',  color = 'm', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf],Mov2[hlf:],yerr = Mov2yerr[hlf:], ls = '--',  color = 'g', alpha = 0.4)
    
    #plt.legend(('mov_F','mov_M', 'mov0.5_F','mov0.5_M', 'mov1_F','mov1_M', 'mov2_F', 'mov2_M'),loc = 2 )    
    plt.subplot(212); plt.xscale('symlog')
    plt.plot(stim_dir[:hlf], Mov[:hlf]- Mov[hlf:], color = 'k')
    plt.fill_between(stim_dir[:hlf], Mov[:hlf]- Mov[hlf:] - (Movyerr[:hlf]+Movyerr[hlf:]), Mov[:hlf]- Mov[hlf:] + (Movyerr[:hlf]+Movyerr[hlf:]), color = 'k', alpha = 0.1)
   # plt.errorbar(stim_dir[:hlf], Mov[:hlf]- Mov[hlf:],yerr = Movyerr[:hlf]+Movyerr[hlf:],  color = 'k', alpha = 0.8)
   # plt.errorbar(stim_dir[:hlf], Mov0[:hlf]- Mov0[hlf:],yerr = Mov0yerr[:hlf]+Mov0yerr[hlf:],  color = 'c', alpha = 0.4)
   # plt.errorbar(stim_dir[:hlf], Mov1[:hlf]- Mov1[hlf:],yerr = Mov1yerr[:hlf]+Mov1yerr[hlf:],  color = 'm', alpha = 0.4)
    #plt.errorbar(stim_dir[:hlf], Mov2[:hlf]- Mov2[hlf:],yerr = Mov2yerr[:hlf]+Mov2yerr[hlf:],  color = 'g', alpha = 0.4)
    #plt.legend(('F-M_Mov', 'F-M_Mov0.5','F-M_Mov1', 'F-M_Mov2'),loc = 2)  
    return(fig)

# TTEST for PvNz
def ttestPvNz(celldf,tp,csvroot,stimulus, celltype, MeanVdf):
    if stimulus == "Sine_PvN_lambda30_zHz_bgstill":
        ctnum = 22
    elif stimulus == "Sine_PvN_lambda30_uHz_bgstill":
        ctnum = 10        
    Pvals = []; Svals = []; P0vals = []; S0vals = []; P1vals = []; S1vals = []; P2vals = []; S2vals = [];
    PWvals = []
    for i in range(ctnum):
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['APmovmean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['APmovmean']
        Svals.append(stats.ttest_rel(s1,s2)[0]); Pvals.append(stats.ttest_rel(s1,s2)[1])
        PWvals.append(stats.wilcoxon(s1,s2)[1])
        
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['APmov0mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['APmov0mean']
        S0vals.append(stats.ttest_rel(s1,s2)[0]); P0vals.append(stats.ttest_rel(s1,s2)[1])
        
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['APmov1mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['APmov1mean']
        S1vals.append(stats.ttest_rel(s1,s2)[0]); P1vals.append(stats.ttest_rel(s1,s2)[1])
        
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['APmov2mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['APmov2mean']
        S2vals.append(stats.ttest_rel(s1,s2)[0]); P2vals.append(stats.ttest_rel(s1,s2)[1])
    ttestdf = pd.DataFrame({'Angles':MeanVdf['Angles'][:ctnum].values,'Pvals':np.asarray(Pvals), 'Svals':np.asarray(Svals),
                           'P0vals':np.asarray(P0vals), 'S0vals':np.asarray(S0vals),
                           'P1vals':np.asarray(P1vals), 'S1vals':np.asarray(S1vals),
                           'P2vals':np.asarray(P2vals), 'S2vals':np.asarray(S2vals),
                           'PWvals': np.asarray(PWvals)})
    dirs = csvroot+'Analysis/'+ stimulus +'/'+ celltype ;  ttestdf.to_csv(dirs+'/ttestdf%s.csv' %tp)
    return(ttestdf)

def ttestPvNzV(celldf,tp,csvroot,stimulus, celltype,MeanVdf):
    if stimulus == "Sine_PvN_lambda30_zHz_bgstill":
        ctnum = 22
    elif stimulus == "Sine_PvN_lambda30_uHz_bgstill":
        ctnum = 10
    Pvals = []; Svals = []; P0vals = []; S0vals = []; P1vals = []; S1vals = []; P2vals = []; S2vals = []; PWvals = []
    for i in range(ctnum):
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['Vmovmean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['Vmovmean']
        Svals.append(stats.ttest_rel(s1,s2)[0]); Pvals.append(stats.ttest_rel(s1,s2)[1])
        PWvals.append(stats.wilcoxon(s1,s2)[1])
        
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['Vmov0mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['Vmov0mean']
        S0vals.append(stats.ttest_rel(s1,s2)[0]); P0vals.append(stats.ttest_rel(s1,s2)[1])
        
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['Vmov1mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['Vmov1mean']
        S1vals.append(stats.ttest_rel(s1,s2)[0]); P1vals.append(stats.ttest_rel(s1,s2)[1])
                
        s1 = celldf.sort_values(by=['Angles','Locs']).loc[[i]]['Vmov2mean']
        s2 = celldf.sort_values(by=['Angles','Locs']).loc[[i+ctnum]]['Vmov2mean']
        S2vals.append(stats.ttest_rel(s1,s2)[0]); P2vals.append(stats.ttest_rel(s1,s2)[1]) 
    
    ttestdf = pd.DataFrame({'Angles':MeanVdf['Angles'][:ctnum].values,'Pvals':np.asarray(Pvals), 'Svals':np.asarray(Svals),
                           'P0vals':np.asarray(P0vals), 'S0vals':np.asarray(S0vals),
                           'P1vals':np.asarray(P1vals), 'S1vals':np.asarray(S1vals),
                           'P2vals':np.asarray(P2vals), 'S2vals':np.asarray(S2vals),
                           'PWvals':np.asarray(PWvals)})
    dirs = csvroot+'Analysis/'+ stimulus +'/'+ celltype ;  ttestdf.to_csv(dirs+'/ttestdf%s.csv' %tp)
    return(ttestdf)

def pltPDNDz(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr, colr):  
    plt.xscale('symlog',subsx = [2, 3, 4, 5, 6, 7, 8, 9]) #;plt.grid()
    plt.plot(stim_dir,Mov, color = colr, alpha = 0.8)
    plt.fill_between(stim_dir, Mov-Movyerr,Mov+Movyerr, color = colr, alpha = 0.1)    
    #plt.errorbar(stim_dir,Mov,yerr = Movyerr,  color = colr, alpha = 0.8)
    #plt.errorbar(stim_dir,Mov0,yerr = Mov0yerr, color = colr, ls = ':', alpha = 0.2)
    #plt.errorbar(stim_dir,Mov1,yerr = Mov1yerr, color = colr, alpha = 0.4)
    #plt.errorbar(stim_dir,Mov2,yerr = Mov2yerr, color = colr, alpha = 0.3) 
    return()

def pltPvNz(csvroot,stimulus,MeanVdf,StdVdf,Meandf,Stddf,norm):
#     if Meandf !=0: 
    spiking = 0
#     else: 
#         spiking = 0
    Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltVval(MeanVdf,StdVdf)
    savedir = csvroot+'Pics/%s' %(stimulus)
    stim_dir = MeanVdf['Angles']
    fig1 = pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
    if norm == True: 
        plt.suptitle(stimulus + '_Vm_norm'); fig1.savefig(savedir + '_Vm_norm.jpeg')
    else:
        plt.suptitle(stimulus + '_Vm'); fig1.savefig(savedir + '_Vm.jpeg')
    if spiking:
        Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPval(Meandf,Stddf)
        fig2 = pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)  
        if norm == True: 
            plt.suptitle(stimulus + '_Ap_norm'); plt.ylabel('normalized firing rate(Hz)'); fig2.savefig(savedir+'norm.jpeg')
        else:
            plt.suptitle(stimulus + '_Ap'); plt.ylabel('firing rate(Hz)'); fig2.savefig(savedir+'.jpeg')

def Cellpidx (Meandf,celldf,pidxbase,tp):
    sti_len = len(Meandf['Angles'])
    n = int(len(celldf['Angles'])/sti_len)
    cellpidx = np.zeros([15,n])
    for i in range(n):
        if tp[:2] == 'Vm':
            indi = celldf['Vmovmean'].values[i*30:int(30*(i+1))]
        else:   
            indi = celldf['APmovmean'].values[i*30:int(30*(i+1))]
        indipidx = (indi[15:] - indi[:15][::-1])/pidxbase
        cellpidx[:,i] = indipidx
    return(cellpidx)

def ttestPDNDz(cellpidxdf,cellpidxLdf):
    Pvals = []; Svals = [];
    for i in range(15):
        tresults = stats.ttest_ind(cellpidxdf.values[i,1:],cellpidxLdf.values[i,1:])
        Svals.append(tresults[0]); Pvals.append(tresults[1])    
    ttestdf = pd.DataFrame({'Angles':cellpidxdf.values[:,0],'Pvals':np.asarray(Pvals), 'Svals':np.asarray(Svals)})
    return(ttestdf)

def pltmulti(meanfile,stdfile,cellfile,tp, csvroot,celltype,stimulus,lf,bf, rf):
    if stimulus == 'Sine_PDND_lambda30_zHz':
        savedir = csvroot+'Pics/%s' %(stimulus)
        plt.figure(figsize=(5,6))
        fdlist = [celltype[:2], celltype[:2]+'_L', celltype[:2]+'_R']
        for i in fdlist: 
            csvpath = csvroot + 'Analysis/' + stimulus+'/'+ i
            MeanVdf = pd.read_csv(csvpath + '/%s.csv' %meanfile)
            StdVdf = pd.read_csv(csvpath + '/%s.csv' % stdfile)
            cellVdf = pd.read_csv(csvpath + '/%s.csv' % cellfile)
            if tp[:2] == 'Vm':
                Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltVval(MeanVdf,StdVdf)
            else:
                Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPval(MeanVdf,StdVdf)
            stim_dir = MeanVdf['Angles']    
            if i == fdlist[0]:
                #pltPDNDz(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr, 'k')
                pltPDNDz(stim_dir, Mov*bf, Movyerr*bf, Mov0*bf, Mov0yerr*bf, Mov1*bf, Mov1yerr*bf, Mov2*bf, Mov2yerr*bf, 'k')
                pidxbase = np.max(Mov[15:]*bf + abs(Mov[:15][::-1])*bf)
                pidx = (Mov[15:] - Mov[:15][::-1])*bf/pidxbase 
                cellpidx = Cellpidx (MeanVdf,cellVdf,pidxbase,tp)*bf
            if i == fdlist[1]:
                movL = Mov*lf
                pltPDNDz(stim_dir, Mov*lf, Movyerr*lf, Mov0*lf, Mov0yerr*lf, Mov1*lf, Mov1yerr*lf, Mov2*lf, Mov2yerr*lf, 'b')
                pidxL = (Mov[15:] - Mov[:15][::-1])*lf/pidxbase 
                cellpidxL = Cellpidx (MeanVdf,cellVdf,pidxbase,tp)*lf
            if i == fdlist[2]:
                movR =Mov*rf 
                #pltPDNDz(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr, 'g')
                pltPDNDz(stim_dir, Mov*rf, Movyerr*rf, Mov0*rf, Mov0yerr*rf, Mov1*rf, Mov1yerr*rf, Mov2*rf, Mov2yerr*rf, 'g')
                pidxR = (Mov[15:] - Mov[:15][::-1])*rf/pidxbase
                cellpidxR = Cellpidx (MeanVdf,cellVdf,pidxbase,tp)*rf
                #plt.legend(('Both', 'Both0.5', 'Ipsi', 'Ipsi0.5','Contra', 'Contra0.5'),loc = 2 )                 
        plt.plot(stim_dir,movL+movR, 'r--', alpha = 0.5)
        plt.legend(('Both', 'Ipsi', 'Contra','Ip+Ct'),loc = 0) 
        plt.title(stimulus + '_' +tp)
        plt.savefig(savedir+'_%s.jpeg' %tp) 
        if celltype[:2] == 'HS':
            pidx= -pidx; pidxL= -pidxL; pidxR= -pidxR; 
            cellpidx = -cellpidx; cellpidxL = -cellpidxL; cellpidxR = -cellpidxR;
        pidxmax = np.max([abs(pidx),abs(pidxL),abs(pidxR)])
        pidxdf = pd.DataFrame({'angles':stim_dir[15:],'pidx': pidx, 'pidxL': pidxL, 'pidxR': pidxR})
        npidxdf = pd.DataFrame({'angles':stim_dir[15:],'pidx': pidx/pidxmax, 'pidxL': pidxL/pidxmax, 'pidxR': pidxR/pidxmax })
        cellpidxdf = pd.concat([pd.DataFrame({'angles':stim_dir[15:].values}), pd.DataFrame(cellpidx/pidxmax)],axis=1,ignore_index=True)
        cellpidxLdf = pd.concat([pd.DataFrame({'angles':stim_dir[15:].values}), pd.DataFrame(cellpidxL/pidxmax)],axis=1,ignore_index=True)
        cellpidxRdf = pd.concat([pd.DataFrame({'angles':stim_dir[15:].values}), pd.DataFrame(cellpidxR/pidxmax)],axis=1,ignore_index=True)
        ttestdf = ttestPDNDz(cellpidxdf,cellpidxLdf); 
        ttestdf.to_csv(csvroot+ 'analysis/%s/ttestpidxBLdf_%s.csv' %(stimulus, tp))
        npidxdf.to_csv(csvroot+ 'analysis/%s/npidxdf_%s.csv' %(stimulus, tp))
        pidxdf.to_csv(csvroot+ 'analysis/%s/pidxdf_%s.csv' %(stimulus, tp))
        cellpidxdf.to_csv(csvroot+ 'analysis/%s/cellpidxdf_%s.csv' %(stimulus, tp))
        cellpidxLdf.to_csv(csvroot+ 'analysis/%s/cellpidxLdf_%s.csv' %(stimulus, tp))
        cellpidxRdf.to_csv(csvroot+ 'analysis/%s/cellpidxRdf_%s.csv' %(stimulus, tp))
        plt.figure(); plt.xscale('symlog',subsx = [2, 3, 4, 5, 6, 7, 8, 9])  #;plt.grid()
        plt.plot(stim_dir[15:],pidx/pidxmax, 'k'); plt.plot(stim_dir[15:],pidxL/pidxmax, 'b')
        plt.plot(stim_dir[15:],pidxR/pidxmax, 'g'); plt.plot(stim_dir[15:],pidxR/pidxmax +pidxL/pidxmax, 'r--', alpha = 0.5)
        plt.legend(('Both', 'Ipsi', 'Contra','Ip+Ct')); plt.title('Preference Index for PD_' + tp)
        [plt.plot(stim_dir[15:],cellpidx[:,i]/pidxmax, 'k',alpha =0.1) for i in range(int(np.shape(cellpidx)[1]))]
        [plt.plot(stim_dir[15:],cellpidxL[:,i]/pidxmax, 'b',alpha =0.1) for i in range(int(np.shape(cellpidxL)[1]))]
        [plt.plot(stim_dir[15:],cellpidxR[:,i]/pidxmax, 'g',alpha =0.1) for i in range(int(np.shape(cellpidxR)[1]))]
        plt.savefig(savedir+'pidx_%s.jpeg' %tp)

def plt12dir(spiking,csvroot,stimulus,celltype,norm): 
    if norm:
        dfName = 'normdf'; PicName = '_norm_'; PolarPicName = '_polar_norm_'
    else:
        dfName = 'df'; PicName = ''; PolarPicName = '_polar_'
    if spiking:
        Meandf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/Mean'+dfName+'.csv')
        Stddf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/Std'+dfName+'.csv')
        Celldf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/cell'+dfName+'.csv')
        stim_dir  = np.around((Meandf['Angles'].values).astype('float'),decimals=1)
                             
        recN = int(len(Celldf)/len(stim_dir))    
        theta = np.array(stim_dir)/360* 2*np.pi 
        theta = np.insert(theta,len(theta),theta[0])
        width = max(stim_dir)/len(stim_dir)
        Mov = Meandf['APmovmean'].values; Movyerr = Stddf['APmovmean'].values  
        movs = Celldf.sort_values(by ='Angles')['APmovmean']
        
        fig, ax = plt.subplots()
        ax.bar(stim_dir, Mov, width,yerr=Movyerr, fc = 'none', ec = 'k', linewidth=2,error_kw=dict(ecolor='k',lw=2))
        for n,i in enumerate(stim_dir):
             ax.scatter(np.ones(recN)*i,movs[n*recN:(n+1)*recN],color='none',edgecolors = 'k',alpha=0.5)          
        plt.savefig(csvroot+'/Pics/%s_'%stimulus+'AP'+PicName+celltype+'.jpeg')

        ax = plt.subplot(111, projection='polar')
        Mov = np.insert(Mov,len(Mov),Mov[0])            
        Movyerr = np.insert(Movyerr,len(Movyerr),Movyerr[0])           
        ax.errorbar(theta, Mov, yerr=Movyerr, color = 'b')
        for n,i in enumerate(theta[:-1]):
            ax.scatter(np.ones(recN)*i,movs[n*recN:(n+1)*recN],color='none',edgecolors = 'b',alpha=0.5)  
        plt.ylabel('firing rate(Hz)')
        plt.savefig(csvroot+'/Pics/%s_'%stimulus+'AP'+PolarPicName+celltype+'.jpeg')                             
                              
    MeanVdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/MeanV'+dfName+'.csv')
    StdVdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/StdV'+dfName+'.csv')
    CellVdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/cellV'+dfName+'.csv')
    stim_dir  = np.around((MeanVdf['Angles'].values).astype('float'),decimals=1)
    
    recN = int(len(CellVdf)/len(stim_dir))    
    theta = np.array(stim_dir)/360* 2*np.pi 
    theta = np.insert(theta,len(theta),theta[0])
    width = max(stim_dir)/len(stim_dir)
    Mov = MeanVdf['Vmovmean'].values; Movyerr = StdVdf['Vmovmean'].values  
    movs = CellVdf.sort_values(by ='Angles')['Vmovmean']

    fig, ax = plt.subplots()
    ax.bar(stim_dir, Mov, width,yerr=Movyerr, fc = 'none', ec = 'k', linewidth=2,error_kw=dict(ecolor='k',lw=2))
    for n,i in enumerate(stim_dir):
         ax.scatter(np.ones(recN)*i,movs[n*recN:(n+1)*recN],color='none',edgecolors = 'k',alpha=0.5)          
    plt.ylabel('Vm (mV)')
    plt.savefig(csvroot+'/Pics/%s_'%stimulus+'V'+PicName+celltype+'.jpeg')

    ax = plt.subplot(111, projection='polar')
    Mov = np.insert(Mov,len(Mov),Mov[0])            
    Movyerr = np.insert(Movyerr,len(Movyerr),Movyerr[0])           
    ax.errorbar(theta, Mov, yerr=Movyerr, color = 'b')
    for n,i in enumerate(theta[:-1]):
        ax.scatter(np.ones(recN)*i,movs[n*recN:(n+1)*recN],color='none',edgecolors = 'b',alpha=0.5)  
    plt.savefig(csvroot+'/Pics/%s_'%stimulus+'V'+PolarPicName+celltype+'.jpeg')

def PltLRfull(spiking,csvroot,stimulus,celltype,norm):
## compare full screen in painted eye conditions
    if norm:
        dfName = 'normdf'; PicName = 'norm_LRfull_'
    else:
        dfName = 'df'; PicName = 'LRfull_'; ratio = 1 
    if spiking:
        MeanLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/Mean'+dfName+'.csv')
        StdLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/Std'+dfName+'.csv')
        MeanRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/Mean'+dfName+'.csv')
        StdRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/Std'+dfName+'.csv')
        if norm:
            # the max before normalization
            bMeanLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/Meandf.csv')
            bMeanRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/Meandf.csv')
            Lmax = np.max(bMeanLdf['APmovmean']); Rmax = np.max(bMeanRdf['APmovmean'])
            # the max from normed value
            LmaxN = np.max(MeanLdf['APmovmean']); RmaxN = np.max(MeanRdf['APmovmean'])
            Lunit = Lmax/LmaxN; Runit = Rmax/RmaxN
            ratio = Runit/Lunit
        
        MovR = MeanRdf['APmovmean'].values*ratio; MovRyerr = StdRdf['APmovmean'].values*ratio
        MovL = MeanLdf['APmovmean'].values; MovLyerr = StdLdf['APmovmean'].values
        fig, ax = plt.subplots()
        ax.bar(stim_dir-width, MovR, width,yerr=MovRyerr, color ="g",alpha=0.7,linewidth=0, error_kw=dict(ecolor='g',lw=0.5))
        ax.bar(stim_dir, MovL, width,yerr=MovLyerr, color ="b",alpha=0.7,linewidth=0, error_kw=dict(ecolor='b',lw=0.5))
        plt.legend(('Contra','Ipsi') ,loc  = 0)
        plt.ylabel('firing rate(Hz)')
        plt.savefig(csvroot+'/Pics/%s_AP_'%stimulus+PicName+celltype[:2]+'.jpeg')
        # polar
        plt.figure()
        ax = plt.subplot(111, projection='polar')
        MovR = np.insert(MovR,len(MovR),MovR[0]); MovRyerr = np.insert(MovRyerr,len(MovRyerr),MovRyerr[0]) 
        MovL = np.insert(MovL,len(MovL),MovL[0]); MovLyerr = np.insert(MovLyerr,len(MovLyerr),MovLyerr[0]) 
        ax.errorbar(theta, MovR, yerr=MovRyerr, color = 'g')
        ax.errorbar(theta, MovL, yerr=MovLyerr, color = 'b')
        plt.savefig(csvroot+'/Pics/%s_AP_Polar_'%stimulus+PicName+celltype[:2]+'.jpeg')

    MeanVLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/MeanV'+dfName+'.csv')
    StdVLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/StdV'+dfName+'.csv')
    MeanVRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/MeanV'+dfName+'.csv')
    StdVRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/StdV'+dfName+'.csv')
    
    if norm:
        # the max before normalization
        bMeanVLdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_L/MeanVdf.csv')
        bMeanVRdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype[:2]+'_R/MeanVdf.csv')        
        Lmax = np.max(bMeanVLdf['Vmovmean']); Rmax = np.max(bMeanVRdf['Vmovmean'])
        # the max from normed value
        LmaxN = np.max(MeanVLdf['Vmovmean']); RmaxN = np.max(MeanVRdf['Vmovmean'])
        Lunit = Lmax/LmaxN; Runit = Rmax/RmaxN
        ratio = Runit/Lunit
        
    stim_dir  = np.around((MeanVLdf['Angles'].values).astype('float'),decimals=1)
    theta = np.array(stim_dir)/360* 2*np.pi 
    theta = np.insert(theta,len(theta),theta[0])
    width = max(stim_dir)/len(stim_dir)/2

    # Vm
    MovR = MeanVRdf['Vmovmean'].values*ratio; MovRyerr = StdVRdf['Vmovmean'].values*ratio 
    MovL = MeanVLdf['Vmovmean'].values; MovLyerr = StdVLdf['Vmovmean'].values
    fig, ax = plt.subplots()
    ax.bar(stim_dir-width, MovR, width,yerr=MovRyerr, color ="g",alpha=0.7,linewidth=0, error_kw=dict(ecolor='g',lw=0.5))
    ax.bar(stim_dir, MovL, width,yerr=MovLyerr, color ="b",alpha=0.7,linewidth=0, error_kw=dict(ecolor='b',lw=0.5))
    plt.legend(('Contra','Ipsi') ,loc  = 0)
    plt.ylabel('Vm (mV)')
    plt.savefig(csvroot+'/Pics/%s_V_'%stimulus+PicName+celltype[:2]+'.jpeg')

    plt.figure()
    ax = plt.subplot(111, projection='polar')
    MovR = np.insert(MovR,len(MovR),MovR[0]); MovRyerr = np.insert(MovRyerr,len(MovRyerr),MovRyerr[0]) 
    MovL = np.insert(MovL,len(MovL),MovL[0]); MovLyerr = np.insert(MovLyerr,len(MovLyerr),MovLyerr[0]) 
    ax.errorbar(theta, MovR, yerr=MovRyerr, color = 'g')
    ax.errorbar(theta, MovL, yerr=MovLyerr, color = 'b')
    plt.savefig(csvroot+'/Pics/%s_V_polar_'%stimulus+PicName+celltype[:2]+'.jpeg')
    
def LRfull(celltype, csvroot, pageName):
## compare full screen in painted eye conditions
    #first, get 12dir data for HS, HS_L, HS_R (or H2)
    for i in [celltype,celltype+'_L',celltype+'_R']:
        csvpath = csvroot+'%s/%s.xlsx' %(i,i)
        xls = pd.ExcelFile(csvpath)
        df = pd.read_excel(xls, pageName)
        df = df.copy()
        if i[:2] == 'HS':
            spiking = 0
        else:
            spiking = 1
            
        stimulus = 'Sine_12dir_lambda30_1Hz'
        if spiking: 
            Meandf,celldf,MeanVdf,cellVdf,Meannormdf,cellnormdf,MeanVnormdf,cellVnormdf,Stddf,StdVdf,Stdnormdf,StdVnormdf = Mergecells(df,csvroot,stimulus,i,spiking)
        else:
            MeanVdf,cellVdf,MeanVnormdf,cellVnormdf,StdVdf,StdVnormdf = Mergecells(df,csvroot,stimulus,i,spiking)
            # simple 12 dir of normed values
        plt12dir(spiking,csvroot,stimulus,i,0); plt12dir(spiking, csvroot,stimulus,i,1)
    
    PltLRfull(spiking,csvroot,stimulus,celltype,0); PltLRfull(spiking,csvroot,stimulus,celltype,1)
    
def pltLR(spiking,csvroot,stimulus,celltype,norm):
    if norm:
        dfName = 'normdf'; PicName = '_norm_';
    else:
        dfName = 'df'; PicName = ''; 
    if spiking: 
        Meandf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/Mean'+dfName+'.csv')
        Stddf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/Std'+dfName+'.csv')
        stim_dir  = np.around((Meandf['Angles'].values).astype('float'),decimals=1)
                             
        recN = int(len(Celldf)/len(stim_dir))    
        theta = np.array(stim_dir)/360* 2*np.pi 
        theta = np.insert(theta,len(theta),theta[0])
        width = max(stim_dir)/len(stim_dir)
        Mov = Meandf['APmovmean'].values; Movyerr = Stddf['APmovmean'].values
        
        fig, ax = plt.subplots()
        ax.bar(stim_dir[:12]-width, Mov[:12], width,yerr=Movyerr[:12], color ="g",alpha=0.7,linewidth=0, error_kw=dict(ecolor='g',lw=0.5))
        ax.bar(stim_dir[12:], Mov[12:], width,yerr=Movyerr[12:], color ="b",alpha=0.7,linewidth=0, error_kw=dict(ecolor='b',lw=0.5))
        plt.legend(('Contra','Ipsi') ,loc  = 0)
        plt.ylabel('firing rate(Hz)')
        plt.savefig(csvroot+'/Pics/%s_AP_'%stimulus+PicName+celltype+'.jpeg')    
    
    #Vm
    MeanVdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/MeanV'+dfName+'.csv')
    StdVdf = pd.read_csv(csvroot + 'analysis/'+stimulus+'/'+celltype+'/StdV'+dfName+'.csv')
    stim_dir  = np.around((MeanVdf['Angles'].values).astype('float'),decimals=1)
    Mov = MeanVdf['Vmovmean'].values; Movyerr = StdVdf['Vmovmean'].values  
    width = max(stim_dir)/len(stim_dir)

    fig, ax = plt.subplots()
    ax.bar(stim_dir[:12]-width, Mov[:12], width,yerr=Movyerr[:12], color ="g",alpha=0.7,linewidth=0, error_kw=dict(ecolor='g',lw=0.5))
    ax.bar(stim_dir[12:], Mov[12:], width,yerr=Movyerr[12:], color ="b",alpha=0.7,linewidth=0, error_kw=dict(ecolor='b',lw=0.5))
    plt.legend(('Contra','Ipsi') ,loc  = 0)
    plt.ylabel('Vm(mV)')
    plt.savefig(csvroot+'/Pics/%s_V_'%stimulus+PicName+celltype+'.jpeg')
    
def LR(celltype, csvroot, pageName):
    #first, get 12dir data for HS, HS_L, HS_R (or H2)
    for i in [celltype,celltype+'_L',celltype+'_R']:
        csvpath = csvroot+'%s/%s.xlsx' %(i,i)
        xls = pd.ExcelFile(csvpath)
        df = pd.read_excel(xls, pageName)
        df = df.copy()
        if i[:2] == 'HS':
            spiking = 0
        else:
            spiking = 1
            
        stimulus = 'Sq_LR_12dir_lambda30_1Hz_bgstill'
        if spiking: 
            Meandf,celldf,MeanVdf,cellVdf,Meannormdf,cellnormdf,MeanVnormdf,cellVnormdf,Stddf,StdVdf,Stdnormdf,StdVnormdf = Mergecells(df,csvroot,stimulus,i,spiking)
        else:
            MeanVdf,cellVdf,MeanVnormdf,cellVnormdf,StdVdf,StdVnormdf = Mergecells(df,csvroot,stimulus,i,spiking)
            # simple 12 dir of normed values
        pltLR(spiking,csvroot,stimulus,i,0);  pltLR(spiking,csvroot,stimulus,i,1)
    
def pltH2HSkir(meanfile,stdfile,cellfile,tp, csvroot,celltype,stimulus):
    if stimulus == 'Sine_PDND_lambda30_zHz':
        savedir = csvroot+'Pics/%s/%s/' %(stimulus,celltype); makedir(savedir)
        plt.figure(figsize=(10,6));plt.xscale('symlog',subsx = [2, 3, 4, 5, 6, 7, 8, 9])
        csvpath = csvroot + 'Analysis/' + stimulus+'/'+ celltype
        fdlist = ['control','HS_kir']; clist = ['g','lime']
        for n,i in enumerate(fdlist):
            MeanVdf = pd.read_csv(csvpath + '/%s/%s.csv' %(i,meanfile))
            StdVdf = pd.read_csv(csvpath +  '/%s/%s.csv' %(i,stdfile))
            cellVdf = pd.read_csv(csvpath +  '/%s/%s.csv' %(i,cellfile))
            if tp[:2] == 'Vm':
                Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltVval(MeanVdf,StdVdf)
            else:
                Mov, Movyerr, Mov0, Mov0yerr,Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPval(MeanVdf,StdVdf)
            stim_dir = MeanVdf['Angles']    
            #pltPDNDz(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr, clist[n])
            sti_len = len(MeanVdf['Angles'])
            m = int(len(cellVdf['Angles'])/sti_len)
            for j in range(m):
                if tp[:2] == 'Vm':
                    indi = cellVdf['Vmovmean'].values[j*30:int(30*(j+1))]
                else:   
                    indi = cellVdf['APmovmean'].values[j*30:int(30*(j+1))]
                plt.plot(stim_dir,indi,clist[n],alpha=0.4)

            pidxbase = np.max(Mov[15:] + abs(Mov[:15][::-1]))
            pidx = (Mov[15:] - Mov[:15][::-1])/pidxbase 
            cellpidx = Cellpidx (MeanVdf,cellVdf,pidxbase,tp)

        legend_elements = [Line2D([0], [0], color=clist[0],label='control'),
                          Line2D([0], [0], color=clist[1],label='HS_kir')]

        plt.legend(handles=legend_elements,loc = 0) 
        plt.title(stimulus + '_' +tp); plt.xlabel('Hz'); plt.ylabel('response')
        plt.savefig(savedir+'HS_kir_%s.jpeg' %tp) 

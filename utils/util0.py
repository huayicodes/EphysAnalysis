
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import os, errno
from igor.binarywave import load
from statistics import mode
import scipy.signal as signal
import pandas as pd
get_ipython().magic('matplotlib nbagg')

def readibw(fdname,fname):
    # the code needs to be a founder outside individual expmt folder
    # Get the current directory
    _thisDir = os.getcwd()
    # name of the folder & the file. e.g. fdname = "041917fly2cell1", # fname = "ch0_2.ibw"
    # directory of the file
    f_dir = os.path.join(_thisDir, fdname, 'waves', fname).replace("\\","/")
    #print(f_dir)
    # load the wave
    wave = load(f_dir)
    v = wave['wave']['wData'] # voltage
    af = wave['wave']['wave_header']['sfA'][0] # 1/(acquisition frequency in s).
    t = np.linspace(0,len(v)*af,len(v)) # generate time
    return(v,t)

def readtraces(fdname, recN):     
    fname = "ch0_%d.ibw" % recN 
    phoname = "ch1_%d.ibw" % recN # photodiode channel 
    v,t = readibw(fdname,fname)
    pho,_ = readibw(fdname,phoname) # all the "t" should be the same, thus only read once
    return(t,v,pho)

def detectAP(v,t,th1,th2):
    vdiff = np.diff(v)
    vdiff1= np.argwhere(vdiff>th1)
    vdiff1diff = np.diff(vdiff1,axis =0)
    vdiff1diff2=np.argwhere(vdiff1diff>th2)[:,0]
    spike_idx = np.arange(0,len(t)+1)[1:][vdiff1][vdiff1diff2]
    return(spike_idx)

def plotrange(t,ta,tb,v,pho,spike_idx):     
    plt.figure()
    plt.subplot(211)
    plt.plot(t[ta:tb],v[ta:tb])
    if np.sum(spike_idx) != 0: 
        plt.scatter(t[spike_idx][ta:tb],v[spike_idx][ta:tb], c='r')
    plt.title('Response')
    plt.subplot(212)
    plt.plot(t[ta:tb],pho[ta:tb])
    plt.title('photodiode')
    plt.xlabel('time (s)')
    plt.ylabel('V (mV)')
    plt.tight_layout()
    plt.show()
    
def dftEpochTT(stimulus): 
    if any(stimulus in x  for x in ['Sq_12dir_lambda30_1Hz', 'Sq_12dir_lambda30_1Hz_grey',
                                    'Sq_0_lambda30_mHz', 'Sine_0_lambda30_mHz','Sine_0_lambda30_mHz_0.5c','Sine_0_lambda30_nHz',
                                    'flowfielddots', 'flowfielddots_dark', 'flowfielddots_0.1', 'flowfielddots_dark_0.1',
                                   'flowfielddots_0.3', 'flowfielddots_dark_0.3', 'flowfielddots_0.6', 'flowfielddots_dark_0.6',
                                   'flowfielddots_1', 'flowfielddots_dark_1']):
        deftotal = 73
        NoBinned = [ 48.,   0.,   0.,  24.]
    elif any(stimulus in x  for x in ['Sine_12dir_lambda30_1Hz', 'Sine_12dir_lambda30_1Hz_0.5c',
                                      'Sine_12dir_lambda30_3Hz', 'Sine_12dir_lambda30_3Hz_0.5c',
                                      'Sine_12dir_lambda30_5Hz', 'Sine_12dir_lambda30_5Hz_0.5c',
                                      'Sine_12dir_lambda30_7Hz', 'Sine_12dir_lambda30_7Hz_0.5c']):
        deftotal = 109
        NoBinned = [ 72.,   0.,   0.,  36.]
    elif any(stimulus in x for x in ['Sq_LR_12dir_lambda30_1Hz','Sq_LR_12dir_lambda30_1Hz_bgstill_15gap', 'Sq_LR_12dir_lambda30_1Hz_bgstill_grey', 'Sq_LR_12dir_lambda30_1Hz_bgstill','Sq_LR_12dir_lambda30_1Hz_bgstill_45gap']):
        deftotal = 73*2-1
        NoBinned = np.array([ 48.,   0.,   0.,  24.])*2
    elif any(stimulus in x for x in ['Sq_LR_local_4dir','Sq_LR_local_4dir_1_bgstill','Sq_LR_local_4dir_bgstill']):
        deftotal = 433 #18*3*4+1*2
        NoBinned = [ 288.,    0.,    0.,  144.] 
    elif any(stimulus in x  for x in ['Sq_PvN_lambda30_1Hz','Sq_PvN_lambda30_1Hz_bgstill_15gap','Sq_PvN_lambda30_1Hz_bgstill_45gap']):
        deftotal = 49
        NoBinned = [ 32.,    0.,    0.,  16.] 
    elif any(stimulus in x  for x in ['yawdots_mHz', 'yawdots_dark_mHz','yawdots_nHz', 'yawdots_dark_nHz']):
        deftotal = 55
        NoBinned = [ 36.,  0.,   0.,   18.]
    elif any(stimulus in x  for x in ['movingBar_vel','movingbar_vel']): 
        deftotal = 31
        NoBinned = [ 24.,   3.,   0.,   3.]
    elif any(stimulus in x  for x in['Sine_PDND_lambda30_xHz','Sine_PDND_lambda30_xHz_grey']):
        deftotal = 271
        NoBinned = [180.,   0.,   0.,  90.]               
    elif any(stimulus in x  for x in['Sine_PDND_lambda30_yHz','Sine_PDND_lambda15_yHz','Sine_PDND_lambda60_yHz',
                                     'Sine_PDND_lambda30_yHz_grey']):
        deftotal = 361
        NoBinned = [240.,   0.,   0., 120.]              
    elif any(stimulus in x  for x in['Sine_0_lambda30_ACC','Sine_0_lambda30_ACC_grey']):
        deftotal = 331
        NoBinned = [264.,   0.,   0.,  66.]   
    elif any(stimulus in x  for x in['Sine_0_lambda30_ACC_yHz', 'Sine_0_lambda30_ACC_yHz_grey']):
        deftotal = 451
        NoBinned = [360.,   0.,   0.,  90.]     
    elif any(stimulus in x  for x in['Sine_0_lambda30_1Hz_xC', 'Sine_0_lambda30_3Hz_xC','Sine_0_lambda30_5Hz_xC',
                                     'Sine_0_lambda30_7Hz_xC']):
        deftotal = 109
        NoBinned = [72.,  0.,  0., 36.]   
    elif any(stimulus in x  for x in['Sine_strips_1Hz','Sine_strips_3Hz','Sine_strips_5Hz','Sine_strips_7Hz']):
        deftotal = 145
        NoBinned = [96.,  0.,  0., 48.]  
    elif any(stimulus in x  for x in ['Edges_light_vel','Edges_dark_vel']): 
        deftotal = 85
        NoBinned = [77.,  1.,  0.,  6.]       
    return (deftotal,NoBinned)

def findepochnames(stimulus,csvpath): 
    df = pd.read_csv(csvpath, usecols =['FrameNumber'])
    # if "epoch no.1" got mixed in: 
    epochNo = df.values[df.values!=1]
    epoch_label = epochNo[:-1][np.diff(epochNo, axis = 0) !=0]
        
    #epoch_label = df.values[1:-1][np.diff(df.values, axis = 0) != 0]
    if any(stimulus in x  for x in ['Sq_12dir_lambda30_1Hz', 'Sq_12dir_lambda30_1Hz_grey']): 
# left, 180;  right,0;   210;     30;         240;       60;       270;         90;       300;       120;       330;     150;     
# [2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16],[17,18,19],[20,21,22], [23,24,25],[26,27,28],[29,30,31],[32,33,34],[35,36,37]
        if len(epoch_label) == 73:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
            
        keys = np.arange(2,38,3)
        epoch_angle = [0,180,30,210,60,240,90,270,120,300,150,330]#[180, 0, 210, 30, 240, 60, 270, 90, 300, 120, 330, 150]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(24,3)[:,0]]
        locs = np.zeros(73)
    elif any(stimulus in x  for x in ['Sine_12dir_lambda30_1Hz', 'Sine_12dir_lambda30_1Hz_0.5c',
                                      'Sine_12dir_lambda30_3Hz', 'Sine_12dir_lambda30_3Hz_0.5c',
                                      'Sine_12dir_lambda30_5Hz', 'Sine_12dir_lambda30_5Hz_0.5c',
                                      'Sine_12dir_lambda30_7Hz', 'Sine_12dir_lambda30_7Hz_0.5c']): 
        if len(epoch_label) == 109:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
            
        keys = np.arange(2,38,3)
        epoch_angle = [0,180,30,210,60,240,90,270,120,300,150,330]#[180, 0, 210, 30, 240, 60, 270, 90, 300, 120, 330, 150]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(36,3)[:,0]]
        locs = np.zeros(109)
    elif any(stimulus in x  for x in ['Sq_0_lambda30_mHz', 'Sine_0_lambda30_mHz','Sine_0_lambda30_mHz_0.5c']):
        if len(epoch_label) == 73:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,38,3)
        epoch_angle = np.arange(0.2,2.6,0.2)
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(24,3)[:,0]]
        locs = np.zeros(73)
    elif any(stimulus in x  for x in ['Sine_0_lambda30_nHz']):
        if len(epoch_label) == 73:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,38,3)
        epoch_angle = [0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(24,3)[:,0]]
        locs = np.zeros(73)
    elif any(stimulus in x  for x in ['Sine_PDND_lambda30_xHz','Sine_PDND_lambda30_xHz_grey']):
        if len(epoch_label) == 271:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,91,3)
        epoch_angle = [0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,5,-5,5.5,-5.5,6,-6,6.5,-6.5,7,-7]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(90,3)[:,0]]
        locs = np.zeros(271)
    elif any(stimulus in x  for x in ['Sine_PDND_lambda30_yHz', 'Sine_PDND_lambda30_yHz_grey',
                                      'Sine_PDND_lambda15_yHz','Sine_PDND_lambda60_yHz',]):
        if len(epoch_label) == 361:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,121,3)
        epoch_angle = [0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,5,-5,5.5,-5.5,6,-6,6.5,-6.5,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(120,3)[:,0]]
        locs = np.zeros(361)
    elif any(stimulus in x  for x in ['Sine_0_lambda30_ACC','Sine_0_lambda30_ACC_grey']):
        if len(epoch_label) == 331:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,111,5)
        epoch_angle = [1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,5,-5,6,-6,8,-8]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(66,5)[:,0]]
        locs = np.zeros(331)      
    elif any(stimulus in x  for x in ['Sine_0_lambda30_ACC_yHz', 'Sine_0_lambda30_ACC_yHz_grey']):
        if len(epoch_label) == 451:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,151,5)
        epoch_angle = [1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,5,-5,6,-6,8,-8,9,-9,10,-10,11,-11,12,-12]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(90,5)[:,0]]
        locs = np.zeros(331)  
    elif any(stimulus in x  for x in ['flowfielddots', 'flowfielddots_dark','flowfielddots_0.1', 'flowfielddots_dark_0.1',
                                   'flowfielddots_0.3', 'flowfielddots_dark_0.3', 'flowfielddots_0.6', 'flowfielddots_dark_0.6',
                                   'flowfielddots_1', 'flowfielddots_dark_1']):
        if len(epoch_label) == 73:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,38,3)
        epoch_angle = np.hstack((np.arange(1,7), - np.arange(1,7)))
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(24,3)[:,0]]
        locs = np.zeros(73)    
    elif any(stimulus in x  for x in ['Sq_PvN_lambda30_1Hz','Sq_PvN_lambda30_1Hz_bgstill_15gap', 'Sq_PvN_lambda30_1Hz_bgstill_45gap']):
        if len(epoch_label) == 49:
            epoch_label = epoch_label[:-1]
        else:            
            print("error: No. of epoch label")
            print(len(epoch_label))
            epoch_label = epoch_label[:-1]
        keys = np.arange(2,25,3)
        epoch_angle = [0,180]*4
        sides =  [1,1,0,0,2,2,3,3]*2# Contra is 0, ipsi is 1; full field is 2, mirror is 3. 
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(int(len(epoch_label)/3),3)[:,0]]
        sDic = dict(zip(keys, sides))
        locs = [sDic[i] for i in epoch_label.reshape(int(len(epoch_label)/3),3)[:,0]]

    elif any(stimulus in x for x in ['Sq_LR_12dir_lambda30_1Hz','Sq_LR_12dir_lambda30_1Hz_bgstill_15gap', 'Sq_LR_12dir_lambda30_1Hz_bgstill_grey','Sq_LR_12dir_lambda30_1Hz_bgstill','Sq_LR_12dir_lambda30_1Hz_bgstill_45gap','Sq_LR_12dir_lambda30_1Hz_bgstill_45gap']):
        if len(epoch_label) == 145:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,73,3)
        epoch_angle = [0,180,30,210,60,240,90,270,120,300,150,330,0,180,30,210,60,240,90,270,120,300,150,330]
        #[180, 0, 210, 30, 240, 60, 270, 90, 300, 120, 330, 150, 180, 0, 210, 30, 240, 60, 270, 90, 300, 120, 330, 150]
        # Contra is 0, ipsi is 1. 
        sides = [1]*12 + [0]*12 
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(48,3)[:,0]]
        sDic = dict(zip(keys, sides))
        locs = [sDic[i] for i in epoch_label.reshape(48,3)[:,0]]
    elif any(stimulus in x for x in ['Sq_LR_local_4dir','Sq_LR_local_4dir_1_bgstill','Sq_LR_local_4dir_bgstill']):
        if len(epoch_label) == 433:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,217,3)
        epoch_angle = [0]*18 + [180]*18 + [90] *18 + [270]*18 
        patches = np.tile(np.concatenate((np.arange(11,20),np.arange(21,30))),4)
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(144,3)[:,0]]
        sDic = dict(zip(keys, patches.tolist()))
        locs = [sDic[i] for i in epoch_label.reshape(144,3)[:,0]]
    #elif any(stimulus in x for x in ['flashOnOff_mHz','flashOffOn_mHz']):
    elif any(stimulus in x  for x in ['yawdots_mHz', 'yawdots_dark_mHz']):
        if len(epoch_label) == 55:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,19,3)
        epoch_angle = np.arange(0.2,1.3,0.2)
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(18,3)[:,0]]
        locs = np.zeros(55)    
    elif any(stimulus in x  for x in ['yawdots_nHz','yawdots_dark_nHz']):
        if len(epoch_label) == 55:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,19,3)
        epoch_angle = [0.05,0.1,0.3,0.5,0.8,1.2]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(18,3)[:,0]]
        locs = np.zeros(55)  
    elif any(stimulus in x  for x in['Sine_0_lambda30_1Hz_xC', 'Sine_0_lambda30_3Hz_xC','Sine_0_lambda30_5Hz_xC',
                                     'Sine_0_lambda30_7Hz_xC']):
        if len(epoch_label) == 109:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,36,3)
        epoch_angle = [0.1,-0.1,0.2,-0.2,0.4,-0.4,0.6,-0.6,0.8,-0.8,1.0,-1.0]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(36,3)[:,0]]
        locs = np.zeros(109)  
    elif any(stimulus in x  for x in['Sine_strips_1Hz','Sine_strips_3Hz','Sine_strips_5Hz','Sine_strips_7Hz']):
        if len(epoch_label) == 145:
            epoch_label = epoch_label[:-1]
        else:
            print("error: No. of epoch label")
            print(len(epoch_label))
        keys = np.arange(2,48,3)
        epoch_angle = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9]
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label.reshape(48,3)[:,0]]
        locs = np.zeros(145)  
        
    elif any(stimulus in x  for x in ['movingBar_vel','movingbar_vel']): 
        while len(epoch_label[:-1]) != 30:
            print("error: No. of epoch label")
            print(len(epoch_label))
            print('correcting')
            keys = np.arange(2,12)
            epoch_idx_dur = np.asarray([818, 273, 164, 117, 91, 74, 63, 55, 48, 45])
            Dic_dur = dict(zip(keys, epoch_idx_dur))
            theo_dur = [Dic_dur[i] for i in epoch_label[:-1]]
            meus_dur = np.diff(np.insert(np.where([np.diff(epochNo, axis = 0) !=0])[1],0,0))[:-1]
            epoch_label = np.insert(epoch_label,np.argwhere(abs(theo_dur-meus_dur)>3).flatten(),epoch_label[np.where(abs(theo_dur-meus_dur)>3)])
            
        if len(epoch_label) == 31:
            epoch_label = epoch_label[:-1]
            print('correct epoch label')               

        keys = np.arange(2,12)
        epoch_angle = np.asarray([20, 60, 100,140,180,220,260,300,340,360])
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label]
        locs = np.zeros(30)
    elif any(stimulus in x  for x in ['flashOnOff_mHz','flashOffOn_mHz']):
        epochNo=df.values
        epoch_label = epochNo[:-1][np.diff(epochNo, axis = 0) !=0]
        keys = np.arange(1,8)
        epoch_idx_dur = np.asarray([180, 60, 120,240,300,360,420])
        Dic_dur = dict(zip(keys, epoch_idx_dur))
        theo_dur = np.asarray([Dic_dur[i] for i in epoch_label])[:-1]
        meus_dur = np.diff(np.insert(np.where([np.diff(epochNo, axis = 0) !=0])[1],0,0))[:-1]
        while sum(abs(theo_dur-meus_dur)>3): 
            epoch_label = np.insert(epoch_label,np.argwhere(abs(theo_dur-meus_dur)>3).flatten(),epoch_label[np.where(abs(theo_dur-meus_dur)>3)])
            wrong_idx = np.argwhere(abs(theo_dur-meus_dur)>3).flatten()
            mtdiff = (meus_dur-theo_dur)[wrong_idx]
            meus_dur[wrong_idx] =theo_dur[wrong_idx]
            meus_dur = np.insert(meus_dur,wrong_idx,mtdiff)
            theo_dur = np.asarray([Dic_dur[i] for i in epoch_label[:-1]])

        epoch_label = epoch_label[:-1] # get rid of the last epoch
        epoch_angle =  np.asarray([1000,333.33,666.67,1333.33,1666.66,2000,2333.33])
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label] # ms           
        angles = np.asarray(angles).repeat(3)
        locs = np.zeros(len(angles)) 
    elif any(stimulus in x  for x in ['Edges_light_vel','Edges_dark_vel']): 
        epochNo =df.values
        epoch_label = epochNo[:-1][np.diff(epochNo, axis = 0) !=0]
        epoch_label = epoch_label[epoch_label!=1]        
        if len(epoch_label) != 42:
            print("error: No. of epoch label")
            print(len(epoch_label))
        
        keys = np.arange(2,16)
        epoch_angle = [12.5,50,100,200,300,500,900,-12.5,-50,-100,-200,-300,-500,-900]        
        Dic = dict(zip(keys, epoch_angle))
        angles = [Dic[i] for i in epoch_label]
        
        epoch_loc = [1296,324,162,81,54,32.4,18,-1296,-324,-162,-81,-54,-32.4,-18]   
        Dic_loc =  dict(zip(keys, epoch_loc))
        locs = [Dic_loc[i] for i in epoch_label]
        
       
    return(angles,locs)        

def epochsplit(start_idxes,len_epoch_dix,v,pre_idx_len, mov_idx_len, post_idx_len,total_idx_len):     
    # organize v into a matrix. 
    shape1= int((len_epoch_dix-1)/3)
    v_pre = np.zeros((shape1,pre_idx_len))
    v_mov = np.zeros((shape1,mov_idx_len))
    v_post = np.zeros((shape1,post_idx_len))
    v_epoch = np.zeros((shape1,total_idx_len))
    for i in range(0,shape1): 
        v_pre[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+pre_idx_len)]
        v_mov[i,:] = v[start_idxes[i,1]: (start_idxes[i,1]+mov_idx_len)]
        v_post[i,:] = v[start_idxes[i,2]: (start_idxes[i,2]+post_idx_len)]
        v_epoch[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+total_idx_len)]
    return(v_pre,v_mov, v_post, v_epoch)

def epochsplit2(start_idxes,len_epoch_dix,v,pre_idx_len, acc_idx_len, mov_idx_len,dcc_idx_len,post_idx_len,total_idx_len):     
    # organize v into a matrix. 
    shape1= int((len_epoch_dix-1)/5)
    v_pre = np.zeros((shape1,pre_idx_len))
    v_acc = np.zeros((shape1,acc_idx_len))
    v_mov = np.zeros((shape1,mov_idx_len))
    v_dcc = np.zeros((shape1,dcc_idx_len))
    v_post = np.zeros((shape1,post_idx_len))
    v_epoch = np.zeros((shape1,total_idx_len))
    for i in range(0,shape1): 
        v_pre[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+pre_idx_len)]
        v_acc[i,:] = v[start_idxes[i,1]: (start_idxes[i,1]+acc_idx_len)]
        v_mov[i,:] = v[start_idxes[i,2]: (start_idxes[i,2]+mov_idx_len)]
        v_dcc[i,:] = v[start_idxes[i,3]: (start_idxes[i,3]+dcc_idx_len)]
        v_post[i,:] = v[start_idxes[i,4]: (start_idxes[i,4]+post_idx_len)]
        v_epoch[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+total_idx_len)]
    return(v_pre, v_acc, v_mov, v_dcc, v_post, v_epoch)

def APepochsplit(start_idxes,len_epoch_dix,spike_dix,pre_idx_len, mov_idx_len, post_idx_len,total_idx_len):     
    # organize v into a matrix. 
    shape1= int((len_epoch_dix-1)/3)
    Ap_pre = np.zeros((shape1,pre_idx_len))
    Ap_mov = np.zeros((shape1,mov_idx_len))
    Ap_post = np.zeros((shape1,post_idx_len))
    Ap_epoch = np.zeros((shape1,total_idx_len))
    for i in range(0,shape1): 
        pre = spike_idx[np.where((spike_idx >= start_idxes[i,0]) & (spike_idx <=start_idxes[i,0]+pre_idx_len))]-start_idxes[i,0]
        mov = spike_idx[np.where((spike_idx >= start_idxes[i,1]) & (spike_idx <=start_idxes[i,1]+mov_idx_len))]-start_idxes[i,1]
        post = spike_idx[np.where((spike_idx >= start_idxes[i,2]) & (spike_idx <=start_idxes[i,2]+post_idx_len))]-start_idxes[i,2]
        epoch = spike_idx[np.where((spike_idx >= start_idxes[i,0]) & (spike_idx <=start_idxes[i,0]+total_idx_len))]-start_idxes[i,0]

        Ap_pre[i,:] = np.pad(pre, (0,pre_idx_len-len(pre)), 'constant', constant_values=0)
        Ap_mov[i,:] = np.pad(mov, (0,mov_idx_len-len(mov)), 'constant', constant_values=0)
        Ap_post[i,:] = np.pad(post, (0,post_idx_len-len(post)), 'constant', constant_values=0)
        Ap_epoch[i,:] = np.pad(epoch, (0,total_idx_len-len(epoch)), 'constant', constant_values=0)
    return(Ap_pre,Ap_mov, Ap_post, Ap_epoch)

def makedfs(angles,locs,v_epoch):
    Vdf = pd.DataFrame(v_epoch)
    Vdf.insert(0, 'Angles', angles)
    Vdf.insert(1, 'Locs', locs)
    return(Vdf)

def checkepoch(thresh_pho,t,pho,stimulus):
    dt = t[1]-t[0]
    # check detected timing for photodiode start signals 
    if any(stimulus in x  for x in ['flashOnOff_mHz','flashOffOn_mHz']):
        epoch_idx = np.argwhere(pho>thresh_pho)[1:][np.diff(np.argwhere(pho>thresh_pho),axis=0)>(1/4/dt)][:-1] 
        tepochs = t[epoch_idx]
        print('Epoch checked later')
    else: 
        if any(stimulus in x  for x in ['Edges_light_vel','Edges_dark_vel']):
            epoch_idx = np.argwhere(pho>thresh_pho)[1:][np.diff(np.argwhere(pho>thresh_pho),axis=0)>(1/10/dt)][1:-1] 
        else:
            epoch_idx = np.argwhere(pho>thresh_pho)[1:][np.diff(np.argwhere(pho>thresh_pho),axis=0)>(1/2/dt)]
        deftotal,NoBinned = dftEpochTT(stimulus)
        tepochs = t[epoch_idx]
        vepochs = pho[epoch_idx]     
        if len(epoch_idx) != deftotal:
            print('total epoch number is wrong')  
            print(len(epoch_idx))
            plt.figure()
            plt.plot(t,pho)
            plt.title('photodiode')
            for i in tepochs: 
                plt.axvline(x=i, linewidth=1, color='g')
            plt.scatter(tepochs,vepochs, c='r', s = 50)
            plt.xlabel('time (s)')
            plt.ylabel('V (mV)')
            plt.tight_layout()
            plt.show()

        # check epoch length
        arr=plt.hist(np.diff(tepochs),bins=4)
        if sum(arr[0] == NoBinned) !=4: 
            print('Error: epoch length')
            print(arr)
            plt.figure()
            plt.hist(np.diff(tepochs),bins=4)    
            plt.title('histogram of epopch length')
            plt.show()
            
    return(epoch_idx,tepochs)

def FiltV(V, N, Wn):
    # design the Buterworth filter
    B, A = signal.butter(N, Wn, output='ba')
    #apply the filter
    fV = signal.filtfilt(B,A,V)  
    return(fV)
def save_eps(tepochs,spiking,epoch_idx,spike_idx,t,v,pho,stimulus,csvpath,fdname,recN,n, N, Wn):
    # idx when each epoch comes on. 
    tepochs_idx =  np.array([int(np.argwhere(t==i)) for i in tepochs])
    if any(stimulus in x  for x in ['Sine_0_lambda30_ACC', 'Sine_0_lambda30_ACC_yHz', 'Sine_0_lambda30_ACC_yHz_grey']):
        # calculating length of pre, ACC, mov, DCC & post. 
        sshape = int((len(epoch_idx)-1)/5)
        pshape =int((len(epoch_idx)-1)/5-1)
        start_idxes = tepochs_idx[0:-1].reshape(sshape,5)
        pre_idx_len = mode(np.diff(start_idxes)[:,0])
        acc_idx_len = mode(np.diff(start_idxes)[:,1])
        mov_idx_len = mode(np.diff(start_idxes)[:,2])
        dcc_idx_len = mode(np.diff(start_idxes)[:,3])
        post_idx_len = int(mode(tepochs_idx[1:-5].reshape(pshape,5)[:,-1] - tepochs_idx[1:-5].reshape(pshape,5)[:,-2]).tolist())
        total_idx_len = pre_idx_len+acc_idx_len+mov_idx_len+dcc_idx_len+post_idx_len

        # splitting into epochs, or sections of epochs
        v_pre, v_acc, v_mov, v_dcc, v_post, v_epoch =  epochsplit2(start_idxes,len(epoch_idx),v, pre_idx_len, acc_idx_len, 
                                                                   mov_idx_len,dcc_idx_len,post_idx_len,total_idx_len)
        pho_pre, pho_acc, pho_mov, pho_dcc, pho_post, pho_epoch =  epochsplit2(start_idxes,len(epoch_idx),pho, pre_idx_len, acc_idx_len, 
                                                                               mov_idx_len,dcc_idx_len,post_idx_len,total_idx_len)
        if spiking: 
            APtrain = np.zeros(len(v))
            APtrain[spike_idx] = 1
            Ap_pre, Ap_acc,Ap_mov, Ap_dcc,Ap_post, Ap_epoch =epochsplit2(start_idxes,len(epoch_idx),APtrain, pre_idx_len, acc_idx_len, 
                                                                         mov_idx_len,dcc_idx_len,post_idx_len,total_idx_len)
          # get angles
        angles,locs = findepochnames(stimulus,csvpath)
        angles = angles[:sshape]
        locs = locs[:sshape]

        # create dfs
        Vdf = makedfs(angles,locs,v_epoch)
        preVdf = makedfs(angles,locs,v_pre)
        accVdf = makedfs(angles,locs,v_acc)
        movVdf = makedfs(angles,locs,v_mov)
        dccVdf = makedfs(angles,locs,v_dcc)
        postVdf = makedfs(angles,locs,v_post)        

        Phodf = makedfs(angles,locs,pho_epoch)
        prePhodf = makedfs(angles,locs,pho_pre)
        accPhodf = makedfs(angles,locs,pho_acc)
        movPhodf = makedfs(angles,locs,pho_mov)
        dccPhodf = makedfs(angles,locs,pho_dcc)
        postPhodf = makedfs(angles,locs,pho_post)
        

        if spiking:
            Apdf = makedfs(angles,locs,Ap_epoch)
            preApdf = makedfs(angles,locs,Ap_pre)
            accApdf = makedfs(angles,locs,Ap_acc)
            movApdf = makedfs(angles,locs,Ap_mov)
            dccApdf = makedfs(angles,locs,Ap_dcc)
            postApdf = makedfs(angles,locs,Ap_post)

        dir_aligned = '%s'%fdname+'/aligned'
        try:
            os.makedirs(dir_aligned)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        #save dfs
        Vdf.to_pickle(dir_aligned+'/rec%dVdf.ASCII'%recN) 
        preVdf.to_pickle(dir_aligned+'/rec%dpreVdf.ASCII'%recN) 
        accVdf.to_pickle(dir_aligned+'/rec%daccVdf.ASCII'%recN)
        movVdf.to_pickle(dir_aligned+'/rec%dmovVdf.ASCII'%recN) 
        dccVdf.to_pickle(dir_aligned+'/rec%ddccVdf.ASCII'%recN)
        postVdf.to_pickle(dir_aligned+'/rec%dpostVdf.ASCII'%recN) 

        #save dfs
        Phodf.to_pickle(dir_aligned+'/rec%dPhodf.ASCII'%recN) 
        prePhodf.to_pickle(dir_aligned+'/rec%dprePhodf.ASCII'%recN)
        accPhodf.to_pickle(dir_aligned+'/rec%daccPhodf.ASCII'%recN) 
        movPhodf.to_pickle(dir_aligned+'/rec%dmovPhodf.ASCII'%recN) 
        dccPhodf.to_pickle(dir_aligned+'/rec%ddccPhodf.ASCII'%recN)
        postPhodf.to_pickle(dir_aligned+'/rec%dpostPhodf.ASCII'%recN) 

        if spiking:
            #save dfs
            Apdf.to_pickle(dir_aligned+'/rec%dApdf.ASCII'%recN) 
            preApdf.to_pickle(dir_aligned+'/rec%dpreApdf.ASCII'%recN)
            accApdf.to_pickle(dir_aligned+'/rec%daccApdf.ASCII'%recN)
            movApdf.to_pickle(dir_aligned+'/rec%dmovApdf.ASCII'%recN) 
            dccApdf.to_pickle(dir_aligned+'/rec%ddccApdf.ASCII'%recN)
            postApdf.to_pickle(dir_aligned+'/rec%dpostApdf.ASCII'%recN) 
        print('all saved!')
        T = 6

    else:
        # calculating length of pre, mov & post. 
        sshape = int((len(epoch_idx)-1)/3)
        pshape =int((len(epoch_idx)-1)/3-1)
        start_idxes = tepochs_idx[0:-1].reshape(sshape,3)
        pre_idx_len = int(mode((start_idxes[:,1] - start_idxes[:,0] ).tolist()))
        mov_idx_len = int(mode(start_idxes[:,2] - start_idxes[:,1]).tolist())
        post_idx_len = int(mode(tepochs_idx[1:-3].reshape(pshape,3)[:,2] - tepochs_idx[1:-3].reshape(pshape,3)[:,1]).tolist())
        total_idx_len = pre_idx_len+mov_idx_len+post_idx_len

        # error messages for checking the length
        if post_idx_len != pre_idx_len:
            print ('Error: mismacth btw pre & post idx length')
        if mov_idx_len/pre_idx_len != 2.0: # 2.0 for local; rest 4. 
            print ('Error: mismacth btw mov & pre idx length')

        # splitting into epochs, or sections of epochs
        v_pre,v_mov, v_post, v_epoch =  epochsplit(start_idxes,len(epoch_idx),v, pre_idx_len, mov_idx_len, post_idx_len,total_idx_len)
        pho_pre,pho_mov, pho_post, pho_epoch =  epochsplit(start_idxes,len(epoch_idx),pho, pre_idx_len, mov_idx_len, 
                                                           post_idx_len,total_idx_len)
        if spiking: 
            APtrain = np.zeros(len(v))
            APtrain[spike_idx] = 1
            Ap_pre,Ap_mov, Ap_post, Ap_epoch =epochsplit(start_idxes,len(epoch_idx),APtrain,pre_idx_len, mov_idx_len, 
                                                         post_idx_len,total_idx_len)

        # get angles
        angles,locs = findepochnames(stimulus,csvpath)
        angles = angles[:sshape]
        locs = locs[:sshape]

        # create dfs
        Vdf = makedfs(angles,locs,v_epoch)
        preVdf = makedfs(angles,locs,v_pre)
        movVdf = makedfs(angles,locs,v_mov)
        postVdf = makedfs(angles,locs,v_post)

        Phodf = makedfs(angles,locs,pho_epoch)
        prePhodf = makedfs(angles,locs,pho_pre)
        movPhodf = makedfs(angles,locs,pho_mov)
        postPhodf = makedfs(angles,locs,pho_post)

        if spiking:
            Apdf = makedfs(angles,locs,Ap_epoch)
            preApdf = makedfs(angles,locs,Ap_pre)
            movApdf = makedfs(angles,locs,Ap_mov)
            postApdf = makedfs(angles,locs,Ap_post)

        dir_aligned = '%s'%fdname+'/aligned'
        try:
            os.makedirs(dir_aligned)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        #save dfs
        Vdf.to_pickle(dir_aligned+'/rec%dVdf.ASCII'%recN) 
        preVdf.to_pickle(dir_aligned+'/rec%dpreVdf.ASCII'%recN) 
        movVdf.to_pickle(dir_aligned+'/rec%dmovVdf.ASCII'%recN) 
        postVdf.to_pickle(dir_aligned+'/rec%dpostVdf.ASCII'%recN) 

        #save dfs
        Phodf.to_pickle(dir_aligned+'/rec%dPhodf.ASCII'%recN) 
        prePhodf.to_pickle(dir_aligned+'/rec%dprePhodf.ASCII'%recN) 
        movPhodf.to_pickle(dir_aligned+'/rec%dmovPhodf.ASCII'%recN) 
        postPhodf.to_pickle(dir_aligned+'/rec%dpostPhodf.ASCII'%recN) 

        if spiking:
            #save dfs
            Apdf.to_pickle(dir_aligned+'/rec%dApdf.ASCII'%recN) 
            preApdf.to_pickle(dir_aligned+'/rec%dpreApdf.ASCII'%recN) 
            movApdf.to_pickle(dir_aligned+'/rec%dmovApdf.ASCII'%recN) 
            postApdf.to_pickle(dir_aligned+'/rec%dpostApdf.ASCII'%recN) 
        print('all saved!')
        T = 4

    # experiment with filter

    plt.figure()
    t = np.linspace(0,T,np.shape(Vdf)[1]-2)
    plt.plot(t,Vdf.iloc[n][2:],'r')
    if spiking:
        tAP = t[Apdf.iloc[n][2:].astype(np.bool)]
        vAP = Vdf.iloc[n][2:][Apdf.iloc[n][2:].astype(np.bool)]
        plt.scatter(tAP,vAP)
        plt.plot(t,FiltV(Vdf.iloc[n][2:].values, N, Wn),'g')
        plt.show()
    print('postApdf')
    print(postApdf.sort_values(by=['Locs', 'Angles']))
    

def edges_ep(recN,v,t,spike_idx,epoch_idx,fdname,stimulus,csvpath,N, Wn, plot):
# for "Edges_light_vel"
    start_idx = epoch_idx[:-1].reshape(int((len(epoch_idx)-1)/2),2)[:,0]
    end_idx = np.hstack([start_idx[1:],epoch_idx[-1]])
    angles,locs = findepochnames(stimulus,csvpath)
    APtrain = np.zeros(len(v))
    APtrain[spike_idx] = 1

    dir_plots = '%s'%fdname+'/%d' %recN
    try:
        os.makedirs(dir_plots)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    Eps = []
    Veps= []
    Aps = []
    Ep_t = []
    for i in np.arange(len(angles)):
        ep = v[start_idx[i]:end_idx[i]]
        vep = FiltV(ep, N, Wn)    
        ap = APtrain[start_idx[i]:end_idx[i]]    
        ep_t = t[start_idx[i]:end_idx[i]]-t[start_idx[i]]
        Eps.append(ep)
        Veps.append(FiltV(ep, N, Wn))
        Aps.append(ap)
        Ep_t.append(ep_t)
        if plot:  
            plt.figure()    
            plt.plot(ep_t,ep,'b')
            plt.scatter(ep_t[ap>0],ep[ap>0],c='r')
            plt.plot(ep_t,vep,'g')
            plt.xlabel('t(s)')
            plt.ylabel('v(mv)')
            plt.axvline(x = 1,ls=":")
            plt.savefig('%s/wave%drec%dspeed%dAP%d.jpeg' %(dir_plots, i,recN, angles[i], sum(ap)))
            plt.close()
            
    # convert to dataframe with speed for each epochs
    Apdf = pd.DataFrame(np.asarray(Aps))
    Vdf = pd.DataFrame(np.asarray(Veps))
    Tdf = pd.DataFrame(np.asarray(Ep_t))
    Apdf.insert(0,'Angles',angles)
    Vdf.insert(0,'Angles',angles)
    Tdf.insert(0,'Angles',angles)
    Apdf.insert(1,'Locs',locs)
    Vdf.insert(1,'Locs',locs)
    Tdf.insert(1,'Locs',locs)

    # save dfs
    dir_aligned = '%s'%fdname+'/aligned'
    try:
        os.makedirs(dir_aligned)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    Apdf = Apdf.sort_values(by=['Angles'])
    Vdf = Vdf.sort_values(by=['Angles'])
    Tdf = Tdf.sort_values(by=['Angles'])
    
    Apdf.to_pickle(dir_aligned+'/rec%dApdf.ASCII'%recN) 
    Vdf.to_pickle(dir_aligned+'/rec%dVdf.ASCII'%recN) 
    Tdf.to_pickle(dir_aligned+'/rec%dTdf.ASCII'%recN)     
    print('all saved')
    return(Apdf,Vdf,Tdf)    

# for 'movingBar_vel'   
def mov_ep(recN,v,t,spike_idx,epoch_idx,fdname,stimulus,csvpath,N, Wn, plot):

    angles,locs = findepochnames(stimulus,csvpath)

    APtrain = np.zeros(len(v))
    APtrain[spike_idx] = 1
    
    # save plots
    dir_plots = '%s'%fdname+'/%d' %recN
    try:
        os.makedirs(dir_plots)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    Eps = []
    Veps= []
    Aps = []
    Ep_t = []
    for i in np.arange(len(angles)):
        ep = v[epoch_idx[i]:epoch_idx[i+1]]
        vep = FiltV(ep, N, Wn)    
        ap = APtrain[epoch_idx[i]:epoch_idx[i+1]]    
        ep_t = t[epoch_idx[i]:epoch_idx[i+1]]-t[epoch_idx[i]]
        Eps.append(ep)
        Veps.append(FiltV(ep, N, Wn))
        Aps.append(ap)
        Ep_t.append(ep_t)
        if plot:  
            plt.figure()    
            plt.plot(ep_t,ep,'b')
            plt.scatter(ep_t[ap>0],ep[ap>0],c='r')
            plt.plot(ep_t,vep,'g')
            plt.xlabel('t(s)')
            plt.ylabel('v(mv)')
            if any(stimulus in x  for x in ['flashOnOff_mHz','flashOffOn_mHz']):
                plt.axvline(x=max(ep_t)*0.2, ls = ":")
                plt.savefig('%s/wave%drec%dperiod%dAP%d.jpeg' %(dir_plots, i,recN, angles[i], sum(ap)))
            else:            
                plt.savefig('%s/wave%drec%dspeed%dAP%d.jpeg' %(dir_plots, i,recN, angles[i], sum(ap)))
            plt.close()
    # convert to dataframe with speed for each epochs
    Apdf = pd.DataFrame(np.asarray(Aps))
    Vdf = pd.DataFrame(np.asarray(Veps))
    Tdf = pd.DataFrame(np.asarray(Ep_t))
    Apdf.insert(0,'Angles',angles)
    Vdf.insert(0,'Angles',angles)
    Tdf.insert(0,'Angles',angles)
    Apdf.insert(1,'Locs',locs)
    Vdf.insert(1,'Locs',locs)
    Tdf.insert(1,'Locs',locs)

    # save dfs
    dir_aligned = '%s'%fdname+'/aligned'
    try:
        os.makedirs(dir_aligned)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    Apdf.to_pickle(dir_aligned+'/rec%dVdf.ASCII'%recN) 
    Vdf.to_pickle(dir_aligned+'/rec%dpreVdf.ASCII'%recN) 
    Tdf.to_pickle(dir_aligned+'/rec%dmovVdf.ASCII'%recN)     
    print('all saved')
    return(Apdf,Vdf,Tdf)
    

def RaRi(RecVlist,fdname):
    R = np.zeros((len(RecVlist),3))
    for n,i in enumerate(RecVlist):
        t,v,pho = readtraces(fdname, i)  
        dt = t[1]-t[0]
        baseline = np.mean(v[:int(0.005/dt)])#np.mean(v[:int(0.003/dt)])
        Ra = 10/abs(min(v)-baseline)*1000
        Rinpt = 10/abs(np.mean(v[int(0.019/dt):int(0.020/dt)])-baseline)*1000#10/abs(np.mean(v[int(0.007/dt):int(0.008/dt)])-baseline)*1000
        R[n][0] = i
        R[n][1] = Ra
        R[n][2] = Rinpt
    #     print('Rec%d: Ra is %f Mohm' %(i,Ra))
    #     print('      Rinput is %f Mohm' %Rinpt)
    Rdf = pd.DataFrame(R)
    Rdf.columns = ['RecN','Ra','Rinput']
    #save
    dir_aligned = 'Analysis/%s'%fdname+'/aligned/Isteps'
    try:
        os.makedirs(dir_aligned)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    Rdf.to_pickle(dir_aligned+'/RaRinput.ASCII')
    
    from pandas.tools.plotting import table
    plt.figure(figsize = [10,1+0.1*len(RecVlist)])
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, Rdf,loc='center')  # where df is your data frame
    
    dir_pic = 'Pics/%s'%fdname+'/Isteps'
    try:
        os.makedirs(dir_pic)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    plt.savefig(dir_pic+'/RaRinput.jpeg')
    plt.close()
    return(Rdf)







# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os, errno
from igor.binarywave import load
from statistics import mode
import scipy.signal as signal
import pandas as pd
import pickle

get_ipython().magic('matplotlib nbagg')
#pd.set_option('io.hdf.default_format','table') # save all .h5 files with HDF5 as PyTables. 

###############
# code for Raw
###############
def readibw(fdname,fname):
        # Get the current directory
        _thisDir = os.getcwd()
        # name of the folder & the file. e.g. fdname = "041917fly2cell1", # fname = "ch0_2.ibw"
        # directory of the file
        f_dir = os.path.join(_thisDir, fdname, 'waves', fname).replace("\\","/")
        # load the wave
        wave = load(f_dir)
        v = wave['wave']['wData'] # voltage
        af = wave['wave']['wave_header']['sfA'][0] # 1/(acquisition frequency in s).
        t = np.linspace(0,len(v)*af,len(v)) # generate time
        return(v,t)
    
def makedfs(angles,locs,v_epoch):
    Vdf = pd.DataFrame(v_epoch) ; Vdf.insert(0, 'Angles', angles); Vdf.insert(1, 'Locs', locs)
    return(Vdf)

def FiltV(V, N, Wn):
    # design the Buterworth filter & then apply the filer. 
    B, A = signal.butter(N, Wn, output='ba'); fV = signal.filtfilt(B,A,V)  
    return(fV)

def epochsplit(start_idxes,len_epoch_idx,v,pre_idx_len, mov_idx_len, post_idx_len,total_idx_len):     
    # organize v into a matrix. 
    shape1= int((len_epoch_idx-1)/3)
    v_pre = np.zeros((shape1,pre_idx_len)) ; v_mov = np.zeros((shape1,mov_idx_len))
    v_post = np.zeros((shape1,post_idx_len)); v_epoch = np.zeros((shape1,total_idx_len))
    for i in range(0,shape1): 
        v_pre[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+pre_idx_len)]
        v_mov[i,:] = v[start_idxes[i,1]: (start_idxes[i,1]+mov_idx_len)]
        v_post[i,:] = v[start_idxes[i,2]: (start_idxes[i,2]+post_idx_len)]
        v_epoch[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+total_idx_len)]
    return(v_pre,v_mov, v_post, v_epoch)

def epochsplit2(start_idxes,len_epoch_idx,v,pre_idx_len, acc_idx_len, mov_idx_len,dcc_idx_len,post_idx_len,total_idx_len):     
    # organize v into a matrix. 
    shape1= int((len_epoch_idx-1)/5)
    v_pre = np.zeros((shape1,pre_idx_len)); v_acc = np.zeros((shape1,acc_idx_len))
    v_mov = np.zeros((shape1,mov_idx_len)); v_dcc = np.zeros((shape1,dcc_idx_len))
    v_post = np.zeros((shape1,post_idx_len)); v_epoch = np.zeros((shape1,total_idx_len))
    for i in range(0,shape1): 
        v_pre[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+pre_idx_len)]
        v_acc[i,:] = v[start_idxes[i,1]: (start_idxes[i,1]+acc_idx_len)]
        v_mov[i,:] = v[start_idxes[i,2]: (start_idxes[i,2]+mov_idx_len)]
        v_dcc[i,:] = v[start_idxes[i,3]: (start_idxes[i,3]+dcc_idx_len)]
        v_post[i,:] = v[start_idxes[i,4]: (start_idxes[i,4]+post_idx_len)]
        v_epoch[i,:] = v[start_idxes[i,0]: (start_idxes[i,0]+total_idx_len)]
    return(v_pre, v_acc, v_mov, v_dcc, v_post, v_epoch)


def makedir(newdir):
    try:
        os.makedirs(newdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def padding(Aps):
    pad = np.empty((len(Aps),max([(len(i)) for i in Aps]))); pad[:] = np.nan
    pad_idx = []
    for n,j in enumerate(Aps): pad[n,:len(j)] = j; pad_idx.append(int(len(j)))
    return(pad, pad_idx)

def fftAP(movApdf,T,fdname,recN,Apidx):
    Aptest = movApdf.iloc[:,2:]; n = 64*2
    t = np.linspace(0,T,np.shape(Aptest)[1])
    APfft = np.empty([np.shape(Aptest)[0],n]); APfft[:] = np.nan # 63*2 because save upto 63Hz.
    for i in range(np.shape(Aptest)[0]): 
        spike_time = t*Aptest.values[i,:]
        if np.size(np.nonzero(spike_time)) != 0: 
            Aptestfft = np.abs(np.fft.fft(spike_time[np.nonzero(spike_time)])[:int(len(spike_time[np.nonzero(spike_time)])/2)])
            if len(Aptestfft) < n:
                fill126 =  np.zeros(int(n - len(Aptestfft)))
                Aptestfft= np.append(Aptestfft,fill126)
            Apx = np.arange(0,len(Aptestfft))/(T-2)
            APfft[i,:] = Aptestfft[:n]
#             plt.figure()
#             plt.bar(Apx,Aptestfft,width =-.1)
#             plt.scatter(Apx,np.abs(Aptestfft),color = 'r')
#             plt.xlabel('Hz'); plt.ylabel('Power'); plt.title('AP: FFT analysis. Angle %d'%movApdf.iloc[i,0])
#             plt.savefig('Pics/%s'%fdname+'/%s/rec%s_angle%d_APfft_n%d.jpeg'%(str(recN),str(recN),movApdf.iloc[i,0],Apidx[i]))
#             plt.close()
    APfftdf = pd.DataFrame(np.hstack((movApdf.iloc[:,:2],APfft)),columns = np.hstack((['Angles','Locs'],[str(i)+'Hz' for i in 
                                                                                                         np.arange(0,n/2,0.5)]))) 
    return(APfftdf)

def fftVm(movVdf,T,fdname,recN, N, Wn,Vidx):
    Vtest = movVdf.iloc[:,2:]; n = 64*2
    filtVtest = FiltV(Vtest,N,Wn)
    Vtestfft = abs(np.fft.fft(filtVtest)[:,:int(np.shape(Vtest)[1]/2)])
    Vx = np.tile(np.arange(np.shape(Vtestfft)[1]),(np.shape(Vtestfft)[0],1))/(T-2)
#     for i in range(np.shape(Vtest)[0]): 
#         plt.figure() # only show from 0.5 to 24.5 with 0.5 increments. 
#         plt.bar(Vx[i,:][1:n], Vtestfft[i,:][1:n],width = 0.1)
#         plt.scatter(Vx[i,:][1:n], Vtestfft[i,:][1:n],color = 'r')
#         plt.xlabel('Hz'); plt.ylabel('Power'); plt.title('Vm: FFT analysis. Angle %d. 0Hz dropped'%movVdf.iloc[i,0])
#         plt.savefig('Pics/%s'%fdname+'/%s/rec%s_angle%d_Vmfft_n%d.jpeg'%(str(recN),str(recN),movVdf.iloc[i,0],Vidx[i]))
#         plt.close()
    Vmfftdf = pd.DataFrame(np.hstack((movVdf.iloc[:,:2],Vtestfft[:,0:n])),columns = np.hstack((['Angles','Locs'],[str(i)+'Hz' for i in 
                                                                                                                  np.arange(0,n/2,0.5)]))) 
    return(Vmfftdf)

class RAW:
    # an object that describe the stimulus_name,v,t,pho,spike_idx of the raw data    
    def __init__(self,stimulus,fdname,recN,spiking,csvroot,csvpath):
        self.stimulus_name = stimulus
        self.fdname = fdname 
        self.recN = recN
        self.spiking = spiking
        self.csvroot = csvroot
        self.csvpath = csvpath
        
    def readtraces(self):  
        fname = "ch0_%d.ibw" % self.recN 
        phoname = "ch1_%d.ibw" % self.recN # photodiode channel 
        self.v,self.t = readibw(self.fdname,fname)
        self.pho,_ = readibw(self.fdname,phoname) # all the "t" should be the same, thus only read once
        return(self.t,self.v,self.pho)

    def detectAP(self,th1,th2,th3):
        vdiff = np.diff(self.v)
        vdiff1= np.argwhere(vdiff>th1)
        vdiff1diff = np.diff(vdiff1,axis =0)
        vdiff1diff2=np.argwhere(vdiff1diff>th2)[:,0]
        self.spike_idx = (np.arange(0,len(self.t)+1)[1:][vdiff1][vdiff1diff2]).flatten()
        self.spike_idx= self.spike_idx[self.v[self.spike_idx] > th3] #th3 is to get rid of noise       
        return(self.spike_idx)

    def plotrange(self,ta,tb):     
        plt.figure()
        plt.subplot(211)
        plt.plot(self.t[ta:tb],self.v[ta:tb])
        if self.spike_idx != None:
            if np.sum(self.spike_idx) != 0: 
                plt.scatter(self.t[self.spike_idx][ta:tb],self.v[self.spike_idx][ta:tb], c='r')
        plt.title('Response')
        plt.subplot(212)
        plt.plot(self.t[ta:tb],self.pho[ta:tb])
        plt.title('photodiode')
        plt.xlabel('time (s)')
        plt.ylabel('V (mV)')
        plt.tight_layout()
        plt.show()
       
    def def_type(self):
    # group stimulus to different types, and define deftotal and NoBinned for each type. 
        stimulus = self.stimulus_name
        Sq_12dir = ['Sq_12dir_lambda30_1Hz', 'Sq_12dir_lambda30_1Hz_grey']
        Sine_12dir = ['Sine_12dir_lambda30_1Hz', 'Sine_12dir_lambda30_1Hz_0.5c',
                      'Sine_12dir_lambda30_3Hz', 'Sine_12dir_lambda30_3Hz_0.5c',
                      'Sine_12dir_lambda30_5Hz', 'Sine_12dir_lambda30_5Hz_0.5c',
                      'Sine_12dir_lambda30_7Hz', 'Sine_12dir_lambda30_7Hz_0.5c']
        Sq_LR_12dir = ['Sq_LR_12dir_lambda30_1Hz','Sq_LR_12dir_lambda30_1Hz_bgstill_15gap', 
                     'Sq_LR_12dir_lambda30_1Hz_bgstill_grey', 'Sq_LR_12dir_lambda30_1Hz_bgstill',
                     'Sq_LR_12dir_lambda30_1Hz_bgstill_45gap']
        Sq_LR_local = ['Sq_LR_local_4dir','Sq_LR_local_4dir_1_bgstill','Sq_LR_local_4dir_bgstill']
        Sq_PvN =  ['Sq_PvN_lambda30_1Hz','Sq_PvN_lambda30_1Hz_bgstill_15gap','Sq_PvN_lambda30_1Hz_bgstill_45gap']
        Sine_PDND_x =['Sine_PDND_lambda30_xHz','Sine_PDND_lambda30_xHz_grey']
        Sine_PDND_y = ['Sine_PDND_lambda30_yHz','Sine_PDND_lambda15_yHz',
                       'Sine_PDND_lambda60_yHz','Sine_PDND_lambda30_yHz_grey']
        Sine_PDND_z = ['Sine_PDND_lambda30_zHz']
        Sine_ACC = ['Sine_0_lambda30_ACC','Sine_0_lambda30_ACC_grey']
        Sine_ACC_y = ['Sine_0_lambda30_ACC_yHz', 'Sine_0_lambda30_ACC_yHz_grey']
        Sine_xC =  ['Sine_0_lambda30_1Hz_xC', 'Sine_0_lambda30_3Hz_xC','Sine_0_lambda30_5Hz_xC','Sine_0_lambda30_7Hz_xC']
        Sine_strips = ['Sine_strips_1Hz','Sine_strips_3Hz','Sine_strips_5Hz','Sine_strips_7Hz']
        Edges_vel = ['Edges_light_vel','Edges_dark_vel', 'Edges_light_vel_grey','Edges_dark_vel_grey']
        Edges_vel_vert = ['Edges_light_vel_vert','Edges_dark_vel_vert','Edges_light_vel_vert_grey','Edges_dark_vel_vert_grey']
        Sine_patches = ['Sine_patches_PDND_1HZ','Sine_patches_UPDN_1HZ', 'Sine_patches_RULD_1HZ', 'Sine_patches_LURD_1HZ',
                       'Sine_patches_PDND_6HZ','Sine_patches_UPDN_6HZ', 'Sine_patches_RULD_6HZ', 'Sine_patches_LURD_6HZ',
                       'Sine_patches_PDND_6HZ_0.2c','Sine_patches_UPDN_6HZ_0.2c', 'Sine_patches_RULD_6HZ_0.2c', 
                        'Sine_patches_LURD_6HZ_0.2c',   'Sine_patches_PDND_6HZ_0.4c','Sine_patches_UPDN_6HZ_0.4c', 
                        'Sine_patches_RULD_6HZ_0.4c', 'Sine_patches_LURD_6HZ_0.4c',
                       'Sine_patches_PDND_1HZ_0.4c','Sine_patches_UPDN_1HZ_0.4c',
                       'Sine_patches_RULD_1HZ_0.4c', 'Sine_patches_LURD_1HZ_0.4c']
        Sine_patches2 = ['Sine_patches2_PDND_6HZ_0.2c','Sine_patches2_UPDN_6HZ_0.2c', 'Sine_patches2_RULD_6HZ_0.2c', 
                'Sine_patches2_LURD_6HZ_0.2c', 'Sine_patches2_PDND_6HZ_0.4c','Sine_patches2_UPDN_6HZ_0.4c', 
                'Sine_patches2_RULD_6HZ_0.4c', 'Sine_patches2_LURD_6HZ_0.4c',
                'Sine_patches2_PDND_1HZ_0.4c','Sine_patches2_UPDN_1HZ_0.4c',
               'Sine_patches2_RULD_1HZ_0.4c', 'Sine_patches2_LURD_1HZ_0.4c']
        Flash = ['flashOnOff_mHz','flashOffOn_mHz','flashOnOff_mHz_grey','flashOffOn_mHz_grey']
        Oppo = ['Sine_LR_Oppo_lambda30_1Hz']
        Sq_PvN =  ['Sq_PvN_lambda30_1Hz','Sq_PvN_lambda30_1Hz_bgstill_15gap','Sq_PvN_lambda30_1Hz_bgstill_45gap']
        Sine_PvN = ['Sine_PvN_lambda30_zHz_bgstill']
        Sine_PvN_u = ['Sine_PvN_lambda30_uHz_bgstill']
        
        if any(stimulus in x  for x in Sq_12dir):
            sti_type = 'Sq_12dir'; sti_set = Sq_12dir; deftotal = 73;  NoBinned = [ 48.,   0.,   0.,  24.]
        elif any(stimulus in x  for x in Sine_12dir):
            sti_type = 'Sine_12dir'; sti_set = Sine_12dir; deftotal = 109; NoBinned = [ 72.,   0.,   0.,  36.]
        elif any(stimulus in x for x in Sq_LR_12dir):
            sti_type = 'Sq_LR_12dir'; sti_set = Sq_LR_12dir;deftotal = 73*2-1; NoBinned = np.array([ 48.,   0.,   0.,  24.])*2
        elif any(stimulus in x for x in Sq_LR_local):
            sti_type = 'Sq_LR_local'; sti_set = Sq_LR_local; deftotal = 433; NoBinned = [ 288.,    0.,    0.,  144.] 
        elif any(stimulus in x  for x in Sq_PvN):
            sti_type = 'Sq_PvN'; sti_set = Sq_PvN; deftotal = 49;NoBinned = [ 32.,    0.,    0.,  16.] 
        elif any(stimulus in x  for x in Sine_PDND_x):
            sti_type = 'Sine_PDND_x'; sti_set = Sine_PDND_x; deftotal = 271; NoBinned = [180.,   0.,   0.,  90.]               
        elif any(stimulus in x  for x in Sine_PDND_y):
            sti_type = 'Sine_PDND_y';sti_set = Sine_PDND_y; deftotal = 361;NoBinned = [240.,   0.,   0., 120.]  
        elif any(stimulus in x  for x in Sine_PDND_z):
            #sti_type = 'Sine_PDND_z';sti_set = Sine_PDND_z; deftotal = 325;NoBinned = [216.,   0.,   0., 108.]
            sti_type = 'Sine_PDND_z';sti_set = Sine_PDND_z; deftotal = 271;NoBinned = [180.,   0.,   0., 90.]       
        elif any(stimulus in x  for x in Sine_ACC):
            sti_type = 'Sine_ACC';sti_set = Sine_ACC; deftotal = 331; NoBinned = [264.,   0.,   0.,  66.]   
        elif any(stimulus in x  for x in Sine_ACC_y):
            sti_type = 'Sine_ACC_y';sti_set = Sine_ACC_y; deftotal = 451; NoBinned = [360.,   0.,   0.,  90.]     
        elif any(stimulus in x  for x in Sine_xC):
            sti_type = 'Sine_xC'; sti_set = Sine_xC; deftotal = 109; NoBinned = [72.,  0.,  0., 36.]   
        elif any(stimulus in x  for x in Sine_strips):
            sti_type = 'Sine_strips'; sti_set = Sine_strips; deftotal = 145; NoBinned = [96.,  0.,  0., 48.]  
        elif any(stimulus in x  for x in Edges_vel): 
            sti_type = 'Edges_vel'; sti_set =Edges_vel; deftotal = 85; NoBinned = [78.,  0.,  0.,  6.]
        elif any(stimulus in x  for x in Edges_vel_vert): 
            sti_type = 'Edges_vel_vert'; sti_set =Edges_vel_vert; deftotal = 85; NoBinned = [78.,  0.,  0.,  6.]
        elif any(stimulus in x  for x in Sine_patches): 
            sti_type = 'Sine_patches'; sti_set = Sine_patches; deftotal = 253; NoBinned = [168.,   0.,  0.,   84.]
        elif any(stimulus in x  for x in Sine_patches2): 
            sti_type = 'Sine_patches2'; sti_set = Sine_patches2; deftotal = 181; NoBinned = [120.,   0.,  0.,   60.]
        elif any(stimulus in x for x in Flash): 
            sti_type = 'Flash'; sti_set = Flash; deftotal = 43; NoBinned = [27.,  5.,  4.,  6.]
        elif any(stimulus in x for x in Oppo): 
            sti_type = 'Oppo'; sti_set = Oppo; deftotal = 82; NoBinned = [54.,  0.,  0., 27.]
        elif any(stimulus in x for x in Sine_PvN): 
            sti_type = 'Sine_PvN'; sti_set = Sine_PvN; deftotal = 397; NoBinned = [264.,   0.,   0., 132.] 
            #deftotal = 433; NoBinned = [288.,   0.,   0., 144.]
        elif any(stimulus in x for x in Sine_PvN_u): 
            sti_type = 'Sine_PvN_u'; sti_set = Sine_PvN_u; deftotal = 181; NoBinned = [120.,   0.,  41.,  19.]
        else:
            print('Stimulus not found')
            sys.exit()        
        self.sti_type = sti_type; self.sti_set = sti_set; self.deftotal = deftotal
        return (self.sti_type, self.sti_set, self.deftotal, NoBinned)
    
    def checkepoch(self,thresh_pho):
        sti_type,sti_set,deftotal,NoBinned = self.def_type()
        dt = self.t[1]-self.t[0]
        if any(self.sti_type in x for x in ['Flash','Edges_vel','Edges_vel_vert']):
            self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/10/dt)][1:-1]#[1:-1] 
        #elif any(self.sti_type in x for x in ['Edges_vel']):
        #    self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/10/dt)][:-1]
        else:
            self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/2/dt)]#[:-1]#[1:]
              
        self.tepochs = self.t[self.epoch_idx]
        vepochs = self.pho[self.epoch_idx]     
        if len(self.epoch_idx) != deftotal:
            print('total epoch number is wrong')  
            print(len(self.epoch_idx))
            plt.figure()
            plt.plot(self.t,self.pho)
            plt.title('photodiode')
            for i in self.tepochs: 
                plt.axvline(x=i, linewidth=1, color='g')
            plt.scatter(self.tepochs,vepochs, c='r', s = 50)
            plt.xlabel('time (s)')
            plt.ylabel('V (mV)')
            plt.tight_layout()
            plt.show()
#             if len(self.epoch_idx) - deftotal == 1:
#                 self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/2/dt)][:-1]
#                 print('new total epoch number is %d, by cutting last' %len(self.epoch_idx))
#             elif len(self.epoch_idx) - deftotal == 2:  
#                 self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/2/dt)][1:-1]
#                 print('new total epoch number is %d, by cutting front & last' %len(self.epoch_idx))
#             elif len(self.epoch_idx) - deftotal == -1:
#                 self.epoch_idx = np.argwhere(self.pho>thresh_pho)[1:][np.diff(np.argwhere(self.pho>thresh_pho),axis=0)>(1/2/dt)]
#                 print('new total epoch number is %d, by adding last' %len(self.epoch_idx))
                
        # check epoch length
        plt.figure()
        arr=plt.hist(np.diff(self.tepochs),bins=4)
        if sum(arr[0] == NoBinned) !=4: 
            print('Error: epoch length'); print(arr)            
            plt.hist(np.diff(self.tepochs),bins=4); plt.title('histogram of epopch length')
            plt.show()
        return(self.epoch_idx,self.tepochs)    
    
    def sti_keys(self): 
        sti_type,sti_set,deftotal,NoBinned = self.def_type()
        if any(sti_type in x for x in ['Sq_12dir','Sine_12dir', 'Sq_PvN','Sq_LR_12dir','Oppo']):
            keys = np.arange(2,int((deftotal-1)/2),3)
            if any(sti_type in x for x in ['Sq_12dir','Sine_12dir']): 
                epoch_angle = [0,180,30,210,60,240,90,270,120,300,150,330]#[180, 0, 210, 30, 240, 60, 270, 90, 300, 120, 330, 150]
                epoch_loc = np.zeros(len(epoch_angle))
            elif sti_type == 'Sq_PvN':
                epoch_angle = [0,180]*4
                epoch_loc =  [1,1,0,0,2,2,3,3]*2# Contra is 0, ipsi is 1; full field is 2, mirror is 3. 
            elif sti_type == 'Sq_LR_12dir':
                epoch_angle = [0,180,30,210,60,240,90,270,120,300,150,330,0,180,30,210,60,240,90,270,120,300,150,330] 
                epoch_loc = [1]*12 + [0]*12 
            elif sti_type == 'Oppo':
                epoch_angle = [180,0,360] *3
                epoch_loc = [2,2,2,1,1,1,0,0,0]           
        elif any(sti_type in x for x in ['Sine_Strips', 'Sine_xC','Sine_PvN', 'Sine_PvN_u']):
            keys = np.arange(2,int((deftotal-1)/3),3)            
            if sti_type == 'Sine_Strips': 
                epoch_angle = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9]
                epoch_loc = np.zeros(len(epoch_angle))
            elif sti_type == 'Sine_xC':
                epoch_angle = [0.1,-0.1,0.2,-0.2,0.4,-0.4,0.6,-0.6,0.8,-0.8,1.0,-1.0]
                epoch_loc = np.zeros(len(epoch_angle))
            elif sti_type == 'Sine_PvN': 
                epoch_angle = [0.1,-0.1,0.1,-0.1,0.5,-0.5,0.5,-0.5,1,-1,1,-1,2,-2,2,-2,3,-3,3,-3,5,-5,5,-5,6,-6,6,-6,
                              8,-8,8,-8,15,-15,15,-15,25,-25,25,-25,32,-32,32,-32] # [25,-25,25,-25,]
                epoch_loc = [2,2,3,3]*11 #12 #full field is 2, mirror is 3. 
            elif sti_type == 'Sine_PvN_u': 
                epoch_angle = [0.1,-0.1,0.1,-0.1,1,-1,1,-1,5,-5,5,-5,6,-6,6,-6,
                              7,-7,7,-7,10,-10,10,-10]
                epoch_loc = [2,2,3,3]*6 #12 #full field is 2, mirror is 3. 
        elif any(sti_type in x for x in ['Sine_PDND_x', 'Sine_PDND_y', 'Sine_PDND_z']):   
            keys = np.arange(2,int((deftotal+2)/3),3)
            if sti_type == 'Sine_PDND_x':
                epoch_angle = [0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,
                               4,-4,4.5,-4.5,5,-5,5.5,-5.5,6,-6,6.5,-6.5,7,-7]
            elif sti_type == 'Sine_PDND_y':
                epoch_angle = [0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,
                               5,-5,5.5,-5.5,6,-6,6.5,-6.5,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12]
            elif sti_type ==  'Sine_PDND_z':
                epoch_angle = [0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,10,-10,15,-15,25,-25,32,-32]
                #[0.1,-0.1,0.5,-0.5,1,-1,1.5,-1.5,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,
                              # 10,-10,15,-15,25,-25,32,-32,40,-40,50,-50,63,-63]
            epoch_loc = np.zeros(len(epoch_angle))
        elif any(sti_type in x for x in ['Sine_ACC', 'Sine_ACC_y']):
            keys = np.arange(2,int((deftotal+2)/3),5)
            if sti_type == 'Sine_ACC':
                epoch_angle = [1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,5,-5,6,-6,8,-8]
            elif sti_type == 'Sine_ACC_y':
                epoch_angle = [1,-1,1.5,-1.5,2,-2,2.5,-2.5,3,-3,3.5,-3.5,4,-4,4.5,-4.5,
                               5,-5,6,-6,8,-8,9,-9,10,-10,11,-11,12,-12]
            epoch_loc = np.zeros(len(epoch_angle))         
        elif sti_type == 'Sine_patches':
            keys = np.arange(2,int((deftotal-1)/3),3) 
            if any(self.stimulus_name in x for x in [sti_set[0], sti_set[4], sti_set[8], sti_set[12], sti_set[16]]):
                epoch_angle = [0]*14 +[180]*14
            elif any(self.stimulus_name in x for x in [sti_set[1], sti_set[5], sti_set[9], sti_set[13], sti_set[17]]): # 'Sine_patches_UPDN_1HZ'
                epoch_angle = [90]*14 +[270]*14
            elif any(self.stimulus_name in x for x in [sti_set[2], sti_set[6], sti_set[10], sti_set[14], sti_set[18]]): # 'Sine_patches_RULD_1HZ'
                epoch_angle = [45]*14 +[225]*14
            elif any(self.stimulus_name in x for x in [sti_set[3], sti_set[7], sti_set[11], sti_set[15], sti_set[19]]): # 'Sine_patches_LURD_1HZ'
                epoch_angle = [135]*14 +[315]*14                
            epoch_loc = [2,3,4,5,12,13,14,15,21,22,23,24,25,26]*2            
        elif sti_type == 'Sine_patches2':
            keys = np.arange(2,int((deftotal-1)/3),3) 
            if any(self.stimulus_name in x for x in [sti_set[0], sti_set[4], sti_set[8]]):
                epoch_angle = [0]*10 +[180]*10
            elif any(self.stimulus_name in x for x in [sti_set[1], sti_set[5],sti_set[9]]): # 'Sine_patches2_UPDN_6HZ'
                epoch_angle = [90]*10 +[270]*10
            elif any(self.stimulus_name in x for x in [sti_set[2], sti_set[6],  sti_set[10]]): # 'Sine_patches2_RULD_6HZ'
                epoch_angle = [45]*10 +[225]*10
            elif any(self.stimulus_name in x for x in [sti_set[3], sti_set[7], sti_set[11]]): # 'Sine_patches2_LURD_6HZ'
                epoch_angle = [135]*10 +[315]*10               
            epoch_loc = [1,2,3,4,5,11,12,13,14,15]*2  
            
        self.epoch_angle = epoch_angle; self.epoch_loc = epoch_loc
        return(keys,self.epoch_angle,self.epoch_loc)        
    
    def findepochnames(self):
        sti_type,sti_set,deftotal,NoBinned = self.def_type()
        df = pd.read_csv(self.csvpath, usecols =['FrameNumber'])
        # if "epoch no.1" got mixed in: 
        epochNo = df.values[df.values!=1]
        epoch_label = epochNo[:-1][np.diff(epochNo, axis = 0) !=0]    
        #epoch_label = df.values[1:-1][np.diff(df.values, axis = 0) != 0
        if any(sti_type in x for x in ['Edges_vel','Edges_vel_vert','Flash']): 
            epochNo =df.values; epoch_label = epochNo[:-1][np.diff(epochNo, axis = 0) !=0]
            epoch_label = epoch_label[epoch_label!=1]  
            if sti_type == 'Edges_vel': 
                len_label = 42; keys = np.arange(2,16)
                epoch_angle = [12.5,50,100,200,300,500,900,-12.5,-50,-100,-200,-300,-500,-900] 
                epoch_loc = [1296,324,162,81,54,32.4,18,-1296,-324,-162,-81,-54,-32.4,-18]  
            elif sti_type == 'Edges_vel_vert': 
                len_label = 42; keys = np.arange(2,16)
                epoch_angle = [12.5,50,100,200,300,500,900,-12.5,-50,-100,-200,-300,-500,-900] 
                epoch_loc = [864,216,108,54,36,21.6,12,-864,-216,-108,-54,-36,-21.6,-12]  
            elif sti_type == 'Flash':
                len_label = 21; keys = np.arange(2,9)
                epoch_angle = [60,120,180,240,300,360,420]
                epoch_loc = np.zeros(len(epoch_angle))
            if len(epoch_label) != len_label:
                print("error: No. of epoch label")
                print(len(epoch_label))  
            Dic = dict(zip(keys, epoch_angle))
            angles = [Dic[i] for i in epoch_label]
            Dic_loc =  dict(zip(keys, epoch_loc))
            locs = [Dic_loc[i] for i in epoch_label]  
        else:

            if len(epoch_label) == deftotal:
                epoch_label = epoch_label[:-1]
            else:
                print("error: No. of epoch label")

            if any(sti_type in x for x in ['Sine_ACC', 'Sine_ACC_y']):
                RS_size = (int((deftotal-1)/5),5)
            else: 
                RS_size = (int((deftotal-1)/3),3)
             
            keys,epoch_angle,epoch_loc = self.sti_keys()
            Dic = dict(zip(keys, epoch_angle))
            angles = [Dic[i] for i in epoch_label.reshape(RS_size)[:,0]]
            sDic = dict(zip(keys, epoch_loc))
            locs = [sDic[i] for i in epoch_label.reshape(RS_size)[:,0]]            

        self.angles = angles; self.locs = locs  
        return(self.angles,self.locs)        
    
    def save_eps_ACC(self): # save the epochs of 'ACC' in df.csv. 
        # translate variables
        tepochs = self.tepochs; spiking = self.spiking; epoch_idx = self.epoch_idx; 
        spike_idx = self.spike_idx; t = self.t;v = self.v; pho = self.pho; 
        stimulus= self.stimulus_name; csvpath = self.csvpath; fdname = self.fdname; 
        recN = self. recN;sti_type = self.sti_type;
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
        angles,locs = self.findepochnames()
        angles = angles[:sshape]; locs = locs[:sshape]

        # create dfs
        Vdf = makedfs(angles,locs,v_epoch).sort_values(by=['Locs', 'Angles'])
        preVdf = makedfs(angles,locs,v_pre).sort_values(by=['Locs', 'Angles'])
        accVdf = makedfs(angles,locs,v_acc).sort_values(by=['Locs', 'Angles'])
        movVdf = makedfs(angles,locs,v_mov).sort_values(by=['Locs', 'Angles'])
        dccVdf = makedfs(angles,locs,v_dcc).sort_values(by=['Locs', 'Angles'])
        postVdf = makedfs(angles,locs,v_post).sort_values(by=['Locs', 'Angles'])        
        Vidx = Vdf.index.values
        
        Phodf = makedfs(angles,locs,pho_epoch).sort_values(by=['Locs', 'Angles'])
        prePhodf = makedfs(angles,locs,pho_pre).sort_values(by=['Locs', 'Angles'])
        accPhodf = makedfs(angles,locs,pho_acc).sort_values(by=['Locs', 'Angles'])
        movPhodf = makedfs(angles,locs,pho_mov).sort_values(by=['Locs', 'Angles'])
        dccPhodf = makedfs(angles,locs,pho_dcc).sort_values(by=['Locs', 'Angles'])
        postPhodf = makedfs(angles,locs,pho_post).sort_values(by=['Locs', 'Angles'])

        if spiking:
            Apdf = makedfs(angles,locs,Ap_epoch).sort_values(by=['Locs', 'Angles'])
            preApdf = makedfs(angles,locs,Ap_pre).sort_values(by=['Locs', 'Angles'])
            accApdf = makedfs(angles,locs,Ap_acc).sort_values(by=['Locs', 'Angles'])
            movApdf = makedfs(angles,locs,Ap_mov).sort_values(by=['Locs', 'Angles'])
            dccApdf = makedfs(angles,locs,Ap_dcc).sort_values(by=['Locs', 'Angles'])
            postApdf = makedfs(angles,locs,Ap_post).sort_values(by=['Locs', 'Angles'])
            Apidx = Apdf.index.values
            
        dir_data = 'Analysis/%s'%fdname; makedir(dir_data)       

        #save dfs
        Vdata = {'Vdf' : Vdf, 'preVdf': preVdf, 'accVdf': accVdf, 'movVdf': movVdf, 'dccVdf': dccVdf, 'postVdf':postVdf, 'Vidx':Vidx}
        Phodata = {'Phodf' : Phodf, 'prePhodf': prePhodf, 'accPhodf': accPhodf, 'movPhodf':movPhodf,
        'dccPhodf': dccPhodf, 'postPhodf': postPhodf}        
        ds = {**self.__dict__,**Vdata, **Phodata}         
        if spiking:
            #save dfs
            Apdata = {'Apdf' : Apdf, 'preApdf': preApdf, 'accApdf': accApdf, 'movApdf': movApdf, 
            'dccApdf': dccApdf, 'postApdf' : postApdf, 'Apidx':Apidx}
            ds = {**ds, **Apdata}            
        with open(dir_data+'/rec%dRaw.pkl'%recN,'wb') as f:
            pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
        print('all saved!')               
        return(ds,Vdf,Apdf)
    
    def save_eps_Edges(self):
    # for "Edges_light_vel"
        # translate variables
        tepochs = self.tepochs; spiking = self.spiking; epoch_idx = self.epoch_idx; 
        spike_idx = self.spike_idx; t = self.t;v = self.v; pho = self.pho; 
        stimulus= self.stimulus_name; csvpath = self.csvpath; fdname = self.fdname; 
        recN = self. recN;sti_type = self.sti_type;
        
        start_idx = epoch_idx[:-1].reshape(int((len(epoch_idx)-1)/2),2)[:,0]
        end_idx = np.hstack([start_idx[1:],epoch_idx[-1]])
        angles,locs = self.findepochnames()
        APtrain = np.zeros(len(v)); APtrain[spike_idx] = 1
        
        Eps = []; Aps = []; 
        for i in np.arange(len(angles)):
            ep = v[start_idx[i]:end_idx[i]]
            ap = APtrain[start_idx[i]:end_idx[i]]    
            Eps.append(ep); Aps.append(ap);
        # pad shorter epochs with np.nan to make it uniform
        v_epoch,pad_idx = padding(Eps); Ap_epoch,pad_idx = padding(Aps);
        dt = t[1] - t[0]; T = dt*np.shape(Ap_epoch)[1]
        if sti_type == 'Flash':
            onidx = int(2/dt)
        else:
            onidx = int(1/dt)            
        # get pre/bl & mov # post
        v_pre = v_epoch[:,int(onidx/2): onidx]; v_mov = v_epoch[:,onidx:]
        v_post = np.vstack((v_epoch[1:,:int(onidx/2)],np.zeros(int(onidx/2))));
        
        Ap_pre = Ap_epoch[:,int(onidx/2): onidx]; Ap_mov = Ap_epoch[:,onidx:]
        Ap_post = np.vstack((Ap_epoch[1:,:int(onidx/2)],np.zeros(int(onidx/2))));
        
        for n,i in enumerate(pad_idx):
            if i != np.shape(v_epoch)[1]:
                pad = np.empty((np.shape(v_epoch)[0], np.shape(v_epoch)[1]-i)); pad[:] = np.nan
                v_epoch[n,:] = np.hstack((v_pre[n,:],v_epoch[n,onidx:i],v_post[n,:],pad[n,:]))
                Ap_epoch[n,:] = np.hstack((Ap_pre[n,:],Ap_epoch[n,onidx:i],Ap_post[n,:],pad[n,:]))
            else:
                v_epoch[n,:] = np.hstack((v_pre[n,:],v_epoch[n,onidx:i],v_post[n,:]))
                Ap_epoch[n,:] = np.hstack((Ap_pre[n,:],Ap_epoch[n,onidx:i],Ap_post[n,:]))

        dur = (np.asarray(pad_idx) - onidx)*dt
        return(v_epoch,v_pre,v_mov,v_post, Ap_epoch, Ap_pre,Ap_mov, Ap_post, T, dur)    
    
    def save_eps(self,N, Wn,bin_size): # master of saving epochs, coordinate btw different stimuli. 
        # translate variables
        tepochs = self.tepochs; spiking = self.spiking; epoch_idx = self.epoch_idx; 
        spike_idx = self.spike_idx; t = self.t;v = self.v; pho = self.pho; 
        stimulus= self.stimulus_name; csvpath = self.csvpath; fdname = self.fdname; 
        recN = self. recN;sti_type = self.sti_type;        
              
        tepochs_idx =  np.array([int(np.argwhere(t==i)) for i in tepochs])
        if any(sti_type in x for x in ['Sine_ACC', 'Sine_ACC_y']):
            ds,Vdf,Apdf = self.save_eps_ACC(); T = 6        
        else:
            angles,locs = self.findepochnames()
            if any(sti_type in x for x in ['Edges_vel','Edges_vel_vert', 'Flash']):
                v_epoch,v_pre,v_mov,v_post, Ap_epoch, Ap_pre,Ap_mov, Ap_post, T, dur = self.save_eps_Edges()
            
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
                    print ('Error: mismatch btw pre & post idx length')
                if mov_idx_len/pre_idx_len != 2.0: # 2.0 for local; rest 4. 
                    print ('Error: mismatch btw mov & pre idx length')

                # splitting into epochs, or sections of epochs
                v_pre,v_mov, v_post, v_epoch =  epochsplit(start_idxes,len(epoch_idx),v, pre_idx_len, mov_idx_len, post_idx_len,total_idx_len)

                if spiking: 
                    APtrain = np.zeros(len(v))
                    APtrain[spike_idx] = 1
                    Ap_pre,Ap_mov, Ap_post, Ap_epoch =epochsplit(start_idxes,len(epoch_idx),APtrain,pre_idx_len, mov_idx_len, 
                                                                 post_idx_len,total_idx_len)
                # get angles  
                angles = angles[:sshape]; locs = locs[:sshape]; T = 4         
            
            dir_pics = 'Pics/%s/%d'%(fdname,recN); makedir(dir_pics)
            # create dfs
            Vdf = makedfs(angles,locs,v_epoch).sort_values(by=['Locs', 'Angles'])
            preVdf = makedfs(angles,locs,v_pre).sort_values(by=['Locs', 'Angles'])
            movVdf = makedfs(angles,locs,v_mov).sort_values(by=['Locs', 'Angles']) 
            postVdf = makedfs(angles,locs,v_post).sort_values(by=['Locs', 'Angles'])
            Vidx = Vdf.index.values
            if any(sti_type in x for x in ['Edges_vel', 'Edges_vel_vert', 'Flash']):
                Vmfftdf = None            
            else:    
                Vmfftdf = fftVm(movVdf,T,fdname,recN, N, Wn,Vidx)

            if spiking:
                Apdf = makedfs(angles,locs,Ap_epoch).sort_values(by=['Locs', 'Angles'])
                preApdf = makedfs(angles,locs,Ap_pre).sort_values(by=['Locs', 'Angles'])
                movApdf = makedfs(angles,locs,Ap_mov).sort_values(by=['Locs', 'Angles'])
                postApdf = makedfs(angles,locs,Ap_post).sort_values(by=['Locs', 'Angles'])
                Apidx = Apdf.index.values                
                if any(sti_type in x for x in ['Edges_vel', 'Edges_vel_vert', 'Flash']):
                    APfftdf = None
                else:
                    APfftdf = fftAP(movApdf,T,fdname,recN,Apidx)
            #save dfs
            Vdata = {'Vdf' : Vdf, 'preVdf': preVdf, 'movVdf': movVdf, 'postVdf' : postVdf, 'Vidx':Vidx, 'Vmfftdf': Vmfftdf}
            
            ds = {**self.__dict__,**Vdata, **{'T':T}}      
            if spiking:
                #save dfs
                Apdata = {'Apdf' : Apdf, 'preApdf': preApdf, 'movApdf' : movApdf, 'postApdf' : postApdf, 'Apidx':Apidx, 
                          'APfftdf': APfftdf}
                ds = {**ds, **Apdata} 
            else: 
                Apdf = None        
            if any(sti_type in x for x in ['Edges_vel', 'Edges_vel_vert', 'Flash']):
                # Tdf contains duration of 'mov'
                Tdf = makedfs(angles,locs,dur).sort_values(by=['Locs', 'Angles'])
                ds = {**ds, **{'Tdf': Tdf}} 
            
            dir_data = 'Analysis/%s'%fdname; makedir(dir_data)
            # save info on filter used
            ds.update({'filter':{'N':N,'Wn':Wn },'bin_size': bin_size})
            with open(dir_data+'/rec%dRaw.pkl'%recN,'wb') as f:
                    pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
                    print('all saved!')
        return(Vdf,Apdf,T)
    
    def checkFilV(self,n, N, Wn,Apdf,Vdf,T):
        # experiment with filter
        plt.figure()        
        t = np.linspace(0,T,np.shape(Vdf)[1]-2); V = Vdf.iloc[n][2:].values;             
        plt.plot(t,V,'r')
        if self.spiking:
            Ap = Apdf.iloc[n][2:]
            tAP = t[Ap.astype(np.bool)]; vAP = V[Ap.astype(np.bool)]
            plt.scatter(tAP,vAP); 
        plt.plot(t,FiltV(Vdf.fillna(0).iloc[n][2:], N, Wn),'g')
        plt.show()
        print('Vdf'); print(Vdf.sort_values(by=['Locs', 'Angles']))        
        
    
    def plotRAW(self,plot,bin_size,N,Wn,Apdf,Vdf,T):
        fdname = self.fdname; recN = self.recN;  spiking = self.spiking
        V = Vdf.iloc[:,2:].values; t = np.linspace(0,T,np.shape(V)[1])     
        if Apdf != None:
             AP = Apdf.iloc[:,2:].values
        
        figdir = 'Pics/'+ fdname +'/' +str(recN); makedir(figdir)    
        if plot:   
            for n in np.arange(0,V.shape[0]):
#                if any(self.sti_type in x for x in ['Edges_vel', 'Flash']): t = T[n] 
                # plot voltage
                plt.close(); plt.figure();plt.plot(t,V[n]); APtotal = 0
                if spiking: # plot AP scatter                    
                    tAP = t[AP[n].astype(np.bool)]; vAP = V[n][AP[n].astype(np.bool)]
                    plt.scatter(tAP,vAP,color = 'r'); APtotal= np.nansum(AP[n].astype(np.int64))
                # plot FiltV & any vertical lines
                plt.plot(t,FiltV(Vdf.fillna(0).iloc[n][2:],N,Wn),'g')
                if any(self.sti_type in x for x in ['Edges_vel','Edges_vel_vert']):
                    plt.axvline(x=0.5,linewidth=2, color='k')
                else:
                    plt.axvline(x=1,linewidth=2, color='k')
                if any(self.sti_type in x for x in ['Flash','Edges_vel','Edges_vel_vert']) !=True:
                    plt.axvline(x=0.5,linewidth=1, ls = ':', color='k') 
                    plt.axvline(x=(T-1),linewidth=2, color='k'); plt.axvline(x=(T-0.5),linewidth=1, ls = ':', color='k')     
                    if T==6: # for ACC etc. 
                        plt.axvline(x=2.0,linewidth=2, color='k'); plt.axvline(x=4.0,linewidth=2, color='k') 
                # save fig   
                plt.xlim(right = np.ceil(max(t)))
                plt.savefig('%s/wave%drec%dangle%0.1floc%dAP%d.png' %(figdir, recN, Vdf.index[n], Vdf.iloc[n,0], Vdf.iloc[n,1], APtotal))
                plt.close()
                
            if spiking: 
                if  bin_size != 0:            
                    # binning the AP 
#                     if any(self.sti_type in x for x in ['Edges_vel', 'Flash']): 
#                         bins = np.arange(0, max([(len(i)) for i in AP]), int(round(bin_size/(t[1]-t[0]))))
#                         pad = np.zeros((len(AP),max([(len(i)) for i in AP])))
#                         for n,j in enumerate(AP): pad[n,:len(j)] = j
#                         AP = pad; T = max([max(i) for i in T])
#                     else:
                    bins = np.arange(0, np.shape(AP)[1],int(round(bin_size/(t[1]-t[0]))))        
                    AP_bin= np.asarray([np.sum(AP[:,bins[i]:bins[i+1]],axis =1) for i in np.arange(np.size(bins)-1)]).transpose()
                    plt.figure(figsize=[12,12]); plt.imshow(AP_bin); plt.colorbar()
                    xticks = np.arange(0,np.shape(AP_bin)[1],int(1/bin_size))
                    xlabels = np.arange(0,np.shape(AP_bin)[1]/int(1/bin_size))
                    yticks = np.arange(0,np.shape(AP_bin)[0],int(T-1))
                    if any(self.sti_type in x for x in ['Edges_vel','Edges_vel_vert']):
                        yticks = np.arange(0,np.shape(AP_bin)[0],3)
                    if self.sti_type == 'Sq_PvN':
                        yticks = np.arange(0,np.shape(AP_bin)[0],2)
                    ylabels = Apdf.iloc[yticks,:2].values
                    plt.xticks(xticks, xlabels)
                    
                        
                    plt.yticks(yticks-0.5, ylabels, rotation = 0)
                    for i in yticks:
                        plt.axhline(y=i-0.5,linewidth=1, ls = ':',color='k')   
                    if any(self.sti_type in x for x in ['Edges_vel','Edges_vel_vert']):
                        plt.axvline(x=0.5/bin_size,ls = ':',linewidth=2, color='k')
                    elif self.sti_type == 'Flash':
                        plt.axvline(x=1.0/bin_size,ls = ':',linewidth=2, color='k')      
                    else: 
                        plt.axvline(x=1.0/bin_size,linewidth=2, color='k')
                        plt.axvline(x=0.5/bin_size,linewidth=1, ls = ':', color='k') 
                        plt.axvline(x=(T-1)/bin_size,linewidth=2, color='k')
                        plt.axvline(x=(T-0.5)/bin_size,linewidth=1, ls = ':', color='k')                    
                        if T==6: # for ACC etc. 
                            plt.axvline(x=2.0/bin_size,linewidth=2, color='k') 
                            plt.axvline(x=4.0/bin_size,linewidth=2, color='k')
                    plt.savefig('%s/wave%dRaster.jpeg' %(figdir, recN))  
                    plt.close()
                    
#####################
# code for avg
####################

def AngleGp(V_al):
## Return index of the last trace for a particular angle
# takes in Vdf.values[:,:2], the first two columns of Vdf. 
    dirn = [] 
    for n,i in enumerate(V_al):
        if sum(i != V_al[n-1]): # if not the same as previous angle
            dirn.append(n)
    dirn = dirn[1:];dirn.append(len(V_al))
    return(dirn)

def twoVorAP(dirn, APmov0, APmov1, APmov2): 
    APmov0mean = np.zeros(len(dirn)); APmov0std = np.zeros(len(dirn))
    APmov1mean = np.zeros(len(dirn)); APmov1std = np.zeros(len(dirn)) 
    APmov2mean = np.zeros(len(dirn)); APmov2std = np.zeros(len(dirn)) 
    for n, i in enumerate(dirn):        
        if n == 0: 
            APmov0mean[n] = np.mean(APmov0[:dirn[n]]); APmov0std[n] = np.std(APmov0[:dirn[n]])
            APmov1mean[n] = np.mean(APmov1[:dirn[n]]); APmov1std[n] = np.std(APmov1[:dirn[n]])
            APmov2mean[n] = np.mean(APmov2[:dirn[n]]); APmov2std[n] = np.std(APmov2[:dirn[n]])
        else:
            APmov0mean[n] = np.mean(APmov0[dirn[n-1]:dirn[n]]); APmov0std[n] = np.std(APmov0[dirn[n-1]:dirn[n]])
            APmov1mean[n] = np.mean(APmov1[dirn[n-1]:dirn[n]]); APmov1std[n] = np.std(APmov1[dirn[n-1]:dirn[n]]) 
            APmov2mean[n] = np.mean(APmov2[dirn[n-1]:dirn[n]]); APmov2std[n] = np.std(APmov2[dirn[n-1]:dirn[n]]) 
    return(APmov0mean/0.5,APmov0std/0.5,APmov1mean,APmov1std,APmov2mean,APmov2std)

def twoVMeanStd(dirn,Vmov_ms): # Vmov0 is the inhitial 0.5s. Vmov1, the 1st s. Vmov2, 2nd s.
    # compare 1st & 2nd seconds of the recording
    # compute length of half pre & half post
    hfmov = int(np.shape(Vmov_ms)[1]/2)
    Vmov0 = Vmov_ms[:,:int(hfmov/2)]; Vmov1 = Vmov_ms[:,:hfmov]; Vmov2 = Vmov_ms[:,hfmov:]
    Vmov0sum = np.mean(Vmov0,axis = 1); Vmov1sum = np.mean(Vmov1,axis = 1); Vmov2sum = np.mean(Vmov2,axis = 1)   
    Vmov0mean,Vmov0std, Vmov1mean,Vmov1std,Vmov2mean,Vmov2std = twoVorAP(dirn,Vmov0,Vmov1, Vmov2)
    return(Vmov0mean,Vmov0std,Vmov1mean,Vmov1std,Vmov2mean,Vmov2std)

def twoAPMeanStd(dirn,APmov): # APmov0 is the inhitial 0.5s. APmov1, the 1st s. APmov2, 2nd s.
    # compute length of halfmov
    hfmov = int(np.shape(APmov)[1]/2)
    APmov0 = np.sum(APmov[:,:int(hfmov/2)],axis =1).astype(np.int64)
    APmov1 = np.sum(APmov[:,:hfmov],axis =1).astype(np.int64)
    APmov2 = np.sum(APmov[:,hfmov:],axis =1).astype(np.int64)    
    APmov0mean,APmov0std,APmov1mean,APmov1std,APmov2mean,APmov2std = twoVorAP(dirn,APmov0,APmov1, APmov2)
    return(APmov0mean,APmov0std,APmov1mean,APmov1std,APmov2mean,APmov2std)

class AVG(): # stimulus_name,fdname,recN,spiking,csvroot,csvpath,
    def __init__(self,fdname,recN):
        self.fdname = fdname
        self.recN = recN    
        
    def loadRAW(self):      
        fdname= self.fdname; recN = self.recN;    
        with open('Analysis/%s'%fdname+'/rec%dRaw.Pkl'%recN, 'rb') as f:
            ds = pickle.load(f)    
        self.ds = ds
        return(self.ds)
 
    def VMeanStd(self,blmode,plot):
        ds = self.loadRAW()
        Vdf = ds['Vdf']; preVdf = ds['preVdf']; postVdf = ds['postVdf']; movVdf = ds['movVdf']; Vmfftdf = ds['Vmfftdf']
        
        dirn = AngleGp(Vdf.values[:,:2])
        dt = ds['t'][1]-ds['t'][0]        
       
        V = Vdf.values[:,2:]; Vpre = preVdf.values[:,2:]; Vmov = movVdf.values[:,2:]; Vpost = postVdf.values[:,2:]
        if ds['spiking']: 
            N = ds['filter']['N']; Wn = ds['filter']['Wn']
            if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']) != True:
                V = FiltV(V, N, Wn); Vpre = FiltV(Vpre, N, Wn); Vmov = FiltV(Vmov, N, Wn); Vpost = FiltV(Vpost, N, Wn)
            else: 
                V = FiltV(Vdf.fillna(0).iloc[:,2:], N, Wn); Vpre = FiltV(preVdf.fillna(0).iloc[:,2:], N, Wn); 
                Vmov = FiltV(movVdf.fillna(0).iloc[:,2:], N, Wn); Vpost = FiltV(postVdf.fillna(0).iloc[:,2:], N, Wn)
                V[Vdf.iloc[:,2:].isnull().values] = np.nan; Vpre[preVdf.iloc[:,2:].isnull().values] = np.nan; 
                Vmov[movVdf.iloc[:,2:].isnull().values] = np.nan; Vpost[postVdf.iloc[:,2:].isnull().values] = np.nan    

        if ds['sti_type'] == ['Edges_vel','Edges_vel_vert']:
            hfpre = int(np.shape(preVdf.iloc[:,2:])[1]); hfpost = int(np.shape(postVdf.iloc[:,2:])[1])
        else:
            hfpre = int(np.shape(preVdf.iloc[:,2:])[1]/2); hfpost = int(np.shape(postVdf.iloc[:,2:])[1]/2)
        # 0.5s before for Vpre. 0.5s after for Vpost    
        Vpre = Vpre[:,hfpre:]; Vpost = Vpost[:,:hfpost] 
            
        # calculate baseline based on 0.5s before stimulus. 
        bltime= int(0.5/dt); blpre = Vpre[:,-bltime:];         
        if blmode == 'indi': 
            bl = np.mean(blpre,axis=1) #bl contains baseline for individual epochs
        else: 
            bl = np.mean(np.mean(blpre,axis=1))
            bl = np.ones(Vpre.shape[0]) *bl

        V_ms = np.vstack([V[i,:] - bl[i] for i in range(np.shape(V)[0])])
        Vmov_ms = np.vstack([Vmov[i,:] - bl[i] for i in range(np.shape(Vmov)[0])])
        Vpost_ms = np.vstack([Vpost[i,:bltime] - bl[i] for i in range(np.shape(Vpost)[0])])
        Vmovsum = np.nanmean(Vmov_ms,axis = 1); Vpostsum = np.nanmean(Vpost_ms,axis = 1)
        if Vmfftdf is not None:
            Vmfft = Vmfftdf.values[:,2:]
        
        if ds['sti_type']== 'ACC':
            accVdf = ds['accVdf']; dccVdf = ds['dccVdf'];
            Vacc = accVdf.values[:,2:]; Vdcc = dccVdf.values[:,2:]
            if ds['spiking']:
                Vacc = FiltV(Vacc, N, Wn); Vdcc = FiltV(Vdcc, N, Wn);
            Vacc_ms = np.vstack([Vacc[i,:bltime] - bl[i] for i in range(np.shape(Vacc)[0])])
            Vdcc_ms = np.vstack([Vdcc[i,:bltime] - bl[i] for i in range(np.shape(Vdcc)[0])])
            Vaccsum = np.mean(Vacc_ms,axis = 1); Vdccsum = np.mean(Vdcc_ms,axis = 1)
            Vaccmean =  np.zeros(len(dirn)); Vaccstd =  np.zeros(len(dirn)); 
            Vdccmean =  np.zeros(len(dirn)); Vdccstd =  np.zeros(len(dirn));
            
        T = ds['T']
        Vmovmean = np.zeros(len(dirn)); Vmovstd = np.zeros(len(dirn)) 
        Vpostmean = np.zeros(len(dirn)); Vpoststd = np.zeros(len(dirn)) 
        blmean = np.zeros(len(dirn)); blstd = np.zeros(len(dirn))
        if Vmfftdf is not None:
            Vfftmean = np.zeros((len(dirn),np.shape(Vmfft)[1])); Vfftstd = np.zeros((len(dirn),np.shape(Vmfft)[1])) 
        
        ep_names = Vdf.values[np.asarray(dirn)-1,:2]
        
        if plot == True: 
            fig1, ax1 = plt.subplots()
            t = np.linspace(0,T,np.shape(V)[1])
        for n in np.arange(0,len(dirn)):   
            dirn_idx = dirn[n]-1
            if n == 0: 
                V0 = V_ms[:dirn[n],:]; 
                Vmovmean[n] = np.mean(Vmovsum[:dirn[n]]); Vmovstd[n] = np.std(Vmovsum[:dirn[n]])
                Vpostmean[n] = np.mean(Vpostsum[:dirn[n]]); Vpoststd[n] = np.std(Vpostsum[:dirn[n]])
                blmean[n] = np.mean(bl[:dirn[n]]); blstd[n] = np.std(bl[:dirn[n]])
                if Vmfftdf is not None:
                    Vfftmean[n] = np.mean(Vmfft[:dirn[n],:], axis = 0); Vfftstd[n] = np.std(Vmfft[:dirn[n],:], axis = 0)
                if ds['sti_type']== 'ACC':
                    Vaccmean[n] = np.mean(Vaccsum[:dirn[n]]); Vaccstd[n] = np.std(Vaccsum[:dirn[n]]);
                    Vdccmean[n] = np.mean(Vdccsum[:dirn[n]]); Vdccstd[n] = np.std(Vdccsum[:dirn[n]]);
            else:
                V0 = V_ms[dirn[n-1]:dirn[n],:]; 
                Vmovmean[n] = np.mean(Vmovsum[dirn[n-1]:dirn[n]]); Vmovstd[n] = np.std(Vmovsum[dirn[n-1]:dirn[n]]) 
                Vpostmean[n] = np.mean(Vpostsum[dirn[n-1]:dirn[n]]); Vpoststd[n] = np.std(Vpostsum[dirn[n-1]:dirn[n]])
                blmean[n] = np.mean(bl[dirn[n-1]:dirn[n]]); blstd[n] = np.std(bl[dirn[n-1]:dirn[n]])
                if Vmfftdf is not None:
                    Vfftmean[n] = np.mean(Vmfft[dirn[n-1]:dirn[n],:],axis = 0); Vfftstd[n] = np.std(Vmfft[dirn[n-1]:dirn[n],:],axis = 0) 
                if ds['sti_type']== 'ACC':
                    Vaccmean[n] = np.mean(Vaccsum[dirn[n-1]:dirn[n]]); Vaccstd[n] = np.std(Vaccsum[dirn[n-1]:dirn[n]]);
                    Vdccmean[n] = np.mean(Vdccsum[dirn[n-1]:dirn[n]]); Vdccstd[n] = np.std(Vdccsum[dirn[n-1]:dirn[n]]);
            V0mean = np.mean(V0, axis = 0)
            V0std = np.std(V0, axis = 0)
 
            if plot == True: 
                fig, ax = plt.subplots()
                for j in np.arange(0,np.shape(V0)[0]): 
                    plt.plot(t,V0[j],':')
                plt.plot(t,V0mean,'k')
                plt.xlabel('t(s)'); plt.ylabel('v(mv)')
                plt.axvline(x=1.0,linewidth=2, color='k')
                plt.axvline(x=0.5,linewidth=1, ls = ':', color='k')
                if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']) != True:
                    plt.axvline(x=(T-1),linewidth=2, color='k')
                    plt.axvline(x=(T-0.5),linewidth=1, ls = ':', color='k')                    
                    if T==6: # for ACC etc. 
                        plt.axvline(x=2.0,linewidth=2, color='k') 
                        plt.axvline(x=4.0,linewidth=2, color='k')          
                plt.title((Vdf.values[dirn_idx,:2],'n=%d'%(np.shape(V0)[0]))) 
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                plt.xlim(right = T)
                fig.savefig('Pics/%s/%d/fig%0.1f,%0.1f.png' %(self.fdname, self.recN, Vdf.values[dirn_idx,0],Vdf.values[dirn_idx,1]))
                ax1.plot(t,V0mean,':')
                plt.close()
        if plot == True:             
            ax1.legend(ep_names,loc  = 0, prop={'size': 8})
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.xaxis.set_ticks_position('bottom')
            ax1.yaxis.set_ticks_position('left') 
            fig1.savefig('Pics/%s/%d/mean' %(self.fdname, self.recN,))
        plt.close()
        if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']):
            Vmov0mean,Vmov0std,Vmov1mean,Vmov1std,Vmov2mean,Vmov2std = [None]*6
        else:
            Vmov0mean,Vmov0std,Vmov1mean,Vmov1std,Vmov2mean,Vmov2std = twoVMeanStd(dirn,Vmov_ms)
        dmean = {'Vmovmean':Vmovmean ,'Vpostmean':Vpostmean,'blmean': blmean,
                 'Vmov0mean':Vmov0mean,'Vmov1mean':Vmov1mean,'Vmov2mean':Vmov2mean} 
        dstd = {'Vmovstd':Vmovstd,'Vpoststd':Vpoststd,'blstd': blstd,'Vmov0std':Vmov0std,'Vmov1std':Vmov1std,'Vmov2std':Vmov2std}
        if ds['sti_type']== 'ACC':
            dmean = {**dmean, **{'Vaccmean':Vaccmean,'Vdccmean': Vdccmean}}
            dstd = {**dstd, **{'Vaccstd':Vaccstd,  'Vdccstd':Vdccstd}}        
        dmeandf = pd.DataFrame.from_dict(dmean);dstddf = pd.DataFrame.from_dict(dstd) 
        if Vmfftdf is not None:
            Vfftmeandf = pd.DataFrame(Vfftmean, columns = Vmfftdf.columns[2:])
            Vfftstddf = pd.DataFrame(Vfftstd, columns = Vmfftdf.columns[2:])
        else:
            Vfftmeandf = None; Vfftstddf = None
        self.Vavg = {'ep_names':ep_names, 'T':T, 'blmode': blmode,'sti_type':ds['sti_type'],'dmeandf' : dmeandf, 
                     'dstddf': dstddf,'fftmeandf': Vfftmeandf, 'fftstddf': Vfftstddf}
        with open('Analysis/%s'%self.fdname+'/rec%dVavg.pkl'%self.recN,'wb') as f:
            pickle.dump(self.Vavg, f, pickle.HIGHEST_PROTOCOL)
            print('all saved!')
        return(self.Vavg)
        
    def ApMeanStd(self):
        ds = self.loadRAW(); T = ds['T']      
        Apdf = ds['Apdf']; preApdf = ds['preApdf']; postApdf = ds['postApdf']; movApdf = ds['movApdf']; Apfftdf = ds['APfftdf']
        
        dirn = AngleGp(Apdf.values[:,:2]); 
        
         # compute length of half pre & half post
        if ds['sti_type'] == ['Edges_vel','Edges_vel_vert']:
            hfpre = int(np.shape(preApdf.iloc[:,2:])[1]); hfpost = int(np.shape(postApdf.iloc[:,2:])[1])
        else:
            hfpre = int(np.shape(preApdf.iloc[:,2:])[1]/2); hfpost = int(np.shape(postApdf.iloc[:,2:])[1]/2)
        
        APtotal = np.sum(Apdf.iloc[:,2:],axis =1).astype(np.int64)
        APpre = np.sum(preApdf.iloc[:,2+hfpre:],axis =1).astype(np.int64) # 0.5s before
        APmov = np.sum(movApdf.iloc[:,2:],axis =1).astype(np.int64)
        APpost = np.sum(postApdf.iloc[:,2:hfpost+2],axis =1).astype(np.int64) # 0.5s after
        if Apfftdf is not None:
            Apfft = Apfftdf.fillna(0).values[:,2:]
        if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']):
            APtotal =  np.divide(APtotal,ds['Tdf'][0].values+2)
            APmov = np.divide(APmov,ds['Tdf'][0].values)
            
        APtotalmean = np.zeros(len(dirn)); APtotalstd = np.zeros(len(dirn))
        APpremean = np.zeros(len(dirn));  APprestd = np.zeros(len(dirn))
        APmovmean = np.zeros(len(dirn)); APmovstd = np.zeros(len(dirn))
        APpostmean = np.zeros(len(dirn)); APpoststd = np.zeros(len(dirn)) 
        if Apfftdf is not None:
            APfftmean = np.zeros((len(dirn),np.shape(Apfft)[1])); APfftstd = np.zeros((len(dirn),np.shape(Apfft)[1]))
        if ds['sti_type']== 'ACC':
            accApdf = ds['accApdf']; dccApdf = ds['dccApdf'];
            APacc = np.sum(accApdf.values[:,2:],axis =1).astype(np.int64); 
            APdcc = np.sum(dccApdf.values[:,2:],axis =1).astype(np.int64); 
            APaccmean =  np.zeros(len(dirn)); APaccstd =  np.zeros(len(dirn)); 
            APdccmean =  np.zeros(len(dirn)); APdccstd =  np.zeros(len(dirn));
          
        ep_names = Apdf.values[np.asarray(dirn)-1,:2]        
        
        for n in np.arange(0,len(dirn)): 
            if n == 0: 
                APtotalmean[n] =  np.mean(APtotal[:dirn[n]]); APtotalstd[n] =  np.std(APtotal[:dirn[n]])
                APpremean[n] =  np.mean(APpre[:dirn[n]]); APprestd[n] =  np.std(APpre[:dirn[n]])
                APmovmean[n] =  np.mean(APmov[:dirn[n]]); APmovstd[n] =  np.std(APmov[:dirn[n]]) 
                APpostmean[n] =  np.mean(APpost[:dirn[n]]); APpoststd[n] =  np.std(APpost[:dirn[n]])
                if Apfftdf is not None:
                    APfftmean[n] = np.mean(Apfft[:dirn[n],:], axis =0); APfftstd[n] =  np.std(Apfft[:dirn[n],:], axis = 0)
                if ds['sti_type']== 'ACC':
                    APaccmean[n] = np.mean(APaccsum[dirn[n-1]:dirn[n]]); APaccstd[n] = np.std(APaccsum[dirn[n-1]:dirn[n]]);
                    APdccmean[n] = np.mean(APdccsum[dirn[n-1]:dirn[n]]); APdccstd[n] = np.std(APdccsum[dirn[n-1]:dirn[n]]);
            else:
                #print(n)
                APtotalmean[n] =  np.mean(APtotal[dirn[n-1]:dirn[n]]); APtotalstd[n] =  np.std(APtotal[dirn[n-1]:dirn[n]])
                APpremean[n] =  np.mean(APpre[dirn[n-1]:dirn[n]]); APprestd[n] =  np.std(APpre[dirn[n-1]:dirn[n]])
                APmovmean[n] =  np.mean(APmov[dirn[n-1]:dirn[n]]); APmovstd[n] =  np.std(APmov[dirn[n-1]:dirn[n]])     
                APpostmean[n] =  np.mean(APpost[dirn[n-1]:dirn[n]]); APpoststd[n] =  np.std(APpost[dirn[n-1]:dirn[n]])
                if Apfftdf is not None:
                    APfftmean[n] =  np.mean(Apfft[dirn[n-1]:dirn[n],:], axis =0); 
                    APfftstd[n] =  np.std(Apfft[dirn[n-1]:dirn[n],:], axis =0)
                if ds['sti_type']== 'ACC':
                    APaccmean[n] = np.mean(APaccsum[dirn[n-1]:dirn[n]]); APaccstd[n] = np.std(APaccsum[dirn[n-1]:dirn[n]]);
                    APdccmean[n] = np.mean(APdccsum[dirn[n-1]:dirn[n]]); APdccstd[n] = np.std(APdccsum[dirn[n-1]:dirn[n]]);
            
        if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']):
            APmov0mean,APmov0std,APmov1mean,APmov1std,APmov2mean,APmov2std = [None]*6        
        else:
            APmov0mean,APmov0std,APmov1mean,APmov1std,APmov2mean,APmov2std = twoAPMeanStd(dirn, movApdf.values[:,2:])
            APmovmean = APmovmean/2; APmovstd= APmovstd/2
        # in Hz. 
        dmean = {'APmovmean':APmovmean,'APpostmean':APpostmean/0.5,'APpremean': APpremean/0.5,
                 'APmov0mean':APmov0mean,'APmov1mean':APmov1mean,'APmov2mean':APmov2mean} 
        dstd = {'APmovstd':APmovstd,'APpoststd':APpoststd/0.5,'APprestd': APprestd/0.5,
                'APmov0std': APmov0std,'APmov1std': APmov1std,'APmov2std':APmov2std}
        if Apfftdf is not None:
            APfftmeandf = pd.DataFrame(APfftmean, columns = Apfftdf.columns[2:])
            APfftstddf = pd.DataFrame(APfftstd, columns = Apfftdf.columns[2:])  
        else: 
            APfftmeandf = None; APfftstddf = None 
        if ds['sti_type']== 'ACC':
            dmean = {**dmean, **{'APaccmean':APaccmean,'APdccmean': APdccmean}}
            dstd = {**dstd, **{'APaccstd':APaccstd,'APdccstd':APdccstd}}
        dmeandf = pd.DataFrame.from_dict(dmean); dstddf = pd.DataFrame.from_dict(dstd)   
        self.Apavg = {'ep_names':ep_names, 'T':T,'sti_type': ds['sti_type'], 'dmeandf' : dmeandf, 
                      'dstddf': dstddf, 'fftmeandf': APfftmeandf, 'fftstddf': APfftstddf}
        if any(ds['sti_type'] in x for x in ['Flash','Edges_vel','Edges_vel_vert']):
            self.Apavg = {**self.Apavg, **{'Tdf':ds['Tdf']}}
        with open('Analysis/%s'%self.fdname+'/rec%dApavg.pkl'%self.recN,'wb') as f:
            pickle.dump(self.Apavg, f, pickle.HIGHEST_PROTOCOL)
            print('all saved!')
        return(self.Apavg)

#############################
# Merge & plot
#############################
def MergeRecs(fdname,recL,norm,avgname):
# avgname = 'Vavg' or 'Apavg'
    Vmeandf = pd.DataFrame(); Vstddf = pd.DataFrame(); Vfftmeandf = pd.DataFrame(); Vfftstddf = pd.DataFrame();
    for i in recL: 
        dfname = 'Analysis/%s'%fdname+'/rec%d%s.pkl'% (i,avgname)
        dset = pd.read_pickle(dfname); dmeandf = dset['dmeandf']; dstddf = dset['dstddf']; 
        dfftmeandf = dset['fftmeandf']; dfftstddf = dset['fftstddf']
        if norm:
            norm_val = np.max(dmeandf.max(axis =0))
            dmeandf = (dmeandf/norm_val).fillna(0) ; dstddf = (dstddf/norm_val).fillna(0) 
        dstddf =  dstddf**2;  
        if dfftmeandf is not None: 
            dfftstddf = dfftstddf**2   # calculate the variance    
        Vmeandf = Vmeandf.add(dmeandf,fill_value=0); Vstddf = Vstddf.add(dstddf,fill_value=0) #adding up
        if dfftmeandf is not None:
            Vfftmeandf = Vfftmeandf.add(dfftmeandf,fill_value=0); Vfftstddf = Vfftstddf.add(dfftstddf,fill_value=0)        
    Vmeandf = Vmeandf/len(recL); Vstddf = Vstddf**(1/2) # averaging
    if dfftmeandf is not None:
        Vfftmeandf = Vfftmeandf/len(recL); Vfftstddf = Vfftstddf**(1/2) # averaging
    else:
        Vfftmeandf = None; Vfftstddf = None
    newdir = 'Analysis/%s'%fdname; makedir(newdir)
    dset['dmeandf'] = Vmeandf; dset['dstddf'] = Vstddf
    dset['dfftmeandf'] = Vfftmeandf; dset['dfftstddf'] = Vfftstddf
    savedir = newdir+'/rec%s%s_merged.pkl'%(''.join(str(e) for e in recL),avgname)
    if norm:
        savedir =  newdir+'/rec%s%s_merged_norm.pkl'%(''.join(str(e) for e in recL),avgname)
    with open(savedir,'wb') as f:
        pickle.dump(dset, f, pickle.HIGHEST_PROTOCOL)
        print('all saved!')
    return(dset)

def pltVval(Vset):
    bl = Vset['dmeandf']['blmean'].values; bl = bl-np.mean(bl); blyerr = Vset['dstddf']['blstd'].values
    Mov = Vset['dmeandf']['Vmovmean'].values; Movyerr = Vset['dstddf']['Vmovstd'].values
    Post = Vset['dmeandf']['Vpostmean'].values; Postyerr = Vset['dstddf']['Vpoststd'].values
    FFTdf = Vset['dfftmeandf']; FFTyerrdf = Vset['dfftstddf']
    return(Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf)

def pltVvaln(Vset):
    Mov = Vset['dmeandf']['Vmovmean'].values; Movyerr = Vset['dstddf']['Vmovstd'].values
    Mov0 = Vset['dmeandf']['Vmov0mean'].values; Mov0yerr = Vset['dstddf']['Vmov0std'].values
    Mov1 = Vset['dmeandf']['Vmov1mean'].values; Mov1yerr = Vset['dstddf']['Vmov1std'].values
    Mov2 = Vset['dmeandf']['Vmov2mean'].values; Mov2yerr = Vset['dstddf']['Vmov2std'].values
    return(Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)

def pltAPval(Apset):
    Mov =  Apset['dmeandf']['APmovmean'].values; Movyerr = Apset['dstddf']['APmovstd'].values
    Pre = Apset['dmeandf']['APpremean'].values; Preyerr = Apset['dstddf']['APprestd'].values
    Post = Apset['dmeandf']['APpostmean'].values; Postyerr = Apset['dstddf']['APpoststd'].values
    FFTdf = Apset['dfftmeandf']; FFTyerrdf = Apset['dfftstddf']
    return(Mov, Movyerr, Pre, Preyerr, Post, Postyerr, FFTdf, FFTyerrdf)

def pltAPvaln(Vset):
    Mov = Vset['dmeandf']['APmovmean'].values; Movyerr = Vset['dstddf']['APmovstd'].values
    Mov0 = Vset['dmeandf']['APmov0mean'].values; Mov0yerr = Vset['dstddf']['APmov0std'].values
    Mov1 = Vset['dmeandf']['APmov1mean'].values; Mov1yerr = Vset['dstddf']['APmov1std'].values
    Mov2 = Vset['dmeandf']['APmov2mean'].values; Mov2yerr = Vset['dstddf']['APmov2std'].values
    return(Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)

def pltLR(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr):
    fig = plt.figure(); 
    plt.errorbar(stim_dir[:12],Pre[:12],yerr = Preyerr[:12],  color = 'c')
    plt.errorbar(stim_dir[:12],Mov[:12],yerr = Movyerr[:12],  color = 'b')
    plt.errorbar(stim_dir[:12],Post[:12],yerr = Postyerr[:12], color = 'm')
    plt.errorbar(stim_dir[:12],Pre[12:],yerr = Preyerr[12:],  ls = ':', color = 'c')
    plt.errorbar(stim_dir[12:],Mov[12:],yerr = Movyerr[12:],   ls = ':',color = 'b')
    plt.errorbar(stim_dir[12:],Post[12:],yerr = Postyerr[12:], ls = ':',  color = 'm')
    plt.legend(('Rpre','Rmov', 'Rpost', 'Lpre', 'Lmov','Lpost') ,loc  = 0)    
    return(fig)
    
def plt12(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr):  
    fig = plt.figure(); plt.xlabel('Direction, 0 is rightwards')
    plt.errorbar(stim_dir,Mov,yerr = Movyerr,  color = 'b')
    plt.errorbar(stim_dir,Pre,yerr = Preyerr, ls = '--', color = 'c')
    plt.errorbar(stim_dir,Post,yerr = Postyerr, ls = '--',  color = 'm')
    plt.legend(('mov', 'pre','post'),loc = 0 )   
    return(fig)
    
def plt12n(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):  
    fig = plt.figure(); plt.xlabel('Direction, 0 is rightwards')
    plt.errorbar(stim_dir,Mov,yerr = Movyerr,  color = 'k', alpha = 0.8)
    plt.errorbar(stim_dir,Mov0,yerr = Mov0yerr, ls = '--', color = 'c', alpha = 0.8)
    plt.errorbar(stim_dir,Mov1,yerr = Mov1yerr, ls = '--',  color = 'm', alpha = 0.8)
    plt.errorbar(stim_dir,Mov2,yerr = Mov2yerr, ls = '--',  color = 'g', alpha = 0.8)
    plt.legend(('Mov', 'Mov0.5','Mov1', 'Mov2'),loc = 0 )   
    return(fig)
    
def plt12fft(stim_dir,FFTdf, FFTyerrdf):
    fig = plt.figure(); plt.xlabel('Direction, 0 is rightwards')
    plt.errorbar(stim_dir, FFTdf['0.5Hz'], yerr = FFTyerrdf['0.5Hz'],color ='c',alpha = 0.5)
    plt.errorbar(stim_dir, FFTdf['1.0Hz'], yerr = FFTyerrdf['1.0Hz'],color ='k')
    plt.errorbar(stim_dir, FFTdf['1.5Hz'], yerr = FFTyerrdf['1.5Hz'],color ='m',alpha = 0.5)
    plt.errorbar(stim_dir, FFTdf['2.0Hz'], yerr = FFTyerrdf['2.0Hz'],color ='b')
    plt.errorbar(stim_dir, FFTdf['2.5Hz'], yerr = FFTyerrdf['2.5Hz'],color ='y',alpha = 0.5)
    plt.legend(['0.5Hz','1.0Hz', '1.5Hz','2.0Hz','2.5Hz'], loc = 0)   
    return(fig)

def instpolar(Pre):
    return(np.insert(Pre,len(Pre),Pre[0]))

def pltLRpolar(theta, Pre, Preyerr, Mov, Movyerr, Post, Postyerr):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar'); ax.grid(True)
    RPre = instpolar(Pre[:12]); RPreyerr = instpolar(Preyerr[:12])
    RMov = instpolar(Mov[:12]); RMovyerr = instpolar(Movyerr[:12])
    RPost =instpolar(Post[:12]); RPostyerr = instpolar(Postyerr[:12])
    LPre = instpolar(Pre[12:]); LPreyerr = instpolar(Preyerr[:12])
    LMov = instpolar(Mov[12:]); LMovyerr = instpolar(Movyerr[12:])
    LPost = instpolar(Post[12:]); LPostyerr = instpolar(Postyerr[12:])
   
    ax = plt.subplot(111, projection='polar'); ax.grid(True)
    ax.errorbar(theta[12:],RPre,yerr = RPreyerr,  color = 'c')
    ax.errorbar(theta[12:],RMov,yerr = RMovyerr,  color = 'b')
    ax.errorbar(theta[12:],RPost,yerr = RPostyerr, color = 'm')
    ax.errorbar(theta[12:],LPre,yerr = LPreyerr,  ls = ':', color = 'c')
    ax.errorbar(theta[12:],LMov,yerr = LMovyerr,   ls = ':',color = 'b')
    ax.errorbar(theta[12:],LPost,yerr = LPostyerr, ls = ':',  color = 'm')
    plt.legend(('Rpre','Rmov', 'Rpost', 'Lpre', 'Lmov','Lpost') ,loc  = 0) 
    return(fig)
    
def plt12polar(theta, Pre, Preyerr, Mov, Movyerr, Post, Postyerr): 
    Mov = instpolar(Mov); Movyerr = instpolar(Movyerr)
    Pre = instpolar(Pre); Preyerr = instpolar(Preyerr)
    Post = instpolar(Post); Postyerr = instpolar(Postyerr)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar'); ax.grid(True)
    ax.errorbar(theta, Mov, yerr=Movyerr, color = 'b'); 
    ax.errorbar(theta, Pre, yerr=Preyerr, color = 'c')
    ax.errorbar(theta, Post, yerr=Postyerr, color = 'm')
    plt.legend(('mov', 'pre','post'),loc = 'upper center' )
    return(fig)

def plt12npolar(theta, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):  
    Mov = instpolar(Mov); Movyerr = instpolar(Movyerr); Mov0 = instpolar(Mov0); Mov0yerr = instpolar(Mov0yerr)
    Mov1 = instpolar(Mov1); Mov1yerr = instpolar(Mov1yerr); Mov2 = instpolar(Mov2); Mov2yerr = instpolar(Mov2yerr);
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar'); ax.grid(True)
    ax.errorbar(theta, Mov,yerr = Movyerr,  color = 'k', alpha = 0.8)
    ax.errorbar(theta, Mov0,yerr = Mov0yerr, ls = '--', color = 'c', alpha = 0.8)
    ax.errorbar(theta, Mov1,yerr = Mov1yerr, ls = '--',  color = 'm', alpha = 0.8)
    ax.errorbar(theta, Mov2,yerr = Mov2yerr, ls = '--',  color = 'g', alpha = 0.8)
    ax.legend(('Mov', 'Mov0.5','Mov1', 'Mov2'),loc = 0 )   
    return(fig)
    
def plt12fftpolar(theta, FFTdf, FFTyerrdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar'); ax.grid(True)    
    ax.errorbar(theta, instpolar(FFTdf['0.5Hz'].values), yerr = instpolar(FFTyerrdf['0.5Hz'].values),color ='c',alpha = 0.5)
    ax.errorbar(theta, instpolar(FFTdf['1.0Hz'].values), yerr = instpolar(FFTyerrdf['1.0Hz'].values),color ='k')
    ax.errorbar(theta, instpolar(FFTdf['1.5Hz'].values), yerr = instpolar(FFTyerrdf['1.5Hz'].values),color ='m',alpha = 0.5)
    ax.errorbar(theta, instpolar(FFTdf['2.0Hz'].values), yerr = instpolar(FFTyerrdf['2.0Hz'].values),color ='b')
    ax.errorbar(theta, instpolar(FFTdf['2.5Hz'].values), yerr = instpolar(FFTyerrdf['2.5Hz'].values),color ='y',alpha = 0.5)
    plt.legend(['0.5Hz','1.0Hz', '1.5Hz','2.0Hz','2.5Hz'], loc = 0); plt.title('fft')
    return(fig)

def pltPDND(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr):  
    fig = plt.figure(); plt.xscale('symlog')  
    plt.errorbar(stim_dir,Mov,yerr = Movyerr,  color = 'b')
    plt.errorbar(stim_dir,Pre,yerr = Preyerr, ls = '--', color = 'c', alpha = 0.5)
    plt.errorbar(stim_dir,Post,yerr = Postyerr, ls = '--',  color = 'm', alpha = 0.5)
    plt.legend(('mov', 'pre','post'),loc = 0 )   
    return(fig)

def pltPDNDn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):  
    fig = plt.figure(); plt.xscale('symlog')  
    plt.errorbar(stim_dir,Mov,yerr = Movyerr,  color = 'k', alpha = 0.8)
    plt.errorbar(stim_dir,Mov0,yerr = Mov0yerr, ls = '--', color = 'c', alpha = 0.8)
    plt.errorbar(stim_dir,Mov1,yerr = Mov1yerr, ls = '--',  color = 'm', alpha = 0.8)
    plt.errorbar(stim_dir,Mov2,yerr = Mov2yerr, ls = '--',  color = 'g', alpha = 0.8)
    plt.legend(('Mov', 'Mov0.5','Mov1', 'Mov2'),loc = 0 )    
    return(fig)

def pltPDNDfft(stim_dir, FFTdf, FFTyerrdf):
    hz_dir = np.copy(stim_dir); hz_dir[abs(stim_dir)< 0.5] = 0.5; 
    # set anything less than 0.5 to 0.5 as no value in FFT between 0 & 0.5. 
    hz_mean = [FFTdf[str(abs(i))+'Hz'][n] for n,i in enumerate(hz_dir)]
    hz_yerr = [FFTyerrdf[str(abs(i))+'Hz'][n] for n,i in enumerate(hz_dir)]
    fig = plt.figure(); plt.xscale('symlog') 
    plt.errorbar(hz_dir, FFTdf['0.5Hz'], yerr = FFTyerrdf['0.5Hz'],color ='c',alpha = 0.5)
    plt.errorbar(hz_dir, FFTdf['1.0Hz'], yerr = FFTyerrdf['1.0Hz'],color ='k')
    plt.errorbar(hz_dir, FFTdf['1.5Hz'], yerr = FFTyerrdf['1.5Hz'],color ='m',alpha = 0.5)
    plt.errorbar(hz_dir, FFTdf['2.0Hz'], yerr = FFTyerrdf['2.0Hz'],color ='b')
    plt.errorbar(hz_dir, FFTdf['3.0Hz'], yerr = FFTyerrdf['3.0Hz'],color ='y')
    plt.errorbar(hz_dir, hz_mean, yerr = hz_yerr,color ='r')
    plt.legend(['0.5Hz','1.0Hz', '1.5Hz','2.0Hz','3Hz','Hz'], loc = 0) 
    plt.xlabel('PDND:log(Hz)'); plt.ylabel('fft:power'); plt.title('FFT_PDND')
    return(fig)    

def loadset(fdname, recN, spiking):
    if spiking:
        Apset = pd.read_pickle('Analysis/'+fdname+'/rec%dApavg_merged.pkl'%recN) 
        Apset_norm = pd.read_pickle('Analysis/'+ fdname+'/rec%dApavg_merged_norm.pkl'%recN)
    Vset = pd.read_pickle('Analysis/'+ fdname+'/rec%dVavg_merged.pkl'%recN)
    if spiking:
        return(Apset, Apset_norm, Vset)
    else:
        return(Vset)
    
def pltPatches(stim_dir, bl, blyerr,Mov, Movyerr, Post, Postyerr ):
    angles = np.unique(stim_dir[:,0])
    idx0 = [stim_dir[:,0] == angles[0]]; idx1 =[stim_dir[:,0] == angles[1]]; 
    loc0 = stim_dir[:,1][idx0]; loc1 = stim_dir[:,1][idx1]
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    wd = 0.4
    ax1.set_title(angles[0]); ax2.set_title(angles[1]); ax2.set_xticks(loc0)
    ax1.bar(loc0-0.3,bl[idx0],yerr = blyerr[idx0], width = wd, color='c', ecolor='c', alpha =0.3)
    ax1.bar(loc0,Mov[idx0],yerr = Movyerr[idx0],width = wd, color='b', ecolor='b', alpha =0.4)
    ax1.bar(loc0+0.3,Post[idx0],yerr = Postyerr[idx0],width = wd, color='m', ecolor='m', alpha =0.3) 
    ax2.bar(loc1-0.3,bl[idx1],yerr = blyerr[idx1],width = wd, color='c', ecolor='c', alpha =0.3)
    ax2.bar(loc1,Mov[idx1],yerr = Movyerr[idx1],width = wd, color='b', ecolor='b', alpha =0.4)
    ax2.bar(loc1+0.3,Post[idx1],yerr = Postyerr[idx1],width = wd, color='m', ecolor='m', alpha =0.3)
    f.legend(('pre','mov','post'), loc = 0 ); plt.xlabel('location')
    
def pltOppo(locplt, stim_dir, Mov, Movyerr, bl, blyerr, Post, Postyerr):  
    fig = plt.figure()
    plt.bar(locplt-0.1,bl,yerr = blyerr, width = 0.1, color ='c',alpha =0.3)
    plt.bar(locplt,Mov,yerr = Movyerr, width = 0.2, alpha =0.4)
    plt.bar(locplt+0.1,Post,yerr = Postyerr, width = 0.1, color = 'm',alpha =0.3)
    plt.xticks(locplt, stim_dir, rotation=20)
    plt.legend(('pre','mov','post'), loc = 0 )
    return(fig)

def pltOppofft(locplt, stim_dir, FFTdf, FFTyerrdf):  
    fig = plt.figure()
    plt.errorbar(locplt, FFTdf['0.5Hz'], yerr = FFTyerrdf['0.5Hz'],color ='c',alpha = 0.5)
    plt.errorbar(locplt, FFTdf['1.0Hz'], yerr = FFTyerrdf['1.0Hz'],color ='k')
    plt.errorbar(locplt, FFTdf['1.5Hz'], yerr = FFTyerrdf['1.5Hz'],color ='m',alpha = 0.5)
    plt.errorbar(locplt, FFTdf['2.0Hz'], yerr = FFTyerrdf['2.0Hz'],color ='b')
    plt.errorbar(locplt, FFTdf['3.0Hz'], yerr = FFTyerrdf['3.0Hz'],color ='y')
    plt.xticks(locplt, stim_dir, rotation=20)
    plt.legend(['0.5Hz','1.0Hz', '1.5Hz','2.0Hz','3Hz'], loc = 0) 
    plt.ylabel('fft:power'); plt.title('FFT')
    return(fig)
    
def pltOppon(locplt, stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):  
    fig = plt.figure()
    plt.bar(locplt,Mov,yerr = Movyerr, width = 0.2, alpha =0.4)
    plt.errorbar(locplt,Mov,yerr = Movyerr,  color = 'k', alpha = 0.4)
    plt.errorbar(locplt,Mov0,yerr = Mov0yerr, ls = '--', color = 'c', alpha = 0.4)
    plt.errorbar(locplt,Mov1,yerr = Mov1yerr, ls = '--',  color = 'm', alpha = 0.4)
    plt.errorbar(locplt,Mov2,yerr = Mov2yerr, ls = '--',  color = 'g', alpha = 0.4)    
    plt.xticks(locplt, stim_dir, rotation=20)
    plt.legend(('Mov','Mov', 'Mov0.5','Mov1', 'Mov2'), loc = 0 )
    return(fig)
 
def pltPvNz(stim_dir,Mov, Movyerr, bl, blyerr, Post, Postyerr):   
    hlf = int(len(stim_dir)/2); fig = plt.figure() 
    plt.subplot(211); plt.xscale('symlog')
    plt.errorbar(stim_dir[:hlf,0], Mov[:hlf],yerr = Movyerr[:hlf],  color = 'b')
    plt.errorbar(stim_dir[hlf:,0], Mov[hlf:],yerr = Movyerr[hlf:], ls = '--', color = 'b')
    plt.errorbar(stim_dir[:hlf,0], bl[:hlf],yerr = blyerr[:hlf],  color = 'c', alpha = 0.5)
    plt.errorbar(stim_dir[hlf:,0], bl[hlf:],yerr = blyerr[hlf:], ls = '--', color = 'c', alpha = 0.5)
    plt.errorbar(stim_dir[:hlf,0], Post[:hlf],yerr = Postyerr[:hlf],  color = 'm', alpha = 0.5)
    plt.errorbar(stim_dir[hlf:,0], Post[hlf:],yerr = Postyerr[hlf:], ls = '--', color = 'm', alpha = 0.5)
    plt.legend(('mov_F','mov_M','pre_F','pre_M','post_F','post_M'), loc = 0 )
    plt.subplot(212); plt.xscale('symlog')
    plt.errorbar(stim_dir[:hlf,0], Mov[:hlf]- Mov[hlf:],yerr = Movyerr[:hlf]+Movyerr[hlf:],  color = 'r')
    plt.legend('F-M', loc = 0)    
    return(fig)

def pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr):  
    hlf = int(len(stim_dir)/2); fig = plt.figure()  
    plt.subplot(211); plt.xscale('symlog')
    plt.errorbar(stim_dir[:hlf,0],Mov[:hlf],yerr = Movyerr[:hlf],  color = 'k', alpha = 0.8)
    plt.errorbar(stim_dir[:hlf,0],Mov[hlf:],yerr = Movyerr[hlf:],  ls = '--', color = 'k', alpha = 0.8)
    plt.errorbar(stim_dir[:hlf,0],Mov0[:hlf],yerr = Mov0yerr[:hlf], color = 'c', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0],Mov1[:hlf],yerr = Mov1yerr[:hlf], color = 'm', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0],Mov2[:hlf],yerr = Mov2yerr[:hlf], color = 'g', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0],Mov0[hlf:],yerr = Mov0yerr[hlf:], ls = '--', color = 'c', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0],Mov1[hlf:],yerr = Mov1yerr[hlf:], ls = '--',  color = 'm', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0],Mov2[hlf:],yerr = Mov2yerr[hlf:], ls = '--',  color = 'g', alpha = 0.4)
    plt.legend(('mov_F','mov_M', 'mov0.5_F','mov0.5_M', 'mov1_F','mov1_M', 'mov2_F', 'mov2_M'),loc = 0 )    
    plt.subplot(212); plt.xscale('symlog')
    plt.errorbar(stim_dir[:hlf,0], Mov[:hlf]- Mov[hlf:],yerr = Movyerr[:hlf]+Movyerr[hlf:],  color = 'k', alpha = 0.8)
    plt.errorbar(stim_dir[:hlf,0], Mov0[:hlf]- Mov0[hlf:],yerr = Mov0yerr[:hlf]+Mov0yerr[hlf:],  color = 'c', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0], Mov1[:hlf]- Mov1[hlf:],yerr = Mov1yerr[:hlf]+Mov1yerr[hlf:],  color = 'm', alpha = 0.4)
    plt.errorbar(stim_dir[:hlf,0], Mov2[:hlf]- Mov2[hlf:],yerr = Mov2yerr[:hlf]+Mov2yerr[hlf:],  color = 'g', alpha = 0.4)
    plt.legend(('F-M_Mov', 'F-M_Mov0.5','F-M_Mov1', 'F-M_Mov2'),loc = 0 )     
    return(fig)
    
def pltTune (Vset, spiking, fdname, recL,*args): 
    if args != ():
        Apset = args[0]; Apset_norm = args[1]
    savedir = 'Pics/%s'%fdname+'/%s_rec%splot.jpeg'%(Vset['sti_type'],str(recL))
    
    if any(Vset['sti_type'] in x for x in ['Oppo','Sq_PvN']):
        stim_dir = Vset['ep_names']
        Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf = pltVval(Vset)
        Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltVvaln(Vset)
        if any(Vset['sti_type'] in x for x in ['Oppo']):
            locplt = stim_dir[:,1] + np.array([0,0.3,0.6]*3)
        elif any(Vset['sti_type'] in x for x in ['Sq_PvN']):
            locplt = stim_dir[:,1] + np.array([0,0.3]*4)
        fig0 =pltOppo(locplt,stim_dir, Mov, Movyerr, bl, blyerr, Post, Postyerr) 
        fig1 = pltOppofft(locplt, stim_dir, FFTdf, FFTyerrdf)
        fig2 = pltOppon(locplt, stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
        fig0.savefig(savedir[:-5] + '_Vm.jpeg'); fig1.savefig(savedir[:-5] + '_Vm_n.jpeg') 
        fig2.savefig(savedir[:-5] + '_Vm_fft.jpeg')
        if spiking:
            for n,i in enumerate(args): 
                Mov, Movyerr, Pre, Preyerr, Post, Postyerr,FFTdf, FFTyerrdf = pltAPval(i) 
                Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPvaln(i)
                fig2 = pltOppon(locplt, stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
                fig0 = pltOppo(locplt,stim_dir,Mov, Movyerr, Pre, Preyerr, Post, Postyerr)                 
                if n ==1: 
                    plt.ylabel('normalized firing rate(Hz)'); fig0.savefig(savedir[:-5]+'_norm.jpeg')
                    fig2.savefig(savedir[:-5]+'_n_norm.jpeg')
                else:
                    plt.ylabel('firing rate(Hz)'); fig0.savefig(savedir); 
                    fig1 = pltOppofft(locplt, stim_dir, FFTdf, FFTyerrdf)
                    fig1.savefig(savedir[:-5]+'_fft.jpeg'); fig2.savefig(savedir[:-5]+'_n.jpeg')
                    
    elif any(Vset['sti_type'] in x for x in ['Sine_PvN', 'Sine_PvN_u']):
        stim_dir = Vset['ep_names']
        Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf = pltVval(Vset)
        Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltVvaln(Vset)
        fig0 =pltPvNz(stim_dir, Mov, Movyerr, bl, blyerr, Post, Postyerr) 
        fig1 = pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
        fig0.savefig(savedir[:-5] + '_Vm.jpeg'); fig1.savefig(savedir[:-5] + '_Vm_n.jpeg')
        if spiking:
            for n,i in enumerate(args): 
                Mov, Movyerr, Pre, Preyerr, Post, Postyerr,FFTdf, FFTyerrdf = pltAPval(i) 
                Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPvaln(i)
                fig2 = pltPvNzn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
                fig0 = pltPvNz(stim_dir,Mov, Movyerr, Pre, Preyerr, Post, Postyerr)                 
                if n ==1: 
                    plt.ylabel('normalized firing rate(Hz)'); fig0.savefig(savedir[:-5]+'_norm.jpeg')
                    fig2.savefig(savedir[:-5]+'_n_norm.jpeg')
                else:
                    plt.ylabel('firing rate(Hz)'); fig0.savefig(savedir); fig2.savefig(savedir[:-5]+'_n.jpeg')
        
    elif any(Vset['sti_type'] in x for x in ['Sine_patches','Sine_patches2']):
        stim_dir = Vset['ep_names']
        Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf = pltVval(Vset)
        pltPatches(stim_dir, bl, blyerr, Mov, Movyerr, Post, Postyerr)
        plt.savefig(savedir[:-5] + '_Vm.jpeg')
        if spiking: 
            for n,i in enumerate(args): 
                Mov, Movyerr, Pre, Preyerr, Post, Postyerr,FFTdf, FFTyerrdf = pltAPval(i)   
                pltPatches(stim_dir, Pre, Preyerr, Mov, Movyerr,  Post, Postyerr)
                if n ==1: 
                    plt.ylabel('normalized firing rate(Hz)'); plt.savefig(savedir[:-5]+'_norm.jpeg')
                else:
                    plt.ylabel('firing rate(Hz)');plt.savefig(savedir)  
                    
    elif any(Vset['sti_type'] in x for x in ['Edges_vel','Edges_vel_vert','Flashes','Sine_PDND_y','Sine_PDND_z','Sine_xC']):
        stim_dir = Vset['ep_names'][:,0]
        Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf = pltVval(Vset)
        Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltVvaln(Vset)
        fig0 = pltPDND(stim_dir, bl, blyerr, Mov, Movyerr, Post, Postyerr) 
        fig2 = pltPDNDn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
        fig0.savefig(savedir[:-5] + '_Vm.jpeg'); fig2.savefig(savedir[:-5] + '_Vm_n.jpeg')
        if any(Vset['sti_type'] in x for x in ['Sine_PDND_y','Sine_PDND_z','Sine_xC']):
            fig1 = pltPDNDfft(stim_dir, FFTdf, FFTyerrdf)
            fig1.savefig(savedir[:-5] + '_Vm_fft.jpeg')
        if spiking:
            for n,i in enumerate(args): 
                Mov, Movyerr, Pre, Preyerr, Post, Postyerr,FFTdf, FFTyerrdf = pltAPval(i)  
                Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPvaln(i)
                fig0 = pltPDND(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr)
                fig2 = pltPDNDn(stim_dir, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
                if n ==1: 
                    plt.ylabel('normalized firing rate(Hz)'); 
                    fig0.savefig(savedir[:-5]+'_norm.jpeg'); fig2.savefig(savedir[:-5]+'_n_norm.jpeg')
                else:
                    plt.ylabel('firing rate(Hz)');fig0.savefig(savedir); fig2.savefig(savedir[:-5]+'_n.jpeg')                   
                    if any(Vset['sti_type'] in x for x in ['Sine_PDND_y','Sine_PDND_z','Sine_xC']):
                        fig1 = pltPDNDfft(stim_dir, FFTdf, FFTyerrdf)
                        fig1.savefig(savedir[:-5]+'_fft.jpeg')
        
    elif any(Vset['sti_type'] in x for x in ['Sq_12dir','Sine_12dir', 'Sq_LR_12dir']): 
        ls_12dirs = ['Sq_12dir','Sine_12dir']; stim_dir = Vset['ep_names'][:,0]
        theta = instpolar(np.array(stim_dir)/360* 2*np.pi) # x-axis for polar plot
        Mov, Movyerr, bl, blyerr, Post, Postyerr, FFTdf, FFTyerrdf = pltVval(Vset)
        Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltVvaln(Vset)
        if any(Vset['sti_type'] in x for x in ls_12dirs): 
            fig1 = plt12fft(stim_dir, FFTdf, FFTyerrdf)
            fig2 = plt12n(stim_dir,Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr); plt.xlabel('Direction, 0 is rightwards')
            fig0 = plt12(stim_dir, bl, blyerr, Mov, Movyerr, Post, Postyerr); plt.xlabel('Direction, 0 is rightwards')   
            fig1.savefig(savedir[:-5] + '_Vm_fft.jpeg') ; fig2.savefig(savedir[:-5] + '_Vm_n.jpeg')
        elif any(Vset['sti_type'] in x for x in ['Sq_LR_12dir']):
            fig0= pltLR(stim_dir, bl, blyerr,Mov, Movyerr, Post, Postyerr); plt.xlabel('Direction, 0 is rightwards')
        fig0.savefig(savedir[:-5] + '_Vm.jpeg') 

        # polar
        if any(Vset['sti_type'] in x for x in ls_12dirs): 
            fig1 = plt12fftpolar(theta, FFTdf, FFTyerrdf)
            fig2 = plt12npolar(theta, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
            fig0 = plt12polar(theta, bl, blyerr, Mov, Movyerr, Post, Postyerr)     
            fig1.savefig(savedir[:-5] + '_Vm_fft_polar.jpeg') ;  fig2.savefig(savedir[:-5] + '_Vm_n_polar.jpeg')
        elif any(Vset['sti_type'] in x for x in ['Sq_LR_12dir']):
            fig0 = pltLRpolar(theta, bl, blyerr, Mov, Movyerr, Post, Postyerr)
         # shade -ve region
#         circle = plt.Circle((0,0), radius = abs(min(min(Mov),min(Post),-10)), transform=ax.transData._b, color = 'k', alpha = 0.3)
#         ax.add_artist(circle)        
        plt.yticks([-10,-5, 0,5,10,15,20]); plt.title('Subthreshold potential(mV)')
        fig0.savefig(savedir[:-5] + '_Vm_polar.jpeg')    
        
                                                 
        if spiking:
            for n,i in enumerate(args): 
                Mov, Movyerr, Pre, Preyerr, Post, Postyerr,FFTdf, FFTyerrdf = pltAPval(i)
                Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr = pltAPvaln(i)
                if any(Vset['sti_type'] in x for x in ls_12dirs):  
                    fig1 = plt12fft(stim_dir, FFTdf, FFTyerrdf)
                    fig2 = plt12n(stim_dir,Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
                    plt.xlabel('Direction, 0 is rightwards')
                    fig0 = plt12(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr); plt.xlabel('Direction, 0 is rightwards')
                elif any(Vset['sti_type'] in x for x in ['Sq_LR_12dir']):
                    fig0 = pltLR(stim_dir, Pre, Preyerr, Mov, Movyerr, Post, Postyerr); plt.xlabel('Direction, 0 is rightwards')   
                if n ==1: 
                    plt.ylabel('normalized firing rate(Hz)'); fig0.savefig(savedir[:-5]+'_norm.jpeg')
                else:
                    plt.ylabel('firing rate(Hz)');fig0.savefig(savedir); fig1.savefig(savedir[:-5]+'_fft.jpeg')
                # polar
                if any(Vset['sti_type'] in x for x in ls_12dirs): 
                    fig1 = plt12fftpolar(theta, FFTdf, FFTyerrdf)
                    fig2 = plt12npolar(theta, Mov, Movyerr, Mov0, Mov0yerr, Mov1, Mov1yerr, Mov2, Mov2yerr)
                    fig0 = plt12polar(theta, Pre, Preyerr, Mov, Movyerr, Post, Postyerr)
                elif any(Vset['sti_type'] in x for x in ['Sq_LR_12dir']): 
                    fig0 = pltLRpolar(theta, Pre, Preyerr, Mov, Movyerr, Post, Postyerr)
                if n ==1: 
                    plt.title('normalized firing rate(Hz)')
                    fig0.savefig(savedir[:-5]+'_norm_polar.jpeg')
                    fig2.savefig(savedir[:-5]+'_n_norm_polar.jpeg')
                else:
                    plt.title('firing rate(Hz)'); 
                    fig0.savefig(savedir[:-5]+'_polar.jpeg')
                    fig1.savefig(savedir[:-5]+'_fft_polar.jpeg')
                    fig2.savefig(savedir[:-5]+'_n_polar.jpeg')
                                 
                        
def calFMap(RecP, fdname,MPtype,spiking):
    if MPtype == 1:
        xcenters = np.arange(-100,122,40); ycenters = np.arange(-33/2, 66+33/2, 33)
        loc_keys = [21,11,1,22,12,2,23,13,3,24,14,4,25,15,5,26,16,6]
    else:
        xcenters = np.arange(-80,82,40); ycenters = np.arange(0, 34, 33)
        loc_keys = [11,1,12,2,13,3,14,4,15,5]
    
    xy = np.asarray([(x,y) for x in xcenters for y in ycenters])
    angles = np.empty(0); locs = np.empty(0); VMov = np.empty(0); VMovyerr = np.empty(0)
    if spiking: 
        ApMov = np.empty(0); ApMovyerr =np.empty(0); ApMov_norm =np.empty(0); ApMovyerr_norm = np.empty(0)
        ApPre = np.empty(0); ApPreyerr = np.empty(0); ApPre_norm = np.empty(0); ApPreyerr_norm = np.empty(0)
    for i in RecP:
        if spiking:
            Apset, Apset_norm, Vset = loadset(fdname, i, spiking)
            apMov, apMovyerr, apPre, apPreyerr, _, _, _,_ = pltAPval(Apset)
            apMov_n, apMovyerr_n, apPre_n, apPreyerr_n, _, _, _,_= pltAPval(Apset_norm)
            ApMov=np.append(ApMov,apMov); ApMovyerr=np.append(ApMovyerr,apMovyerr)
            ApMov_norm=np.append(ApMov_norm,apMov_n); ApMovyerr_norm=np.append(ApMovyerr_norm,apMovyerr_n)
            ApPre=np.append(ApPre,apPre); ApPreyerr=np.append(ApPreyerr,apPreyerr)
            ApPre_norm=np.append(ApPre_norm,apPre_n); ApPreyerr_norm=np.append(ApPreyerr_norm,apPreyerr_n)
        else:
            Vset = loadset(fdname, i, spiking)
        vMov, vMovyerr, _, _, _, _, _,_ = pltVval(Vset) 
        VMov= np.append(VMov,vMov); VMovyerr=np.append(VMovyerr,vMovyerr)
        angles=np.append(angles,Vset['ep_names'][:,0]); locs=np.append(locs,Vset['ep_names'][:,1])
    xf = np.cos(np.deg2rad(angles)); yf = np.sin(np.deg2rad(angles))
    
    Dic = dict(zip(loc_keys, xy)); Locs = np.asarray([Dic[i] for i in locs])
    Locsdf = pd.DataFrame(Locs,columns=['angle','loc']).sort_values(by=['angle','loc'])
    dirn=AngleGp(Locsdf.values); stim_dir = Locsdf.values[np.asarray(dirn)-1]
    if spiking:
        return(VMov, ApMov, ApMov_norm, ApPre, ApPre_norm, Locsdf, xf,yf,Locsdf, dirn, stim_dir, xcenters, ycenters, xy) 
    else:
        return(VMov, Locsdf, xf,yf,Locsdf, dirn, stim_dir, xcenters, ycenters, xy) 

def pltFMap(VMov,xf,yf,Locsdf, dirn, stim_dir, xcenters, ycenters, xy):
    xVMov = xf*VMov; yVMov = yf*VMov
    xVMov=xVMov[Locsdf.index]; yVMov=yVMov[Locsdf.index]
    
    XVMov = np.zeros(len(dirn)); YVMov = np.zeros(len(dirn)) 
    for n in np.arange(0,len(dirn)):        
        if n == 0: 
            XVMov[n] = np.mean(xVMov[:dirn[n]]); YVMov[n] = np.mean(yVMov[:dirn[n]])      
        else:
            XVMov[n] = np.mean(xVMov[dirn[n-1]:dirn[n]]); YVMov[n] = np.mean(yVMov[dirn[n-1]:dirn[n]]) 

    beta = 20/np.max([xVMov,yVMov]) # scale factor to see the point well
    fig = plt.figure(figsize=(8*2, 3.3*2));plt.axvline(x=0, c='g'); plt.grid(True)
    ax = fig.add_subplot(111); ax.set_xticks(np.arange(-120,122,40)); ax.set_yticks(np.arange(-33,67,33))
    for i in ycenters: plt.axhline(y = i, color = 'g', ls = ':',alpha =0.4)
    for i in xcenters: plt.axvline(x = i, color = 'g', ls = ':',alpha =0.4)
    #plt.errorbar(xVMov + Locsdf.values[:,0], yVMov + Locsdf.values[:,1],yerr=VMovyerr[Locsdf.index], color='b', fmt='o', ms=1)
    plt.scatter(xVMov*beta + Locsdf.values[:,0], yVMov *beta+ Locsdf.values[:,1], color='b', marker='o', s=1)
    for n,i in enumerate(stim_dir):   
        ax.arrow(i[0], i[1], XVMov[n]*beta, YVMov[n]*beta, lw= 1,fc='r', ec='r', head_width=1.5, head_length =2) 
        #ax.add_patch(arrMov);
    plt.scatter(xy[:,0],xy[:,1],color='g', s =5, marker = 'o')
    plt.axes().set_aspect('equal')
    plt.xlim([-120,120]); plt.ylim([-33,66]); plt.title('scale = x %f'%beta)
    plt.tight_layout();
    Vecdf = pd.DataFrame({'x':stim_dir[:,0], 'y':stim_dir[:,1], 'Vx': XVMov, 'Vy': YVMov})
    return(Vecdf)

                                              
     
        

       
       

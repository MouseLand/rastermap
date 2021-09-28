import h5py
import numpy as np
import os
from pathlib import Path

def get_fold_list(input_path):
    f_path = Path(input_path)
    f = [e for e in f_path.iterdir() if e.is_dir()]
    return f    

def extract_and_save_spks(dataroot, db, saveroot=None):
    """ load and concatenate mesoscope recordings """
    foldpath = os.path.join(dataroot, db['mname'], db['datexp'], db['blk'],'suite2p')
    file_path = get_fold_list(foldpath)

    ROI_size_temp = np.zeros([len(file_path),2])
    Neuron_Num = np.zeros(len(file_path))

    ROI_size,meanImg,xpos,ypos,spks = [],[],[],[],[]

    xpos = np.zeros((1,0))[0]
    ypos = np.zeros((1,0))[0]
    for i in range(len(file_path)):
        ops = np.load(os.path.join(file_path[i],'ops.npy'),allow_pickle=True).item()  
        stat = np.load(os.path.join(file_path[i],'stat.npy'),allow_pickle=True)    
        spks.append(np.load(os.path.join(file_path[i],'spks.npy')))    
        ROI_size_temp[i,:] = [ops['Ly'],ops['Lx']] # get x pixels and y pixels of each ROI   
        ypos_temp = np.array([stat[k]['med'][0] for k in range(len(stat))])
        xpos_temp = np.array([stat[k]['med'][1] for k in range(len(stat))])
        xpos = np.concatenate((xpos, xpos_temp + ops['dx']), axis=0) 
        ypos = np.concatenate((ypos, ypos_temp + ops['dy']), axis=0)     
        meanImg.append(ops['meanImg'])
        ROI_size.append(ROI_size_temp)
    spks = np.concatenate(spks, axis=0)
    mdict={'meanImg': meanImg, 'xpos':xpos, 'ypos':ypos, 
            'ROI_size': ROI_size, 'spks': spks}  
    if saveroot is not None:
        savename = db['mname'] + '_' + db['datexp'] + '_' + db['blk'] + '.npy'
        if not os.path.exists(saveroot):    
            os.makedirs(saveroot)   
        np.save(os.path.join(saveroot, savename), mdict)
        
    return mdict

def loadh5py(filename):
    Timeline = h5py.File(filename, 'r')['Timeline']
    return Timeline

def h5pyfield(h5pyobject,fieldname):
    fieldArray = np.array(fieldname)
    itN = np.max(fieldArray.shape)
    value = []
    for i in range(itN):
        value.append(np.array(h5pyobject[fieldArray[i][0]]))
    return value

def read_strings(h5file, dataset):
    from builtins import chr
    strlist = [u''.join(chr(c) for c in h5file[obj_ref][()].flatten()) for obj_ref in dataset]
    return strlist

def linearCorridor(root,file):
    import numpy as np
    Timeline = loadh5py(os.path.join(root,file['mname'],file['datexp'],file['blk'],\
                     'Timeline_'+file['mname']+'_'+file['datexp']+'_'+file['blk']+'_RAW.mat'))
    BehaviorResults = Timeline['BehaviorResults']
    BehaviorSettings = Timeline['BehaviorSettings']    
    TunnelLength = np.squeeze(BehaviorSettings['TunnelLength'])
    WhiteSpaceLength = np.squeeze(BehaviorSettings['WhiteSpaceLength'])    
    data = np.array(Timeline['data']).T
    
    TimeTunnelStart = np.squeeze(BehaviorResults['TimeTunnelStart'])
    EnterWhiteSpaceTime = np.squeeze(BehaviorResults['EnterWhiteSpaceTime'])
    
    if 'WallIsProbe' in BehaviorResults.keys():
        WallIsProbe = np.squeeze(BehaviorResults['WallIsProbe'])

    WallType = np.squeeze(BehaviorResults['WallType'])
    ID_wall = np.squeeze(BehaviorResults['ID_wall'])
    refs = Timeline['BehaviorResults']['Name_wall'][()].flatten()
    WallName = read_strings(Timeline['BehaviorResults'], refs)
    WallName = np.array(WallName)
    Category1_names = np.unique(WallName[WallType==1])
    Category2_names = np.unique(WallName[WallType==2])

    uniqWall_1 = np.unique(ID_wall[WallType==1])
    uniqWall_2 = np.unique(ID_wall[WallType==2])

    images = h5pyfield(Timeline,BehaviorResults['AllImages'])
    Image1RewardTime = np.squeeze(BehaviorResults['Image1RewardTime'])
    Image2RewardTime = np.squeeze(BehaviorResults['Image2RewardTime'])
    Licks = np.squeeze(BehaviorResults['Licks'])
    
    SoundTime = np.squeeze(BehaviorResults['SoundTime'])

    VRMove = h5pyfield(Timeline,BehaviorResults['VRMove'])
    VRM = np.hstack(VRMove)
    SubjMove = h5pyfield(Timeline,BehaviorResults['SubjMove'])
    SubjM = np.hstack(SubjMove)
    SubjM[0,:] = np.cumsum(np.abs(SubjM[0,:])) 
    SubjM[1,:] = np.cumsum(SubjM[1,:])
    
    # get frame time using interpolation
    ind = np.where(data[:,1]>0)[0]
    t = data[ind,1]
    frame_start_temp = (data[:-1,0] > 2.5) & (data[1:,0] <= 2.5)
#     frame_start = np.where(frame_start_temp==1)[0][:frames]
    frame_start = np.where(frame_start_temp==1)[0]
    ft = np.interp(frame_start, ind, t)    
    # interpolate movement
    subRun = np.interp(ft, SubjM[4,:], np.abs(SubjM[0,:]))
    subRun = np.insert(np.diff(subRun),0,0)
    subRun = subRun/subRun.max()
    subRot = np.interp(ft, SubjM[4,:], SubjM[1,:])   
    subRot = np.insert(np.diff(subRot),0,0)
    subRot = subRot/subRot.max()
    
#     frameInd = list(range(frames))
    frameInd = list(range(ft.shape[0]))
    TunnelStartInd = np.interp(TimeTunnelStart,ft,frameInd)
    whiteSpcFrameInd = np.interp(EnterWhiteSpaceTime, ft,frameInd)    
    
    Image1RewInd = np.interp(Image1RewardTime[0,np.invert(np.isnan(Image1RewardTime[0,:]))],ft,frameInd)
    Image2RewInd = np.interp(Image2RewardTime[0,np.invert(np.isnan(Image2RewardTime[0,:]))],ft,frameInd)
    LickInd = np.interp(Licks[0,:],ft,frameInd)    
    
    soundInd = np.interp(SoundTime[2,:],ft,frameInd)
    
    VRpos = np.interp(ft,VRM[-1,:],np.abs(VRM[-2,:]))
    VRpos = VRpos/VRpos.max()
    
    WallType1 = TunnelStartInd[WallType==1]
    WallType2 = TunnelStartInd[WallType==2]  
    WallType1_stim, WallType2_stim = [],[]
    for i in range(uniqWall_1.shape[0]):
        WallType1_stim.append(TunnelStartInd[(WallType==1) & (ID_wall==uniqWall_1[i])])
    for i in range(uniqWall_2.shape[0]):
        WallType2_stim.append(TunnelStartInd[(WallType==2) & (ID_wall==uniqWall_2[i])])     
        
        
    results = {}    
    #results = GetWall(Timeline)
    results['imgs'] = Timeline['img']
    results['subRun'] = subRun
    results['subRot'] = subRot
    results['whiteSpcStart'] = whiteSpcFrameInd  
    
    results['Image1Rew'] = Image1RewInd
    results['Image2Rew'] = Image2RewInd
    results['LickInd'] = LickInd      
    
    results['VRpos'] = VRpos
    results['ID_wall'] = ID_wall
    results['WallType1_img'] = images[2]
    results['WallType2_img'] = images[3]
    results['WallType1'] = WallType1
    results['WallType2'] = WallType2
    results['WallType1_stim'] = WallType1_stim
    results['WallType2_stim'] = WallType2_stim    
    results['Category1_names'] = Category1_names
    results['Category2_names'] = Category2_names
    
    results['Time_licks'] = Licks
    results['TimeTunnelStart'] = TimeTunnelStart
    results['TimeWhiteSpaceStart'] = EnterWhiteSpaceTime
    results['WallType'] = WallType
    if 'WallIsProbe' in BehaviorResults.keys():
        results['WallIsProbe'] = WallIsProbe
    results['SoundTime'] = SoundTime
    results['SoundInd'] = soundInd
    
    results['FrameInd'] = frameInd
    
    return results
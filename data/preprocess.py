import librosa
import os
import numpy as np
import pickle
import random
import pandas as pd

from tqdm.notebook import tqdm
import joblib

import torch
import torchlibrosa as tl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_Metadata():
    #get list of all files with labels
    data_path = './birdclef-2021/train_short_audio/'
    
    df = pd.read_csv('./birdclef-2021/train_metadata.csv')
    blist=df['primary_label'].unique()
    
    call_list=list()
    for sec,pri,filename in zip(df['secondary_labels'],df['primary_label'],df['filename']):
        sec_2 = list(sec.replace("'",'').replace('[','').replace(']','').split(','))
        sec_2.append(pri)
        if sec_2[0]=='':
            sec_2=sec_2[1:]
        call_list.append([os.path.join(data_path,pri,filename),sec_2])
    
    blist=list(blist)
    
    return call_list, blist

def get_feature_extractor():
    SPEC_HEIGHT = 128
    SPEC_WIDTH = 256
    NUM_MELS = SPEC_HEIGHT
    HOP_LENGTH = int(32000 * 5 / (SPEC_WIDTH - 1)) # sample rate * duration / spec width - 1 == 627
    FMIN = 500
    FMAX = 12500
    
    feature_extractor = torch.nn.Sequential(
    tl.Spectrogram(
        n_fft=2048,
        hop_length=HOP_LENGTH,
        freeze_parameters=True
    ), tl.LogmelFilterBank(
        sr=32000,
        n_mels=NUM_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )).to(device)
        
    return feature_extractor

def get_call_window(mel,duration=256,mode='precise'):
    #Todo : implementing get multiple windows option would be needed
    
    if not mode in ['precise','fast','collect']:
      raise ValueError('get_call_window mode parameter allow "precise" and "fast"')

    mean = mel.mean()
    if mode=='collect':
      call = list()
    else:
      call = None

    call_mean = 0
    i = 0
    while (i+duration)<len(mel):
      i_mean = mel[i:i+duration].mean()
      if i_mean > mean:
        if mode=='collect':
          call.append(mel[i:i+duration].T[np.newaxis,:])
          i+=256
          continue
        elif mode=='precise':
          if call_mean < i_mean:
            call = mel[i:i+duration].T[np.newaxis,:]
            call_mean = i_mean
        else:
          if call is None:
            call = mel[i:i+duration].T[np.newaxis,:]
      if mode == 'fast' and call is not None:
        return call
      i+=50
      
    if(len(call) > 1):
      return np.concatenate(call)
    elif(len(call) == 1):
      return np.array(call).squeeze(axis=0)
    else:
      return call

def _normalize(S, min_db):
    return np.clip((S - min_db) / -min_db, 0, 1)

def preprocess(data, feature_extractor, blist):
    wav, _ =librosa.load(data[0],sr=32000)
    wav = torch.from_numpy(wav.reshape((1,)+ wav.shape))

    if torch.cuda.is_available():
        mel_spec = feature_extractor(wav.cuda()).cpu()
    else:
        mel_spec = feature_extractor(wav)

    mel_spec = mel_spec.squeeze().numpy().T
    mel_spec = _normalize(mel_spec, mel_spec.min())

    mel_list = get_call_window(mel_spec.T, mode='collect')

    BCencoding = np.zeros((len(blist)))

    for bird in data[1]:
        if bird in blist:
            BCencoding[blist.index(bird)] = 1

    return [mel_list, BCencoding]

def main():
    call_list, blist = load_Metadata()
    feature_extractor = get_feature_extractor()
    
    PART_INDEXES = [0,5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, -1]
    random.shuffle(call_list)

    for i in range(0, len(PART_INDEXES)-1):
      new_list = joblib.Parallel(n_jobs=8,prefer="threads")\
      (joblib.delayed(preprocess)(data, feature_extractor, blist)\
       for data in tqdm(call_list[PART_INDEXES[i]:PART_INDEXES[i+1]]))
    
      with open('./asset/temp_{0}.pkl'.format(i),'wb') as f:
          pickle.dump(new_list,f)
    
      del new_list
      
if __name__ == "__main__":
    main()
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import pickle

class BirdCallDataset(Dataset):
    
    def __init__(self, pkl_dir):
        if not os.path.exists(pkl_dir):
            raise OSError(f'No such Pickle directory {pkl_dir}')
        
        pkl_list = [i for i in os.listdir(pkl_dir) if i.split('.')[-1]=='pkl']
        
        pkl_dic = dict()
        num_all_file = 0
        self.dataset=list()
        for pkl in pkl_list:
            print(f'File_{pkl} Loading...')
            with open(os.path.join(pkl_dir,pkl), 'rb') as f:
                temp = pickle.load(f)
                for bird in temp:
                    self.dataset.append(bird)
                pkl_dic[pkl] = len(temp)
                num_all_file += len(temp)
        
        self.pkl_dir = pkl_dir
        self.pkl_dic = pkl_dic
        self.num_all_file = num_all_file
        
    def __len__(self):
        return self.num_all_file
    
    
    def __getitem__(self, idx):
        wav = self.dataset[idx][0][np.newaxis,:,:]
        bird = self.dataset[idx][1]
        return (wav,bird)
    
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

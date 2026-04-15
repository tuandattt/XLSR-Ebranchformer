import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature
from utils import pad
import soundfile as sf
			
class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=66800
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        try:
            X, fs = sf.read(self.base_dir+'flac/'+utt_id+'.flac')
        except:
            try:
                X, fs = librosa.load(path, sr=16000)
            except Exception as e:
                print(f"[ERROR] Cannot read file: {path}")
                print(f"Reason: {e}")
                raise e
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target
class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        path = self.base_dir + '/' + utt_id + '.flac'

        X,fs = librosa.load(path, sr=16000) 


        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id

class Dataset_train_var(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args   
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_rs = np.reshape(Y,(1,-1))
        target = self.labels[utt_id]
        return X_rs, target, None

class Dataset_eval_var(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.track=track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
        X_rs = np.reshape(X,(1,-1))
        return X_rs, None, utt_id  

class Dataset_eval_in_the_wild(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id

class Dataset_eval_in_the_wild_var(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.track=track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_rs = np.reshape(X,(1,-1))
        return X_rs, None, utt_id  

class Dataset_train5(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=66800
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        speaker = utt_id.split("_")[0]
        try:
            X, fs = sf.read(self.base_dir + utt_id + '.flac')
        except:
            X, fs = librosa.load(self.base_dir + utt_id + '.flac', sr=16000)
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target
        
class Dataset_eval5(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        try:
            X, fs = sf.read(self.base_dir + utt_id + '.flac')
        except:
            X, fs = librosa.load(self.base_dir + utt_id + '.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


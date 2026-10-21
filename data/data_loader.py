import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag="train", size=None,
                 features="MS", data_path="ETTh1.csv",
                 target="OT", scale=True, inverse=False,
                 timeenc=0, freq="h", cols=None):
        # Default seq_len=96 (matches training-cell --seq_len 96)
        if size is None:
            self.seq_len   = 96
            self.label_len = 48
            self.pred_len  = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size
        self.flag      = flag
        self.features  = features
        self.target    = target
        self.scale     = scale
        self.inverse   = inverse
        self.timeenc   = timeenc
        self.freq      = freq
        self.cols      = cols
        self.root_path = root_path
        self.data_path = data_path
        self.data_scaler   = StandardScaler()
        self.target_scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            other = list(df_raw.columns)
            other.remove("date")
            other.remove(self.target)
            df_raw = df_raw[["date"] + other + [self.target]]

        train_len = 12 * 30 * 24
        val_len   =  4 * 30 * 24
        test_len  =  4 * 30 * 24
        border1s = [0,
                    train_len - self.seq_len,
                    train_len + val_len - self.seq_len,
                    len(df_raw) - self.seq_len]
        border2s = [train_len,
                    train_len + val_len,
                    train_len + val_len + test_len,
                    len(df_raw)]
        idx = {"train": 0, "val": 1, "test": 2, "pred": 3}[self.flag]
        border1, border2 = border1s[idx], border2s[idx]
        tb1, tb2 = border1s[0], border2s[0]

        if self.features == "MS":
            df_x = df_raw[df_raw.columns[1:]]
            df_y = df_raw[[self.target]]
        elif self.features == "M":
            df_x = df_raw[df_raw.columns[1:]]
            df_y = df_x
        else:
            df_x = df_raw[[self.target]]
            df_y = df_raw[[self.target]]

        if self.scale:
            self.data_scaler.fit(df_x.iloc[tb1:tb2].values)
            self.target_scaler.fit(df_y.iloc[tb1:tb2].values)
            data_x = self.data_scaler.transform(df_x.values)
            data_y = self.target_scaler.transform(df_y.values)
        else:
            data_x = df_x.values
            data_y = df_y.values

        self.data_x    = data_x[border1:border2]
        self.data_y    = data_y[border1:border2]
        df_stamp       = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

    def __getitem__(self, idx):
        s  = idx
        e  = s + self.seq_len
        r  = e - self.label_len
        r2 = r + self.label_len + self.pred_len
        return self.data_x[s:e], self.data_y[r:r2], self.data_stamp[s:e], self.data_stamp[r:r2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag="train", size=None,
                 features="MS", data_path="ETTm1.csv",
                 target="OT", scale=True, inverse=False,
                 timeenc=0, freq="t", cols=None):
        if size is None:
            self.seq_len   = 96
            self.label_len = 48
            self.pred_len  = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag; self.features = features; self.target = target
        self.scale = scale; self.inverse = inverse; self.timeenc = timeenc
        self.freq = freq; self.cols = cols
        self.root_path = root_path; self.data_path = data_path
        self.data_scaler = StandardScaler(); self.target_scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy(); cols.remove(self.target)
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            other = list(df_raw.columns); other.remove("date"); other.remove(self.target)
            df_raw = df_raw[["date"] + other + [self.target]]
        train_len = 12*30*24*4; val_len = 4*30*24*4; test_len = 4*30*24*4
        border1s = [0, train_len-self.seq_len, train_len+val_len-self.seq_len, len(df_raw)-self.seq_len]
        border2s = [train_len, train_len+val_len, train_len+val_len+test_len, len(df_raw)]
        idx = {"train":0,"val":1,"test":2,"pred":3}[self.flag]
        border1, border2 = border1s[idx], border2s[idx]; tb1, tb2 = border1s[0], border2s[0]
        if self.features=="MS":
            df_x = df_raw[df_raw.columns[1:]]; df_y = df_raw[[self.target]]
        elif self.features=="M":
            df_x = df_raw[df_raw.columns[1:]]; df_y = df_x
        else:
            df_x = df_raw[[self.target]]; df_y = df_raw[[self.target]]
        if self.scale:
            self.data_scaler.fit(df_x.iloc[tb1:tb2].values); self.target_scaler.fit(df_y.iloc[tb1:tb2].values)
            data_x = self.data_scaler.transform(df_x.values); data_y = self.target_scaler.transform(df_y.values)
        else:
            data_x = df_x.values; data_y = df_y.values
        self.data_x = data_x[border1:border2]; self.data_y = data_y[border1:border2]
        df_stamp = df_raw[["date"]][border1:border2]; df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

    def __getitem__(self, idx):
        s=idx; e=s+self.seq_len; r=e-self.label_len; r2=r+self.label_len+self.pred_len
        return self.data_x[s:e], self.data_y[r:r2], self.data_stamp[s:e], self.data_stamp[r:r2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag="train", size=None,
                 features="MS", data_path="custom.csv",
                 target="OT", scale=True, inverse=False,
                 timeenc=0, freq="h", cols=None):
        if size is None:
            self.seq_len=96; self.label_len=48; self.pred_len=24
        else:
            self.seq_len, self.label_len, self.pred_len = size
        self.flag=flag; self.features=features; self.target=target
        self.scale=scale; self.inverse=inverse; self.timeenc=timeenc
        self.freq=freq; self.cols=cols; self.root_path=root_path; self.data_path=data_path
        self.data_scaler=StandardScaler(); self.target_scaler=StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols=self.cols.copy(); cols.remove(self.target)
        else:
            cols=list(df_raw.columns); cols.remove("date"); cols.remove(self.target)
        df_raw = df_raw[["date"]+cols+[self.target]]
        total=len(df_raw); num_train=int(total*0.7); num_val=int(total*0.1); num_test=total-num_train-num_val
        border1s=[0,num_train-self.seq_len,num_train+num_val-self.seq_len,total-self.seq_len]
        border2s=[num_train,num_train+num_val,total,total]
        idx={"train":0,"val":1,"test":2,"pred":3}[self.flag]
        border1,border2=border1s[idx],border2s[idx]; tb1,tb2=border1s[0],border2s[0]
        if self.features=="MS":
            df_x=df_raw[df_raw.columns[1:]]; df_y=df_raw[[self.target]]
        elif self.features=="M":
            df_x=df_raw[df_raw.columns[1:]]; df_y=df_x
        else:
            df_x=df_raw[[self.target]]; df_y=df_raw[[self.target]]
        if self.scale:
            self.data_scaler.fit(df_x.iloc[tb1:tb2].values); self.target_scaler.fit(df_y.iloc[tb1:tb2].values)
            data_x=self.data_scaler.transform(df_x.values); data_y=self.target_scaler.transform(df_y.values)
        else:
            data_x=df_x.values; data_y=df_y.values
        self.data_x=data_x[border1:border2]; self.data_y=data_y[border1:border2]
        df_stamp=df_raw[["date"]][border1:border2]; df_stamp["date"]=pd.to_datetime(df_stamp.date)
        self.data_stamp=time_features(df_stamp,timeenc=self.timeenc,freq=self.freq)

    def __getitem__(self,idx):
        s=idx;e=s+self.seq_len;r=e-self.label_len;r2=r+self.label_len+self.pred_len
        return self.data_x[s:e],self.data_y[r:r2],self.data_stamp[s:e],self.data_stamp[r:r2]

    def __len__(self):
        return len(self.data_x)-self.seq_len-self.pred_len+1

    def inverse_transform(self,data):
        return self.target_scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag="pred", size=None,
                 features="MS", data_path="ETTm1.csv",
                 target="OT", scale=True, inverse=False,
                 timeenc=0, freq="h", cols=None):
        assert flag == "pred"
        if size is None:
            self.seq_len=96; self.label_len=48; self.pred_len=24
        else:
            self.seq_len,self.label_len,self.pred_len=size
        self.features=features; self.target=target; self.scale=scale
        self.inverse=inverse; self.timeenc=timeenc; self.freq=freq
        self.cols=cols; self.root_path=root_path; self.data_path=data_path
        self.data_scaler=StandardScaler(); self.target_scaler=StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw=pd.read_csv(os.path.join(self.root_path,self.data_path))
        if self.cols:
            cols=self.cols.copy(); cols.remove(self.target)
        else:
            cols=list(df_raw.columns); cols.remove("date"); cols.remove(self.target)
        df_raw=df_raw[["date"]+cols+[self.target]]
        total=len(df_raw); border1=total-self.seq_len; border2=total; train_end=int(total*0.7)
        if self.features=="MS":
            df_x=df_raw[df_raw.columns[1:]]; df_y=df_raw[[self.target]]
        elif self.features=="M":
            df_x=df_raw[df_raw.columns[1:]]; df_y=df_x
        else:
            df_x=df_raw[[self.target]]; df_y=df_raw[[self.target]]
        if self.scale:
            self.data_scaler.fit(df_x.iloc[:train_end].values); self.target_scaler.fit(df_y.iloc[:train_end].values)
            data_x=self.data_scaler.transform(df_x.values); data_y=self.target_scaler.transform(df_y.values)
        else:
            data_x=df_x.values; data_y=df_y.values
        self.data_x=data_x[border1:border2]; self.data_y=data_y[border1:border2]
        df_stamp=df_raw[["date"]][border1:border2]; df_stamp["date"]=pd.to_datetime(df_stamp.date)
        self.data_stamp=time_features(df_stamp,timeenc=self.timeenc,freq=self.freq)

    def __getitem__(self,idx):
        s=idx;e=s+self.seq_len;r=e-self.label_len;r2=r+self.label_len+self.pred_len
        return self.data_x[s:e],self.data_y[r:r+self.label_len],self.data_stamp[s:e],self.data_stamp[r:r2]

    def __len__(self):
        return len(self.data_x)-self.seq_len+1

    def inverse_transform(self,data):
        return self.target_scaler.inverse_transform(data)

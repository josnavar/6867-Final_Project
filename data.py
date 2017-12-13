import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import csv
dirty_data_dir="bump.csv"
out_dir="nuts"


def cleanRepetitions(input_dir,output_dir):
    with open(input_dir,"rb") as csvfile:
        with open(output_dir, "wb") as filer:
            writer = csv.writer(filer)
            reader=csv.reader(csvfile)
            previous_ts=0
            previous_price=-1
            previous_vol=-1
            count=0.0
            for row in reader:
                if int(row[0])>=1483312024:
                    if int(row[0])==previous_ts:
                        #Need to average out ts info
                        previous_price+=float(row[1])
                        previous_vol+=float(row[2])
                        count+=1.0
                    elif previous_price!=-1:
                        temp=[str(previous_ts/count),str(previous_price/count),
                                  str(previous_vol/count)]
                        writer.writerow(temp)
                    previous_ts=int(row[0])
                    previous_price=float(row[1])
                    previous_vol=float(row[2])
                    count=1.0
train_length=500
predict_length=50
class Test(data.Dataset):
    def __init__(self,dir,train_length,predict_length,cache_size,is_test=False):
        self.is_test=is_test
        self.dir=dir
        self.length=None
        self.train_length=train_length
        self.predict_length=predict_length
        self.cache_size=cache_size
        # Maintain a "cache" for the next cache_len calls
        self.cache=[]
        #The right is none-inclusive
        self.range=(0,cache_size)
        with open(dir,"rb") as csvfile:
            count=0
            reader=csv.reader(csvfile)
            for elt in reader:
                if count<cache_size+predict_length:
                    self.cache.append(elt)
                count+=1
            self.length=count
        csvfile.close()
        self.cache=np.asarray(self.cache,dtype="double")
    def __len__(self):
        return 1000
    def __getitem__(self,i):
        if self.is_test:
            i+=211000
        else:
            i+=210000
        if i < self.range[1] - self.train_length and i >= self.range[0]:
            x = self.cache[i - self.range[0]:i + self.train_length - self.range[0], :]
        else:
            # Update the cache
            temp = []
            with open(self.dir, "rb") as csvfile:
                reader = csv.reader(csvfile)
                for count, elt in enumerate(reader):
                    if count >= i and count < i + self.cache_size + self.predict_length:
                        temp.append(elt)
                    if count >= i + self.cache_size + self.predict_length:
                        break
            csvfile.close()
            self.cache = np.asarray(temp, dtype="double")
            self.range = (i, i + self.cache_size)
            x = self.cache[i - self.range[0]:i - self.range[0] + self.train_length, :]
        ultimo = [x[-1, 1]]
        y = self.get_predictors(i)
        return {"x": torch.from_numpy(np.asarray(x, dtype="double")[:, 1:]),
                "y": y, "ultimo": torch.from_numpy(np.asarray(ultimo, dtype="double"))}
    def get_predictors(self, i):
        # always called after __getitem__ hence always in bounds
        stuff = self.cache[
                i - self.range[0] + self.train_length:i - self.range[0] + self.train_length + self.predict_length, 1]
        return torch.from_numpy(np.asarray(stuff))
class Data(data.Dataset):
    #Dir to dataset that has none repeating ts entries
    def __init__(self,dir,train_length,predict_length,cache_size):
        self.dir=dir
        self.length=None
        self.train_length=train_length
        self.predict_length=predict_length
        self.cache_size=cache_size
        # Maintain a "cache" for the next cache_len calls
        self.cache=[]
        #The right is none-inclusive
        self.range=(0,cache_size)
        with open(dir,"rb") as csvfile:
            count=0
            reader=csv.reader(csvfile)
            for elt in reader:
                if count<cache_size+predict_length:
                    self.cache.append(elt)
                count+=1
            self.length=count
        csvfile.close()
        self.cache=np.asarray(self.cache,dtype="double")
    def __len__(self):
        #return 24000
        return 10000
    def __getitem__(self,i):
        i+=100000
        if i<self.range[1]-self.train_length and i>=self.range[0]:
            x=self.cache[i-self.range[0]:i+self.train_length-self.range[0],:]
        else:
            #Update the cache
            temp=[]
            with open(self.dir,"rb") as csvfile:
                reader=csv.reader(csvfile)
                for count,elt in enumerate(reader):
                    if count>=i and count<i+self.cache_size+self.predict_length:
                        temp.append(elt)
                    if count>=i+self.cache_size+self.predict_length:
                        break
            csvfile.close()
            self.cache=np.asarray(temp,dtype="double")
            self.range=(i,i+self.cache_size)

            x=self.cache[i-self.range[0]:i-self.range[0]+self.train_length,:]
        ultimo=[x[-1,1]]
	primero=[x[0,1]]
        y=self.get_predictors(i)
        return {"x":torch.from_numpy(np.asarray(x,dtype="double")[:,1:]),
        "y":y, "ultimo":torch.from_numpy(np.asarray(ultimo,dtype="double")),
	"primero":torch.from_numpy(np.asarray(primero,dtype="double"))}
    def get_predictors(self,i):
        #always called after __getitem__ hence always in bounds
        stuff=self.cache[i-self.range[0]+self.train_length:i-self.range[0]+self.train_length+self.predict_length,1]
        return torch.from_numpy(np.asarray(stuff))




















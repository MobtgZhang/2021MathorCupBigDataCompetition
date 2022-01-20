import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def vectorize(item_dataset,entity_dict,names_dict,train,other):
    discrete_data = item_dataset[names_dict["discrete"]]
    continue_data = item_dataset[names_dict["continue"]]
    time_data = item_dataset[names_dict["time"]]
    
    discrete_list = []
    tmp_values = continue_data.values
    names_list = continue_data.index.values.tolist()
    continue_list = [names_list,tmp_values]
    time_data = time_data.tolist()
    time_data[-1] = str(time_data[-1])[:-2]+"-"+str(time_data[-1])[-2:] + "-01"
    time_data = [item.split("-") for item in time_data]
    time_list = [continue_data.index.values.tolist(),time_data]
    for name,value in list(zip(discrete_data.index.values,discrete_data.values)):
        ent_index = entity_dict[name,value]
        discrete_list.append(ent_index)
    if train and not other:
        target_data = item_dataset["价格"]
        return discrete_list,continue_list,time_list,target_data
    elif other:
        target_data = item_dataset["成交结果"]
        return discrete_list,continue_list,time_list,target_data
    else:
        return discrete_list,continue_list,time_list
def batchfy(batch):
    ent_list = [ex[0] for ex in batch]
    val_list = [ex[1][1] for ex in batch]
    time_year_list = [[int(ex[2][1][k][0])-2000 for k in range(len(ex[2][1]))] for ex in batch]
    time_month_list = [[int(ex[2][1][k][1])-1 for k in range(len(ex[2][1]))] for ex in batch]
    time_day_list = [[int(ex[2][1][k][2])-1 for k in range(len(ex[2][1]))] for ex in batch]
    target = [ex[3] for ex in batch]
    ent_tensor = torch.tensor(ent_list,dtype=torch.long)
    val_tensor = torch.tensor(val_list,dtype=torch.float)
    year_tensor = torch.tensor(time_year_list,dtype=torch.long)
    month_tensor = torch.tensor(time_month_list,dtype=torch.long)
    day_tensor = torch.tensor(time_day_list,dtype=torch.long)
    target_tensor = torch.tensor(target,dtype=torch.float)
    return ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor,target_tensor
def test_batchfy(batch):
    ent_list = [ex[0] for ex in batch]
    val_list = [ex[1][1] for ex in batch]
    time_year_list = [[int(ex[2][1][k][0])-2000 for k in range(len(ex[2][1]))] for ex in batch]
    time_month_list = [[int(ex[2][1][k][1])-1 for k in range(len(ex[2][1]))] for ex in batch]
    time_day_list = [[int(ex[2][1][k][2])-1 for k in range(len(ex[2][1]))] for ex in batch]

    ent_tensor = torch.tensor(ent_list,dtype=torch.long)
    val_tensor = torch.tensor(val_list,dtype=torch.float)
    year_tensor = torch.tensor(time_year_list,dtype=torch.long)
    month_tensor = torch.tensor(time_month_list,dtype=torch.long)
    day_tensor = torch.tensor(time_day_list,dtype=torch.long)
    return ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor
class CarDataset(Dataset):
    def __init__(self,save_dataset_file,ent_dict_file,names_file,normalize = True,method="std",train=True,other=False):
        assert method in ["std","maxmin"]
        super(CarDataset,self).__init__()
        self.ent_dict_file = ent_dict_file
        self.save_dataset_file = save_dataset_file
        self.normalize = normalize
        self.train = train
        self.other = other
        self.dataset = pd.read_excel(save_dataset_file)
        self.entity_dict = Dictionary.load(ent_dict_file)
        def tp_func(value):
            length,width,height = value.split("*")
            length,width,height = int(length),int(width),int(height)
            volume = length*width*height
            circle = 4*(length+width+height)
            square = 4*(length*width+length*height+width*height)
            return length,width,height,volume,square,circle
        output = np.array(list(map(tp_func,self.dataset["匿名特征12"])))
        name_list = ["车辆长度","车辆宽度","车辆高度","车辆占有体积","车辆占有表面积","车辆占有空间周长"]
        for index,name in enumerate(name_list):
            self.dataset[name] = output[:,index]
        with open(names_file,mode="r",encoding="utf-8") as rfp:
            self.names_dict = json.load(rfp)
        self.names_dict["continue"] += name_list
        self.values_list = dict()
        if normalize:
            if method == "std":
                for name in self.names_dict["continue"]:
                    df = self.dataset[name]
                    std_value = df.std()
                    mean_value = df.mean()
                    self.dataset[name] = ((df-mean_value)/std_value).tolist()
            elif method == "maxmin":
                for name in self.names_dict["continue"]:
                    df = self.dataset[name]
                    max_value = df.max()
                    min_value = df.min()
                    self.dataset[name] = ((df-min_value)/(max_value-min_value)).tolist()
            if train and normalize and not other:
                df = self.dataset["价格"]
                std_value = df.std()
                mean_value = df.mean()
                self.values_list["价格"] = [std_value,mean_value]
                self.dataset["价格"] = ((df-mean_value)/std_value).tolist()
            
    def __getitem__(self,item):
        return vectorize(self.dataset.loc[item,:],self.entity_dict,self.names_dict,self.train,self.other)
    def __len__(self):
        return len(self.dataset)
class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = []
        self.token2ind = {}
    def __getitem__(self,item):
        #print(type(item),item)
        if type(item) == tuple and len(item) == 2:
            word = item[1]
            name = item[0]
            item_label = str(name) + ":" +str(word)
            return self.token2ind.get(item_label)
        elif type(item) == str:
            item_label = item
            return self.token2ind.get(item_label)
        elif type(item) == int:
            item_label = self.ind2token[item]
            name,word = item_label.split(":")
            return word
        else:
            raise IndexError()
    def add(self,word,name=None):
        if name is not None:
            word_label = name + ":" +str(word)
        else:
            word_label = word
        if word_label not in self.token2ind:
            self.token2ind[word_label] = len(self.ind2token)
            self.ind2token.append(word_label)
    def save(self,save_file):
        with open(save_file,"w",encoding="utf-8") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
            }
            json.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding="utf-8") as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
        return tp_dict
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))

class CarDatasetExtenstion(Dataset):
    def __init__(self,save_dataset_file,ent_dict_file,names_file,normalize = True,method="std",train=True):
        assert method in ["std","maxmin"]
        super(CarDatasetExtenstion,self).__init__()
        self.ent_dict_file = ent_dict_file
        self.save_dataset_file = save_dataset_file
        self.normalize = normalize
        self.train = train
        self.dataset = pd.read_excel(save_dataset_file)
        self.entity_dict = Dictionary.load(ent_dict_file)
        def tp_func(value):
            length,width,height = value.split("*")
            length,width,height = int(length),int(width),int(height)
            volume = length*width*height
            circle = 4*(length+width+height)
            square = 4*(length*width+length*height+width*height)
            return length,width,height,volume,square,circle
        output = np.array(list(map(tp_func,self.dataset["匿名特征12"])))
        name_list = ["车辆长度","车辆宽度","车辆高度","车辆占有体积","车辆占有表面积","车辆占有空间周长"]
        for index,name in enumerate(name_list):
            self.dataset[name] = output[:,index]
        with open(names_file,mode="r",encoding="utf-8") as rfp:
            self.names_dict = json.load(rfp)
        self.names_dict["continue"] += name_list
        self.values_list = dict()
        if normalize:
            if method == "std":
                for name in self.names_dict["continue"]:
                    df = self.dataset[name]
                    std_value = df.std()
                    mean_value = df.mean()
                    self.dataset[name] = ((df-mean_value)/std_value).tolist()
            elif method == "maxmin":
                for name in self.names_dict["continue"]:
                    df = self.dataset[name]
                    max_value = df.max()
                    min_value = df.min()
                    self.dataset[name] = ((df-min_value)/(max_value-min_value)).tolist()
            
    def __getitem__(self,item):
        return vectorize_ext(self.dataset.loc[item,:],self.entity_dict,self.names_dict,self.train)
    def __len__(self):
        return len(self.dataset)
def vectorize_ext(item_dataset,entity_dict,names_dict,train):
    discrete_data = item_dataset[names_dict["discrete"]]
    continue_data = item_dataset[names_dict["continue"]]
    time_data = item_dataset[names_dict["time"]]
    
    discrete_list = []
    tmp_values = continue_data.values
    names_list = continue_data.index.values.tolist()
    continue_list = [names_list,tmp_values]
    time_data = time_data.tolist()
    time_data[-1] = str(time_data[-1])[:-2]+"-"+str(time_data[-1])[-2:] + "-01"
    time_data = [item.split("-") for item in time_data]
    time_list = [continue_data.index.values.tolist(),time_data]
    for name,value in list(zip(discrete_data.index.values,discrete_data.values)):
        ent_index = entity_dict[name,value]
        discrete_list.append(ent_index)
    if train :
        target_data_a = item_dataset["成交时间"]
        target_data_b = item_dataset["成交价格"]
        time_data = [item.split("-") for item in time_data]
        return discrete_list,continue_list,time_list,target_data
    else:
        return discrete_list,continue_list,time_list

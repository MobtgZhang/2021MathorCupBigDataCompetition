from cmath import log
from dataclasses import replace
import os
import json
import time
import datetime
from traceback import print_tb
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import sklearn


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
from .headers import get_headers_quesA,get_headers_quesB
from .data import Dictionary
def to_device(item,device):
    value_list = []
    for v_val in item:
        if type(v_val)==list or type(v_val)==tuple:
            tp_list = []
            for v in v_val:
                tp_list.append(v.to(device))
            value_list.append(tp_list)
        else:
            value_list.append(v_val.to(device))
    return value_list
def change_txt_to_xlsx(data_dir,save_dir):
    raw_dataset_file = os.path.join(save_dir,"raw_dataset.xlsx")
    if not os.path.exists(raw_dataset_file):
        train_file1 = os.path.join(data_dir,"附件1：估价训练数据.txt")
        header_list,meaning_list = get_headers_quesA()
        output = pd.read_csv(train_file1,sep="\t",header=None)
        
        output.columns = meaning_list
        
        output.to_excel(raw_dataset_file,index=None)
        logger.info("File saved in %s"%raw_dataset_file)
    
    test_dataset_file = os.path.join(save_dir,"test_dataset.xlsx")
    if not os.path.exists(test_dataset_file):
        train_file2 = os.path.join(data_dir,"附件2：估价验证数据.txt")
        header_list,meaning_list = get_headers_quesA()
        output = pd.read_csv(train_file2,sep="\t",header=None)
        output.columns = meaning_list[0:-1]
        output.to_excel(test_dataset_file,index=None)
        logger.info("File saved in %s"%test_dataset_file)
    other_dataset_file = os.path.join(save_dir,"other_dataset.xlsx")
    if not os.path.exists(other_dataset_file):
        train_file3 = os.path.join(data_dir,"附件4：门店交易训练数据.txt")
        all_dataset = []
        header_list,meaning_list = get_headers_quesB()
        with open(train_file3,mode="r") as rfp:
            for line in rfp:
                tp_list=line.strip().split("\t")
                if len(tp_list) == 5:
                    tp_list = tp_list + [""]
                elif len(tp_list) == 6:
                    pass
                else:
                    raise ValueError()
                all_dataset.append(tp_list)
        output = pd.DataFrame(all_dataset)
        output.columns = meaning_list
        output.to_excel(other_dataset_file,index=None)
        logger.info("File saved in %s"%other_dataset_file)
def build_dictionary(data_dir,result_dir,train_dataset,dev_dataset):
    property_file = os.path.join(data_dir,"property_zh.json")
    with open(property_file,mode="r",encoding="utf-8") as rfp:
        name_dict = json.load(rfp)
    discrete_list = name_dict["discrete"]
    continue_list = name_dict["continue"]
    all_set_list = dict()
    for name in discrete_list:
        name_setA = set(train_dataset.loc[:,name])
        name_setB = set(dev_dataset.loc[:,name])
        tp_set = name_setA | name_setB
        if name !='匿名特征11' and name !='匿名特征12':
            all_set_list[name] = [int(item) for item in tp_set]
        else:
            all_set_list[name] = list(tp_set)
    save_dictionary_file = os.path.join(result_dir,"dictionary.json")
    with open(save_dictionary_file,mode="w",encoding="utf-8") as wfp:
        json.dump(all_set_list,wfp)
    #data_dict = dict(zip(range(len(discrete_list)),discrete_list))
    #save_keys_file = os.path.join(result_dir,"keys.json")
    #with open(save_keys_file,mode="w",encoding="utf-8") as wfp:
    #    json.dump(data_dict,wfp)
    
def fix_dataset(result_dir):
    raw_dataset_file = os.path.join(result_dir,"raw_dataset.xlsx")
    test_dataset_file = os.path.join(result_dir,"test_dataset.xlsx")
    train_dataset = pd.read_excel(raw_dataset_file)
    dev_dataset = pd.read_excel(test_dataset_file)
    datasets = [train_dataset,dev_dataset]
    file_names = [raw_dataset_file,test_dataset_file]
    for tp_file,item_dataset in zip(file_names,datasets):
        # 2.国标码中有9个是空白，将其中修正为-1
        item_dataset.loc[:,"国标码"].fillna(value = -1,inplace=True)
        # 3.国别有3757个是空白，将其中修正为-1
        item_dataset.loc[:,"国别"].fillna(value = -1,inplace=True)
        # 4.厂商类型有3641个是空白，将其修正为-1
        item_dataset.loc[:,"厂商类型"].fillna(value = -1,inplace=True)
        # 5.年款有312个是空白，修正为-1
        item_dataset.loc[:,"年款"].fillna(value = -1,inplace=True)
        # 6.车辆id为48665 d的变速箱为空白，修正为-1
        item_dataset.loc[:,"变速箱"].fillna(value = 0,inplace=True)
        # 7.匿名特征1中有1582个是空白，将其修正为-1
        item_dataset.loc[:,"匿名特征1"].fillna(value = -1,inplace=True)
        # 8.匿名特征4中有12108个是空白，将其修正为-1
        item_dataset.loc[:,"匿名特征4"].fillna(value = -1,inplace=True)
        # 9.匿名特征7中有大量空白，将其修正为2000-01-01
        item_dataset.loc[:,"匿名特征7"].fillna(value = "2000-01-01",inplace=True)
        # 10.匿名特征8中有大量空白，将其修正为-1
        item_dataset.loc[:,"匿名特征8"].fillna(value = -1,inplace=True)
        # 11.匿名特征9中有大量空白，将其修正为-1
        item_dataset.loc[:,"匿名特征9"].fillna(value = -1,inplace=True)
        # 12.匿名特征10中有大量空白，将其修正为-1
        item_dataset.loc[:,"国别"].fillna(value = -1,inplace=True)
        # 13.匿名特征11中有大量空白，将其修正为-1
        item_dataset.loc[:,"匿名特征10"].fillna(value = -1,inplace=True)
        # 14.匿名特征13中有大量空白，将其修正为190001
        item_dataset.loc[:,"匿名特征13"].fillna(value = "200001",inplace=True)
        # 15.匿名特征15中有大量空白，将其修正为1900-01-01
        item_dataset.loc[:,"匿名特征15"].fillna(value = "2000-01-01",inplace=True)
        # 15.匿名特征15中有大量空白，将其修正为 -1
        item_dataset.loc[:,"匿名特征11"].fillna(value = "-1",inplace=True)
        item_dataset.loc[:,"匿名特征12"].fillna(value = "0*0*0",inplace=True)
        item_dataset.to_excel(tp_file,index=None)
def build_graph(data_dir,result_dir):
    save_rel_ent_dir = os.path.join(result_dir,"graph")
    dictionary_file = os.path.join(result_dir,"dictionary.json")
    property_file = os.path.join(data_dir,"property_zh.json")
    # 创建关系
    save_relation2idx = os.path.join(save_rel_ent_dir,"relation2idx.json")
    # 创建实体
    save_entity2idx = os.path.join(save_rel_ent_dir,"entity2idx.json")
    # 创建实体关系
    save_head2relation2tail = os.path.join(save_rel_ent_dir,"head2relation2tail.csv")
    # 创建实体关系
    save_head2relation2tail_names = os.path.join(save_rel_ent_dir,"head2relation2tail-names.csv")
    with open(dictionary_file,mode="r",encoding="utf-8") as rfp:
        data_dict = json.load(rfp)
    entity_dict = Dictionary()
    relation_dict = Dictionary()
    for ri,name in enumerate(data_dict):
        relation_dict.add(name)
        for rj,item in enumerate(data_dict[name]):
            entity_dict.add(item,name)
    entity_dict.save(save_entity2idx)
    relation_dict.save(save_relation2idx)
    # 构建实体关系图
    raw_file = os.path.join(result_dir,"raw_dataset.xlsx")
    test_file = os.path.join(result_dir,"test_dataset.xlsx")
    train_dataset = pd.read_excel(raw_file)
    dev_dataset = pd.read_excel(test_file)
    train_len = len(train_dataset)
    name_list = list(data_dict.keys())
    ent2rel2ent = set()
    ent2rel2ent_withnames = set()
    for index in range(train_len):
        item = train_dataset.iloc[index,:]
        for id_i in range(len(name_list)):
            head = name_list[id_i]
            key_head = str(item[head])
            ent_head_id = entity_dict[head,key_head]
            for id_j in range(len(name_list)):
                if id_i==id_j:
                    continue
                tail = name_list[id_j]
                key_tail = str(item[tail])
                ent_tail_id = entity_dict[tail,key_tail]
                rel_id = relation_dict[tail]
                tp_tuple = (ent_head_id,rel_id,ent_tail_id)
                head_str = str(head) + ":" +str(key_head)
                tail_str = str(tail) + ":" +str(key_tail)
                tp_tuple_name = (head_str,tail,tail_str)
                ent2rel2ent_withnames.add(tp_tuple_name)
                ent2rel2ent.add(tp_tuple)
                    
    dev_len = len(dev_dataset)
    for index in range(dev_len):
        item = dev_dataset.iloc[index,:]
        for id_i in range(len(name_list)):
            head = name_list[id_i]
            key_head = str(item[head])
            ent_head_id = entity_dict[head,key_head]
            for id_j in range(id_i+1,len(name_list)):
                if id_i==id_j:
                    continue
                tail = name_list[id_j]
                key_tail = str(item[tail])
                ent_tail_id = entity_dict[tail,key_tail]
                rel_id = relation_dict[tail]
                head_str = str(head) + ":" +str(key_head)
                tail_str = str(tail) + ":" +str(key_tail)
                tp_tuple_name = (head_str,tail,tail_str)
                ent2rel2ent_withnames.add(tp_tuple_name)
                tp_tuple = (ent_head_id,rel_id,ent_tail_id)
                ent2rel2ent.add(tp_tuple)
                    
    e2r2e_dataset = pd.DataFrame(ent2rel2ent)
    e2r2e_withnames_dataset = pd.DataFrame(ent2rel2ent_withnames)
    e2r2e_dataset.columns = ['head_entity_index','relation_index','tail_entity_index']
    e2r2e_dataset.to_csv(save_head2relation2tail,index=None)
    e2r2e_withnames_dataset.columns = ['head_entity_index','relation_index','tail_entity_index']
    e2r2e_withnames_dataset.to_csv(save_head2relation2tail_names,index=None)
def split_dataset(raw_file,train_file,dev_file,percentage=0.70):
    df_dataset = pd.read_excel(raw_file)
    df_dataset = sklearn.utils.shuffle(df_dataset)
    train_len = int(percentage*len(df_dataset))
    train_dataset = df_dataset.iloc[:train_len,:]
    dev_dataset = df_dataset.iloc[train_len:,:]
    train_dataset.to_excel(train_file,index=None)
    logger.info("File saved in %s"%train_file)
    dev_dataset.to_excel(dev_file,index=None)
    logger.info("File saved in %s"%dev_file)

def write_to_txt(data_list,save_txt_filename):
    with open(save_txt_filename,mode="w",encoding="utf-8") as wfp:
        for item in data_list:
            wfp.write(str(item)+"\n")
    logger.info("Saved in file:%s"%save_txt_filename)
def cal_delta_time(date1,date2):
    date1=time.strptime(date1,"%Y-%m-%d")
    date2=time.strptime(date2,"%Y-%m-%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    date2=datetime.datetime(date2[0],date2[1],date2[2])
    return date2-date1

def filter_other_dataset(result_dir):
    other_data_filename = os.path.join(result_dir,"other_dataset.xlsx")
    raw_data_filename = os.path.join(result_dir,"raw_dataset.xlsx")
    # raw_data_filename = os.path.join(result_dir,"test_dataset.xlsx")
    raw_dataset = pd.read_excel(raw_data_filename)
    other_dataset = pd.read_excel(other_data_filename)
    raw_dataset.drop('价格',axis = 1,inplace = True)
    
    df_list = []
    result_list = []
    for index in tqdm(range(len(other_dataset))):
        result_data = raw_dataset[raw_dataset["车辆id"]==other_dataset.loc[index,"车辆id"]]
        value = 1 if not pd.isna(other_dataset.loc[index,"成交时间"]) else 0
        result_list.append(value)
        df_list.append(result_data)
    dp_data = pd.concat(df_list)
    result_list = np.array(result_list)
    dp_data["成交结果"] = result_list
    other_dataset.drop('车辆id',axis = 1,inplace = True)
    other_dataset.drop('{价格调整时间：调整后价格}',axis = 1,inplace = True)
    other_dataset.drop('下架时间(成交车辆下架时间和成交时间相同)',axis = 1,inplace = True)
    other_dataset.drop('成交时间',axis = 1,inplace = True)

    headers = np.hstack([dp_data.columns.values,other_dataset.columns.values])
    out_data = np.hstack([dp_data.values,other_dataset.values])
    fixed_dataset = pd.DataFrame(out_data,columns=headers)
    fixed_filename = os.path.join(result_dir,"fixed_dataset.xlsx")

    fixed_dataset.to_excel(fixed_filename,index=None)
    logger.info("File saved in %s"%fixed_filename)
def create_dataset(result_dir):
    other_data_filename = os.path.join(result_dir,"other_dataset.xlsx")
    raw_data_filename = os.path.join(result_dir,"raw_dataset.xlsx")
    # raw_data_filename = os.path.join(result_dir,"test_dataset.xlsx")
    raw_dataset = pd.read_excel(raw_data_filename)
    other_dataset = pd.read_excel(other_data_filename)
    raw_dataset.drop('价格',axis = 1,inplace = True)
    
    df_list = []
    df_other_list = []
    for index in tqdm(range(len(other_dataset))):
        result_data = raw_dataset[raw_dataset["车辆id"]==other_dataset.loc[index,"车辆id"]]
        if pd.isna(other_dataset.loc[index,"成交时间"]):
            continue
        df_other_list.append(other_dataset.loc[index,:].values)
        df_list.append(result_data)
    dp_data = pd.concat(df_list)
    df_other_list = pd.DataFrame(df_other_list,columns=other_dataset.columns)
    df_other_list.drop('车辆id',axis = 1,inplace = True)
    df_other_list.drop('下架时间(成交车辆下架时间和成交时间相同)',axis = 1,inplace = True)

    headers = np.hstack([dp_data.columns.values,df_other_list.columns.values])
    out_data = np.hstack([dp_data.values,df_other_list.values])
    fixed_dataset = pd.DataFrame(out_data,columns=headers)
    fixed_filename = os.path.join(result_dir,"fixed_dataset_ext.xlsx")

    fixed_dataset.to_excel(fixed_filename,index=None)
    logger.info("File saved in %s"%fixed_filename)
""""""
def fix_ext_dataset(result_dir):
    fixed_filename = os.path.join(result_dir,"fixed_dataset_ext.xlsx")
    dataset = pd.read_excel(fixed_filename)
    price_list = []
    for k in range(len(dataset)):
        json_str = dataset.loc[k,"{价格调整时间：调整后价格}"]
        str_line = json_str.replace('"',"").replace('{','').replace('}','')
        value = str_line.split(":")
        if len(value)>=2:
            price_list.append(float(value[-1].strip()))
        elif len(value) == 1:
            value = dataset.loc[k,"上架价格"]
            price_list.append(float(value))
        else:
            print(value)
    dataset.drop("{价格调整时间：调整后价格}",axis=1,inplace=True)
    dataset["成交价格"] = price_list
    dataset.to_excel(fixed_filename)

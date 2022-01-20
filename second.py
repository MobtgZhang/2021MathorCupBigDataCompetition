from ast import arg
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_parse_args
from src.utils import filter_other_dataset,split_dataset,to_cuda,write_to_txt,create_dataset,fix_ext_dataset
from src.data import CarDataset,batchfy
from src.model import TabNet,TEIGANNClassifier
from src.evaluate import evaluate_probability

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
def train_dnn_judge_model(args):
    save_keys_file = os.path.join(args.data_dir,"property_fixed_zh.json")
    train_data_file = os.path.join(args.result_dir,"fixed_dataset.xlsx")
    dev_data_file = os.path.join(args.result_dir,"dev_fixed_dataset.xlsx")
    save_keys_file = os.path.join(args.data_dir,"property_zh.json")
    save_ent_dict_file = os.path.join(args.result_dir,"graph","entity2idx.json")
    #save_rel_dict_file = os.path.join(args.result_dir,"graph","relation2idx.json")
    train_dataset = CarDataset(train_data_file,ent_dict_file=save_ent_dict_file,names_file=save_keys_file,normalize=args.normalize,other=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size = args.batch_size,collate_fn=batchfy)
    logger.info("The length of train dataset is %d."%len(train_dataset))
    dev_dataset = CarDataset(dev_data_file,ent_dict_file=save_ent_dict_file,names_file=save_keys_file,other=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,shuffle=True,batch_size = args.batch_size,collate_fn=batchfy)
    logger.info("The length of dev dataset is %d."%len(dev_dataset))
    ent_vocab_size = len(train_dataset.entity_dict)
    value_size = len(train_dataset.names_dict["continue"])
    ent_size = len(train_dataset.names_dict["discrete"])
    time_size = len(train_dataset.names_dict["time"])
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    torch.cuda.set_device(0)
    # 模型准备
    
    model = TEIGANNClassifier(ent_vocab_size=ent_vocab_size,value_size=value_size,ent_size=ent_size,time_embedding_dim=30,time_size=time_size,
            ent_dim=args.embedding_dim,year_span = args.year_span,linear_dim=args.linear_dim,hidden_dim = args.hidden_dim)
    model_dir = os.path.join(args.result_dir,"model")
    embedding_file = os.path.join(model_dir,'embedding.npz')
    output = np.load(embedding_file)
    if os.path.exists(embedding_file):
        model.ent_embed.from_pretrained(torch.from_numpy(output["ent_embedding"]))
    
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adamax(model.parameters(),lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.6)
    loss_list = []
    f1_list = []
    for epoch in range(args.epoch_times):
        model.to(device)
        model.train()
        loss_all = 0.0
        for item in train_dataloader:
            optimizer.zero_grad()
            ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor,target_tensor = to_cuda(item,device)
            predict_tensor = model(ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor)
            
            loss = loss_fn(predict_tensor,target_tensor)
            loss.backward()
            optimizer.step()
            val_loss = loss.item()
            loss_all += val_loss
        f1_score_value = evaluate_probability(dev_dataloader,model,device)
        f1_list.append(f1_score_value)
        loss_all /= len(train_dataloader)
        loss_list.append(loss_all)
        logger.info("epoch is %d, loss is %0.4f,f1 score is %0.4f"%(epoch+1,loss_all,f1_score_value))
        scheduler.step()
    save_txt_filename = os.path.join(args.result_dir,"pearson.txt")
    write_to_txt(f1_list,save_txt_filename)
    save_txt_filename = os.path.join(args.result_dir,"loss.txt")
    write_to_txt(loss_list,save_txt_filename)
    
def train_tabnet_model(args):
    ent_rel_dir = os.path.join(args.result_dir,"graph")
    device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    torch.cuda.set_device(0)

if __name__ == "__main__":
    args = get_parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = os.path.join(args.log_dir,rq + '.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.info(str(args))
    fixed_filename = os.path.join(args.result_dir,"fixed_dataset.xlsx")
    if not os.path.exists(fixed_filename):
        filter_other_dataset(args.result_dir)
        raw_file = os.path.join(args.result_dir,"fixed_dataset.xlsx")
        train_file = os.path.join(args.result_dir,"train_fixed_dataset.xlsx")
        dev_file = os.path.join(args.result_dir,"dev_fixed_dataset.xlsx")
        split_dataset(raw_file,train_file,dev_file,percentage=args.percentage)
        create_dataset(args.result_dir)
        raw_file = os.path.join(args.result_dir,"fixed_dataset_ext.xlsx")
        train_file = os.path.join(args.result_dir,"train_fixed_dataset_ext.xlsx")
        dev_file = os.path.join(args.result_dir,"dev_fixed_dataset_ext.xlsx")
        split_dataset(raw_file,train_file,dev_file,percentage=args.percentage)
    fix_ext_dataset(args.result_dir)
    #train_dnn_judge_model(args)
    train_tabnet_model(args)
    
    
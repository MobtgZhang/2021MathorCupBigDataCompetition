import os
import uuid
import time
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from src.data import CarDataset
from src.data import batchfy,test_batchfy
from src.utils import to_device,write_to_txt
from src.evaluate import evaluate_dataset,draw_result,test_raw_dataset
from src.model import IGANN,TEIGANN
from config import get_parse_args

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train_dnn_model(args):
    train_data_file = os.path.join(args.result_dir,"train_dataset.xlsx")
    dev_data_file = os.path.join(args.result_dir,"dev_dataset.xlsx")
    save_keys_file = os.path.join(args.data_dir,"property_zh.json")
    save_ent_dict_file = os.path.join(args.result_dir,"graph","entity2idx.json")
    #save_rel_dict_file = os.path.join(args.result_dir,"graph","relation2idx.json")
    train_dataset = CarDataset(train_data_file,ent_dict_file=save_ent_dict_file,names_file=save_keys_file,normalize=args.normalize)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size = args.batch_size,collate_fn=batchfy)
    logger.info("The length of train dataset is %d."%len(train_dataset))
    dev_dataset = CarDataset(dev_data_file,ent_dict_file=save_ent_dict_file,names_file=save_keys_file)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,shuffle=True,batch_size = args.batch_size,collate_fn=batchfy)
    logger.info("The length of dev dataset is %d."%len(dev_dataset))
    ent_vocab_size = len(train_dataset.entity_dict)
    value_size = len(train_dataset.names_dict["continue"])
    ent_size = len(train_dataset.names_dict["discrete"])
    time_size = len(train_dataset.names_dict["time"])
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    torch.cuda.set_device(0)
    if args.model_name == "IGANN":
        model = IGANN(ent_vocab_size=ent_vocab_size,value_size=value_size,ent_size=ent_size,time_size=time_size,
                ent_dim=args.embedding_dim,time_dim_list =args.time_dim_list,year_span = args.year_span,linear_dim=args.linear_dim,hidden_dim = args.hidden_dim)
    elif args.model_name == "TEIGANN":
        model = TEIGANN(ent_vocab_size=ent_vocab_size,value_size=value_size,ent_size=ent_size,time_embedding_dim=30,time_size=time_size,
            ent_dim=args.embedding_dim,year_span = args.year_span,linear_dim=args.linear_dim,hidden_dim = args.hidden_dim)
        
    else:
        raise Exception("Error for the model name %s"%args.model_name)
    model_dir = os.path.join(args.result_dir,"model")
    embedding_file = os.path.join(model_dir,'embedding.npz')
    output = np.load(embedding_file)
    if os.path.exists(embedding_file):
        model.ent_embed.from_pretrained(torch.from_numpy(output["ent_embedding"]))
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.6)
    score_list = []
    loss_list = []
    accuracy_list = []
    mape_list = []
    best_score = 0.0
    model_uuid = uuid.uuid1()
    for epoch in range(args.epoch_times):
        model.to(device)
        model.train()
        loss_all = 0.0
        for item in train_dataloader:
            optimizer.zero_grad()
            ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor,target_tensor = to_device(item,device)
            predict_tensor = model(ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor)
            loss = loss_fn(predict_tensor,target_tensor)
            loss.backward()
            optimizer.step()
            
            val_loss = loss.item()
            loss_all += val_loss
        # evaluate the dataset
        score,acc_score,mape_score,mse_score = evaluate_dataset(model,dev_dataloader,device)
        if best_score<acc_score and score <1.0 and score>0.0:
            model_file_name = os.path.join(model_dir,"IGNN_"+ str(model_uuid).upper() +"_model.ckpt")
            torch.save(model.cpu(),model_file_name)
            best_score = acc_score
            args.best_score = score
            args.model_file_name = model_file_name
            logger.info(str(args))
        loss_all /= len(dev_dataloader)
        score_list.append(score)
        loss_list.append(loss_all)
        accuracy_list.append(acc_score)
        mape_list.append(mape_score)
        logger.info("epoch is %d, loss is %0.4f,accuracy is %0.4f,purge accuracy is %0.4f,mape is %0.4f,mse is %0.4f"%(epoch+1,loss_all,score,acc_score,mape_score,mse_score))
        scheduler.step()
    title = "IGNN Net training process"
    ylabel = "Accuracy score"
    xlabel = "Model training times"
    save_fig_name = os.path.join(model_dir,"IGNN-accuracy.png")
    draw_result(score_list,title,ylabel,xlabel,save_fig_name)
    save_txt_name = os.path.join(model_dir,"IGNN-accuracy.txt")
    write_to_txt(score_list,save_txt_name)

    ylabel = "Loss value"
    save_fig_name = os.path.join(model_dir,"IGNN-loss.png")
    draw_result(loss_list,title,ylabel,xlabel,save_fig_name)
    save_txt_name = os.path.join(model_dir,"IGNN-loss.txt")
    write_to_txt(loss_list,save_txt_name)

    ylabel = "Accuracy<0.05 value"
    save_fig_name = os.path.join(model_dir,"IGNN-accuracy-ext.png")
    draw_result(accuracy_list,title,ylabel,xlabel,save_fig_name)
    save_txt_name = os.path.join(model_dir,"IGNN-IGNN-accuracy-ext.txt")
    write_to_txt(accuracy_list,save_txt_name)

    ylabel = "Mape value"
    save_fig_name = os.path.join(model_dir,"IGNN-accuracy-mape.png")
    draw_result(mape_list,title,ylabel,xlabel,save_fig_name)
    save_txt_name = os.path.join(model_dir,"IGNN-accuracy-mape.txt")
    write_to_txt(mape_list,save_txt_name)
def test_IGNN(args):
    # 测试模型
    model = torch.load(args.model_file_name)
    test_data_file = os.path.join(args.result_dir,"test_dataset.xlsx")
    save_keys_file = os.path.join(args.data_dir,"property_zh.json")
    save_ent_dict_file = os.path.join(args.result_dir,"graph","entity2idx.json")
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    torch.cuda.set_device(0)
    test_dataset = CarDataset(test_data_file,ent_dict_file=save_ent_dict_file,names_file=save_keys_file,normalize=args.normalize,train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = args.test_batch_size,collate_fn=test_batchfy)
    test_raw_dataset(args,model,test_dataloader,device)
def main():
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
    model_dir = os.path.join(args.result_dir,"model")
    args.embedding_file = os.path.join(model_dir,'embedding.npz')
    args.model_file_name = None
    for filename in os.listdir(model_dir):
        name_list = filename.split(".")
        filename = os.path.join(model_dir,filename)
        if os.path.isfile(filename) and name_list[1] == "ckpt": 
            args.model_file_name = filename
            break
    if args.model_file_name is None:
        train_dnn_model(args)
    test_IGNN(args)
if __name__ == "__main__":
    main()

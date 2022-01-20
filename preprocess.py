import json
import os
import time
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm,trange
import matplotlib.pyplot as plt

import torch
import torch_geometric

from src.utils import split_dataset,write_to_txt
from src.utils import change_txt_to_xlsx
from src.model import RGCN
from src.utils import build_dictionary,fix_dataset,build_graph
from src.samples import load_split_data,build_test_graph,generate_sampled_graph_and_labels
from src.evaluate import valid_mrr_dataset,draw_result
from config import get_parse_args

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train_rgcn_model(args):
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        torch.cuda.set_device(0)
    else:
         torch.device('cpu')
    
    ent_dict, rel_dict, train_triplets, valid_triplets = load_split_data(args.result_dir,args.percentage)    
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets)))

    test_graph = build_test_graph(len(ent_dict), len(rel_dict), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    """
    test_graph_dataloader = torch_geometric.loader.NeighborLoader(
        test_graph,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[15]*2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=args.neighbor_batch_size,#args.batch_size,
    )
    """
    model = RGCN(len(ent_dict), len(rel_dict),embedding_dim=args.embedding_dim, num_bases=args.n_bases, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logger.info(model)
    model.to(device)
    num_entities = len(ent_dict)
    num_relations = len(rel_dict)
    mrr_list = []
    best_mrr = 0.0
    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):

        model.train()
        optimizer.zero_grad()

        train_data = generate_sampled_graph_and_labels(train_triplets, args.graph_batch_size, args.graph_split_size, num_entities, num_relations, args.negative_sample)

        train_data.to(device)

        entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
        loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + args.reg_ratio * model.reg_loss(entity_embedding)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        
        if epoch % args.evaluate_every == 0:
            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))
            model.eval()
            valid_mrr = valid_mrr_dataset(valid_triplets, model, test_graph, all_triplets,device)
            mrr_list.append(valid_mrr)
            if valid_mrr > best_mrr:
                
                best_mrr = valid_mrr
                model_dir = os.path.join(args.result_dir,"model") # 
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_file = os.path.join(model_dir,'best_mrr_model.pth')
                embedding_file = os.path.join(model_dir,'embedding.npz')
                np.savez(embedding_file,ent_embedding= model.entity_embedding.weight.numpy(),
                                rel_embedding = model.relation_embedding.numpy())
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_file)
    
    model_dir = os.path.join(args.result_dir,"model") # 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    embedding_file = os.path.join(model_dir,'embedding.npz')
    np.savez(embedding_file,ent_embedding= model.cpu().entity_embedding.weight.detach().numpy(),
                                rel_embedding = model.cpu().relation_embedding.detach().numpy())
    """
    title = "RGCN Net training for mrr score."
    ylabel = "MRR score"
    xlabel = "Model training times"
    save_fig_name = os.path.join(args.result_dir,"RGCN-mrr.png")
    draw_result(mrr_list,title,ylabel,xlabel,save_fig_name)
    save_txt_filename = os.path.join(args.result_dir,"RGCN-mrr.txt")
    write_to_txt(mrr_list,save_txt_filename)
    """
def preprocess(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        change_txt_to_xlsx(args.data_dir,args.result_dir)    
        fix_dataset(args.result_dir)
        # 将数据分为训练数据与测试数据
        raw_file = os.path.join(args.result_dir,"raw_dataset.xlsx")
        train_file = os.path.join(args.result_dir,"train_dataset.xlsx")
        dev_file = os.path.join(args.result_dir,"dev_dataset.xlsx")
        split_dataset(raw_file,train_file,dev_file,percentage=args.percentage)
    logger.info("The dataset saved in path:%s"%args.result_dir)
    logger.info("Fixed the dataset !")
    raw_data_file = os.path.join(args.result_dir,"raw_dataset.xlsx")
    test_data_file = os.path.join(args.result_dir,"test_dataset.xlsx")
    # 创建字典
    dictionary_file = os.path.join(args.result_dir,"dictionary.json")
    if not os.path.exists(dictionary_file):
        train_dataset = pd.read_excel(raw_data_file)
        dev_dataset = pd.read_excel(test_data_file)
        build_dictionary(args.data_dir,args.result_dir,train_dataset,dev_dataset)
    logger.info("Dictionary save in %s"%args.result_dir)
    # Embedding 以及问题一预处理数据
    # 创建关系图
    save_rel_ent_dir = os.path.join(args.result_dir,"graph")
    if not os.path.exists(save_rel_ent_dir):
        os.makedirs(save_rel_ent_dir)
        build_graph(args.data_dir,args.result_dir)
    logger.info("Graphs save in %s"%save_rel_ent_dir)
def pre_statistics(args):
    raw_dataset_file = os.path.join(args.result_dir,"raw_dataset.xlsx")
    save_keys_file = os.path.join(args.data_dir,"property_zh.json")
    raw_dataset = pd.read_excel(raw_dataset_file)
    with open(save_keys_file,mode="r",encoding="utf-8") as rfp:
        keys_data = json.load(rfp)
    discrete_path = os.path.join(args.result_dir,"pictures","discrete")
    continue_path = os.path.join(args.result_dir,"pictures","continue")
    time_path = os.path.join(args.result_dir,"pictures","time")
    if not os.path.exists(discrete_path):
        os.makedirs(discrete_path)
    if not os.path.exists(continue_path):
        os.makedirs(continue_path)
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    for name in keys_data["discrete"]:
        data = raw_dataset[name]
        plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel("The "+name+" of the dataset")
        plt.ylabel("Number")
        plt.title("The " + name + " statistics")
        save_fig_file = os.path.join(discrete_path,name + "_"+"statistics.png")
        plt.savefig(save_fig_file)
        plt.close()
    for name in keys_data["continue"]:
        data = raw_dataset[name]
        plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel("The "+name+" of the dataset")
        plt.ylabel("Number")
        plt.title("The " + name + " statistics")
        save_fig_file = os.path.join(continue_path,name + "_"+"statistics.png")
        plt.savefig(save_fig_file)
        plt.close()
    data = raw_dataset.loc[:,"价格"]
    plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("The span of the dataset")
    plt.ylabel("Number")
    plt.title("Data statistics")
    save_fig_file = os.path.join(continue_path,"price_statistics.png")
    plt.savefig(save_fig_file)
    plt.close()
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
    preprocess(args)
    pre_statistics(args)
    model_dir = os.path.join(args.result_dir,"model")
    embedding_file = os.path.join(model_dir,'embedding.npz')
    if not os.path.exists(embedding_file):
        train_rgcn_model(args)
    logger.info("The file saved in %s"%embedding_file)
if __name__ == "__main__":
    main()

from __future__ import print_function
import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import f1_score
from .utils import to_cuda
from .data import CarDataset

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
def cal_score(y_pred,y_target):
    # mape
    ape = np.abs(y_target-y_pred)/y_pred
    mape = np.mean(ape)
    a_val = ape<=0.05
    b_val = len(ape)
    accuracy = a_val/b_val
    return 0.2*(1-mape)+0.8*accuracy
def cal_score_extension(y_pred,y_target):
    # mape
    ape = np.abs(y_target-y_pred)/y_pred
    mape = np.mean(ape)
    a_val = ape<=0.5
    b_val = len(ape)
    accuracy = a_val/b_val
    return 0.2*(1-mape)+0.8*accuracy
def cal_accuracy(y_pred,y_target):
    ape = np.abs(y_target-y_pred)/y_pred
    a_val = ape<=0.05
    b_val = len(ape)
    accuracy = a_val/b_val
    return accuracy
def cal_mape(y_pred,y_target):
    ape = np.abs(y_target-y_pred)/y_pred
    mape = np.mean(ape)
    return mape
def cal_mse(y_pred,y_target):
    return np.abs(y_pred-y_target).mean()
def evaluate_dataset(model,dev_dataloader,device):
    model.eval()
    score_list = []
    acc_list = []
    mape_list = []
    mse_list = []
    for item in dev_dataloader:
        ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor,target_tensor = to_cuda(item,device)
        predict_tensor = model(ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor)
        target = target_tensor.detach().cpu().numpy()
        predict = predict_tensor.detach().cpu().numpy()
        score1 = cal_score(predict,target)
        score2 = cal_accuracy(predict,target)
        score3 = cal_mape(predict,target)
        score4 = cal_mse(predict,target)
        mape_list.append(score3)
        score_list.append(score1)
        acc_list.append(score2)
        mse_list.append(score4)
    score_list = np.hstack(score_list)
    acc_list = np.hstack(acc_list)
    mape_list = np.hstack(mape_list)
    mse_list = np.hstack(mse_list)
    return score_list.mean(),acc_list.mean(),mape_list.mean(),mse_list.mean()
def test_raw_dataset(args,model,test_dataloader,device):
    model.to(device)
    value_all_list = []
    raw_data_file = os.path.join(args.result_dir,"raw_dataset.xlsx")
    raw_dataset = pd.read_excel(raw_data_file)
    """
    min_value = raw_dataset["价格"].min()
    max_value = raw_dataset["价格"].max()
    """
    std_value = raw_dataset["价格"].std()
    mean_value = raw_dataset["价格"].mean()
    
    for item in test_dataloader:
        ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor = to_cuda(item,device)
        predict_tensor = model(ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor)
        predict = predict_tensor.detach().cpu().numpy()
        
        if args.normalize:
            predict = predict*std_value+mean_value
        """
        if args.normalize:
            predict = predict*(max_value-min_value)+min_value
        """
        value_all_list.append(predict)
    value_all_list = np.hstack(value_all_list)
    test_data_file = os.path.join(args.result_dir,"result.xlsx")
    test_dataset = pd.read_excel(test_data_file)
    test_dataset["价格"] = value_all_list
    predict_file_name = os.path.join(args.result_dir,"predict.txt")
    with open(predict_file_name,mode="w",encoding="utf-8") as wfp:
        for k in range(len(test_dataset)):
            wfp.write(str(test_dataset.loc[k,"车辆id"]) +"\t"+str(test_dataset.loc[k,"价格"])+"\n")
    test_dataset.to_excel(test_data_file,index=None)
def draw_result(score_list,title,ylabel,xlabel,save_fig_name):
    x = np.linspace(0,len(score_list)-1,len(score_list))
    plt.figure(figsize=(10,8))
    plt.plot(x,score_list)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(save_fig_name)
    logger.info("File saved in %s"%save_fig_name)
def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices
# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():
        
        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            
            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)
            
            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)
            
            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
            
    return mrr.item()
def valid_mrr_dataset(valid_triplets, model, test_graph, all_triplets,device):
    model.to(device)
    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr
def pearson(y_target,y_predict):
    x = y_predict-y_predict.mean()
    y = y_target-y_target.mean()
    return np.dot(x,y)/(np.sqrt(np.power(x,2).sum()*np.power(y,2).sum()))
def evaluate_probability(dev_dataloader,model,device):
    model.eval()
    target_list = []
    predict_list = []
    for item in dev_dataloader:
        ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor,target_tensor = to_cuda(item,device)
        predict_tensor = model(ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor)
        target = target_tensor.detach().cpu().numpy()
        target_list.append(target)
        predict = predict_tensor.detach().cpu().numpy()
        predict_list.append(predict)
    target = np.hstack(target_list)
    predict = np.hstack(predict_list)
    return pearson(target,predict)


import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TimeEmbedding
class _Linear_KB(torch.autograd.Function):
    # 前向运算
    def forward(self,input_x,a=1.0):
        self.a_val = a
        assert abs(a)>1e-3
        self.save_for_backward(input_x)
        output = a*input_x + 0.5
        if output >1.0:
            return 1.0
        elif output <0.0:
            return 0.0
        else:
            return output                              
                                         
    def backward(self, grad_output):                             
        input_x = self.saved_tensors
        output = self.a_val*input_x + 0.5
        if output >1.0 or output <0.0:
            grad_x = 0.0
        else:
            grad_x = self.a_val
        return grad_x 
linear_kb = _Linear_KB.apply
class IGANN(nn.Module):
    """
    The Interactive Gate Attention for Deep Nerual Network(IGANN)
    """
    def __init__(self,ent_vocab_size,ent_size,time_size,value_size,ent_dim=80,time_dim_list = [30,30,30],
                    year_span = 101,linear_dim=12,hidden_dim = 8):
        super(IGANN, self).__init__()
        # Embedding 区域
        assert len(time_dim_list) == 3
        self.ent_vocab_size = ent_vocab_size
        self.ent_size = ent_size
        self.time_size = time_size
        self.value_size = value_size
        
        self.year_span = year_span
        self.ent_dim = ent_dim
        self.year_dim = time_dim_list[0]
        self.month_dim = time_dim_list[1]
        self.day_dim = time_dim_list[2]

        self.ent_embed = nn.Embedding(self.ent_vocab_size,self.ent_dim)
        self.year_embed = nn.Embedding(self.year_span,self.year_dim)
        self.month_embed = nn.Embedding(12,self.month_dim)
        self.day_embed = nn.Embedding(31,self.day_dim)

        nn.init.xavier_uniform_(self.ent_embed.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.year_embed.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.month_embed.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.day_embed.weight, gain=nn.init.calculate_gain('relu'))
        # 线性变换
        self.linear_dim = linear_dim
        self.hidden_dim = hidden_dim
        self.ent_linear = nn.Linear(self.ent_dim,self.linear_dim)
        self.year_linear = nn.Linear(self.year_dim,self.linear_dim)
        self.month_linear = nn.Linear(self.month_dim,self.linear_dim)
        self.day_linear = nn.Linear(self.day_dim,self.linear_dim)
        self.val_linear = nn.Linear(self.value_size,self.linear_dim)
        # 交互式计算
        
        self.bais_wet = nn.Linear(self.ent_size*self.linear_dim, hidden_dim)
        self.bais_wtt = nn.Linear(self.time_size*self.linear_dim*3, hidden_dim)
        self.bais_wvt = nn.Linear(self.linear_dim*self.hidden_dim,hidden_dim)
        
        self.wrt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wzt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wqt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        # 输出层
        self.out = nn.Linear(3*hidden_dim,1)
    def forward(self,ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor):
        '''
        :param ent_tensor: size of (batch_size,ent_size)
        :param val_tensor: size of (batch_size,value_size)
        :param year_tensor: size of (batch,year_dim)
        :param month_tensor: size of (batch,month_dim)
        :param day_tensor: size of (batch,day_dim)
        :return:
        '''
        batch_size = ent_tensor.shape[0]
        hid_ent = torch.tanh(self.ent_linear(self.ent_embed(ent_tensor))) # (batch_size,seq_len,linear_dim)
        hid_year = torch.tanh(self.year_linear(self.year_embed(year_tensor))) # (batch_size,seq_len,linear_dim)
        hid_month = torch.tanh(self.month_linear(self.month_embed(month_tensor))) # (batch_size,seq_len,linear_dim)
        hid_day = torch.tanh(self.day_linear(self.day_embed(day_tensor))) # (batch_size,seq_len,linear_dim)
        hid_time = torch.cat([hid_year,hid_month,hid_day], dim=2)  # (batch,seq_len,3*linear_dim)
        
        val_tensor = val_tensor.unsqueeze(1).repeat(1,self.hidden_dim,1)

        hid_value = torch.tanh(self.val_linear(val_tensor))
        # 属性对齐
        
        pe = torch.tanh(self.bais_wet(hid_ent.view(batch_size,-1)))
        pt = torch.tanh(self.bais_wtt(hid_time.view(batch_size,-1)))
        pv = torch.tanh(self.bais_wvt(hid_value.view(batch_size,-1)))
        
        # 交互式计算
        zpt = torch.sigmoid(self.wrt(torch.cat([pe,pt],dim=1)))
        o_rt = (1 - zpt)*pt + zpt*pe
        zpt = torch.sigmoid(self.wzt(torch.cat([pe,pv],dim=1)))
        o_zt = (1 - zpt)*pe + zpt*pv
        zpt = torch.sigmoid(self.wqt(torch.cat([pt,pv],dim=1)))
        o_qt = (1 - zpt)*pv + zpt*pt
        # output layer
        cat_tensor = torch.cat([o_rt,o_zt,o_qt],dim=1)
        #return torch.relu(self.out(cat_tensor)).squeeze()
        return self.out(cat_tensor).squeeze()

class TEIGANN(nn.Module):
    """
    The Time Extension Interactive Gate for Deep Nerual Network(TEIGNN)
    """
    def __init__(self,ent_vocab_size,ent_size,time_size,value_size,ent_dim=80,time_embedding_dim=30,
                    year_span = 101,linear_dim=12,hidden_dim = 8,emb_dropout=0.2):
        super(TEIGANN, self).__init__()
        # Embedding 区域
        self.ent_vocab_size = ent_vocab_size
        self.ent_size = ent_size
        self.time_size = time_size
        self.value_size = value_size
        
        self.year_span = year_span
        self.ent_dim = ent_dim
        self.time_embedding_dim = time_embedding_dim

        self.time_embed = TimeEmbedding(years_num=year_span,embedding_dim=time_embedding_dim,
                                        emb_dropout=emb_dropout)
        self.ent_embed = nn.Embedding(self.ent_vocab_size,self.ent_dim)

        nn.init.xavier_uniform_(self.ent_embed.weight, gain=nn.init.calculate_gain('relu'))
        # 线性变换
        self.linear_dim = linear_dim
        self.hidden_dim = hidden_dim
        self.ent_linear = nn.Linear(self.ent_dim,self.linear_dim)
        self.time_linear = nn.Linear(self.time_embedding_dim,self.linear_dim)
        self.val_linear = nn.Linear(self.value_size,self.linear_dim)
        # 交互式计算
        
        self.bais_wet = nn.Linear(self.ent_size*self.linear_dim, hidden_dim)
        self.bais_wtt = nn.Linear(self.time_size*self.linear_dim, hidden_dim)
        self.bais_wvt = nn.Linear(self.value_size*self.linear_dim,hidden_dim)
        
        self.wrt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wzt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wqt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        # 输出层
        self.out = nn.Linear(3*hidden_dim,1)
    def forward(self,ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor):
        '''
        :param ent_tensor: size of (batch_size,ent_size)
        :param val_tensor: size of (batch_size,value_size)
        :param year_tensor: size of (batch,year_dim)
        :param month_tensor: size of (batch,month_dim)
        :param day_tensor: size of (batch,day_dim)
        :return:
        '''
        batch_size = ent_tensor.shape[0]
        hid_ent = torch.relu(self.ent_linear(self.ent_embed(ent_tensor))) # (batch_size,seq_len,linear_dim)
        #hid_time = torch.tanh(self.)
        hid_time = torch.relu(self.time_linear(self.time_embed(year_tensor,month_tensor,day_tensor))) # (batch_size,seq_len,linear_dim)
        
        val_tensor = val_tensor.unsqueeze(1).repeat(1,self.value_size,1)
        
        hid_value = torch.relu(self.val_linear(val_tensor))

        
        # 属性对齐
        pe = torch.tanh(self.bais_wet(hid_ent.reshape(batch_size,-1)))
        pt = torch.tanh(self.bais_wtt(hid_time.reshape(batch_size,-1)))
        pv = torch.tanh(self.bais_wvt(hid_value.reshape(batch_size,-1)))

        # 交互式计算
        zpt = torch.sigmoid(self.wrt(torch.cat([pe,pt],dim=1)))
        
        o_rt = (1 - zpt)*pt + zpt*pe
        zpt = torch.sigmoid(self.wzt(torch.cat([pe,pv],dim=1)))
        o_zt = (1 - zpt)*pe + zpt*pv
        zpt = torch.sigmoid(self.wqt(torch.cat([pt,pv],dim=1)))
        o_qt = (1 - zpt)*pv + zpt*pt
        # output layer
        cat_tensor = torch.cat([o_rt,o_zt,o_qt],dim=1)
        # return torch.relu(self.out(cat_tensor)).squeeze()
        # return torch.abs(self.out(cat_tensor)).squeeze()
        return self.out(cat_tensor).squeeze()

class TEIGANNClassifier(nn.Module):
    """
    The Time Extension Interactive Gate for Deep Nerual Classifier Network(TEIGNN)
    """
    def __init__(self,ent_vocab_size,ent_size,time_size,value_size,ent_dim=80,time_embedding_dim=30,
                    year_span = 101,linear_dim=12,hidden_dim = 8,emb_dropout=0.2):
        super(TEIGANNClassifier, self).__init__()
        # Embedding 区域
        self.ent_vocab_size = ent_vocab_size
        self.ent_size = ent_size
        self.time_size = time_size
        self.value_size = value_size
        
        self.year_span = year_span
        self.ent_dim = ent_dim
        self.time_embedding_dim = time_embedding_dim

        self.time_embed = TimeEmbedding(years_num=year_span,embedding_dim=time_embedding_dim,
                                        emb_dropout=emb_dropout)
        self.ent_embed = nn.Embedding(self.ent_vocab_size,self.ent_dim)

        nn.init.xavier_uniform_(self.ent_embed.weight, gain=nn.init.calculate_gain('relu'))
        # 线性变换
        self.linear_dim = linear_dim
        self.hidden_dim = hidden_dim
        self.ent_linear = nn.Linear(self.ent_dim,self.linear_dim)
        self.time_linear = nn.Linear(self.time_embedding_dim,self.linear_dim)
        self.val_linear = nn.Linear(self.value_size,self.linear_dim)
        # 交互式计算
        
        self.bais_wet = nn.Linear(self.ent_size*self.linear_dim, hidden_dim)
        self.bais_wtt = nn.Linear(self.time_size*self.linear_dim, hidden_dim)
        self.bais_wvt = nn.Linear(self.value_size*self.linear_dim,hidden_dim)
        
        self.wrt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wzt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        self.wqt = nn.Linear(2*hidden_dim, hidden_dim)#, bias=False)
        # 输出层
        self.out = nn.Linear(3*hidden_dim,1)
    def forward(self,ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor):
        '''
        :param ent_tensor: size of (batch_size,ent_size)
        :param val_tensor: size of (batch_size,value_size)
        :param year_tensor: size of (batch,year_dim)
        :param month_tensor: size of (batch,month_dim)
        :param day_tensor: size of (batch,day_dim)
        :return:
        '''
        batch_size = ent_tensor.shape[0]
        hid_ent = torch.relu(self.ent_linear(self.ent_embed(ent_tensor))) # (batch_size,seq_len,linear_dim)
        #hid_time = torch.tanh(self.)
        hid_time = torch.relu(self.time_linear(self.time_embed(year_tensor,month_tensor,day_tensor))) # (batch_size,seq_len,linear_dim)
        
        val_tensor = val_tensor.unsqueeze(1).repeat(1,self.value_size,1)
        
        hid_value = torch.relu(self.val_linear(val_tensor))

        
        # 属性对齐
        pe = torch.relu(self.bais_wet(hid_ent.reshape(batch_size,-1)))
        pt = torch.relu(self.bais_wtt(hid_time.reshape(batch_size,-1)))
        pv = torch.relu(self.bais_wvt(hid_value.reshape(batch_size,-1)))

        # 交互式计算
        zpt = torch.tanh(self.wrt(torch.cat([pe,pt],dim=1)))
        o_rt = (1 - zpt)*pt + zpt*pe
        zpt = torch.tanh(self.wzt(torch.cat([pe,pv],dim=1)))
        o_zt = (1 - zpt)*pe + zpt*pv
        zpt = torch.tanh(self.wqt(torch.cat([pt,pv],dim=1)))
        o_qt = (1 - zpt)*pv + zpt*pt
        # output layer
        cat_tensor = torch.cat([o_rt,o_zt,o_qt],dim=1)
        return torch.sigmoid(self.out(cat_tensor)).squeeze()



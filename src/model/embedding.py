import torch
import torch.nn as nn
import torch.nn.functional as F
class TimeEmbedding(nn.Module):
    def __init__(self,years_num = 30,embedding_dim=50,emb_dropout = 0.2):
        super(TimeEmbedding,self).__init__()
        self.emb_dropout = emb_dropout
        self.years_num = years_num
        self.months_num = 12
        self.days_num = 31
        self.year_embed = nn.Embedding(self.years_num,embedding_dim)
        self.month_embed = nn.Embedding(self.months_num,embedding_dim)
        self.day_embed = nn.Embedding(self.days_num,embedding_dim)
        nn.init.xavier_uniform_(self.year_embed.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.month_embed.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.day_embed.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self,years,months,days):
        '''
        :params time_str: The formater of the time is 'XXXX-XX-XX'
        '''
        y_emb = self.year_embed(years)
        m_emb = self.month_embed(months)
        d_emb = self.day_embed(days)
        combine_emb = torch.relu(y_emb + m_emb + d_emb)
        return F.dropout(combine_emb,p=self.emb_dropout)




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadDifAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_k = torch.tensor(d_k)

        self.W_q = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1) #0.1 apparently?
        self.mask = self.get_mask(self.seq_len)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads)
        self._lambda_init = torch.rand(1)
        self._lambda = nn.Parameter(self._lambda_init.clone())

    def forward(self, x, dropout=0.0):
        query = self.W_q(query).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        key = self.W_k(key).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        values = self.W_v(values).view(-1, self.num_heads, self.seq_len, self.num_hidden)

        #split query into [q1;q2] and same for keys [k1;k2]
        query_1 = query[:, :, :, :self.num_hidden]
        query_2 = query[:, :, :, self.num_hidden:]

        key_1 = key[:, :, :, :self.num_hidden]
        key_2 = key[:, :, :, self.num_hidden:]

        QK_T_1 = torch.matmul(query_1, key_1.mT) / torch.sqrt(self.d_k)
        QK_T_2 = torch.matmul(query_2, key_2.mT) / torch.sqrt(self.d_k)

        QK_T_1_norm = self.softmax(QK_T_1)
        QK_T_2_norm = self.softmax(QK_T_2)

        #eq 1
        attention_scores = (QK_T_1_norm - self._lambda * QK_T_2_norm)

        attention_scores = self.dropout(attention_scores) 
        output = torch.matmul(attention_scores, values)  
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads , self.seq_len, self.num_hidden)  
        
        output = self.group_norm(output)
        
        output = output * (1 - self._lambda_init)
        output = torch.cat([output[:, i, :, :] for i in range(self.num_heads)], dim=-1)

        output = self.W_o(output)  
        return output, attention_scores #need two outputs for transformer forward method

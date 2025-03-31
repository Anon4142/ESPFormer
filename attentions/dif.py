import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadDifAttention(nn.Module):
    #num_hidden is d_model
    def __init__(self, num_hidden, num_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = -1 #useless, its just x.size(1); Assumes max_seq_len is same for inference as training (common practice)
        # d_k is converted to a float tensor for proper computation.
        self.d_k = torch.tensor(d_k, dtype=torch.float32)

        self.W_q = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)  # 0.1 dropout rate
        # self.mask = self.get_mask(self.seq_len)  # Unused here, but kept if needed
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads)
        self._lambda_init = torch.rand(1)
        self._lambda = nn.Parameter(self._lambda_init.clone())

    def forward(self, x, attn_mask=None, dropout=0.0):
        # x: (batch_size, seq_len, num_hidden)
        # Compute query, key, and values from input x
        seq_len = x.size(1)  # derive seq_len from x
        query = self.W_q(x).view(-1, self.num_heads, seq_len, 2 * self.num_hidden)
        key   = self.W_k(x).view(-1, self.num_heads, seq_len, 2 * self.num_hidden)
        values = self.W_v(x).view(-1, self.num_heads, seq_len, self.num_hidden)

        # Split query and key into two parts along the last dimension
        query_1 = query[:, :, :, :self.num_hidden]
        query_2 = query[:, :, :, self.num_hidden:]

        key_1 = key[:, :, :, :self.num_hidden]
        key_2 = key[:, :, :, self.num_hidden:]

        # Compute attention logits for each split
        QK_T_1 = torch.matmul(query_1, key_1.transpose(-1, -2)) / torch.sqrt(self.d_k)
        QK_T_2 = torch.matmul(query_2, key_2.transpose(-1, -2)) / torch.sqrt(self.d_k)

        # If an attention mask is provided, apply it to both QK_T matrices.
        if attn_mask is not None:
            # If mask is of shape (batch_size, seq_len), expand it to (batch_size, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, seq_len, -1)
            # Now unsqueeze the mask for heads dimension: (batch_size, 1, seq_len, seq_len)
            QK_T_1 = QK_T_1.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
            QK_T_2 = QK_T_2.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        # Apply softmax to get normalized attention scores
        QK_T_1_norm = self.softmax(QK_T_1)
        QK_T_2_norm = self.softmax(QK_T_2)

        # Compute the differential attention scores using the lambda parameter.
        attention_scores = (QK_T_1_norm - self._lambda * QK_T_2_norm)
        attention_scores = self.dropout(attention_scores)
        
        # Compute the output by weighted sum of the values.
        output = torch.matmul(attention_scores, values)
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads, seq_len, self.num_hidden)
        
        output = self.group_norm(output)
        output = output * (1 - self._lambda_init)
        # Concatenate outputs from all heads.
        output = torch.cat([output[:, i, :, :] for i in range(self.num_heads)], dim=-1)
        output = self.W_o(output)
        
        # Return output and attention_scores (for consistency with the Transformer forward interface)
        return output, attention_scores

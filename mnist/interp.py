import torch
from torch import nn

def interp(M,N):
    
    x_src = torch.linspace(0, 1, M)  
    x_tgt = torch.linspace(0, 1, N)

    interpolation_matrix = torch.zeros((N, M))

    for i in range(N):
        for j in range(M - 1):
            if x_src[j] <= x_tgt[i] <= x_src[j + 1]:
                # Linear interpolation weights
                weight_right = (x_tgt[i] - x_src[j]) / (x_src[j + 1] - x_src[j])
                weight_left = 1 - weight_right

                # Assign weights to the interpolation matrix
                interpolation_matrix[i, j] = weight_left
                interpolation_matrix[i, j + 1] = weight_right
                break
                
    row_sums = interpolation_matrix.sum(dim=1, keepdim=True)
    return interpolation_matrix
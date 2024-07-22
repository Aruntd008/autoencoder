import torch 
import torch.nn as nn

class VQEMA(nn.Module):
    def __init__(self, embedding_dim, k,) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k = k
        
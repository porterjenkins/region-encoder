import torch
import torch.nn as nn




class RegionEncoder(nn.Module):
    """
    Implementatino of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self):
        super(RegionEncoder, self).__init__()


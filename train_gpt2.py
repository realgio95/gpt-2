from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 565
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config


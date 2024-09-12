import torch
import torch.nn as nn
from opt_einsum import contract
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import json


class Encoder(nn.Module):
    pass
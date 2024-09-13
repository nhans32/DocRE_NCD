import torch
import torch.nn as nn
from opt_einsum import contract
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import json
import const
from losses import ATLoss
from encoding import encode


class RelationEncoder(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_class,
                 embed_size=768, # Intermediary embedding for head and tail entities
                 out_embed_size=768, # Final embedding for the relationship
                 max_labels=4, # Max number of classes to predict for one entity pair
                 block_size=64,):
        super(RelationEncoder, self).__init__()

        if model_name not in [const.LUKE_BASE, const.LUKE_LARGE, const.LUKE_LARGE_TACRED]:
            raise ValueError(f'Invalid encoder name: {model_name}')
        self.model_name = model_name

        self.start_tok_ids = [tokenizer.cls_token_id]
        self.end_tok_ids = [tokenizer.sep_token_id]

        self.embed_size = embed_size
        self.block_size = block_size
        self.hidden_size = 768 if self.model_name == const.LUKE_BASE else 1024
        self.out_embed_size = out_embed_size

        self.loss_fn = ATLoss()
        self.num_class = num_class

        self.luke_model = AutoModel.from_pretrained(self.model_name) # Base model
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.bilinear = nn.Linear(self.embed_size * self.block_size, self.out_embed_size)

    
    def forward(self, batch):
        seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels = encode(model=self.luke_model,
                                                                                      batch=batch,
                                                                                      start_tok_ids=self.start_tok_ids,
                                                                                      end_tok_ids=self.end_tok_ids)
        # NOTE: Keep in mind that each "entity" at this point (after collate_fn) is simply a single mention of an entity.
        #       A single mention of an entity does not necessarily represent the entire entity (there can be multiple mentions).
        #       Therefore, we pool entity mentions in the hrt_pool function to operate with single entity representations.
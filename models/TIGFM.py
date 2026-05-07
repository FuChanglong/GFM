import numpy as np
import torch
from transformers import AutoModel

from .gt import GPS


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TIGFM(torch.nn.Module):
    def __init__(self, graph_input_dim, graph_hid_dim, graph_num_layer, attn_kwargs, text_id=None):
        super().__init__()
        self.graph_model = GPS(in_dim=graph_input_dim, channels=graph_hid_dim, out_dim=graph_hid_dim, pe_dim=8,
                               num_layers=graph_num_layer, attn_type='multihead', attn_kwargs=attn_kwargs)
        text_model = AutoModel.from_pretrained(text_id, local_files_only=True)
        self.text_model = text_model
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prototype = PrototypicalLoss(128, 384)

    def encode_graph(self, batch):
        graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs,_

    def encode_text(self, input_ids, token_type_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def freeze_text(self):
        for k, v in self.text_model.named_parameters():
            v.requires_grad = False


import torch.nn.functional as F


class PrototypicalLoss(torch.nn.Module):
    def __init__(self, n_prototypes, embedding_dim, temperature=0.07):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.prototypes = torch.nn.Parameter(torch.randn(n_prototypes, embedding_dim))

    def forward(self, features):
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        logits = torch.matmul(features, prototypes.T) / self.temperature
        return logits

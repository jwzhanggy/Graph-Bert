'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from code.MethodBertComp import BertEmbeddings, BertEncoder


BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBert(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(MethodGraphBert, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()


    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, head_mask=None, residual_h=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(raw_features=raw_features, wl_role_ids=wl_role_ids, init_pos_ids=init_pos_ids, hop_dis_ids=hop_dis_ids)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def run(self):
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import BertPreTrainedModel, BertModel
from MethodGraphBert import MethodGraphBert
from MethodBertComp import NodeConstructOutputLayer

import time
import numpy as np



BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4

    def __init__(self, config):
        super(MethodGraphBertNodeConstruct, self).__init__(config)

        self.bert = MethodGraphBert(config)
        self.cls = NodeConstructOutputLayer(config)

        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        return prediction_scores

    def train_model(self, max_epoch=1000):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()

            reconstruction_features = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'])
            loss_train = F.mse_loss(reconstruction_features, self.data['raw_embeddings'])
            loss_train.backward()
            optimizer.step()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}

            if epoch % 50 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def run(self):
        self.train_model()
        return self.learning_record_dict
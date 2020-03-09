import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeConstruct, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = self.cls_y(sequence_output)

        return x_hat


    def train_model(self, max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()

            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'])

            loss_train = F.mse_loss(output, self.data['X'])

            loss_train.backward()
            optimizer.step()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 50 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):

        self.train_model(self.max_epoch)

        return self.learning_record_dict
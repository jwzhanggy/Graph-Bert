import torch

from transformers.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

from sklearn.cluster import KMeans

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertGraphClustering(BertPreTrainedModel):
    learning_record_dict = {}
    use_raw_feature = True
    cluster_number = 0
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertGraphClustering, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        kmeans = KMeans(n_clusters=self.cluster_number, max_iter=self.max_epoch)
        if self.use_raw_feature:
            clustering_result = kmeans.fit_predict(self.data['X'])
        else:
            clustering_result = kmeans.fit_predict(sequence_output.tolist())

        return {'pred_y': clustering_result, 'true_y': self.data['y']}

    def train_model(self, max_epoch):
        t_begin = time.time()

        clustering = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
                              self.data['hop_embeddings'])

        self.learning_record_dict = clustering

    def run(self):

        self.train_model(self.max_epoch)

        return self.learning_record_dict
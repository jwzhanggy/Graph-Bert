'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.method import method
import networkx as nx
import pickle


class MethodHopDistance(method):
    data = None
    k = None
    dataset_name = None

    def run(self):
        node_list = self.data['idx']
        link_list = self.data['edges']
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(link_list)

        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        hop_dict = {}
        for node in batch_dict:
            if node not in hop_dict: hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except:
                    hop = 99
                hop_dict[node][neighbor] = hop
        return hop_dict
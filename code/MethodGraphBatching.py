'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.method import method


class MethodGraphBatching(method):
    data = None
    k = 5

    def run(self):
        S = self.data['S']
        index_id_dict = self.data['index_id_map']

        user_top_k_neighbor_intimacy_dict = {}
        for node_index in index_id_dict:
            node_id = index_id_dict[node_index]
            s = S[node_index]
            s[node_index] = -1000.0
            top_k_neighbor_index = s.argsort()[-self.k:][::-1]
            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in top_k_neighbor_index:
                neighbor_id = index_id_dict[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))
        return user_top_k_neighbor_intimacy_dict





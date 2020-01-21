'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import v_measure_score

class EvaluateClustering(evaluate):
    data = None
    
    def evaluate(self):
        eval_result_dict = {}
        eval_result_dict['ami'] = adjusted_mutual_info_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['rand'] = adjusted_rand_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['comp'] = completeness_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['fow'] = fowlkes_mallows_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['hom'] = homogeneity_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['nmi'] = normalized_mutual_info_score(self.data['true_y'], self.data['pred_y'])
        eval_result_dict['v_score'] = v_measure_score(self.data['true_y'], self.data['pred_y'])
        return eval_result_dict
        
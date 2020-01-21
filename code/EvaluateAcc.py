'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class EvaluateAcc(evaluate):
    data = None
    
    def evaluate(self):
        
        return accuracy_score(self.data['true_y'], self.data['pred_y'])
        
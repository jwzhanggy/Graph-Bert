'''
Base evaluate class for all evaluation metrics and methods
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


import abc


class evaluate:
    """ 
    evaluate: Abstract Class
    Entries: 
    """
    
    evaluate_name = None
    evaluate_description = None
    
    data = None
    
    # initialization function
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    @abc.abstractmethod
    def evaluate(self):
        return

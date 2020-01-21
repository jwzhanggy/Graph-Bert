'''
Base IO class for all datasets
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


import abc

class dataset:
    """ 
    dataset: Abstract Class 
    Entries: dataset_name: the name of the dataset
             dataset_description: the textual description of the dataset
    """
    
    dataset_name = None
    dataset_descrition = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    data = None
    
    # initialization function
    def __init__(self, dName=None, dDescription=None):
        '''
        Parameters: dataset name: dName, dataset description: dDescription
        Assign the parameters to the entries of the base class
        '''
        self.dataset_name = dName
        self.dataset_descrition = dDescription
    
    # information print function
    def print_dataset_information(self):
        '''
        Print the basic information about the dataset class
        inclduing the dataset name, and dataset description
        '''
        print('Dataset Name: ' + self.dataset_name)
        print('Dataset Description: ' + self.dataset_descrition)

    # dataset load abstract function
    @abc.abstractmethod
    def load(self):
        return
    
    
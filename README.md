# Graph-Bert
Source code of "Graph-Bert: Only Attention is Needed for Learning Graph Representations". 

The paper at arXiv is available at https://arxiv.org/abs/2001.05140

At our group homepage, we also have a copy of our latest paper released: http://www.ifmlab.org/files/paper/graph_bert.pdf


************************************************************************************************

##How to Run the code?

(1) The Graph-Bert model takes (1) node WL code, (2) intimacy based subgraph batch, (3) node hop distance as the prior inputs. These can be computed with the script_1_preprocessing.py.

(2) Pre-training of Graph-Bert based on node attribute reconstruction and graph structure recovery is provided by script_2_pre_training.py.

(3) Please check the script_3_fine_tuning.py as the entry point to run the model on node classification and graph clustering. 

(4) script_4_evaluate_plot.py is used for plots drawing and results evaluation purposes.

We suggest to run the code with Pycharm instead.


######Several toolkits may be needed to run the code
(1) pytorch (https://anaconda.org/pytorch/pytorch)
(2) sklearn (https://anaconda.org/anaconda/scikit-learn) 
(3) transformers (https://anaconda.org/conda-forge/transformers) 
(4) networkx (https://anaconda.org/anaconda/networkx) 

************************************************************************************************

Learning results of Graph-Bert with graph-raw residual on Cora dataset.

![Learning Results of Graph-Bert with Graph Residual on Cora](./result/screenshot/cora_graph_residual_k_7.png)


************************************************************************************************

Learning results of Graph-Bert with raw residual on Cora dataset.

![Learning Results of Graph-Bert with Raw Residual on Cora](./result/screenshot/cora_raw_residual_k_7.png)

************************************************************************************************

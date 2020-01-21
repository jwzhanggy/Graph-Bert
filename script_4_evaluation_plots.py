import matplotlib.pyplot as plt
from code.ResultSaving import ResultSaving

#---------- clustering results evaluation -----------------

dataset_name = 'pubmed'

if 0:
    pre_train_task = 'node_reconstruction+structure_recovery'

    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = 'clustering_' + dataset_name + '_' + pre_train_task
    loaded_result = result_obj.load()

    eval_obj = EvaluateClustering()
    eval_obj.data = loaded_result
    eval_result = eval_obj.evaluate()

    print(eval_result)




#--------------- Graph Bert Pre-Training Records Convergence --------------

dataset_name = 'cora'

if 0:
    if dataset_name == 'cora':
        k = 7
    elif dataset_name == 'citeseer':
        k = 5
    elif dataset_name == 'pubmed':
        k = 30

    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GraphBert/'
    best_score = {}

    result_obj.result_destination_file_name = dataset_name + '_' + str(k) + '_node_reconstruction'
    record_dict = result_obj.load()
    epoch_list = sorted(record_dict.keys())
    print(record_dict)

    plt.figure(figsize=(4, 3))
    train_loss = [record_dict[epoch]['loss_train'] for epoch in epoch_list]
    plt.plot(epoch_list, train_loss, label='GraphBert')

    plt.xlim(0, 200)
    plt.ylabel("training loss")
    plt.xlabel("epoch (iter. over data set)")
    plt.legend(loc="upper right")
    plt.show()



#--------------- Graph Bert Learning Convergence --------------

dataset_name = 'cora'

if 0:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    depth_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GraphBert/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
        print(result_obj.result_destination_file_name)
        depth_result_dict[depth] = result_obj.load()
    print(depth_result_dict)

    x = range(150)

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth][i]['acc_train'] for i in x]
        plt.plot(x, train_acc, label='GraphBert(' + str(depth) + '-layer)')

    plt.xlim(0, 150)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth][i]['acc_test'] for i in x]
        plt.plot(x, test_acc, label='DifNet(' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)

    plt.xlim(0, 150)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    print(best_score)
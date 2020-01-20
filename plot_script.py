import matplotlib.pyplot as plt
from ResultSaving import ResultSaving

#--------------- DifNet --------------

dataset_name = 'cora'

if 1:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    depth_list = [1, 2, 3, 10, 20, 30]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
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
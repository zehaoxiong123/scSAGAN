from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import umap
import numpy as np
import cluster
from  matplotlib.colors import  rgb2hex
import data_Preprocess
from sklearn.manifold import TSNE
import compare
import draw_heatmap
import result_class
import pickle
import utils
def test():
    digits = load_digits()
    fig, ax_array = plt.subplots(20, 20)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray_r')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.01)
    plt.show()
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(digits.data)
    print(embedding.shape)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    print(digits.target)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset')
    plt.show()

def tsne_for_data(data,function_name,kind_name,cell_class):
    data_class = np.shape(data[0])[1]
    # cmap = plt.get_cmap('viridis', data_class)
    embedding = TSNE(random_state=50).fit_transform(data[1])
    y_true = np.argmax(data[0], axis=1)
    fig, ax = plt.subplots()
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=data_class)
    for i in range(data_class):
        need_idx = np.where(y_true==i)[0]
        ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(i)), s=cell_class,label = kind_name[i])

    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('tsne_1')
    plt.ylabel('tsne_2')
    plt.title(function_name+' projection of the dataset')
    legend = ax.legend(loc='upper right',framealpha=0.5)
    plt.show()
    return embedding

def umap_for_data(data,function_name,kind_name,cell_class,data_set_name):
    data_class = np.shape(data[0])[1]
    #cmap = plt.get_cmap('viridis', data_class)
    reducer = umap.UMAP(random_state= 20150101)
    embedding = reducer.fit_transform(data[1])
    y_true = np.argmax(data[0], axis=1)
    fig, ax = plt.subplots()
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=data_class)
    for i in range(data_class):
        need_idx = np.where(y_true==i)[0]
        ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(i)), s=1,label = kind_name[i])

    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.title(function_name+' projection of '+str(data_set_name))
    legend = ax.legend(loc='upper right',fontsize =5)
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" + str(function_name) + ".png")
    plt.show()

    return embedding
def umap_for_z_bar(data,gene_name,function_name,gene_count_z,data_set_name):
    cm = plt.cm.get_cmap('PuBu')
    data_num = np.shape(data[1])[0]
    # cmap = plt.get_cmap('viridis', data_class)
    reducer = umap.UMAP(random_state=20150101)
    embedding = reducer.fit_transform(data[1])
    for i in range(data_num):
        sc = plt.scatter(embedding[i, 0], embedding[i, 1], c=gene_count_z[i], vmin=0, vmax=1, s=1, cmap=cm)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.title(gene_name + ' rich in ' + str(data_set_name))
    plt.colorbar(sc)
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" +str(function_name)+"_"+ str(gene_name) + ".png")
    plt.show()

def show_result_h5(data_set_name,compare_list,compare_function_list,label_file):
    # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
    result_list = []
    label_list = []
    embeddings_list = []
    for i in range(len(compare_list)):
        mat, data_label, cell_type, cell_type_num, obs, var = data_Preprocess.read_cell_for_h5_imputed(
            compare_list[i], label_file)
        compare_result = result_class.result_for_impute(compare_function_list[i], mat, data_label, cell_type,
                                                        cell_type_num, obs, var)
        result_list.append(compare_result)
    # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
    # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
    for i in range(len(compare_list)):
        embeddings = umap_for_data((result_list[i].data_label, result_list[i].data_mat), compare_function_list[i],
                                   cell_type, cell_type_num,data_set_name)
        embeddings_list.append(embeddings)

        label_1 = cluster.k_means(embeddings, result_list[i].cell_type_num)
        label_list.append(label_1)
    # 在通过聚类算法对指标进行测评
    label = np.array(label_list)
    # 对function_name进行赋值
    function_name = compare_function_list
    # 通过各种指标对聚类结果评测
    # AUC = AUC.compare_for_auc(function_name, data_label, label_score)
    compare.get_indicator(function_name, data_label, label, data_set_name, embeddings_list)
    # PR.compare_for_PR(function_name, data_label, label_score)
    # draw_heatmap.draw_heatmap(var,obs,mat)
    #data_array,data_label,gene_name,cell_name,data_for_count=data_Preprocess.read_cell_to_image("./test_csv/splatter_exprSet_test.csv", "./test_csv/splatter_exprSet_test_label.csv", 4)

def show_result_interpretable(data_set_name,compare_list,compare_function_list,label_file):
    # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
    result_list = []
    label_list = []
    embeddings_list = []
    for i in range(len(compare_list)):
        data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_cell_for_interpretable_imputed(
             compare_list[i], label_file,2,"COVID-19")
        # data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_interpretable_for_train_T2D_imputed(compare_list[i],2,i)
        compare_result = result_class.result_for_impute(compare_function_list[i], data_array, data_label, ["disease","comparison"],
                                                        2, cell_name, gene_name)
        result_list.append(compare_result)
    # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
    # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
    for i in range(len(compare_list)):
        embeddings = umap_for_data((result_list[i].data_label, result_list[i].data_mat), compare_function_list[i],
                                   ["disease", "comparison"],
                                   2, data_set_name)
        embeddings_list.append(embeddings)
        label_1 = cluster.k_means(embeddings, result_list[i].cell_type_num)
        label_list.append(label_1)
    # 在通过聚类算法对指标进行测评
    label = np.array(label_list)
    # 对function_name进行赋值
    function_name = compare_function_list
    # 通过各种指标对聚类结果评测
    # AUC = AUC.compare_for_auc(function_name, data_label, label_score)
    compare.get_indicator(function_name, data_label, label, data_set_name, embeddings_list)
    draw_heatmap_for_interpretable(data_set_name,result_list[0].gene_name)
    # PR.compare_for_PR(function_name, data_label, label_score)
    # draw_heatmap.draw_heatmap(var,obs,mat)

def draw_heatmap_for_interpretable(data_set_name,gene_name=None):
    t_e = np.load("./compare_funtion/sagan/models/"+str(data_set_name)+"/_topic_embeding.npy")
    c_e = np.load("./compare_funtion/sagan/models/"+str(data_set_name)+"/_class_embeding.npy")

    # 首先找出与疾病有关联的主题是哪几个
    d_c = np.array(["disease","comparison"])
    topic = np.array(range(50))
    png_name = "comparsion_png"
    draw_heatmap.draw_heatmap(d_c,topic,c_e[:,:2],data_set_name,png_name)
    #挑选对比度最高的10个主题,0是得病高富集的主题，1是健康高富集的主题
    topic_diff = np.array([c_e[i,0]-c_e[i,1] for i in range(50)])
    k = np.argsort(topic_diff,kind='quicksort',axis=-1)[0:3]
    #找出在这10个主题中表达量都相对较高的基因
    t_e_comp = t_e[:gene_name.shape[0],k]
    t_e_diff = np.argsort(-np.mean(t_e_comp,axis=1),axis=-1)[0:500]
    #挑选出的200个基因对于此疾病有重大意义，进行热图绘画，和保存
    png_name_gene = "comparsion_png_gene"
    df = draw_heatmap.draw_heatmap( topic,gene_name[t_e_diff], t_e[t_e_diff,:], data_set_name, png_name_gene)
    f = open('./result_txt/result_'+str(data_set_name)+'_genes.txt', mode='w')
    f.write("id"+"\n")
    for i in range(500):
        f.write(str(gene_name[t_e_diff][i])+"\n")
    f.close()

def find_gene_in_set(data_path,label_file,funtion_name,find_gene,data_type,data_set_name):
    result_list = []
    for i in range(len(compare_list)):
        data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_cell_for_interpretable_imputed(
            compare_list[i], label_file, 2, "COVID-19")
        compare_result = result_class.result_for_impute(compare_function_list[i], data_array, data_label,
                                                        ["disease", "comparison"],
                                                        2, cell_name, gene_name)
        result_list.append(compare_result)
    x = np.argwhere(result_list[0].gene_name == find_gene)
    for i in range(len(compare_list)):
        gene_count = result_list[i].data_mat[:,x]
        print(gene_count)
        gene_count_z = utils.count_z_score(gene_count)
        embeddings = umap_for_z_bar(( result_list[i].data_label, result_list[i].data_mat), find_gene,funtion_name[i],
                                   gene_count_z, data_set_name)




if __name__=="__main__":
    #data_array,data_label,gene_name,cell_name=data_Preprocess.read_cell("./test_csv/splatter_exprSet_test.csv", "./test_csv/splatter_exprSet_test_label.csv", 4)
    #-----------对h5数据集进行可视化处理-------------------------
    data_set_name = "Quake_10x_Spleen"
    compare_list = ["./compare_funtion/sagan/result/Quake_10x_Spleen_0.1-imputed-scsagan(filted).csv"]
    compare_function_list=["scSAGAN_0.5"]
    show_result_h5(data_set_name,compare_list,compare_function_list,"./compare_funtion/sagan/test_csv/Quake_10x_Spleen/data.h5")
    # -----------对HP数据集进行可解释性研究-------------------------
    # data_set_name = "COVID-19_interpretable"
    # compare_list = ["./compare_funtion/sagan/result/COVID-19_interpretable/GSE164948_covid_control_RNA_counts.csv","./compare_funtion/sagan/result/COVID-19_interpretable/COVID-19_interpretable-Imputed-scSAGAN.csv"]
    # compare_function_list = ["dropout", "scSAGANs"]
    # label_file = "./compare_funtion/sagan/test_csv/COVID-19_interpretable/GSE164948_covid_control_count_metadata.csv"
    # show_result_interpretable(data_set_name,compare_list,compare_function_list,label_file)
    # find_gene_in_set(compare_list,label_file,compare_function_list,"RPLP0","COVID-19",data_set_name)
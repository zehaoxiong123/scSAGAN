import csv
import numpy as np
import pandas as pd
import cmath as cm
import h5py
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight
import sklearn
from sklearn import preprocessing
#read_cell_to_image用于图片读取成可以放入模型的张量_用于模拟单细胞数据集
def read_cell_to_image(data_path,label_path,class_num):
    data = pd.read_csv(data_path, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    label = np.array(label)[:, 1]
    data_array = []
    data_for_count = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('double'))
        #数据归一化操作
        gene_list_all = np.sum(gene_list_for_count)
        gene_list_median = np.median(gene_list_for_count)
        gene_list_for_count = gene_list_for_count*(gene_list_median/gene_list_all)
        data_for_count.append(gene_list_all/gene_list_median)
        gene_list_for_count = np.log2(gene_list_for_count+1)
        #把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        gene_len =  len(gene_list)
        figure_size = int(gene_len**0.5)
        if figure_size*figure_size == gene_len:
            data = np.array(gene_list).reshape(figure_size,figure_size,1).astype('double')
        else:
            for j in range((figure_size+1)**2-gene_len):
                gene_list.append(0)
            data =  np.array(gene_list).reshape(figure_size+1, figure_size+1, 1).astype('double')
        data_array.append(data)
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[int(label[i][5]) - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array,data_label,gene_name,cell_name,data_for_count

#read_cell用于单细胞数据未补全时的预处理_用于模拟单细胞数据集
def read_cell(data_path,label_path,class_num):
    data = pd.read_csv(data_path, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    label = np.array(label)[:, 1]
    data_array = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('float64'))
        #数据归一化操作
        gene_list_all = np.sum(gene_list_for_count)
        gene_list_median = np.median(gene_list_for_count)
        gene_list_for_count = gene_list_for_count * (gene_list_median / gene_list_all)
        gene_list_for_count = np.log2(gene_list_for_count + 1)
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        data_array.append(np.array(gene_list))

    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[int(label[i][5]) - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    print(data_array)
    return data_array, data_label,gene_name,cell_name

def read_cell_for_h5(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max()+1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())

        print(X)
    return X,data_label,cell_type,cell_type.shape[0],obs,var

def read_cell_for_h5_to_image(filename,rate,gene_num, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_count_array=[]
        data_array = []
        semi_label_index = []
        np.random.seed(1)
        semi_label = np.random.permutation(cell_name.shape[0])
        # rate表示取多少标签进行训练
        semi_label_index = int((1 - rate) * cell_name.shape[0])
        semi_label_train = semi_label[:semi_label_index]
        #为了测试准确率用
        semi_label_real = cell_label[semi_label_train]
        weight_label_train = semi_label[semi_label_index+1:]
        cell_label[semi_label_train] = class_num
        class_weight = 'balanced'
        weight = compute_class_weight(class_weight, np.array(range(class_num)), cell_label[weight_label_train])
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num+1)
            x[cell_label[i]] = 1
            data_label.append(x)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        indicator = np.where(X > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()

        data = X[:, sum_gene>10]


        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-gene_num:]
        data = data[:, index]
        # print(data.shape)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        data = scaler.fit_transform(data)
        for i in range(cell_label.shape[0]):
            gene_list_for_count = np.array(data[i,0:].astype('double'))
            # 把单细胞表达数据转化为图片
            gene_list_max = np.max(gene_list_for_count)
            data_count_array.append(gene_list_max)

            gene_list = gene_list_for_count.tolist()
            gene_len = len(gene_list)
            figure_size = int(gene_len ** 0.5)
            if figure_size * figure_size == gene_len:
                data_train = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
                figure_size_real = figure_size
            else:
                for j in range((figure_size + 1) ** 2 - gene_len):
                    gene_list.append(0)
                data_train = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
                figure_size_real = figure_size + 1
            data_array.append(data_train)

        test_data = np.array(data_array)[semi_label_train]
    return data_array,data_label,cell_type,cell_type.shape[0],obs,var,figure_size_real,data_count_array,weight,scaler,semi_label_real,test_data

def read_cell_for_h5_imputed(array_file,filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max()+1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        data = pd.read_csv(array_file, header=None, sep=",")
        arr1 = np.array(data)
        data_array = []
        cell_num = np.shape(cell_name)[0]
        print(cell_num)

        for i in range(cell_num):
            gene_list_for_count = np.array(arr1[1:, i + 1].astype('float64'))
            # 数据归一化操作
            # gene_list_all = np.sum(gene_list_for_count)
            # gene_list_median = np.median(gene_list_for_count)
            # gene_list_for_count = gene_list_for_count * (gene_list_median / gene_list_all)
            # gene_list_for_count = np.log2(gene_list_for_count + 1)
            # 把单细胞表达数据转化为图片
            gene_list = gene_list_for_count.tolist()
            gene_len = len(gene_list)
            data_array.append(np.array(gene_list))
        data_array = np.array(data_array)
    return data_array,data_label,cell_type,cell_type.shape[0],obs,var

def read_cell_for_interpretable_imputed(data_path,label_path,class_num,data_set, sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=None, sep=",")
    if data_set == "AD" or data_set == "HP":
        label = pd.read_csv(label_path, header=0, sep=",")
    elif data_set == "COVID-19":
        label = pd.read_csv(label_path, header=0, sep=";")
    elif data_set == "T2D":
        label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    if data_set == "AD" or data_set == "HP":
        label = np.array(label)[:, 1]
    elif data_set == "COVID-19":
        label = np.array(label)[:, 2]
    elif data_set == "T2D":
        label = np.array(label)[:,1]
    if data_set == "AD":
        label2number = {"AD": 1, "ct": 2}
    elif data_set == "HP":
        label2number = {"""disease: Type 2 Diabetic""": 1, """disease: Non-Diabetic""": 2}
    elif data_set == "COVID-19":
        label2number = {"C-1": 1, "C-2": 1, "C-3": 1, "C-4": 1, "COV-1": 2, "COV-2": 2, "COV-3": 2, "COV-4": 2,
                        "COV-5": 2, "COV-6": 2, "COV-7": 2, "COV-8": 2}
    elif data_set == "T2D":
        label2number = {"Control": 1, "T2D": 2}
    data_array = []
    data_for_count = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('double'))
        # 数据归一化操作
        gene_list_max = np.max(gene_list_for_count)
        data_for_count.append(gene_list_max)
        gene_list_for_count = gene_list_for_count / gene_list_max
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        data_array.append(np.array(gene_list))
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array, data_label, gene_name, cell_name, data_for_count

def read_interpretable_for_train(data_path,label_path,class_num,data_set,rate,gene_num,sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=0, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    data_label = []
    label = np.array(label)[:, 1]
    if data_set == "AD":
        label2number = {"AD": 1, "ct": 2}
    elif data_set == "HP":
        label2number = {"""disease: Type 2 Diabetic""": 1, """disease: Non-Diabetic""": 2}
    for i in range(len(label)):
        x = np.zeros(class_num+1)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    data_label_index = np.argmax(data_label,1)
    arr1 = np.array(data)
    X = arr1[:,1:].T
    indicator = np.where(X > 0, 1, 0)
    sum_gene = np.sum(indicator, axis=0).flatten()

    data = X[:, sum_gene > 10]

    var_gene = np.var(X, axis=0)
    index = np.argsort(var_gene)[-gene_num:]
    data = data[:, index]
    # print(data.shape)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.fit_transform(data)

    gene_name = np.array(arr1[index, 0])
    cell_name = np.array(arr1[0, 1:])

    data_array = []
    data_for_count = []
    np.random.seed(1)
    semi_label = np.random.permutation(cell_name.shape[0])
    # rate表示取多少标签进行训练
    semi_label_index = int((1 - rate) * cell_name.shape[0])
    semi_label_train = semi_label[:semi_label_index]
    # 为了测试准确率用
    semi_label_real = data_label_index[semi_label_train]
    weight_label_train = semi_label[semi_label_index + 1:]
    data_label[semi_label_train]=np.array([0,0,1])
    class_weight = 'balanced'
    weight = compute_class_weight(class_weight, np.array(range(class_num)), data_label_index[weight_label_train])
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(data[i, 0:].astype('double'))
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        figure_size = int(gene_len ** 0.5)
        if figure_size * figure_size == gene_len:
            data_x = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
            figure_size_real = figure_size
        else:
            for j in range((figure_size + 1) ** 2 - gene_len):
                gene_list.append(0)
            data_x = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
            figure_size_real = figure_size + 1
        data_array.append(data_x)
    data_array = np.array(data_array)

    test_data = np.array(data_array)[semi_label_train]
    return data_array, data_label, gene_name, cell_name, figure_size_real,weight,scaler,semi_label_real,test_data

def read_interpretable_for_train_COVID19(data_path,label_path,class_num,sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=";")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    label = np.array(label)[:, 2]
    label2number = {"C-1":1,"C-2":1,"C-3":1,"C-4":1,"COV-1":2,"COV-2":2,"COV-3":2,"COV-4":2,"COV-5":2,"COV-6":2,"COV-7":2,"COV-8":2}
    data_array = []
    data_for_count = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('double'))
        # 数据归一化操作
        gene_list_max = np.max(gene_list_for_count)
        data_for_count.append(gene_list_max)
        gene_list_for_count = gene_list_for_count / gene_list_max
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        figure_size = int(gene_len ** 0.5)
        if figure_size * figure_size == gene_len:
            data = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
            figure_size_real = figure_size
        else:
            for j in range((figure_size + 1) ** 2 - gene_len):
                gene_list.append(0)
            data = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
            figure_size_real = figure_size + 1
        data_array.append(data)
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real

def read_interpretable_for_train_T2D(data_path,class_num,sparsify = False, skip_exprs = False):
    data = pd.read_table(data_path, header=None, sep="\t")
    arr1 = np.array(data)
    gene_name = np.array(arr1[7:, 0])
    cell_name = np.array(arr1[6, 1:])
    label = np.array(arr1[1, 1:])
    label2number = {"Control":1,"T2D":2}
    data_array = []
    data_for_count = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[7:, i + 1].astype('double'))
        # 数据归一化操作
        gene_list_max = np.max(gene_list_for_count)
        data_for_count.append(gene_list_max)
        gene_list_for_count = gene_list_for_count / gene_list_max
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)

        figure_size = int(gene_len ** 0.5)
        if figure_size * figure_size == gene_len:
            data = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
            figure_size_real = figure_size
        else:
            for j in range((figure_size + 1) ** 2 - gene_len):
                gene_list.append(0)
            data = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
            figure_size_real = figure_size + 1
        data_array.append(data)
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real

# def read_interpretable_for_train_T2D_imputed(data_path,class_num,type,sparsify = False, skip_exprs = False):
#     data = pd.read_table(data_path, header=None, sep="\t")
#     arr1 = np.array(data)
#     if(type == 1):
#         gene_name = np.array(arr1[7:, 0])
#         cell_name = np.array(arr1[6, 1:])
#         label = np.array(arr1[1, 1:])
#     else:
#         gene_name = np.array(arr1[1:, 0])
#         cell_name = np.array(arr1[0, 1:])
#     label2number = {"Control":1,"T2D":2}
#     data_array = []
#     data_for_count = []
#     cell_num = np.shape(cell_name)[0]
#     for i in range(cell_num):
#         gene_list_for_count = np.array(arr1[7:, i + 1].astype('double'))
#         # 数据归一化操作
#         gene_list_max = np.max(gene_list_for_count)
#         data_for_count.append(gene_list_max)
#         gene_list_for_count = gene_list_for_count / gene_list_max
#         # 把单细胞表达数据转化为图片
#         gene_list = gene_list_for_count.tolist()
#         gene_len = len(gene_list)
#         data_array.append(np.array(gene_list))
#     data_array = np.array(data_array)
#     data_label = []
#     for i in range(len(label)):
#         x = np.zeros(class_num)
#         x[label2number[label[i]] - 1] = 1
#         data_label.append(x)
#     data_label = np.array(data_label)
#     return data_array, data_label, gene_name, cell_name, data_for_count

if __name__=="__main__":
    #read_cell_to_image("./test_csv/splatter_exprSet_test.csv","./test_csv/splatter_exprSet_test_label.csv",4)
    #data_array,data_label,cell_type,cell_class,obs,var,figure_size_real,data_count_array,weight,index,semi_label_real,test_data=read_cell_for_h5_to_image("./compare_funtion/sagan/test_csv/Adam/data.h5",0.2)
    # print(semi_label_real.shape)
    # print(test_data.shape)

    data_array, data_label, gene_name, cell_name, figure_size_real,weight,scaler,semi_label_real,test_data= read_interpretable_for_train("./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_counts.csv","./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_covariates.csv",2,"AD",0.5,2500)
    print(data_array.shape)
    print(gene_name)
    print(weight)
    #data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = read_interpretable_for_train(
       # "./compare_funtion/sagan/test_csv/HP_interpretable/HP.csv",
       # "./compare_funtion/sagan/test_csv/HP_interpretable/HP_label.csv", 2, "HP")
   #  data_array, data_label, gene_name, cell_name, data_for_count = read_cell_for_interpretable_imputed(
   #      "./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_counts.csv",
   #      "./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD")
   #  data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = read_interpretable_for_train_COVID19(
   #      "./compare_funtion/sagan/test_csv/COVID-19_interpretable/GSE164948_covid_control_RNA_counts.csv",
   #      "./compare_funtion/sagan/test_csv/COVID-19_interpretable/GSE164948_covid_control_count_metadata.csv", 2)

    # data_array, data_label, gene_name, cell_name, data_for_count,figure_size = read_interpretable_for_train_T2D("./compare_funtion/sagan/test_csv/t2d_interpretable/T2D_raw.tsv",2)
    # print(cell_name.shape)
    # print(data_array.shape)
    # print(gene_name.shape)
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def transpose_mat(filepath):
    data = pd.read_csv(filepath,header = None,sep=",")
    data_t = np.array(data)[1:,1:]
    data_t = data_t.transpose(-1,0)
    a = pd.DataFrame(data_t)
    a.to_csv(filepath)

def transpose_tsv_to_csv(filepath):
    data = pd.read_table(filepath, header=None, sep="\t")
    arr1 = np.array(data)
    gene_name = np.array(arr1[7:, 0])
    cell_name = np.array(arr1[6, 1:])
    label = np.array(arr1[1, 1:])
    data_array = []
    cell_num = np.shape(cell_name)[0]
    data_array.append(gene_name)
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[7:, i + 1].astype('double'))
        gene_list = gene_list_for_count.tolist()
        data_array.append(np.array(gene_list))
    a = pd.DataFrame(data_array).T
    a.to_csv('./test_csv/t2d_interpretable/t2d_raw.csv',index=0)
    b = pd.DataFrame(label)
    b.to_csv('./test_csv/t2d_interpretable/t2d_label.csv')

if __name__=="__main__":
    transpose_mat("./result/Young/Young-imputed-DeepImpute.csv")
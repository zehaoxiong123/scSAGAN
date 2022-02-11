import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import data_Preprocess
def test():
    sns.set()
    #用行和列标签绘制
    flights_long = sns.load_dataset("flights")
    flights = flights_long.pivot("month", "year", "passengers")
    # 绘制x-y-z的热力图，比如 年-月-销量 的热力图
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(flights, ax=ax)
    #设置坐标字体方向
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.show()

def draw_heatmap(gene_name,cell_name,reads,data_set_name,png_name):
    sns.set()
    # 绘制x-y-z的热力图，比如 年-月-销量 的聚类热图
    gene_name=gene_name.tolist()
    cell_name=cell_name.tolist()
    reads = reads.astype(float).tolist()
    df = pd.DataFrame(reads,columns=gene_name,index=cell_name)
    g = sns.clustermap(df, fmt="d", cmap='YlOrRd')
    ax = g.ax_heatmap
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='left')
    plt.savefig("./compare_funtion/sagan/result/"+str(data_set_name)+"/"+str(png_name)+".png")
    plt.show()
    return df

if __name__=="__main__":
    data_array, data_label, gene_name, cell_name = data_Preprocess.test()
    draw_heatmap( gene_name, cell_name,data_array)

from compare_funtion.sagan.parameter import *
from compare_funtion.sagan.trainer import Trainer
# from tester import Tester
from torch.backends import cudnn
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn
import data_loader
import numpy as np
from compare_funtion.sagan import sagan_models
from compare_funtion.sagan.utils import *
import data_Preprocess
import pandas as pd
import sklearn

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    #data_array, data_label,cell_type,cell_type_num, obs, var,figure_size,date_for_count,weight,scaler,semi_label_real,test_data = data_Preprocess.read_cell_for_h5_to_image("./test_csv/Quake_10x_Spleen/data.h5",0.3,2500)
    data_array, data_label, gene_name, cell_name, figure_size_real, weight, scaler, semi_label_real, test_data =  data_Preprocess.read_interpretable_for_train(
        "./test_csv/AD_interpretable/GSE138852_counts.csv",
        "./test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD", 0.6, 2500)
    # data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = data_Preprocess.read_interpretable_for_train(
    #     "./test_csv/AD_interpretable/GSE138852_counts.csv",
    #     "./test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD")
    # data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = data_Preprocess.read_interpretable_for_train_T2D(
    #     "./test_csv/t2d_interpretable/T2D_raw.tsv", 2)
    print(test_data.shape)
    print(semi_label_real.shape)
    # np.random.seed(0)
    # np.random.shuffle(data_array)
    # np.random.seed(0)
    # np.random.shuffle(data_label)
    dataset = data_loader.cell_Dataset(data_array, data_label)
    dataloader = DataLoader.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        #设置两种不同模型进行训练
        if config.model=='sagan':
            trainer = Trainer(dataloader, config,weight,semi_label_real,test_data)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

    #与scGAN类似利用k近邻算法补全
def count_acc(orgdict_d):
    config = get_parameters()
    discriminator = sagan_models.Discriminator(config.class_num, config.batch_size, config.imsize, config.d_conv_dim,
                                               config.z_dim).cuda()
    discriminator.load_state_dict(torch.load(orgdict_d))
    data_array, data_label, cell_type, cell_type_num, obs, var, figure_size, data_for_count,weight,index,_,_  = data_Preprocess.read_cell_for_h5_to_image(
        "./test_csv/Quake_10x_Spleen/data.h5",0.1,2500)
    data_array_r, data_label_r, cell_type_r, cell_type_num_r, obs_r, var_r, figure_size_r, data_for_count_r, weight_r,index_r,_,_  = data_Preprocess.read_cell_for_h5_to_image(
        "./test_csv/Quake_10x_Spleen/data.h5", 1,2500)
    data_label_np = np.array(data_label)
    data_label_com =np.array(data_label_r)
    data_label_semi = np.argmax(data_label_np,axis=1)
    data_label_semi_index = np.where(data_label_semi==config.class_num)
    data_array_semi = np.array(data_array)[data_label_semi_index]
    data_label_r = data_label_com[data_label_semi_index]
    data_label_test = []
    data_label_ture = []
    dataset = data_loader.cell_Dataset(data_array_semi, data_label_r)
    dataloader = DataLoader.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    for i in range(0, step_per_epoch):
        real_images, image_label = next(data_iter)
        data_label_d, _, _ = discriminator(tensor2var(real_images).permute(0, 3, 1, 2))
        data_label_d = data_label_d.detach().data.cpu().numpy()
        data_label_d = np.argmax(data_label_d, axis=1)
        image_label =  np.argmax(image_label, axis=1)
        data_label_test.extend(data_label_d)
        data_label_ture.extend(image_label.detach().data.cpu().numpy())
    data_label_ture = np.array(data_label_ture)
    data_label_test = np.array(data_label_test)
    print(data_label_test.shape)
    print(data_label_ture.shape)
    print(np.mean(data_label_ture==data_label_test))


def imputation(orgdict_g,orgdict_d):
    config = get_parameters()
    generator = sagan_models.Generator(config.batch_size,config.class_num, config.imsize, config.z_dim, config.g_conv_dim).cuda()
    generator.load_state_dict(torch.load(orgdict_g))
    discriminator = sagan_models.Discriminator(config.class_num, config.batch_size, config.imsize,config.d_conv_dim,config.z_dim).cuda()
    discriminator.load_state_dict(torch.load(orgdict_d))
    # data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_cell_to_image(
    #     "./compare_funtion/sagan/test_csv/splatter_exprSet_test.csv", "./compare_funtion/sagan/test_csv/splatter_exprSet_test_label.csv", 4)
    # data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = data_Preprocess.read_interpretable_for_train_COVID19(
    #     "./test_csv/COVID-19_interpretable/GSE164948_covid_control_RNA_counts.csv",
    #     "./test_csv/COVID-19_interpretable/GSE164948_covid_control_count_metadata.csv", 2)
    data_array, data_label, gene_name, cell_name, figure_size_real, weight, scaler, semi_label_real, test_data = data_Preprocess.read_interpretable_for_train(
        "./test_csv/AD_interpretable/GSE138852_counts.csv",
        "./test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD", 0.8, 2500)
    data_array_r, data_label_r, gene_name_r, cell_name_r, figure_size_real_r, weight_r, scaler_r, semi_label_real_r, test_data_r = data_Preprocess.read_interpretable_for_train(
        "./test_csv/AD_interpretable/GSE138852_counts.csv",
        "./test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD", 1.0, 2500)
    # data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = data_Preprocess.read_interpretable_for_train_T2D(
    #     "./test_csv/t2d_interpretable/T2D_raw.tsv", 2)
    # data_array, data_label, cell_type, cell_type_num, obs, var, figure_size, data_for_count,weight,scaler,_,_ = data_Preprocess.read_cell_for_h5_to_image(
    #     "./test_csv/Quake_10x_Spleen/data.h5",0.1,2500)
    # data_array_r, data_label_r, cell_type_r, cell_type_num_r, obs_r, var_r, figure_size_r, data_for_count_r,weight_r,scaler,_,_ = data_Preprocess.read_cell_for_h5_to_image(
    #      "./test_csv/Quake_10x_Spleen/data.h5",1,2500)
    data_array_np = np.array(data_array)
    data_label_np = np.array(data_label)
    data_label_semi = np.argmax(data_label_np,axis=1)
    data_label_semi_index = np.where(data_label_semi==config.class_num)
    data_array_semi = np.array(data_array)[data_label_semi_index]
    data_label = data_label_np[data_label_semi_index]
    dataset = data_loader.cell_Dataset(data_array_semi, data_label)
    dataloader = DataLoader.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    k = 0
    for i in range(0,step_per_epoch):
        real_images, image_label = next(data_iter)
        data_label_d,_,_ = discriminator(tensor2var(real_images).permute(0,3,1,2))
        data_label_d = data_label_d.detach().data.cpu().numpy()
        data_label_d = np.argmax(data_label_d[:,:config.class_num],axis=1)
        for j in range(data_label_d.shape[0]):
            x = np.zeros(config.class_num + 1)
            x[data_label_d[j]] = 1
            data_label_np[data_label_semi_index[0][k]]=x
            #data_label_np.append(x)
            k = k+1
    print(np.mean(np.argmax(np.array(data_label_np), axis=1)==np.argmax(np.array(data_label_r), axis=1)))
    dataset = data_loader.cell_Dataset(data_array_np, data_label_np)
    data_imp_org = np.asarray([np.array(dataset[i][0]).reshape((config.imsize*config.imsize))for i in range(len(dataset))])
    rels=[]
    # t_e =  tensor2var(torch.tensor(np.load("./models/Muraro/Muraro_0.1/_topic_embeding.npy")))
    # c_e = tensor2var(torch.tensor(np.load("./models/Muraro/Muraro_0.1/_class_embeding.npy")))
    for j in range(len(data_imp_org)):
        label_num = np.argmax(dataset[j][1])
        label =  tensor2var(torch.tensor(dataset[j][1]).repeat(50,1))
        z = tensor2var(torch.randn(50, config.z_dim))
        fake_images,_,_ = generator(z,label)
        #到这一步生成假的图片
        fake_images = fake_images.detach().data.cpu().numpy()
        #out对真实的图片进行复制
        out = data_imp_org[j].copy()
        out = out
        #qlk将真实的图片进行变维，维度变化为(19046,1)
        q1k =  data_imp_org[j].reshape((config.imsize*config.imsize,1))
        #找出真实图片的数据值中大于0的值,维度为(19046,1)
        q1kl = np.int8(q1k>0)
        #将0值再乘0不变，维度变化为(50,19046,1)
        qlkn = np.repeat(q1k*q1kl,repeats=50,axis = 1)
        #sim_out_tmp将生成的虚假图片进行变维再进行转置，维度变化为(19046,50)
        sim_out_tmp = fake_images.reshape((50,config.imsize*config.imsize)).T
        sim_outn = sim_out_tmp*np.repeat(q1kl,repeats=50,axis=1)
        diff = qlkn - sim_outn
        diff = diff*diff
        rel = np.sum(diff,axis=0)
        locs =np.where(q1kl==0)[0]
        sim_out_c = np.median(sim_out_tmp[:, rel.argsort()[0:10]], axis=1)
        #sim_out_c = np.median(sim_out_tmp, axis=1)
        out[locs] =sim_out_c[locs]
        #print(sim_out_c[locs])
        #对数据进行逆向归一化
        out_res = np.where(out>0,out,0)
        rels.append(out_res[:2500])
        #print(out_res)
    # rels = scaler.inverse_transform(rels)
    a=pd.DataFrame(rels).T
    a.to_csv("./result/Quake_10x_Spleen_0.1-imputed-scsagan(filted).csv")
if __name__ == '__main__':
    config = get_parameters()
    main(config)
    # imputation('./models/Quake_10x_Spleen/Quake_10x_Spleen_0.1/best_G.pth','./models/Quake_10x_Spleen/Quake_10x_Spleen_0.1/best_D.pth')

    #count_acc('./models/Quake_10x_Spleen/Quake_10x_Spleen_0.1/best_D.pth')
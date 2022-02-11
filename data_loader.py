import torch
import torchvision.datasets as dsets
from torchvision import transforms
import torch.utils.data.dataset as Dataset
import data_Preprocess
import torch.utils.data.dataloader as DataLoader
class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        #设置dataLoader的数据集类型
        self.dataset = dataset
        #设置dataset的图片路径
        self.path = image_path
        #设置图片尺寸
        self.imsize = image_size
        #设置图片批训练量
        self.batch = batch_size
        #设置数据的随机打乱顺序
        self.shuf = shuf

        self.train = train


    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()

        #加载图片数据
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader


class cell_Dataset(Dataset.Dataset):
    def __init__(self,cell_image,cell_label):
        self.Data = cell_image
        self.Label = cell_label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data,label


if __name__=='__main__':
    data_array,data_label,gene_name,cell_name,data_for_count=data_Preprocess.read_cell_to_image("./test_csv/splatter_exprSet_test.csv", "./test_csv/splatter_exprSet_test_label.csv", 4)
    dataset = cell_Dataset(data_array,data_label)
    dataloader  = DataLoader.DataLoader(dataset,batch_size=8, shuffle = True, num_workers= 4)
    for i, item in enumerate(dataloader):
        print('i:', i)
        data, label = item
        print('data:', data)
        print('label:', label)
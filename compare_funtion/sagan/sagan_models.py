import torch
import torch.nn as nn
from compare_funtion.sagan.spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,width,height):
        super(Self_Attn,self).__init__()
        #chanel_in设置浅层变量维度
        self.chanel_in = in_dim
        #activation设置激活函数
        self.activation = activation
        #设置q，k，v三个卷积层进行图片卷积操作
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        #设置4个属性用于存储图片属性的4个不同维度特性，C是图片通道数
        m_batchsize,C,width ,height = x.size()
        #计算magic层的注意力
        #计算同标签类别的遮罩

        #x_label_matrix = torch.mm(x_label,x_label.view(m_batchsize,class_number).permute(1,0))
        #x_label_matrix_byte = x_label_matrix.repeat(C,1,1).byte()
        #magic_matrix_1 = self.magic_conv(x).view(m_batchsize,-1,width*height).permute(1,0,2)
        #magic_matrix_2 = self.magic_conv(x).view(m_batchsize,-1,width*height).permute(1,2,0)
        #magic_atten = torch.bmm(magic_matrix_1, magic_matrix_2)
        #magic_atten = self.softmax(magic_atten)
        #magic_atten.masked_fill(x_label_matrix_byte == 0, value=torch.tensor(0))
        #y_label_out = self.label_atten(x_label).view(m_batchsize,-1,width*height)
        #y_label_out_t =y_label_out.permute(0,2,1)
        #y_label_energy = torch.bmm(y_label_out_t, y_label_out)
        #先将图片转变为C*(width*height)的形式，再对张量进行转置转变成(width*height)*C的形式
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        #将图片转变为key_conv，张量转变为C*(width*height)的形式
        #引入MAGIC层注意力
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        #利用torch将key和query相乘，得到(width*height)*(width*height)的张量
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        #energy_atten = torch.bmm(energy,y_label_energy)
        #给注意力矩阵通过softmax函数
        #attention = self.softmax(energy_atten) # BX (N) X (N)
        attention = self.softmax(energy)  # BX (N) X (N)
        #把图片输入给value卷积层得到C*(width*height)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) ).view(m_batchsize,-1,width,height)
        out = self.gamma*out + x

        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self,batch_size,class_num, image_size=122, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.softmax = nn.Softmax(dim=-1)
        self.class_num = class_num
        #设置超参数
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        #设置重复次数ls
        repeat_num = int(np.log2(self.imsize))
        mult = 8 # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim+self.class_num+1, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())


        layer4 = []
        curr_dim = int(curr_dim / 2)
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())
        self.l4 = nn.Sequential(*layer4)
        curr_dim = int(curr_dim / 2)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.stride = int(self.imsize/31)
        self.kernel = self.imsize+6-31*self.stride
        last.append(nn.ConvTranspose2d(curr_dim, 1, self.kernel, self.stride, 3))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)
        self.attn1 = Self_Attn( self.imsize*2, 'relu',64,64)
        self.attn2 = Self_Attn(self.imsize,  'relu',128,128)

    def forward(self, z,x_label):
        #type_embeding_repeat = c_e.repeat(z.size(0), 1, 1).permute(0,2,1)
        x_label = x_label.view(x_label.size(0),x_label.size(1),1,1)
        #x_topic_exp = torch.bmm(x_label,type_embeding_repeat).view(z.size(0), z.size(1), 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        z_ex = torch.cat((z,x_label),1)
        out=self.l1(z_ex)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)
        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self,class_num, batch_size=64, image_size=29, conv_dim=64,z_dim=50):
        super(Discriminator, self).__init__()

        self.imsize = image_size
        self.gene_num = self.imsize*self.imsize
        self.topic_num =z_dim
        self.cell_type = class_num
        self.alpha = 0.3
        self.z_dim = z_dim
        self.softmax = nn.Softmax(dim=-1)  #
        # introduce the learned parameter
        self.topic_embeding = nn.Parameter(torch.randn(self.gene_num, self.z_dim))
        self.type_embeding = nn.Parameter(torch.randn(self.z_dim, self.cell_type + 1))
        self.repeat_num = int(np.log2(self.imsize))
        layer1 = []
        last = []
        curr_dim = 1
        for i in range(self.repeat_num):
            layer1.append(SpectralNorm(nn.Conv2d(curr_dim, conv_dim, 4, 2, 1)))
            layer1.append(nn.LeakyReLU(0.1))
            curr_dim = conv_dim
            conv_dim = conv_dim*1


        self.l1 = nn.Sequential(*layer1)

        #指定判别器变为k+1维的分类器
        last.append(nn.Conv2d(conv_dim, self.z_dim, 1))
        last.append(nn.Softmax())
        self.last = nn.Sequential(*last)
        self.attn2 = Self_Attn(64, 'relu',1,1)

    def forward(self, x):
        #加入基因-主题嵌入
        m_batches,C,width,height = x.size()
        x_embedding = x.view(m_batches,-1,width*height)
        topic_embeding_repeat = self.topic_embeding.repeat(m_batches,1,1)
        type_embeding_repeat = self.type_embeding.repeat(m_batches,1,1)
        topic_dis = torch.bmm(x_embedding,topic_embeding_repeat)
        topic_dis = self.softmax(topic_dis)
        out = self.l1(x)
        out,p2 = self.attn2(out)
        out=self.last(out)
        #融合主题嵌入
        out = out.view(m_batches,-1,self.topic_num)
        out_emerge = (1-self.alpha)*out + self.alpha*topic_dis
        out_dis = self.softmax(torch.bmm(out_emerge,type_embeding_repeat))

        out_dis = out_dis.view(-1,self.cell_type+1)
        return out_dis, self.topic_embeding, self.type_embeding

import time
import datetime
import torch.nn as nn
import numpy as np
import sys
from torchvision.utils import save_image
from compare_funtion.sagan.sagan_models import Generator, Discriminator
from compare_funtion.sagan.utils import *
import data_loader
import torch.utils.data.dataloader as DataLoader
class Trainer(object):
    def __init__(self, data_loader, config ,weight,semi_label_real,test_data):


        # Data loader
        self.data_loader = data_loader
        self.class_num = config.class_num
        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss
        self.weight =weight
        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.kt = config.kt
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel
        self.beta = 1.0
        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.gamma = config.gamma
        self.dataset = config.dataset
        self.param = config.params
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        #test
        self.semi_label_real = semi_label_real
        self.test_data = test_data
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()




    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)
        lambda_k = 0.001
        k = self.kt
        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        max_M = sys.float_info.max
        min_dM = 1e-08
        dM = 1
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        test_label = tensor2var(torch.tensor(self.semi_label_real))

        for step in range(start, self.total_step):
            cur_M = 0
            cur_dM = 1
            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:

                #real_images, _ = next(data_iter)
                real_images, image_label = next(data_iter)
                step_batches = image_label.shape[0]
            except:
                data_iter = iter(self.data_loader)
                #real_images, _ = next(data_iter)
                real_images,image_label = next(data_iter)
                step_batches = image_label.shape[0]

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            #在每一轮中测试准确率
            max_accuracy = 0
            test_data = tensor2var(torch.tensor(self.test_data)).float().permute(0,3,1,2)


            #将最后一类的损失调0
            image_label=tensor2var(image_label)
            image_label_mmd = torch.argmax(image_label)
            image_real_label = torch.argmax(image_label,dim=1)
            image_real_label_star = image_real_label
            image_mask = torch.tensor(image_label, dtype=bool)
            loss_weight = torch.tensor(self.weight)
            image_label_weight_ = image_label[:,:self.class_num] * tensor2var(loss_weight)
            image_label_weight = torch.masked_select(image_label_weight_,image_mask[:,:self.class_num])
            image_mmd_index = torch.where(image_real_label != self.class_num)
            image_semi_index = torch.where(image_real_label == self.class_num)
            real_images = tensor2var(real_images).permute(0,3,1,2)
            d_out_real,t_e,c_e = self.D(real_images)
            image_d_label = torch.argmax(d_out_real[:,:self.class_num],dim=1)
            image_d_label_g = image_d_label
            image_d_label_g_train =  tensor2var(torch.nn.functional.one_hot(image_d_label_g, self.class_num+1)).float()
            image_d_label_g_train_mask = torch.tensor(image_d_label_g_train, dtype=bool)
            image_d_label_g_train_weight_ = image_d_label_g_train[:,:self.class_num]*tensor2var(loss_weight)
            image_d_label_g_train_weight = torch.masked_select(image_d_label_g_train_weight_,image_d_label_g_train_mask[:,:self.class_num])
            #find star in data
            # image_real_label_star[image_semi_index] = image_d_label[image_semi_index]
            # image_find_star = []
            # for i in range(self.class_num):
            #     index = torch.where(image_real_label == i)
            #     star = torch.mean(real_images[index],dim=0,keepdim=True)
            #     star = torch.where(star == 'nan',0,star)
            #     print(star)
            #     image_find_star.append(star)
            #
            # image_find_star_g = tensor2var(torch.tensor(image_find_star))

            #对参与损失计算的标签进行筛选
            image_mask = image_mask[image_mmd_index]
            d_out_real_label = d_out_real[image_mmd_index]
            d_out_real_unlabel = d_out_real[image_semi_index]
            image_real_label = image_real_label[image_mmd_index]
            image_d_label = image_d_label[image_mmd_index]
            if self.adv_loss == 'wgan-gp':
                #全新的损失计算方法：

                first_loss_ = torch.masked_select(d_out_real_label[:,:self.class_num],image_mask[:,:self.class_num])
                first_loss = torch.mul(first_loss_,image_label_weight).mean()
                second_loss = torch.log(torch.sum(torch.exp(d_out_real_label[:,:self.class_num]),dim=1)).mean()
                d_loss_real_label = -1*first_loss + second_loss
                third_loss = torch.log(torch.sum(torch.exp(d_out_real_unlabel[:,:self.class_num]),dim=1)).mean()
                firth_loss = torch.log( 1 + torch.sum(torch.exp(d_out_real_unlabel[:,:self.class_num]),dim=1)).mean()
                d_loss_real_unlabel = -1*third_loss + firth_loss
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
           #  z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
           #  fake_images,gf1,gf2 = self.G(z,image_label,t_e,c_e)
           # # d_out_fake_2, dr3, dr4 = self.D(real_images, z_label_tensor)
           #  d_out_fake,df1,df2 = self.D(fake_images)
            #d_out_fake2,df3,df5 = self.D(fake_images,image_label)
            # if self.adv_loss == 'wgan-gp':
            #     d_loss_fake = torch.log(torch.sum(torch.exp(d_out_real[:,:self.class_num]),dim=1) + 1).mean()
                #d_loss_fake2 = d_out_fake_2.mean()

               # d_loss_fake2 = torch.nn.ReLU()(1.0 + d_out_fake_2).mean()
           ##if self.adv_loss == 'wgan-gp':
            ##     d_out_fake_2 = d_out_fake_2.mean()
            ## elif self.adv_loss == 'hinge':
            ##     d_out_fake_2 = torch.nn.ReLU()(1.0 + d_out_fake_2).mean()
            ##
            # Backward + Optimize
            d_loss = d_loss_real_label + 0.3*d_loss_real_unlabel
            # d_pesudo_label = tensor2var(torch.nn.functional.one_hot(image_d_label_g, self.class_num+1)).float()
            # d_pesudo_label_mask = torch.tensor(d_pesudo_label,dtype=bool)
            # d_pesudo_label_weight_ = d_pesudo_label[:,:self.class_num] * tensor2var(loss_weight)
            # d_pesudo_label_weight = torch.masked_select(d_pesudo_label_weight_,d_pesudo_label_mask[:,:self.class_num])
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            ##半监督学习
            d_test_label_,_,_ =self.D(test_data)
            d_test_label = torch.argmax(d_test_label_[:,:self.class_num],dim=1)
            accracy_t = torch.mean((d_test_label==test_label).float())
            '''
            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated,image_label)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step() 
            '''
            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_,_ = self.G(z,image_d_label_g_train)
            # fake_images_cucal = fake_images.view(real_images.size(0),1,fake_images.shape[-1]*fake_images.shape[-1])
            # fake_image_mmd = torch.bmm(fake_images_cucal[image_mmd_index],t_e.repeat(image_mmd_index[0].shape[0],1,1))
            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images)  # batch x n
            image_g_real = torch.argmax(g_out_fake,dim=1)
            #
            # g_out_fake = g_out_fake[image_mmd_index]
            # image_g_real = image_g_real[image_mmd_index]
            if self.adv_loss == 'wgan-gp':
                g_first_loss_ = torch.masked_select(g_out_fake[:,:self.class_num],image_d_label_g_train_mask[:,:self.class_num])
                g_first_loss = torch.mul(g_first_loss_,image_d_label_g_train_weight).mean()
                g_second_loss = torch.log(torch.sum(torch.exp(g_out_fake[:, :self.class_num]), dim=1)).mean()
                # mmd_loss = mmd(fake_image_mmd.view(fake_image_mmd.shape[0],self.z_dim), image_g_real,self.class_num, self.beta)
                # for j in range(self.class_num):
                #     g_index = torch.where(image_g_real==j)
                #     g_out_fake_star_mean = torch.mean(g_out_fake[g_index])
                #
                #     star_loss_j = torch.dist(image_find_star_g,g_out_fake_star_mean)
                #     star_loss.append(star_loss_j)
                # star_loss = tensor2var(torch.tensor(star_loss).mean())
                g_loss_fake = -1 * g_first_loss + g_second_loss
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            accracy = torch.mean((image_d_label==image_real_label).float())
            accracy_g = torch.mean((image_g_real==image_d_label_g).float())
            # Print out log info

            if (step + 1) % 10 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, accyacy_g: {:.4f}, accyacy: {:.4f},accuracy_t : {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss,
                             self.G.attn1.gamma.mean().data,accracy_g,accracy,accracy_t))
                max_accuracy = accracy_t
            '''
            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(z,image_label,t_e,c_e)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
            '''
            if accracy_t >= max_accuracy and accracy_g >= max_accuracy:
                torch.save(self.G.state_dict(),
                               os.path.join(self.model_save_path, 'best_G.pth'))
                torch.save(self.D.state_dict(),
                               os.path.join(self.model_save_path, 'best_D.pth'))
                np.save(os.path.join(
                        self.param, 'best_topic_embeding.npy'),t_e.data.cpu().numpy())
                np.save(os.path.join(
                        self.param, 'best_class_embeding.npy'), c_e.data.cpu().numpy())
                min_dM = accracy
            # diff = torch.mean(self.gamma * d_loss_real_label - g_loss_fake)
            # M = (d_loss_real_label + torch.abs(diff)).item()
            # cur_M += M
            # cur_M = cur_M / step_batches
            #
            # if cur_M < max_M:
            #     torch.save(self.G.state_dict(),
            #                os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
            #     torch.save(self.D.state_dict(),
            #                os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
            #     np.save(os.path.join(
            #         self.param, '_topic_embeding.npy'),t_e.data.cpu().numpy())
            #     np.save(os.path.join(
            #         self.param, '_class_embeding.npy'), c_e.data.cpu().numpy())
            #     dM = min(max_M - cur_M, cur_M)
            #     if dM < min_dM:  # if convergence threthold meets, stop training
            #         print("Training was stopped after " + str(
            #             step + 1) + " epoches since the convergence threthold (" + str(min_dM) + ".) reached: " + str(
            #             dM))
            #         break
            #     cur_dM = max_M - cur_M
            #     max_M = cur_M
            if step + 1 == self.total_step and cur_dM > min_dM:
                print("Training was stopped after " + str(
                    step + 1) + " epoches since the maximum epoches reached: " + str(self.total_step  + "."))
                print("WARNING: the convergence threthold (" + str(min_dM) + ") was not met. Current value is: " + str(
                    cur_dM))
                print("You may need more epoches to get the most optimal model!!!")


    def build_model(self):

        #self.G = Generator(self.topic_embeding,self.class_embeding,self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        #self.D = Discriminator(self.topic_embeding,self.class_embeding,self.class_num,self.batch_size,self.imsize, self.d_conv_dim).cuda()
        self.G = Generator(self.batch_size,self.class_num, self.imsize, self.z_dim,self.g_conv_dim).cuda()
        self.D = Discriminator(self.class_num, self.batch_size, self.imsize,self.d_conv_dim,self.z_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def save_params(self):
        np.save(os.path.join(
            self.param, '{}_topic_embeding.npy'),self.topic_embeding.data.cpu().numpy())
        np.save(os.path.join(
            self.param, '{}_class_embeding.npy'), self.class_embeding.data.cpu().numpy())

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

    # def kl_sinkhorn(self,c_e):
    #     c_e_count = c_e.view(self.class_num+1,-1)
    #     loss = []
    #     for i in range(c_e_count.shape[0]):
    #         loss_class = 0
    #         for j in range(c_e_count.shape[0]-i-1):
    #             dist = torch.dist(c_e_count[i],c_e_count[j+1+i])
    #             loss_class+=dist
    #         loss.append(loss_class)
    #     loss = tensor2var(torch.tensor(loss).mean())
    #     return 0.01*loss

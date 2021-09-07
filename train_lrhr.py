import os
import time
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset
from imresize import imresize
from gkernel import *
import model
import mymodel
from config import *
from utils import get_root_logger, mkdir_or_exist
from torch.utils.data import Dataset, DataLoader

import cv2


class Train(object):
    def __init__(self, args, size, scale_list, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr,
                 task_iter, loadfrom_dir, config, bz):
        print('[*] Initialize Training')
        self.trial = args.trial  # 0
        self.step = args.step
        self.current_epoch = 0

        self.HEIGHT = size[0]
        self.WIDTH = size[1]
        self.CHANNEL = size[2]
        self.scale_list = scale_list
        # self.scale=scale_list[0] #2

        self.META_BATCH_SIZE = meta_batch_size
        self.META_LR = meta_lr
        self.META_ITER = meta_iter

        self.TASK_BATCH_SIZE = task_batch_size
        self.TASK_LR = task_lr  # 1e-2
        self.TASK_ITER = task_iter

        # self.data_generator=data_generator
        self.loadfrom_dir = loadfrom_dir
        self.work_dir = args.work_dir  # ./experiments
        self.use_weighted_loss = args.use_weighted_loss
        self.bz = bz
        self.multi_step_loss_num_epochs = args.multi_step_loss_num_epochs
        self.total_epoch = args.total_epoch
        print('use weighted loss?', self.use_weighted_loss)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.work_dir = os.path.join(self.work_dir, timestamp)  # 创建work_dir
        mkdir_or_exist(os.path.abspath(self.work_dir))

        # init the logger
        log_file = os.path.join(self.work_dir, 'root.log')
        self.logger = get_root_logger(log_file=log_file, log_level='INFO')
        if self.bz is not None:
            self.logger.info(self.bz)
        # log some basic info
        # logger.info('training gpus num: {}'.format(args.gpu_ids))
        # logger.info('Config:\n{}'.format(arg_text))

        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])

        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if torch.cuda.is_available() else torch.device(
            'cpu')

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''model'''
        self.config = config
        self.model = model.Learner(self.config)
        # self.model = mymodel.Learner(self.config)

        # self.model = mymodel.MYMODEL(args)
        if self.loadfrom_dir is not None:
            self.pretrain_weights = torch.load(self.loadfrom_dir)  # 加载预训练模型
            self.vars = nn.ParameterList()
            for name, param in self.pretrain_weights.items():  # 装着可以放进Model的权重
                # print(name, param.requires_grad)
                if 'laplacian' in name:  # 有三个laplacian的权重也会被不小心放进来，但是它们实际上是不需要学习的
                    continue
                self.vars.append(nn.Parameter(param))
            print('pretrain vars', len(self.vars))
            self.model.vars = self.vars  # 将预训练的参数放入
            self.logger.info('load checkpoint from %s', self.loadfrom_dir)
            # print(self.model.vars[15])  #看一下是否Load成功

        param_nums = 0
        for item in self.model.parameters():  # 得到类的参数
            param_nums += np.prod(np.array(item.shape))
        self.logger.info("model: {} 's total parameter nums: {}".format(self.model.__class__.__name__,
                                                                        param_nums))  # 如果用model.__class__ 则输出的类名为 DataParallel 用model.module则为similarity类

        if len(args.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.model.to(args.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, args.gpu_ids)  # multi-GPUs
        # self.model = self.model.to(self.device)

        '''loss'''
        self.loss_fn = nn.L1Loss()

        '''Optimizers'''
        self.pretrain_op = optim.Adam(self.model.parameters(), lr=self.META_LR)
        self.opt = optim.Adam(self.model.parameters(), lr=self.META_LR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt, T_max=64, eta_min=0)
        # print(self.model.module.parameters())

    def construct_model(self, lr,hr):
        def task_meta_learning(lr,hr):  # hr size 8,3,64,64 这个batch应该属于同一个任务
            lr_son = []  #存放从lr得到的lr_son，用于Dtr inputa labela
            lr_=[]  #存放从hr得到的lr，用于Dte inputb labelb

            #from hr to lr Dtr和Dte用不同的Kernel
            scale = np.random.choice(self.scale_list, 1)[0]  # 2
            Kernel = generate_kernel(k1=scale * 2.5, ksize=15)  # 随机生成Kernel
            noise_std = 0.0
            for i in range(hr.shape[0]):  # 8
                hr_tmp = hr[i].permute(1, 2, 0).cpu().detach().numpy()  # 64,64,3
                clean_img_LR = imresize(hr_tmp, scale=1. / scale, kernel=Kernel)
                img_LR = np.clip(clean_img_LR + np.random.randn(*clean_img_LR.shape) * noise_std, 0., 1.)  # 加了噪声的
                # used cubic upsample
                img_ILR = imresize(img_LR, scale=scale, output_shape=hr_tmp.shape, kernel='cubic')
                lr_.append(img_ILR)

            lr_ = np.asarray(lr_)  # 8,64,64,3
            lr_ = torch.as_tensor(lr_).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)  # 8,3,64,64

            scale = np.random.choice(self.scale_list, 1)[0]  # 2
            Kernel = generate_kernel(k1=scale * 2.5, ksize=15)  # 随机生成Kernel
            #from lr to lr_son
            for i in range(lr.shape[0]):  # 8
                lr_tmp = lr[i].permute(1, 2, 0).cpu().detach().numpy()  # 64,64,3
                clean_img_LR_son = imresize(lr_tmp, scale=1. / scale, kernel=Kernel)
                img_LR_son = np.clip(clean_img_LR_son + np.random.randn(*clean_img_LR_son.shape) * noise_std, 0., 1.)  # 加了噪声的
                # used cubic upsample
                img_ILR_son = imresize(img_LR_son, scale=scale, output_shape=lr_tmp.shape, kernel='cubic')
                lr_son.append(img_ILR_son)

            lr_son = np.asarray(lr_son)  # 8,64,64,3
            lr_son = torch.as_tensor(lr_son).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)  # 8,3,64,64


            inputa = lr_son  # 8,3,64,64
            labela = lr  # 8,3,64,64

            inputb = lr_  # 8,3,64,64
            labelb = hr  # 8,3,64,64

            self.LW = self.get_loss_weights().to(self.device)  # 可以求出来，用不用再说

            self.model.train()
            task_outputbs = []
            task_lossesb = []
            task_outputa = self.model(inputa, vars=None)  # support set
            # print(self.model.module.vars[0][0][0]) 证明是在梯度更新的
            task_lossa = self.loss_fn(labela, task_outputa)
            grad = torch.autograd.grad(task_lossa, filter(lambda p: p.requires_grad, self.model.parameters()))
            fast_weights = list(map(lambda p: p[1] - self.TASK_LR * p[0], zip(grad, self.model.parameters())))

            '''
            # this is the loss before first update use the model's original parameters
            with torch.no_grad():
                output = self.model(inputb,self.model.parameters())  # query set
                task_outputbs.append(output)
                task_lossesb.append(self.loss_fn(labelb, output))
            '''

            # this is the loss after the first update use the fast weights
            with torch.no_grad():
                output = self.model(inputb, fast_weights)
                task_outputbs.append(output)  # 存放test的输出

                if self.use_weighted_loss and self.current_epoch < self.multi_step_loss_num_epochs:  # 启用加权Loss
                    # task_lossesb += self.loss_fn(labelb, output) * self.LW[0]
                    task_lossesb.append(self.loss_fn(labelb, output) * self.LW[0])

                # task_lossesb += self.loss_fn(labelb, output) * self.LW[0]

            '''
            对于每个任务，优化好的参数要使得在Dte上的Loss最小
            '''
            for i in range(self.TASK_ITER - 1):
                output_s = self.model(inputa, fast_weights)  # inputa是Dtrain  support set
                loss = self.loss_fn(labela, output_s)
                grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, fast_weights))
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.TASK_LR * p[0], zip(grad, fast_weights)))
                '''
                修改前
                # output = self.model(inputb, fast_weights)  # inputb是Dtest  query set
                # task_outputbs.append(output)  # 存放test的输出
                # task_lossesb += self.loss_fn(labelb, output) * self.LW[i + 1]  # 1~4
                '''

                if self.use_weighted_loss and self.current_epoch < self.multi_step_loss_num_epochs:  # 启用加权Loss 并且只在前15个epoch

                    output = self.model(inputb, fast_weights)  # inputb是Dtest  query set
                    task_outputbs.append(output)  # 存放test的输出
                    # task_lossesb += self.loss_fn(labelb, output) * self.LW[i + 1]  # 1~4
                    task_lossesb.append(self.loss_fn(labelb, output) * self.LW[i + 1])  # 1~4
                else:
                    if i == self.TASK_ITER - 2:  # 3,也就是最后一个step时，MAML原始代码只使用了最后一次step时候的Loss
                        output = self.model(inputb, fast_weights)  # inputb是Dtest  query set
                        task_outputbs.append(output)  # 存放test的输出
                        task_lossesb.append(self.loss_fn(labelb, output))

            task_lossesb = torch.sum(torch.stack(task_lossesb))
            # task_lossesb /= self.TASK_ITER
            task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
            return task_output  # hr size 16,3,

        self.total_lossa = 0.
        self.total_lossesb = []

        for i in range(self.META_BATCH_SIZE):  # 5
            res = task_meta_learning(lr[i],hr[i])  # hr[i] (8, 3, 64, 64)
            self.total_lossa += res[2]  # 先求和，然后再除以meta batch size
            self.total_lossesb.append(res[3])  # 5个iter loss的均值

        self.total_lossa /= self.META_BATCH_SIZE  # 训练的Loss
        # self.total_lossesb /= self.META_BATCH_SIZE  # 相当于除以META_BATCH_SIZE
        self.total_lossesb = torch.mean(torch.stack(self.total_lossesb))  # 相当于除以META_BATCH_SIZE

        print(self.total_lossesb)
        self.opt.zero_grad()
        self.total_lossesb.backward()
        self.opt.step()  # 该优化器的学习率是meta lr =1e-4 β


    def get_loss_weights(self):
        loss_weights = torch.ones(size=[self.TASK_ITER]) * (1.0 / self.TASK_ITER)
        decay_rate = 1.0 / self.TASK_ITER / self.multi_step_loss_num_epochs  # 使用加权loss多少个epoch，就设置相应的退化率！
        min_value = 0.03 / self.TASK_ITER

        loss_weights_pre = torch.maximum(
            loss_weights[:-1] - (torch.multiply(torch.tensor(self.current_epoch), torch.tensor(decay_rate))),
            torch.tensor(min_value))

        loss_weight_cur = torch.minimum(
            loss_weights[-1] + (
                torch.multiply(torch.tensor(self.current_epoch), torch.tensor((self.TASK_ITER - 1) * decay_rate))),
            torch.tensor(1.0 - ((self.TASK_ITER - 1) * min_value)))

        loss_weight_cur = loss_weight_cur.reshape(-1)  # torch.Size([1])
        # torch.Size([4])
        loss_weights = torch.cat((loss_weights_pre, loss_weight_cur), axis=0)  # torch.Size([5])
        # print(loss_weights)
        return loss_weights

    def __call__(self):
        # init the logger

        PRINT_ITER = 100
        SAVE_ITET = 1000
        self.logger.info('Training Starts')
        SECOND_ORDER_GRAD_ITER = 0  # For the 1st-order approximation. Until this step, 1st-order approximation is used for fast training

        step = self.step
        t2 = time.time()

        meta_ds = dataset.metaTrainDataset_lrhr("/data", 'DIV2K', self.scale_list)
        dataloader = DataLoader(meta_ds, batch_size=self.META_BATCH_SIZE * self.TASK_BATCH_SIZE, shuffle=True,
                                pin_memory=True, drop_last=True)  # bs=40
        self.logger.info('Len of dataset is %d , totally %d batches ' % (len(meta_ds), len(dataloader)))
        while True:
            self.scheduler.step(epoch=self.current_epoch)
            self.current_epoch += 1  # 应该计算epoch?

            for lr,hr in dataloader:  # len=10
                lr = lr.unsqueeze(1).view(self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, 3, 64, 64).to(self.device)  # 5,8,3,64,64
                hr = hr.unsqueeze(1).view(self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, 3, 64, 64).to(self.device)  # 5,8,3,64,64

                self.construct_model(lr,hr)  # 80张一个iter,一个epoch需要10个iter,20000个iter等同于2000个epoch
                step += 1

                if step % PRINT_ITER == 0:
                    t1 = t2
                    t2 = time.time()
                    # self.logger.info('Iter: %d - Pre loss %.3f - Post weighted loss: %.3f - Post loss: %.3f - Time: %.2f'%(step,self.total_lossa,self.weighted_total_lossesb,self.total_lossesb,t2 - t1))
                    self.logger.info('Iter: %d - Pre loss %.4f - Post loss: %.4f - Time: %.2f' % (
                        step, self.total_lossa, self.total_lossesb, t2 - t1))

                if step % SAVE_ITET == 0:
                    save_path = os.path.join(self.work_dir, 'checkpoint')
                    save(self.model, save_path, step)  # trial=0  step=2500

                    self.logger.info('Save model to %s' % (save_path))

                if step == self.META_ITER:  # 100000
                    self.logger.info('Done Training - Time: %s' % (strftime('%b-%d %H:%M:%S', localtime())))
                    break

            if (self.current_epoch == self.total_epoch):
                self.logger.info('Done Training - Time: %s' % (strftime('%b-%d %H:%M:%S', localtime())))
                break



def save(model, checkpoint_dir, step):
    model_name = 'model_{}.pth'.format(str(step))  # model_2500.pth
    mkdir_or_exist(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, model_name))  # checkpoint/Model0/model_2500.pth

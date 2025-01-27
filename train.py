import os
import time
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset

import model
import mymodel
from config import *
from utils import get_root_logger,mkdir_or_exist

class Train(object):
    def __init__(self, args, size, scale_list, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr,
                 task_iter,checkpoint_dir, loadfrom_dir,config):
        print('[*] Initialize Training')
        self.trial = args.trial  #0
        self.step = args.step
        self.global_step = args.step
        
        self.HEIGHT = size[0]
        self.WIDTH = size[1]
        self.CHANNEL = size[2]
        self.scale_list = scale_list
        # self.scale=scale_list[0] #2

        self.META_BATCH_SIZE = meta_batch_size
        self.META_LR = meta_lr
        self.META_ITER = meta_iter

        self.TASK_BATCH_SIZE = task_batch_size
        self.TASK_LR = task_lr  #1e-2
        self.TASK_ITER = task_iter

        # self.data_generator=data_generator
        self.checkpoint_dir = checkpoint_dir
        self.loadfrom_dir = loadfrom_dir
        self.work_dir=args.work_dir  #./experiments
        self.use_weighted_loss=args.use_weighted_loss
        print('use weighted loss?',self.use_weighted_loss)
        if self.use_weighted_loss:
            self.LW = self.get_loss_weights()
        else: 
            self.LW=torch.ones((self.TASK_ITER))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.work_dir = os.path.join(self.work_dir, timestamp)  # 创建work_dir
        mkdir_or_exist(os.path.abspath(self.work_dir))

        # init the logger
        log_file = os.path.join(self.work_dir, 'root.log')
        self.logger = get_root_logger(log_file=log_file, log_level='INFO')
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


        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''model'''
        self.config=config
        self.model = model.Learner(self.config)

        # self.model = mymodel.MYMODEL(args)
        if self.loadfrom_dir is not None:
            self.pretrain_weights = torch.load(self.loadfrom_dir)  # 加载预训练模型
            # from collections import OrderedDict
            # for key,value in checkpoint.items():
            #     print(key)

            # new_state_dict = OrderedDict()
            # for k, v in checkpoint['net'].items():
            #     name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            # self.model.load_state_dict(self.pretrain_weights)
            # self.logger.info('load checkpoint from %s', self.loadfrom_dir)

            self.vars=nn.ParameterList()
            for name, param in self.pretrain_weights.items():  #装着可以放进Model的权重
                print(name, param)
                self.vars.append(nn.Parameter(param))
            self.model.vars=self.vars #将预训练的参数放入
            print(self.model.vars[15])  #看一下是否Load成功 



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
        # print(self.model.module.parameters())

    def construct_model(self, inf):
        def task_meta_learning(inputa, labela, inputb, labelb):
            self.model.train()

            inputa = torch.as_tensor(inputa).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)  # 8,3,64,64
            labela = torch.as_tensor(labela).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)
            inputb = torch.as_tensor(inputb).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)  # 8,3,64,64
            labelb = torch.as_tensor(labelb).type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)

            task_outputbs= []
            task_lossesb=0
            
            # self.model.load_state_dict(self.pretrain_weights)  #每个任务的起点都一样，都用的是加载进来的参数作为权重
            task_outputa = self.model(inputa,vars=None)  #support set
            # print('see var[15]',self.model.module.vars[15])  # 看一下每一轮训练的时候，参数是否一样

            task_lossa = self.loss_fn(labela, task_outputa)
            grad = torch.autograd.grad(task_lossa, self.model.parameters())
            # print(grad)
            fast_weights = list(map(lambda p: p[1] - self.TASK_LR * p[0], zip(grad, self.model.parameters())))
            # print(fast_weights[0])

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
                # task_lossesb.append(self.loss_fn(labelb, output))  # 先把task_lossb都放到列表中 最后再来更新meta_learner的梯度
                task_lossesb+=self.loss_fn(labelb, output)*self.LW[0]
                
                

            for i in range(self.TASK_ITER - 1):
                output_s = self.model(inputa,fast_weights)  # inputa是train
                loss = self.loss_fn(labela, output_s)
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.TASK_LR * p[0], zip(grad, fast_weights)))

                output = self.model(inputb,fast_weights)  # inputb是test
                task_outputbs.append(output)  # 存放test的输出
                # task_lossesb.append(self.loss_fn(labelb, output))  # 先把task_lossb都放到列表中 最后再来更新meta_learner的梯度
                task_lossesb+= self.loss_fn(labelb, output)*self.LW[i+1]  #1~4
                # print('task_lossesb',task_lossesb[-1].requires_grad) TRUE
                #task_lossesb 中总共存放了 TASK_ITER个loss,每个Loss都是一个Meta batch的数据的Loss之和
                # print(task_lossesb[-1]) #看看是不是对batchsize=8个数据求了均值后的数据，是的，是一个数
            # print("len(task_lossesb)",len(task_lossesb))  #=5
            task_lossesb/=self.TASK_ITER
            
            task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

            return task_output

        self.total_lossa = 0.
        self.total_lossesb = 0.
        inputa, labela, inputb, labelb = inf
        for i in range(self.META_BATCH_SIZE):  #5
            res = task_meta_learning(inputa[i], labela[i], inputb[i], labelb[i])  # inputa[i] (8, 64, 64, 3)
            # print(res[2])
            self.total_lossa += res[2]  #先求和，然后再除以meta batch size
            # self.total_lossesb.append(res[3][-1])  #5个iter loss的均值
            self.total_lossesb+= res[3] #5个iter loss的均值
            # self.total_lossesb.append(sum(res[3]) / self.TASK_ITER)  #5个iter loss的均值
        self.total_lossa /= self.META_BATCH_SIZE  # 训练的Loss

        # print("len(self.total_lossesb)", len(self.total_lossesb))  #理论上等于META_BATCH_SIZE
        # self.total_lossesb=torch.tensor(self.total_lossesb,requires_grad=True)  #5,5 torch.tensor 默认为false
        # self.total_lossesb=torch.mean(self.total_lossesb,dim=0)  #相当于除以META_BATCH_SIZE  torch.Size([5])
        self.total_lossesb/= self.META_BATCH_SIZE#相当于除以META_BATCH_SIZE  torch.Size([5])

        print(self.total_lossesb)



        # print(self.weighted_total_lossesb.requires_grad)  #为啥还是false????
        # self.weighted_total_lossesb.requires_grad=True

        # print(self.weighted_total_lossesb.requires_grad)
        self.opt.zero_grad()
        self.total_lossesb.backward()
        self.opt.step()  # 该优化器的学习率是meta lr =1e-4 β
        # print(self.model.module.parameters()[15])
        # print(self.model.module.vars[15])

    def get_loss_weights(self):
        
        loss_weights = torch.ones(size=[self.TASK_ITER]) * (1.0 / self.TASK_ITER)
        decay_rate = 1.0 /  self.TASK_ITER / (10000 / 3)
        min_value = 0.03 /  self.TASK_ITER

        loss_weights_pre = torch.maximum(
            loss_weights[:-1] - (torch.multiply(torch.tensor(self.global_step), torch.tensor(decay_rate))),
            torch.tensor(min_value))

        loss_weight_cur = torch.minimum(
            loss_weights[-1] + (torch.multiply(torch.tensor(self.global_step), torch.tensor((self.TASK_ITER - 1) * decay_rate))),
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
        while True:
            inf = dataset.make_data_tensor(self.scale_list)  # 采样任务Batch

            self.construct_model(inf)

            step += 1
            self.global_step+=1

            if step % PRINT_ITER == 0:
                t1 = t2
                t2 = time.time()
                # self.logger.info('Iter: %d - Pre loss %.3f - Post weighted loss: %.3f - Post loss: %.3f - Time: %.2f'%(step,self.total_lossa,self.weighted_total_lossesb,self.total_lossesb,t2 - t1))
                self.logger.info('Iter: %d - Pre loss %.3f - Post loss: %.3f - Time: %.2f'%(step,self.total_lossa,self.total_lossesb,t2 - t1))


            if step % SAVE_ITET == 0:
                save_path=os.path.join(self.work_dir,'checkpoint')
                save(self.model, save_path, step)  #trial=0  step=2500
                self.logger.info('Save model to %s' % (save_path))

            if step == self.META_ITER:
                self.logger.info('Done Training - Time: %s'%(strftime('%b-%d %H:%M:%S', localtime())))
                break



def save(model, checkpoint_dir,step):
    model_name = 'model_{}.pth'.format(str(step)) #model_2500.pth
    mkdir_or_exist(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, model_name))  #checkpoint/Model0/model_2500.pth

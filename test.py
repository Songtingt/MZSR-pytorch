import model
import torch
import time
import imageio
from utils import *
import numpy as np

class Test(object):
    def __init__(self, model_path, save_path,kernel, scale, method_num, num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        self.max_iters=num_of_adaptation   #1 or 10
        self.display_iter = 1

        self.upscale_method= 'cubic'
        self.noise_level = 0.0

        self.back_projection=False
        self.back_projection_iters=4

        self.model_path=model_path
        self.save_path=save_path
        self.method_num=method_num #默认为0

        self.ds_method=methods[self.method_num]

        self.kernel = kernel
        self.scale=scale  #2.0
        self.scale_factors = [self.scale, self.scale]


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.Net()  #还要加载参数！！

        self.learning_rate = 0.02
        self.loss_fn = torch.nn.L1Loss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate) #和tf的tf.train.GradientDescentOptimizer 对应


        if not os.path.exists('%s/%02d' % (self.save_path, self.max_iters)):
            os.makedirs('%s/%02d' % (self.save_path, self.max_iters))
        self.log_file = open(get_path('%s/%02d' % (self.save_path, self.max_iters), 'log.txt'), 'a+')
        self.log_file.write('************************************************************************************\n')

    def initialize(self):
        checkpoint = torch.load(self.model_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        self.model.load_state_dict(new_state_dict)  # 每次都加载模型？？还是放在init里面好一些
        print('load network')
        self.log_file.write('load from {}\n'.format(self.model_path))
        self.model = self.model.to(self.device)
        self.loss = [None] * self.max_iters  # 记录下每次迭代的Loss
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.psnr = []  # 每次都要重新初始化，如果放在类的初始化中，就会叠加结果
        self.iter = 0

    def __call__(self, img, gt, img_name):  #每次送入一张图片
        self.img = img
        self.gt = modcrop(gt, self.scale)  #使得gt与 img上采样scale后的图像尺寸一致
        # print(self.img.shape, self.gt.shape)
        
        self.img_name = img_name
        print('** Start Adaptation for X', self.scale, os.path.basename(self.img_name), ' **')
        self.log_file.write('** Start Adaptation for X {}  {}  **\n'.format(self.scale, os.path.basename(self.img_name)))
        self.initialize()


        self.sf = np.array(self.scale_factors)
        self.output_shape = np.uint(np.ceil(np.array(self.img.shape[0:2]) * self.scale))

        # Train the network
        self.quick_test()  # forward一次


        print('[*] Baseline ')
        self.log_file.write('[*] Baseline \n')
        self.train()

        post_processed_output = self.final_test()
        if self.save_results:
            if not os.path.exists('%s/%02d' % (self.save_path, self.max_iters)):
                os.makedirs('%s/%02d' % (self.save_path, self.max_iters))

            imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4]),
                                  post_processed_output)


        print('** Done Adaptation for X', self.scale, os.path.basename(self.img_name),', PSNR: %.4f' % self.psnr[-1], ' **')
        self.log_file.write('** Done Adaptation for X {}  {}  PSNR: {:.4f} **\n'.format(self.scale,os.path.basename(self.img_name),self.psnr[-1]))
        print('')

        return post_processed_output, self.psnr



    def train(self):
        self.hr_father = self.img
        self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method) #下采样  先利用Kernel对 lr图像做下采样
        self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.) #加噪声



        t1=time.time()
        for self.iter in range(self.max_iters):
            if self.method_num == 0:  # 看一下论文 ，这里是什么鬼东西。。。
                '''direct'''
                if self.iter == 0:
                    self.learning_rate = 2e-2
                elif self.iter < 4:
                    self.learning_rate = 1e-2
                else:
                    self.learning_rate = 5e-3

            elif self.method_num == 1:
                '''Multi-scale'''
                if self.iter < 3:
                    self.learning_rate = 1e-2
                else:
                    self.learning_rate = 5e-3

            elif self.method_num == 2:
                '''bicubic'''
                if self.iter == 0:
                    self.learning_rate = 0.01
                elif self.iter < 3:
                    self.learning_rate = 0.01
                else:
                    self.learning_rate = 0.001

            elif self.method_num == 3:
                ''''scale 4'''
                if self.iter == 0:
                    self.learning_rate = 1e-2
                elif self.iter < 5:
                    self.learning_rate = 5e-3
                else:
                    self.learning_rate = 1e-3

            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father)
            # Display information
            if self.iter % self.display_iter == 0:
                print('Scale: ', self.scale, ', iteration: ', (self.iter + 1), ', loss: ', self.loss[self.iter].cpu().item())
                self.log_file.write(
                    "Scale: {}- iteration: {} -  loss: {:.3f} \n".format(self.scale, (self.iter + 1),
                                                                                            self.loss[self.iter].cpu().item()))

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))

    def quick_test(self):
        # 1. True MSE
        self.sr = self.forward_pass(self.img, self.gt.shape)

        self.mse = self.mse + [np.mean((self.gt - self.sr)**2)]

        '''Shave'''
        scale = int(self.scale)
        PSNR = psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])
        self.psnr.append(PSNR)
        
        # 2. Reconstruction MSE 这是啥？？？ 对lr图像下采样，然后 通过网络输出预测的lr图像
        self.reconstruct_output = self.forward_pass(self.hr2lr(self.img), self.img.shape)
        self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))  #维度要一致

        processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)

        print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)
        self.log_file.write("iteration: {} - recon mse: {:.3f} - true mse: {:.3f}  - PSNR: {:.3f}\n".format(self.iter,self.mse_rec[-1],(self.mse[-1] if self.mse else None),PSNR))
        return processed_output

    def forward_pass(self, input, output_shape=None): #只输出结果，不回传梯度
        ILR = imresize(input, self.scale, output_shape, self.upscale_method) #输入必须是numpy格式  scale*h,scale*w,c
        ILR = torch.as_tensor(ILR).type(torch.FloatTensor).to(self.device).permute(2,0,1)  # c,scale*h,scale*w
        ILR= ILR[None, :, :, :]  #1,3,512,512

        
        output = self.model(ILR).cpu().detach().numpy() #1,3,512,512
        output=np.squeeze(output,0).transpose(1,2,0) #512,512,3
        # print(output.shape)

        return np.clip(output, 0., 1.)

    def forward_backward_pass(self, input, hr_father):
        self.model.train()
        ILR = imresize(input, self.scale, hr_father.shape, self.upscale_method)  #先对LR上采样

        HR = torch.tensor(hr_father).type(torch.FloatTensor).to(self.device).permute(2,0,1)[None,:,:,:]
        ILR = torch.as_tensor(ILR).type(torch.FloatTensor).to(self.device).permute(2, 0, 1)  # c,scale*h,scale*w
        ILR = ILR[None, :, :, :]  # 1,3,512,512

        train_output = self.model(ILR)
        self.loss[self.iter] = self.loss_fn(train_output, HR)
        
        self.opt.zero_grad()
        self.loss[self.iter].backward()
        self.opt.step()

        train_output = train_output.cpu().detach().numpy()  # 1,3,512,512
        train_output = np.squeeze(train_output, 0).transpose(1, 2, 0)  # 512,512,3
        # print(train_output.shape)

        return np.clip(train_output, 0., 1.)
    
    def hr2lr(self, hr):  #scale=2.0
        lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
        return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)

    def final_test(self):
    
        output = self.forward_pass(self.img, self.gt.shape)
        if self.back_projection == True:
            for bp_iter in range(self.back_projection_iters):
                output = back_projection(output, self.img, down_kernel=self.kernel,
                                                  up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)

        processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)

        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(processed_output)[scale:-scale, scale:-scale])

        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
        #           rgb2y(processed_output))

        self.psnr.append(PSNR)

        return processed_output

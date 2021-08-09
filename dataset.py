import os

import numpy as np
import random
from scipy.misc import imresize as resize

import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from gkernel import *
from imresize import imresize
from source_target_transforms import *
import imageio
from PIL import Image
from imresize import imresize
META_BATCH_SIZE = 5
TASK_BATCH_SIZE = 8

def random_crop(hr,size):
    h, w = hr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_hr = hr[y:y+size, x:x+size].copy()

    return crop_hr


def random_flip_and_rotate(im1):
    if random.random() < 0.5:
        im1 = np.flipud(im1)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)

    # have to copy before be called by transform function
    return im1.copy()


class preTrainDataset(data.Dataset):
    def __init__(self, hr_path, lr_path,patch_size=96, scale=2):  #96，2
        super(preTrainDataset, self).__init__()
        self.patch_size = patch_size
        self.hr_path=hr_path
        self.lr_path=lr_path

        # h5f = h5py.File(path, 'r')
        self.hr=[os.path.join(self.hr_path,v) for v in os.listdir(self.hr_path)]  #'/data/DIV2K/DIV2K_train_HR'
        # self.lr=os.listdir(lr_path)  #'/data/DIV2K/DIV2K_train_LR_bicubic'
        # self.hr = [v[:] for v in h5f["HR"].values()]
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v for v in os.listdir(os.path.join(lr_path,"X{}".format(i)))] for i in self.scale]
        else:
            self.scale = scale
            lr_root_path=os.path.join(self.lr_path, "X{}".format(self.scale))
            self.lr = [os.path.join(lr_root_path,v) for v in os.listdir(lr_root_path)]
        
        # h5f.close()

        self.transform = transforms.Compose([  #对两张图，做一样的数据增强
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(self.patch_size),
            ToTensor(),
            Normalize()
        ])
    def __getitem__(self, index):
        
        hr_img = Image.open(self.hr[index]) 
        # hr_img = hr_img / 255.
        
        lr_img = Image.open(self.lr[index]) 
        # lr_img = lr_img / 255.
        lr_img = np.asarray(lr_img)
        hr_img= np.asarray(hr_img)
        # print(type(lr_img))
        # lr_img=resize(lr_img, self.scale*100, interp='cubic')
        # print(lr_img.shape)
        item = [(hr_img, resize(lr_img, self.scale*100, interp='cubic'))] #先对lr做一个上采样，然后后面crop的时候就可以和HRcrop一样的大小
        # return [(self.transform(hr), self.transform(imresize(lr, 400, interp='cubic'))) for hr, lr in item]
        # for hr, lr in item:  #此时一个是
        #     print(type(hr))  #<class 'numpy.ndarray'>   
        #     print(type(lr))
        return [self.transform([hr,lr]) for hr, lr in item]

    def __len__(self):
        return len(self.hr)

class metaTrainDataset(data.Dataset):
    def __init__(self, path, patch_size=64, scale=4):
        super(metaTrainDataset, self).__init__()
        self.size = patch_size
        self.path=path
        # h5f = h5py.File(path, 'r')
        # self.hr = [v[:] for v in h5f["HR"].values()]
        # self.hr = random.sample(self.hr, TASK_BATCH_SIZE*2*META_BATCH_SIZE) #随机采样 80个
        # h5f.close()
        self.img_name = os.listdir(path)
        self.hr=random.sample(self.img_name, TASK_BATCH_SIZE*2*META_BATCH_SIZE) #随机采样 80个
        # print(len(img_name))
        # pass
        # self.tansform = transforms.Compose([
        #     transforms.RandomCrop(64),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip()
        #     # transforms.ToTensor()
        # ])

    def __getitem__(self, index):
        img_path=os.path.join(self.path,self.hr[index])
        # print(img_path)
        item = imageio.imread(img_path).astype(np.float32)
        item = [item/255.]
        item = [random_crop(hr,self.size) for hr in item]
        return [random_flip_and_rotate(hr) for hr in item]
    def __len__(self):
        return len(self.hr)

def make_data_tensor(scale, noise_std=0.0):
    label_train = metaTrainDataset('/data/DIV2K/DIV2K_train_HR')  #只需要hr的路径   data/DIV2K_train.h5
    input_meta = []
    label_meta = []

    for t in range(META_BATCH_SIZE): #5
        input_task = []
        label_task = []

        Kernel = generate_kernel(k1=scale*2.5, ksize=15)  #随机生成Kernel
        for idx in range(TASK_BATCH_SIZE*2): #16
            img_HR = label_train[t*TASK_BATCH_SIZE*2 + idx][-1]
            # add isotropic and anisotropic Gaussian kernels for the blur kernels 
            # and downsample 
            clean_img_LR = imresize(img_HR, scale=1./scale, kernel=Kernel)  #默认是direct x2 因此测试时选择direct x2
            # add noise
            img_LR = np.clip(clean_img_LR + np.random.randn(*clean_img_LR.shape)*noise_std, 0., 1.)
            # used cubic upsample 
            img_ILR = imresize(img_LR,scale=scale, output_shape=img_HR.shape, kernel='cubic')

            input_task.append(img_ILR) #16个
            label_task.append(img_HR)
        
        input_meta.append(np.asarray(input_task))
        label_meta.append(np.asarray(label_task))
    
    input_meta = np.asarray(input_meta) #(5, 16, 64, 64, 3)
    label_meta = np.asarray(label_meta) #(5, 16, 64, 64, 3)

    inputa = input_meta[:,:TASK_BATCH_SIZE,:,:] #(5, 8, 64, 64, 3)
    labela = label_meta[:,:TASK_BATCH_SIZE,:,:] #
    inputb = input_meta[:,TASK_BATCH_SIZE:,:,:] #(5, 8, 64, 64, 3)
    labelb = label_meta[:,TASK_BATCH_SIZE:,:,:]

    return inputa, labela, inputb, labelb

if __name__ == '__main__':
    inputa, labela, inputb, labelb=make_data_tensor(4)
    # print(inputa.shape)
    print(labela.shape)

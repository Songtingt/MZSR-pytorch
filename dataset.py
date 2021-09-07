import os

import numpy as np
import random
from scipy.misc import imresize as resize

import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from gkernel import *
from source_target_transforms import *
import imageio
from PIL import Image
from imresize import imresize
import utils
import glob
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
META_BATCH_SIZE = 5
TASK_BATCH_SIZE = 8


def random_crop(hr, size):
    h, w = hr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    crop_hr = hr[y:y + size, x:x + size].copy()

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


class preTrainDataset(Dataset):
    def __init__(self, dir_data,name, patch_size=96, scale=2):  # 96，2
        super(preTrainDataset, self).__init__()
        self.patch_size = patch_size
        self.no_augment=False
        self.name = name  # DIV2K
        self.scale=scale
        self.input_large = False
        self._set_filesystem(dir_data)

        path_bin = os.path.join(self.apath, 'bin')  # /data/DIV2K/bin
        list_hr, list_lr = self._scan()  # 得到图片名
        self.images_hr, self.images_lr = [], []
        for h in list_hr:
            b = h.replace(self.apath, path_bin)
            b = b.replace(self.ext[0], '.pt')
            self.images_hr.append(b)
            self._check_and_load('sep', h, b, verbose=True)
        for l in list_lr:
            b = l.replace(self.apath, path_bin)
            b = b.replace(self.ext[1], '.pt')
            self.images_lr.append(b)
            self._check_and_load('sep', l, b, verbose=True)
        
        n_patches = 16*1000
        n_images = len(self.images_hr)  #800
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = 1 #max(n_patches // n_images, 1)  # 16000/800=20 次



        
    def _set_filesystem(self, dir_data):  #../../../dataset
        self.apath = os.path.join(dir_data, self.name)  #/data/DIV2K
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')
        
    def _scan(self):
        names_hr = sorted( #读取所有图片名称
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )[0:800]
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        self.scale, filename, self.scale, self.ext[1]
                    )
                ))

        return names_hr, names_lr
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:  #verbose 冗长的
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)
        
    def __getitem__(self, index):
        lr, hr, filename = self._load_file(index)
        pair = self.get_patch(lr, hr)  # 得到一对patch
        # print(self.hr[index], self.lr[index])
        # print(lr_img.shape, hr_img.shape)  #(48,48,3)  <class 'imageio.core.util.Array'>
        # print(pair[0].shape, pair[1].shape)  #(48,48,3)  <class 'imageio.core.util.Array'>
        # print(type(pair[0]), type(pair[1]))

        pair =utils.set_channel(*pair, n_channels=3) #如果通道为3就啥也不做
        pair[0]=imresize(pair[0],scale=self.scale,kernel='cubic') #先上采样
        pair_t = utils.np2Tensor(*pair, rgb_range=255)  # 转化成tensor
        # print(pair[0].shape, pair[1].shape)
        
        # item = [(hr_img, resize(lr_img, self.scale * 100, interp='cubic'))]  # 先对lr做一个上采样，然后后面crop的时候就可以和HRcrop一样的大小
        return pair_t[0]/255., pair_t[1]/255.
    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))

        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f)
        with open(f_lr, 'rb') as _f:
            lr = pickle.load(_f)
        return lr, hr, filename

    def get_patch(self, lr, hr):
        lr, hr = utils.get_patch(
                lr, hr,
                patch_size=self.patch_size,
                scale=self.scale,
                multi=False,
                input_large=False
            )
        if not self.no_augment: lr, hr = utils.augment(lr, hr)
        return lr, hr

    def __len__(self):
        return len(self.images_hr) * self.repeat  # 800*20=16000


class preTrainDataset_old(Dataset):
    def __init__(self, hr_path, lr_path, patch_size=96, scale=2):  # 96，2
        super(preTrainDataset_old, self).__init__()
        self.patch_size = patch_size
        self.hr_path = hr_path
        self.lr_path = lr_path

        self.hr = sorted( #读取所有图片名称
            glob.glob(os.path.join(self.hr_path, '*.png')))  # '/data/DIV2K/DIV2K_train_HR'
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v for v in os.listdir(os.path.join(lr_path, "X{}".format(i)))] for i in self.scale]
        else:
            self.scale = scale
            self.lr = []
            for f in self.hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                self.lr.append(os.path.join(
                    self.lr_path, 'X{}/{}x{}{}'.format(
                        self.scale, filename, self.scale, '.png'
                    )
                ))
        '''
        self.hr = [os.path.join(self.hr_path, v) for v in os.listdir(self.hr_path)]  # '/data/DIV2K/DIV2K_train_HR'
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v for v in os.listdir(os.path.join(lr_path, "X{}".format(i)))] for i in self.scale]
        else:
            self.scale = scale
            lr_root_path = os.path.join(self.lr_path, "X{}".format(self.scale))
            self.lr = [os.path.join(lr_root_path, v) for v in os.listdir(lr_root_path)]'''


        self.transform = transforms.Compose([  # 对两张图，做一样的数据增强
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(self.patch_size),
            ToTensor(),
            Normalize()
        ])


    def __getitem__(self, index):
        # print(self.hr[index],self.lr[index])
        hr_img = Image.open(self.hr[index])
        lr_img = Image.open(self.lr[index])
        lr_img = np.asarray(lr_img)
        hr_img = np.asarray(hr_img)
        # print(type(lr_img))
        # lr_img=resize(lr_img, self.scale*100, interp='cubic')
        # print(lr_img.shape)
        item = [(hr_img, resize(lr_img, self.scale * 100, interp='cubic'))]  # 先对lr做一个上采样，然后后面crop的时候就可以和HRcrop一样的大小
        # return [(self.transform(hr), self.transform(imresize(lr, 400, interp='cubic'))) for hr, lr in item]
        # for hr, lr in item:  #此时一个是
        #     print(type(hr))  #<class 'numpy.ndarray'>
        #     print(type(lr))
        return [self.transform([hr, lr]) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


class metaTrainDataset(Dataset):
    def __init__(self, dir_data,name, scale_list,patch_size=64):
        super(metaTrainDataset, self).__init__()
        self.size = patch_size
        self.scale_list=scale_list
        self.name=name

        # self.hr = sorted(glob.glob(os.path.join(self.path, '*.png')))[0:800]  #完整路径
        self._set_filesystem(dir_data)

        path_bin = os.path.join(self.apath, 'bin')  # /data/DIV2K/bin
        list_hr=sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))  # 得到图片名 [0:800]
        self.images_hr= []
        for h in list_hr:
            b = h.replace(self.apath, path_bin)
            b = b.replace(self.ext[0], '.pt')
            self.images_hr.append(b)
            self._check_and_load('sep', h, b, verbose=True)
        n_patches = 16 * 1000
        n_images = len(self.images_hr)  # 800
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = 1 #max(n_patches // n_images, 1)  # 16000/800=20 次
        self.transform=transforms.Compose([
                transforms.ToTensor()]
            )


    def _set_filesystem(self, dir_data):  #/data
        self.apath = os.path.join(dir_data, self.name)  #/data/DIV2K
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.ext = ('.png', '.png')
    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f) #读取

        return hr, filename
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:  #verbose 冗长的
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)  #把图片保存到 _f中，也就是 .pt文件


    def __getitem__(self, index):
        hr,  filename = self._load_file(index)  #要先crop?

        hr=random_crop(hr/255., self.size)  #64,64,3 忘了除以255，loss就爆炸了？
        hr=random_flip_and_rotate(hr)   #64,64,3
        hr=np.ascontiguousarray(hr.transpose((2, 0, 1)))
        hr = torch.from_numpy(hr).float()

        return hr

    def __len__(self):
        return len(self.images_hr) * self.repeat  # 800*20=16000

class metaTrainDataset_old(Dataset):
    def __init__(self, path, patch_size=64, scale=2):
        super(metaTrainDataset_old, self).__init__()
        self.size = patch_size
        self.path = path  #'/data/DIV2K/DIV2K_train_HR'
        self.img_name = os.listdir(path)
        self.hr = random.sample(self.img_name, TASK_BATCH_SIZE * 2 * META_BATCH_SIZE)  # 随机采样 80个

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.hr[index])
        # print(img_path)
        item = imageio.imread(img_path).astype(np.float32)
        item = [item / 255.]
        item = [random_crop(hr, self.size) for hr in item]
        return [random_flip_and_rotate(hr) for hr in item]

    def __len__(self):
        return len(self.hr)

'''
产生Lr和hr
lr用作Dtr
hr用作Dte
'''
class metaTrainDataset_lrhr(Dataset):
    def __init__(self, dir_data,name, scale_list,patch_size=64):
        super(metaTrainDataset_lrhr, self).__init__()
        self.size = patch_size
        self.scale_list=scale_list
        self.scale=int(self.scale_list[0])
        self.name=name

        # self.hr = sorted(glob.glob(os.path.join(self.path, '*.png')))[0:800]  #完整路径
        self._set_filesystem(dir_data)

        path_bin = os.path.join(self.apath, 'bin')  # /data/DIV2K/bin
        list_hr, list_lr = self._scan()  # 得到图片名
        self.images_hr, self.images_lr = [], []
        for h in list_hr:
            b = h.replace(self.apath, path_bin)
            b = b.replace(self.ext[0], '.pt')
            self.images_hr.append(b)
            self._check_and_load('sep', h, b, verbose=True)

        for l in list_lr:
            b = l.replace(self.apath, path_bin)
            b = b.replace(self.ext[1], '.pt')
            self.images_lr.append(b)
            self._check_and_load('sep', l, b, verbose=True)

        n_patches = 16 * 1000
        n_images = len(self.images_hr)  # 800
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = 1 #max(n_patches // n_images, 1)  # 16000/800=20 次
        self.transform=transforms.Compose([
                transforms.ToTensor()]
            )


    def _set_filesystem(self, dir_data):  #/data
        self.apath = os.path.join(dir_data, self.name)  #/data/DIV2K
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = ('.png', '.png')

    def _scan(self):
        names_hr = sorted( #读取所有图片名称
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )[0:800]
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        self.scale, filename, self.scale, self.ext[1]
                    )
                ))

        return names_hr, names_lr
    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f) #读取
        with open(f_lr, 'rb') as _f:
            lr = pickle.load(_f)

        return lr, hr, filename
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:  #verbose 冗长的
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)  #把图片保存到 _f中，也就是 .pt文件


    '''
    因为lr和hr用作不同的任务，所以不需要一一对应
    '''
    def __getitem__(self, index):
        lr, hr, filename = self._load_file(index)

        hr=random_crop(hr/255., self.size)  #64,64,3 忘了除以255，loss就爆炸了？
        hr=random_flip_and_rotate(hr)   #64,64,3
        hr=np.ascontiguousarray(hr.transpose((2, 0, 1)))
        hr = torch.from_numpy(hr).float()

        lr = random_crop(lr / 255., self.size)  # lr的裁剪尺寸和hr可以不同吗？
        lr = random_flip_and_rotate(lr)  # 64,64,3
        lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        lr = torch.from_numpy(lr).float()

        return lr,hr

    def __len__(self):
        return len(self.images_hr) * self.repeat  # 800*20=16000

def make_data_tensor(scale_list, noise_std=0.0):
    # print('scale=',scale_list)
    label_train = metaTrainDataset_old('/data/DIV2K/DIV2K_train_HR')  # 只需要hr的路径   data/DIV2K_train.h5 这里每次都会重新读取数据，是不是会导致gpu利用率降低
    input_meta = []
    label_meta = []
    #每次构造80张 
    for t in range(META_BATCH_SIZE):  # 5 1个meta batch 为一个任务 所以每次5个任务
        input_task = []
        label_task = []
        scale = np.random.choice(scale_list, 1)[0]  #2
        Kernel = generate_kernel(k1=scale * 2.5, ksize=15)  # 随机生成Kernel
        for idx in range(TASK_BATCH_SIZE * 2):  # 16  task batch size使用的都是基于同一个kernel
            img_HR = label_train[t * TASK_BATCH_SIZE * 2 + idx][-1]
            # add isotropic and anisotropic Gaussian kernels for the blur kernels 
            # and downsample 
            clean_img_LR = imresize(img_HR, scale=1. / scale, kernel=Kernel)  # 默认是direct x2 因此测试时选择direct x2
            # add noise
            img_LR = np.clip(clean_img_LR + np.random.randn(*clean_img_LR.shape) * noise_std, 0., 1.)  #加了噪声的
            # used cubic upsample 
            img_ILR = imresize(img_LR, scale=scale, output_shape=img_HR.shape, kernel='cubic')

            input_task.append(img_ILR)  # 16个
            label_task.append(img_HR)

        input_meta.append(np.asarray(input_task)) #5个  (16, 64, 64, 3)
        label_meta.append(np.asarray(label_task)) #5个  (16, 64, 64, 3)

    input_meta = np.asarray(input_meta)  # (5, 16, 64, 64, 3)
    label_meta = np.asarray(label_meta)  # (5, 16, 64, 64, 3)

    inputa = input_meta[:, :TASK_BATCH_SIZE, :, :]  # (5, 8, 64, 64, 3)
    labela = label_meta[:, :TASK_BATCH_SIZE, :, :]  #
    inputb = input_meta[:, TASK_BATCH_SIZE:, :, :]  # (5, 8, 64, 64, 3)
    labelb = label_meta[:, TASK_BATCH_SIZE:, :, :]

    return inputa, labela, inputb, labelb


if __name__ == '__main__':
    ds=metaTrainDataset_bs80('/data/DIV2K/DIV2K_train_HR')
    dataloader=DataLoader(ds, batch_size=80, shuffle=True,
                                pin_memory=True)  # bs=80
    for hr in dataloader:
        print(hr.shape)  #80,3,64,64
        for i in hr.shape[0]:
            scale = np.random.choice(scale_list, 1)[0]  # 2
            Kernel = generate_kernel(k1=scale * 2.5, ksize=15)  # 随机生成Kernel
            img_HR=hr[i] #3,64,64
            clean_img_LR = imresize(img_HR, scale=1. / scale, kernel=Kernel)  # 默认是direct x2 因此测试时选择direct x2

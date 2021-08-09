import dataset
import train
from config import *
from utils import *
import glob
import scipy.io
import os
import test


def main():
    if args.is_train == True:

        Trainer = train.Train(args, size=[HEIGHT, WIDTH, CHANNEL],
                              scale_list=SCALE_LIST, meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR,
                              meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER, checkpoint_dir=CHECKPOINT_DIR,
                              loadfrom_dir=LOAD_FROM)

        Trainer()
    else:
        img_path = sorted(glob.glob(os.path.join(args.inputpath, '*.png')))  # 读取所有的输入图像  256x256
        gt_path = sorted(glob.glob(os.path.join(args.gtpath, '*.png')))  # 读取所有的gt 512x512

        scale = 2.0  # 根据输入图像的尺寸来修改 scale

        try:
            kernel = scipy.io.loadmat(args.kernelpath)['kernel']  # 读取kernel
        except:
            kernel = 'cubic'

        Tester = test.Test(LOAD_FROM_meta, args.savepath, kernel, scale, args.model, args.num_of_adaptation)
        P = []
        for i in range(len(img_path)):  # 遍历完所有图片
            img = imread(img_path[i])
            gt = imread(gt_path[i])
            _, pp = Tester(img, gt, img_path[i])  # 返回PSNR pp是当前图片的 初始psnr 以及中间训练过程的Psnr
            P.append(pp)
        avg_PSNR = np.mean(P, 0)

        print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))
        Tester.log_file.write("[*] Average PSNR ** Initial: {:.4f}, Final :{:.4f}\n".format(avg_PSNR[0],avg_PSNR[1]))


if __name__ == '__main__':
    main()

import imageio
import os
import numpy as np
import re
import math
from imresize import imresize
import logging
logger_initialized = {}
from time import strftime, localtime
import os.path as osp
import random
import torch
def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]
def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2] #得到lr的高和宽

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size  #大图的patch尺寸要比小图大scale倍
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy #大图的采样点位置也要在小图裁剪的位置上x2.
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :], #裁剪小图
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]] #裁剪大图
    ]

    return ret
def get_path(root_path,subdir):
    return os.path.join(root_path, subdir)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.
    The logger will be initialized if it has not been initialized.
    By default a StreamHandler will be added.
    If `log_file` is specified, a FileHandler will also be added.
    The name of the root logger is the top-level package name, e.g., "edit".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        logging.Logger: The root logger.
    """
    # root logger name: mmedit
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger
def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger


    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    # fix stream twice bug
    # while logger.handlers:
    #     logger.handlers.pop()
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    rank = 0

    # only rank 0 will add a FileHandler

    file_handler = logging.FileHandler(log_file, 'w')
    handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger
def imread(path):
    img=imageio.imread(path).astype(np.float32)
    img=img/255.
    return img


def save(saver, sess, checkpoint_dir, trial, step):
    model_name='model'
    checkpoint = os.path.join(checkpoint_dir, 'Model%d'% trial)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    saver.save(sess, os.path.join(checkpoint, model_name), global_step=step)

def count_param(scope=None):
    N=np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print('Model Params: %d K' % (N/1000))

def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def print_time():
    print('Time: ', strftime('%b-%d %H:%M:%S', localtime()))

''' color conversion '''
def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255

    return y


def modcrop(imgs, modulo):  #modulo=2 scale
    sz=imgs.shape
    sz=np.asarray(sz)

    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:int(sz[0]), 0:int(sz[1])]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:int(szt[0]), 0:int(szt[1]),:]

    return out

def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None, ds_method='direct'):
    y_sr += imresize(y_lr - imresize(y_sr, scale=1.0/sf, output_shape=y_lr.shape, kernel=down_kernel, ds_method=ds_method),
                     scale=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)

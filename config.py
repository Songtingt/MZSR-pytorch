from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--n_feats', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--work_dir', type=str, default='./experiments_m',
                    help='the directory to save the results of experiments')

# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default='TestSet/Set5/g13/LR/')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='TestSet/Set5/GT_crop/')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='TestSet/Set5/g13/kernel.mat')
parser.add_argument('--savepath', type=str, dest='savepath', default='results/Set5')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)

# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--train', dest='is_train', default=False, action='store_true')

# for loss
parser.add_argument('--use_weighted_loss', dest='use_weighted_loss', default=False, action='store_true')

args= parser.parse_args()
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
#Transfer Learning From Pre-trained model.
IS_TRANSFER = True
TRANS_MODEL = 'Pretrained/Pretrained'

# Dataset Options
HEIGHT=64
WIDTH=64
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[2.0]

META_ITER=100000
META_BATCH_SIZE=5
META_LR=1e-4

TASK_ITER=5
TASK_BATCH_SIZE=8
TASK_LR=1e-2

# Loading tfrecord and saving paths
CHECKPOINT_DIR='SR'
LOAD_FROM= './checkpoint1/model_latest.pth' #'/home/songtingting02/stt/EDSR-PyTorch-master/experiment/edsr_x2/model/model_best.pt'
LOAD_FROM_meta='./experiments_m_weighted/20210806_162412/checkpoint/model_15000.pth'  #  ./experiments_m/20210806_102824/checkpoint/model_2000.pth
#./experiments_m_weighted/20210806_162412/checkpoint/model_15000.pth

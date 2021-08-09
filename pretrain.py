import os

import dataset
import torch
import model
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train():
    learning_rate = 4e-4
    net = model.Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=net.to(device)
    # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    '''data'''
    hr_path = '/data/DIV2K/DIV2K_train_HR'
    lr_path = '/data/DIV2K/DIV2K_train_LR_bicubic'
    predataset = dataset.preTrainDataset(hr_path,lr_path)
    dataloader = DataLoader(predataset,
                           batch_size=32,
                           num_workers=6,
                           shuffle=True,
                           drop_last=True)
    

    step = 0
    with tqdm.tqdm(total=100000, miniters=1, mininterval=0) as progress:
        while True:
            for inputs in dataloader:
                net.train()
                hr, lr = inputs[-1][0], inputs[-1][1]
                # print(hr.shape)  #64,3,64,64 32,3,96,96
                # print(lr.shape)  #64,3,64,64

                hr = hr.to(device)
                lr = lr.to(device)
                # print(hr.device)
                # print(lr.device)

                out = net(lr)
                loss = loss_fn(hr, out)

                progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                                         iter=step, loss=loss.item(), lr=learning_rate))

                progress.update()

                if step > 0 and step % 30000 == 0:
                    learning_rate = learning_rate / 10
                    adjust_learning_rate(optimizer, new_lr=learning_rate)
                    print("Learning rate reduced to {lr}".format(lr=learning_rate))
                if step > 0 and step % 100 == 0:
                    save_path = os.path.join('./checkpoint_96', 'model_latest.pth')
                    torch.save(net.state_dict(), save_path)
                    print("Model is saved !")

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                optimizer.step()

                step += 1

            if step > 100000:
                print('Done training.')
                break

    save_path = os.path.join('./checkpoint_96','Pretrain.pth')
    torch.save(net.state_dict(), save_path)
    print("Model is saved !")
    
    
def save(model, checkpoint_dir, trial, step):
    model_name = 'model_{}.pth'.format(str(step))
    checkpoint = os.path.join(checkpoint_dir, 'Model%d' % trial)
    if not os.path.join(checkpoint):
        os.makedirs(checkpoint)
    torch.save(model.state_dict(), os.path.join(checkpoint, model_name))
if __name__ == '__main__':
    train()





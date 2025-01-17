# -*- coding: utf-8 -*-
# create time:2022/6/25
# author:Wx
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import model2
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import NH_HazeDataset
from loss import CustomLoss_function

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=16)
parser.add_argument("-c", "--cropsize", type=int, default=60)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "./checkpoints6/SDNet4"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
#1-2-4WU
CROP_SIZE = args.cropsize


def save_dehaze_images(images, iteration, epoch):
    filename = METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_dehazed.png"
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(0.5 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    print("init data folders")

    encoder = {}
    decoder = {}
    encoder_optim = {}
    decoder_optim = {}
    encoder_scheduler = {}
    decoder_scheduler = {}
    for s in ['s1', 's2', 's3', 's4']:
        encoder[s] = {}
        decoder[s] = {}
        encoder_optim[s] = {}
        decoder_optim[s] = {}
        encoder_scheduler[s] = {}
        decoder_scheduler[s] = {}
        for lv in ['lv1', 'lv2', 'lv3']:
            encoder[s][lv] = model2.Encoder()
            decoder[s][lv] = model2.Decoder()
            encoder[s][lv].apply(weight_init).cuda(GPU)
            decoder[s][lv].apply(weight_init).cuda(GPU)
            encoder_optim[s][lv] = torch.optim.Adam(encoder[s][lv].parameters(), lr=LEARNING_RATE)
            encoder_scheduler[s][lv] = StepLR(encoder_optim[s][lv], step_size=1000, gamma=0.1)
            decoder_optim[s][lv] = torch.optim.Adam(decoder[s][lv].parameters(), lr=LEARNING_RATE)
            decoder_scheduler[s][lv] = StepLR(decoder_optim[s][lv], step_size=1000, gamma=0.1)
            if os.path.exists(str(METHOD + "/encoder_" + s + "_" + lv + ".pkl")):
                encoder[s][lv].load_state_dict(
                    torch.load(str(METHOD + "/encoder_" + s + "_" + lv + ".pkl")))
                print("load encoder_" + s + "_" + lv + " successfully!")
            if os.path.exists(str(METHOD + "/decoder_" + s + "_" + lv + ".pkl")):
                decoder[s][lv].load_state_dict(
                    torch.load(str(METHOD + "/decoder_" + s + "_" + lv + ".pkl")))
                print("load decoder_" + s + "_" + lv + " successfully!")

    if os.path.exists(METHOD) == False:
        os.makedirs(METHOD)

    for epoch in range(args.start_epoch, EPOCHS):
        for s in ['s1', 's2', 's3', 's4']:
            for lv in ['lv1', 'lv2', 'lv3']:
                encoder_scheduler[s][lv].step(epoch)
                decoder_scheduler[s][lv].step(epoch)

        print("Training...")

        train_dataset = NH_HazeDataset(
            hazed_image_files='new_dataset3/train_patch_hazy.txt',
            # make changes here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dehazed_image_files='new_dataset3/train_patch_gt.txt',
            root_dir='new_dataset3/',
            crop=True,
            crop_size=CROP_SIZE,
            rotation=True,
            color_augment=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        start = 0
        for iteration, inputs in enumerate(train_dataloader):
            #mse = nn.MSELoss().cuda(GPU)
            custom_loss_fn = CustomLoss_function().cuda(GPU)

            images = {}
            feature = {}
            residual = {}
            for s in ['s1', 's2', 's3', 's4']:
                feature[s] = {} # encoder output
                residual[s] = {} # decoder output

            images['gt'] = Variable(inputs['dehazed_image'] - 0.5).cuda(GPU)
            images['lv1'] = Variable(inputs['hazed_image'] - 0.5).cuda(GPU)
            H = images['lv1'].size(2)
            W = images['lv1'].size(3)

            images['lv2_1'] = images['lv1'][:, :, 0:int(H / 2), :]
            images['lv2_2'] = images['lv1'][:, :, int(H / 2):H, :]
            images['lv3_1'] = images['lv2_1'][:, :, :, 0:int(W / 2)]
            images['lv3_2'] = images['lv2_1'][:, :, :, int(W / 2):W]
            images['lv3_3'] = images['lv2_2'][:, :, :, 0:int(W / 2)]
            images['lv3_4'] = images['lv2_2'][:, :, :, int(W / 2):W]

            s = 's1'
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1'])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2'])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3'])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4'])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3)
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3)
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top']) + \
                                    feature[s]['lv3_top']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot']) +  \
                                    feature[s]['lv3_bot']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1'] + residual[s]['lv2']) + feature[s]['lv2']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's2'
            ps = 's1'
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1']+residual[ps]['lv1'][:, :, 0:int(H / 2), 0:int(W / 2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2']+residual[ps]['lv1'][:, :, 0:int(H / 2), int(W / 2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3']+residual[ps]['lv1'][:, :, int(H / 2):H, 0:int(W / 2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4']+residual[ps]['lv1'][:, :, int(H / 2):H, int(W / 2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1']+residual[ps]['lv1'][:, :, 0:int(H / 2), :] + residual[s]['lv3_top']) + \
                                    feature[s]['lv3_top'] + \
                                    feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2']+residual[ps]['lv1'][:, :, int(H / 2):H, :] + residual[s]['lv3_bot']) + \
                                    feature[s]['lv3_bot'] + \
                                    feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1']+ residual[ps]['lv1'] + residual[s]['lv2']) + \
                                feature[s]['lv2'] + \
                                feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's3'
            ps = 's2'
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1']+residual[ps]['lv1'][:, :, 0:int(H / 2), 0:int(W / 2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2']+residual[ps]['lv1'][:, :, 0:int(H / 2), int(W / 2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3']+residual[ps]['lv1'][:, :, int(H / 2):H, 0:int(W / 2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4']+residual[ps]['lv1'][:, :, int(H / 2):H, int(W / 2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1']+residual[ps]['lv1'][:, :, 0:int(H / 2), :]+residual[s]['lv3_top']) + \
                                    feature[s]['lv3_top'] + \
                                    feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2']+residual[ps]['lv1'][:, :, int(H / 2):H, :]+residual[s]['lv3_bot']) + \
                                    feature[s]['lv3_bot'] + \
                                    feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])# + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1']+residual[ps]['lv1']+residual[s]['lv2']) +\
                                feature[s]['lv2'] + \
                                feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's4'
            ps = 's3'
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1']+residual[ps]['lv1'][:, :, 0:int(H / 2), 0:int(W / 2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2']+residual[ps]['lv1'][:, :, 0:int(H / 2), int(W / 2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3']+residual[ps]['lv1'][:, :, int(H / 2):H, 0:int(W / 2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4']+residual[ps]['lv1'][:, :, int(H / 2):H, int(W / 2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1']+
                residual[ps]['lv1'][:, :, 0:int(H / 2), :] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + \
                                  feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2']+
                residual[ps]['lv1'][:, :, int(H / 2):H, :] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + \
                                  feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1']+
                residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + \
                                feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            loss = custom_loss_fn(residual['s4']['lv1'], images['gt']) + \
                    custom_loss_fn(residual['s3']['lv1'], images['gt']) + \
                        custom_loss_fn(residual['s2']['lv1'], images['gt']) + \
                            custom_loss_fn(residual['s1']['lv1'], images['gt'])

            for s in ['s1', 's2', 's3', 's4']:
                for lv in ['lv1', 'lv2', 'lv3']:
                    encoder[s][lv].zero_grad()
                    decoder[s][lv].zero_grad()

            loss.backward()

            for s in ['s1', 's2', 's3', 's4']:
                for lv in ['lv1', 'lv2', 'lv3']:
                    encoder_optim[s][lv].step()
                    decoder_optim[s][lv].step()

            if (iteration + 1) % 10 == 0:
                stop = time.time()
                print(METHOD + "   epoch:", epoch, "iteration:", iteration + 1, "loss:", loss.item(), 'time:%.4f'%(stop-start))
                start = time.time()

        if (epoch) % 200 == 0:
            if os.path.exists(METHOD + '/epoch' + str(epoch)) == False:
                os.makedirs(METHOD + '/epoch' + str(epoch))

            for s in ['s1', 's2', 's3', 's4']:
                for lv in ['lv1', 'lv2', 'lv3']:
                    torch.save(encoder[s][lv].state_dict(), str(
                        METHOD + '/epoch' + str(epoch) + "/encoder_" + s + "_" + lv + ".pkl"))
                    torch.save(decoder[s][lv].state_dict(), str(
                        METHOD + '/epoch' + str(epoch) + "/decoder_" + s + "_" + lv + ".pkl"))


        for s in ['s1', 's2', 's3', 's4']:
            for lv in ['lv1', 'lv2', 'lv3']:
                torch.save(encoder[s][lv].state_dict(),
                           str(METHOD + "/encoder_" + s + "_" + lv + ".pkl"))
                torch.save(decoder[s][lv].state_dict(),
                           str(METHOD + "/decoder_" + s + "_" + lv + ".pkl"))


if __name__ == '__main__':
    main()







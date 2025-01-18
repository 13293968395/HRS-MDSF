from zhangUtils.train import *
from model import DMPHN
from loss import CustomLoss_function
from datasets import NH_HazeDataset
from torchvision import transforms
import torch
import argparse
from torch.optim.lr_scheduler import StepLR



parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=500)
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

model = DMPHN()
criterion = CustomLoss_function()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
train_dataset = NH_HazeDataset(
            hazed_image_files='/root/lanyun-tmp/NH-HAZE2/Train/train.txt',
            # make changes here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dehazed_image_files='/root/lanyun-tmp/NH-HAZE2/Train/train.txt',
            root_dir='/root/lanyun-tmp/NH-HAZE2/Train/',
            crop=True,
            crop_size=CROP_SIZE,
            rotation=True,
            color_augment=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataset = NH_HazeDataset(
            hazed_image_files='/root/lanyun-tmp/NH-HAZE2/Test/test.txt',
            dehazed_image_files='/root/lanyun-tmp/NH-HAZE2/Test/test.txt',
            root_dir='/root/lanyun-tmp/NH-HAZE2/Test/',
            crop=True,
            crop_size=CROP_SIZE,
            rotation=True,
            color_augment=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

train(model=model, criterion=criterion, optimizer=optimizer, trainloader=train_loader, 
      epochs=EPOCHS, testloader=test_loader, testEpoch=5, scheduler=scheduler)

#  1.12.1 py3.10_cuda11.3_cudnn8.3.2_0
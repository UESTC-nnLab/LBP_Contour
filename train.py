# -*- coding: utf-8 -*
import torch as t
from sklearn.metrics import confusion_matrix
from torch import optim
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from tensorboardX import SummaryWriter
import os
#import wx

from cellnn import mseLoss, mcellnn, DiceLoss, bceLoss, cellnn
from tools import myDataset, Config, show_plot, imshow, getFOV  # ,showFrame

writer = SummaryWriter()


def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


def mytest():
    folder_dataset_test = dataset.ImageFolder(root=Config.testing_dir)
    myDatasets_test = myDataset(imageFolderDataset=folder_dataset_test,
                                transform=transforms.Compose(
                                    [transforms.ToTensor()]),
                                target_transform=transforms.Compose([transforms.ToTensor()]))

    Data = DataLoader(myDatasets_test,
                      shuffle=False,
                      num_workers=0,
                      batch_size=1)
    dataiter = iter(Data)
    labels = []
    preds = []
    mean = 0
    for data in dataiter:
        img, label, mask = data
        label = label.to("cuda")
        img = img.to("cuda")
        mask = mask.to("cuda")
        output = net(img)
        loss = mycriterion(output, label, mask)
        mean += loss.item()
        labels.append(getFOV(label, mask))
        preds.append(getFOV(output, mask))
    labels = flatten(labels)
    preds = flatten(preds)
    y_pred = np.empty((len(preds)))
    for i in range(len(labels)):
        if preds[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(labels, y_pred)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] +
                         confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    print("dice"+str(mean/20))
    return accuracy


if __name__ == '__main__':
    gpus = [0]
    cuda_gpu = t.cuda.is_available()
    net = mcellnn()
    if (cuda_gpu):
        net = t.nn.DataParallel(net, device_ids=gpus).cuda()
    checkpoint = t.load("model_temp.ph")
    net.load_state_dict(checkpoint['net'], strict=False)
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    folder_dataset = dataset.ImageFolder(root=Config.training_dir)
    myDatasets = myDataset(imageFolderDataset=folder_dataset,
                           transform=transforms.Compose([
                               transforms.RandomChoice([
                                   transforms.RandomRotation((0, 0)),
                                   transforms.RandomHorizontalFlip(p=1),
                                   transforms.RandomVerticalFlip(p=1),
                                   transforms.RandomRotation((90, 90)),
                                   transforms.RandomRotation((180, 180)),
                                   transforms.RandomRotation((270, 270)),
                                   # transforms.CenterCrop(400),
                                   transforms.Compose([
                                       transforms.RandomHorizontalFlip(p=1),
                                       transforms.RandomRotation((90, 90)),
                                   ]),
                                   transforms.Compose([
                                       transforms.RandomHorizontalFlip(p=1),
                                       transforms.RandomRotation((270, 270)),
                                   ])
                               ]), transforms.ToTensor()]),
                           target_transform=transforms.Compose([
                               transforms.RandomChoice([
                                   transforms.RandomRotation((0, 0)),
                                   transforms.RandomHorizontalFlip(p=1),
                                   transforms.RandomVerticalFlip(p=1),
                                   transforms.RandomRotation((90, 90)),
                                   transforms.RandomRotation((180, 180)),
                                   transforms.RandomRotation((270, 270)),
                                   transforms.Compose([
                                       transforms.RandomHorizontalFlip(p=1),
                                       transforms.RandomRotation((90, 90)),
                                   ]),
                                   transforms.Compose([
                                       transforms.RandomHorizontalFlip(p=1),
                                       transforms.RandomRotation((270, 270)),
                                   ])
                               ]), transforms.ToTensor()]))

    trainData = DataLoader(myDatasets,
                           shuffle=True,
                           num_workers=5,
                           batch_size=Config.train_batch_size)

    vis_dataloader = DataLoader(myDatasets,  #
                                shuffle=True,
                                num_workers=2,
                                batch_size=5)

    mycriterion = mseLoss()
    counter = []
    loss_history = []
    iteration_number = 0
    highest = 0
    for epoch in range(0, Config.train_number_epochs):
        epoch_loss = []
        for i, data in enumerate(trainData, 0):
            img, label, mask = data
            label = label.to("cuda")
            img = img.to("cuda")
            mask = mask.to("cuda")
            optimizer.zero_grad()
            output = net(img)
            loss = mycriterion(output, label, mask)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(np.mean(np.array(epoch_loss)))
        scheduler.step(np.mean(np.array(epoch_loss)))
        if epoch % 1 == 0:
            print("Epoch number {}\n Current loss {}\n".format(
                epoch, np.mean(np.array(epoch_loss))))
            iteration_number += 10  # 每10个iteration进行一次loss记录
            counter.append(iteration_number)
            loss_history.append(np.mean(np.array(epoch_loss)))
            acc = mytest()
            writer.add_scalar("./scalar/test", loss.item(), epoch)
            if acc > highest:
                highest = acc
                state = {'net': net.state_dict(
                ), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                t.save(state, "model_new_temp.ph")
                print("highest:"+str(highest))
    t.save(state, "model_ori_{}.ph".format(loss.item()))

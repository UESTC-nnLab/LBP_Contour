# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch as t

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.metrics import accuracy_score
import types
#import wx
#from matplotlib.backends import backend_wxagg
#from matplotlib.figure import Figure


# class showFrame(wx.Frame):
#     def __init__(self):
#         wx.Frame.__init__(self, None)

#         self.panel = backend_wxagg.FigureCanvasWxAgg(self, -1, Figure())
#         self.axes = self.panel.figure.gca()

#     def draw(self,image):
#         self.axes.cla()
#         image = image.cpu().clone()  # we clone the tensor to not do changes on it
#         image = image.squeeze(0)  # remove the fake batch dimension
#         image = (image - t.min(image)) / (t.max(image) - t.min(image))
#         unloader = transforms.ToPILImage()
#         image = unloader(image)
#         self.axes.imshow(image,cmap = plt.cm.gray)
#         self.panel.draw()
        #plt.pause(0.001)


class Config():
    training_dir = "./training/"
    testing_dir = "./testing/"
    train_batch_size = 1
    train_number_epochs = 10000


class myDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None, target_transform = None,keepOrder=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.target_transform = target_transform
        self.keepOrder = keepOrder

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
    def clahe_equalized(self,img):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #img_equalized = np.empty(img.shape)
        img_equalized = clahe.apply(np.array(img, dtype=np.uint8))
        return img_equalized

    # ===== normalize over the dataset
    def dataset_normalized(self,img):
        #img_normalized = np.empty(img.shape)
        img_std = np.std(img)
        img_mean = np.mean(img)
        img_normalized = (img - img_mean) / img_std

        img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
        return img_normalized

    def adjust_gamma(self,img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        #new_img = np.empty(img.shape)
        new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)
        return new_img

    def __getitem__(self, index):
        if self.keepOrder:
            index = index%20
            label_tuple = self.imageFolderDataset.imgs[index]
            img_tuple = self.imageFolderDataset.imgs[index+20]
            mask_tuple = self.imageFolderDataset.imgs[index+40]
            img = Image.open(img_tuple[0])
            label = Image.open(label_tuple[0])
            mask = Image.open(mask_tuple[0])

            img = img.convert("L")
            img = np.asarray(img)
            img = self.dataset_normalized(img)
            img = self.clahe_equalized(img)
            img = self.adjust_gamma(img, 1.2)
            img = Image.fromarray(np.uint8(img))
            seed = np.random.randint(2147483647)
            if self.transform is not None:
                random.seed(seed)
                img = self.transform(img)
            if self.target_transform is not None:
                random.seed(seed)
                label = self.target_transform(label)
                random.seed(seed)
                mask = self.target_transform(mask)
            #img = t.sub(t.mul(img/t.max(img),2),1)
            #0黑 255白
            #label = t.sub(1,label)
            # label = t.mul(label,2)
            # label = t.sub(label,1)
            return img,label,mask

    def __len__(self):
        return int(len(self.imageFolderDataset.imgs)/3)
        #return 1

def imshow(tensor, file_name,title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    #image = image.numpy()
    image = image.squeeze(0)  # remove the fake batch dimension
    image = (image - t.min(image))/(t.max(image)-t.min(image))
    image = image.float()
    unloader = transforms.ToPILImage()
    image = unloader(image)

    plt.imshow(image,cmap = plt.cm.gray)
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    if title is not None:
       plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def getFOV(img,mask):#返回视野内一维的数据
    mask = mask.ge(0.5)  # 确定没问题
    img = t.masked_select(img, mask)
    img = img.view(img.size(0), -1)
    img = img.cpu()
    img = img.detach().numpy()
    img = img.flatten()
    img = img.tolist()
    return img

def clahe_equalized(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #img_equalized = np.empty(img.shape)
    img_equalized = clahe.apply(np.array(img, dtype=np.uint8))
    return img_equalized

# ===== normalize over the dataset
def dataset_normalized(img):
    #img_normalized = np.empty(img.shape)
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std

    img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
    return img_normalized

def adjust_gamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    #new_img = np.empty(img.shape)
    new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)
    return new_img


class SBS(object):
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.2, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, X_test, y_train, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, X_test, y_train, y_test, p)
                scores.append(score)
                subsets.append(p)
                #print(scores)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
            print(str(dim))
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, X_test, y_train, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

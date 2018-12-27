
# Following is the data preparator from the images downloaded using OIDv4_Toolkit
# from ("https://github.com/EscVM/OIDv4_ToolKit") for any custom classes.
# Please tweak it according to your need

#%%
import time, os, copy, argparse, collections, sys, numpy as np

import torch, torch.nn as nn, torch.optim as optim, torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

# import model
from anchors import Anchors
import losses
from datagen import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval
import csv
# assert torch.__version__.split('.')[1] == '4'


#%%
oid_path = "../OIDv4_ToolKit/OID/data_ads/"
oid_path = "data/"

#%%
# Training/Testing/Valid File
csv_final = []
train_path = oid_path+"test/per_mp_lap"
for idx, file in enumerate(os.scandir(train_path+"/Label")):
#     if(idx > 3):
#         break
    with open(train_path+"/Label/"+file.name, "r") as f:
        annot = [x[:-1] for x in f.readlines()]
        for an in annot:
            spl = an.split(" ")

            # Removing the Label with 2 word(Mobile phone) here
            if('phone' in spl):
                spl.remove('phone')
#             print(spl)
            csv_final.append(["data/test/per_mp_lap/{}.jpg".format(file.name.split(".")[0])]+
                             spl[1:] + [spl[0]])
print(csv_final[:20])


#%%
with open("data/test/test_annot.csv", "w") as f:
#     f.write("\n".join(csv_final) + "\n")
    spamwriter = csv.writer(f, delimiter=',')
    for row in csv_final:
        spamwriter.writerow(row)


#%%
c = csv.reader(open("data/test/test_annot.csv", "r"), delimiter=',')
dic = {}
for row in c:
    if row[0] not in dic:
        dic[row[0]] = 1
print(len(dic))

#%%
# Rectifier for bad images(only having 1 dimension)
# I have corrected it using ImageMagick during the training only.

import skimage
train_path = oid_path+"test/per_mp_lap"
for idx, file in enumerate(os.scandir(train_path)):
    if not file.name == "Label":
        img = skimage.io.imread(train_path + "/" + file.name)
        if(len(img.shape) < 2):
            print(file.name)
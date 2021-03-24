import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import pandas as pd
import random

import glob
import os
import argparse
from sklearn.model_selection import StratifiedKFold
from collections import Counter

names = ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none')

def get_class_distribution(dataset_obj):
	count_dict = {k:0 for k, v in dataset_obj.class_to_idx.items()}
	idx2class = {v:k for k,v in dataset_obj.class_to_idx.items()}
	for element in tqdm(dataset_obj, desc="get_class_distribution"):
		y_lbl = element[1]
		y_lbl = idx2class[y_lbl]
		count_dict[y_lbl] += 1
		
	return count_dict


def weight_sampler(dataset_obj):
    class_count = [i for i in get_class_distribution(dataset_obj).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    
    #print(class_count, class_weights)

    sample_weights = [0] * len(dataset_obj)  # initializing을 하는거라는데,,?
    #print("len dataset_obj:", len(dataset_obj))
    #print(len(sample_weights))
    
    #all_label = []
    for idx, (img, label) in enumerate(tqdm(dataset_obj, desc="weight_sampler")):
        #label = dataset_obj[idx][1]
        #all_label.append(label)
        #if(idx == len(dataset_obj)): break
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    #print(Counter(all_label))
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


def set_seed(device, random_seed):
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    np.random.seed(random_seed)
    random.seed(random_seed)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

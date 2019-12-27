# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from .datasets.Caltech256 import Caltech
from .transforms.augmentations import ClassifyAugmentation
from config.config import dataset_base, train_config, val_config

train_transform = ClassifyAugmentation(train_config)
val_transform = ClassifyAugmentation(val_config)


def getData(mode):
    if mode == "train":
        data = Caltech(dataset_base.root, dataset_base.train, train_transform)
    elif mode == "val":
        data = Caltech(dataset_base.root, dataset_base.val, val_transform)
    elif mode == "test":
        data = Caltech(dataset_base.root, dataset_base.test, val_transform)
    else:
        raise Exception("Invalid mode!")
    return data



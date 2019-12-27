"""
Created on: 2019-09-17 15:09:41
@author: yang
"""
import os
from PIL import Image
from torch.utils import data


class Caltech(data.Dataset):
    def __init__(self, root=None, file=None, transform=None):
        self.root = root
        self.file = file
        self.transform = transform
        self.img_targets = self.read_files()

    def __len__(self):
        return len(self.img_targets)

    def read_files(self):
        img_targets = []
        for line in open(self.file):
            image_path, target = line.strip().split()
            img_targets.append({
                "path": image_path,
                "target": int(target)
            })
        return img_targets

    def __getitem__(self, index):
        item = self.img_targets[index]
        image = Image.open(os.path.join(self.root, item["path"])).convert("RGB")  # 一定要加上，因为灰度图读进来只有一个通道
        target_int = item["target"]
        if self.transform:
            image = self.transform(image)

        return image, target_int - 1

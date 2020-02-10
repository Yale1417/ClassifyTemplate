from config import train_config
from data.transforms.augmentations import ClassifyAugmentation2
import cv2
import albumentations

img = cv2.imread("/Users/yang/PycharmProjects/notebooks/images/parrot.jpg")

print(img.shape)
res = ClassifyAugmentation2(train_config)(img)

print(res[0])


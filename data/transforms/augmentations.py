from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    Normalize, ToTensor, Compose
from .customTransforms import PadResize, PadResizeCv
import albumentations
import torch

def do_nothing(img=None):
    return img


def enable_if(condition, obj):
    return obj if condition else do_nothing


class ClassifyAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, cfg=None):
        self.augment = Compose([
            PadResize(cfg.resize),
            enable_if(cfg.augment_horizontal_flip, RandomHorizontalFlip()),
            enable_if(cfg.augment_vertical_flip, RandomVerticalFlip()),
            enable_if(cfg.augment_rotation, RandomRotation(15)),
            ToTensor(),
            Normalize(cfg.MEAN, cfg.STD),

        ])

    def __call__(self, img):
        return self.augment(img)


class ClassifyAugmentation2(object):
    """ Transform to be used when training. """

    def __init__(self, cfg=None):
        self.augment = albumentations.Compose([
            albumentations.RandomRotate90(),
            albumentations.Flip(),
            albumentations.Transpose(),
            albumentations.OneOf([
                albumentations.IAAAdditiveGaussianNoise(),
                albumentations.GaussNoise(),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.MotionBlur(p=.2),
                albumentations.MedianBlur(blur_limit=3, p=0.1),
                albumentations.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            albumentations.OneOf([
                albumentations.OpticalDistortion(p=0.3),
                albumentations.GridDistortion(p=.1),
                albumentations.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
                albumentations.RandomBrightnessContrast(),
            ], p=0.3),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.OneOf([
                # albumentations.Resize(cfg.resize[0], cfg.resize[1]),
                PadResizeCv(cfg.resize)
            ], p=1),
            albumentations.Normalize(),
        ], p=1)

    def __call__(self, img):
        pic = self.augment(image=img)['image']
        return torch.from_numpy(pic.transpose((2, 0, 1)))


class ClassifyAugmentation3(object):
    """ Transform to be used when training. """

    def __init__(self, cfg=None):
        self.augment = albumentations.Compose([
            albumentations.OneOf([
                # albumentations.Resize(cfg.resize[0], cfg.resize[1]),
                PadResizeCv(cfg.resize)
            ], p=1),
            albumentations.Normalize(),
        ], p=1)

    def __call__(self, img):
        pic = self.augment(image=img)['image']
        return torch.from_numpy(pic.transpose((2, 0, 1)))
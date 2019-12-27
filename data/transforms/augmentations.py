from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
                                    Normalize, ToTensor, Compose
from .customTransforms import PadResize

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

# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import math
import random
import cv2
import torchvision.transforms.functional as F


__all__ = ["PadResize"]


class PadResize(object):
    """"
    PIL 实现方式
    resize的时候保持原图的有效像素长宽比不变，其他位置以填充值替代
    """
    def __init__(self, size, fill=(127, 127, 127)):
        assert (isinstance(size, tuple) and len(size) == 2)
        self.size = size  # (h, w) 与 openCV 的相反
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        pad_w, pad_h = 0, 0
        rate_dst = self.size[0] / self.size[1]
        rate_src = height / width
        if rate_dst > rate_src:
            pad_h = int(rate_dst*width-height)
        else:
            pad_w = int(height/rate_dst-width)
        return F.resize(F.pad(img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), self.fill),
                        self.size)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# class PadResize(object):
#     """"
#     cv2 实现方式
#     resize的时候保持原图的有效像素长宽比不变，其他位置以填充值替代
#     """
#     def __init__(self, size, padvalue=(127, 127, 127)):
#         assert (isinstance(size, tuple) and len(size) == 2)
#         self.size = size  # (w, h)
#         self.padvalue = padvalue
#
#     def __call__(self, img):
#         height, width = img.shape[:2]
#         pad_w, pad_h = 0, 0
#         rate_dst = self.size[1] / self.size[0]
#         rate_src = height / width
#         if rate_dst > rate_src:
#             pad_h = int(rate_dst*width-height)
#         else:
#             pad_w = int(height/rate_dst-width)
#         img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
#                                  cv2.BORDER_CONSTANT, value=self.padvalue)
#         img = cv2.resize(img, self.size)
#
#         return img
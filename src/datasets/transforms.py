
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

import cv2


def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets


def train_validation_split(root, train=True, ratio=0.3):
    imgs = sorted(glob.glob(os.path.join(root,'train', '*.png')))

    if train:
        boxes = sorted(glob.glob(os.path.join(root,'train', '*.txt')))

    train_x, train_y, val_x, val_y = train_test_split(imgs, boxes, test_size=ratio)

    return [train_x, train_y], [val_x, val_y]


class CustomDataset(Dataset):
    def __init__(self, img_list, boxes_list=None, transforms=None):
        self.transforms = transforms
        self.imgs = img_list
        
        if boxes_list:
            self.boxes = boxes_list

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1 # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
                
            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim

from utils.train_helper import model_snapshot, load_model

import yaml
from utils.train_helper import edict2dict
from utils.util_fnc import box_denormalize


class Runner(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.seed = config.seed
        self.device = config.device

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')
        self.metrics_file = os.path.join(config.exp_sub_dir, 'results.json')


        self.train_conf = config.train
        self.dataset_conf = config.dataset

        # # Get Loss Function
        # self.criterion = nn.MSELoss()
    
        # Choose the model
        if self.config.model_name == 'faster_rcnn':
            pass
        else:
            raise ValueError("Non-supported Model")
        
        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.train_conf.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum)
        elif self.train_conf.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        self.model = self.model.to(device=self.device)
        # self.criterion = self.criterion.to(device=self.device)
        
    def train(self, train_dataloader, val_dataloader):
        best_val = int(1e9)
        train_losses = {}
        val_losses = {}

        for epoch in range(1, self.train_conf.epoch+1):
            # Train the model for one epoch
            train_loss = self._train_epoch(train_dataloader)
            sum_train_loss = train_loss['total']

            for key in train_loss:
                if key not in train_losses:
                    train_losses[key] = []
                train_losses[key].append(train_loss[key].item())

            # Compute validation metric
            val_loss = self._evaluate(val_dataloader)
            sum_val_loss = val_loss['total']

            for key in val_loss:
                if key not in val_losses:
                    val_losses[key] = []
                val_losses[key].append(val_loss[key].item())

            # Save the weights of the best model
            if val_loss > best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)
            
            self.logger.info(f'Epoch {epoch:03d}: train_loss = {sum_train_loss:.4f} | val_loss = {sum_val_loss:.4f}')
            self.logger.info(f'Best Validation Loss: {best_val:.4f}')
                
        # Save the training and validation metrics to a file
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        
    def _train_epoch(self, dataloader):
        self.model.train()
        train_loss_dict = {}

        for inputs, targets in tqdm(dataloader):
            inputs = [img.to(self.device) for img in inputs]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            # Forward pass
            loss_dict = self.model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()
            self.optimizer.step()

            # Log the loss for this batch
            for key in loss_dict:
                if key not in train_loss_dict:
                    train_loss_dict[key] = []
                train_loss_dict[key].append(loss_dict[key].item())
            if 'total' not in train_loss_dict:
                train_loss_dict['total'] = []
            train_loss_dict['total'].append(losses.item())

        for key in train_loss_dict:
            train_loss_dict[key] = sum(train_loss_dict[key]) / len(train_loss_dict[key])

        return train_loss_dict

    
    def _evaluate(self, dataloader):
        val_loss_dict = {}
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader):
                inputs = [img.to(self.device) for img in inputs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = self.model(inputs, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Log the loss for this batch
                for key in loss_dict:
                    if key not in val_loss_dict:
                        val_loss_dict[key] = []
                    val_loss_dict[key].append(loss_dict[key].item())
                if 'total' not in val_loss_dict:
                    val_loss_dict['total'] = []
                val_loss_dict['total'].append(losses.item())

            for key in val_loss_dict:
                val_loss_dict[key] = sum(val_loss_dict[key]) / len(val_loss_dict[key])

            return val_loss_dict
    
    def inference(self, test_loader):
        self.model.eval()
        # Load the weights of the best model
        self.model.load_state_dict(torch.load(self.best_model_dir))
        
        results = pd.read_csv(self.dataset_conf.dir+'/sample_submission.csv')

        for img_files, images, img_width, img_height in tqdm(iter(test_loader)):
            images = [img.to(device) for img in images]

            with torch.no_grad():
                outputs = self.model(images)

            for idx, output in enumerate(outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = box_denormalize(x1, y1, x2, y2, img_width[idx], img_height[idx])
                    results = results.append({
                        "file_name": img_files[idx],
                        "class_id": label-1,
                        "confidence": score,
                        "point1_x": x1, "point1_y": y1,
                        "point2_x": x2, "point2_y": y1,
                        "point3_x": x2, "point3_y": y2,
                        "point4_x": x1, "point4_y": y2
                    }, ignore_index=True)

        # 결과를 CSV 파일로 저장
        results.to_csv(self.config.exp_sub_dir+'/baseline_submit.csv', index=False)
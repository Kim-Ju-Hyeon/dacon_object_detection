import click
import traceback
import os
import datetime
import pytz
from easydict import EasyDict as edict
import yaml
import pandas as pd

from utils.logger import setup_logging
from utils.train_helper import set_seed, mkdir, edict2dict
from datasets.my_dataset import *
from datasets.transforms import *
from runner.baseline_runner import Runner
from torch.utils.data import DataLoader


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    try:
        config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
        for train_transforms, aug_list in grid_search_train_transforms(augmentation_config=config.augmentations):
            now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            sub_dir = now.strftime('%m%d_%H%M%S')
            sub_dir = str(config.exp_name) + '_' + sub_dir
            
            config.seed = set_seed(config.seed)

            config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
            config.model_save = os.path.join(config.exp_sub_dir, "model_save")
            mkdir(config.model_save)

            save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
            config.execute_augmentations_list = aug_list
            yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
            log_save_name = f"log_exp_{config.seed}.txt"
                

            log_file = os.path.join(config.exp_sub_dir, log_save_name)
            logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
            logger.info("Writing log file to {}".format(log_file))
            logger.info("Exp instance id = {}".format(config.exp_name))


            runner = Runner(config=config, logger=logger)        
            train_data, val_data = train_validation_split(config.dataset.dir, train=True, ratio=config.dataset.val_size)

            train_dataset = CustomDataset(img_list=train_data[0], boxes_list=train_data[1], transforms=train_transforms, train=True)
            val_dataset = CustomDataset(img_list=val_data[0], boxes_list=val_data[1], transforms=get_validation_transforms(config.dataset.img_size), train=True)
            test_dataset = CustomDataset(img_list=None, train=False, transforms=get_test_transforms(config.dataset.img_size), root=config.dataset.dir+'/test')

            train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

            runner.train(train_loader, val_loader)
            runner.inference(test_loader)
        
    except:
        logger.error(traceback.format_exc())
        
if __name__ == '__main__':
    main()
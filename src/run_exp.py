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
from runner.runner import Runner
from torch.utils.data import DataLoader


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--inference', type=bool, default=False)
@click.option('--train_resume', type=bool, default=False)
def main(conf_file_path, inference, train_resume):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    
    if not inference and not train_resume:
        now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        sub_dir = now.strftime('%m%d_%H%M%S')
        sub_dir = str(config.exp_name) + '_' + sub_dir
        
        config.seed = set_seed(config.seed)

        config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
        config.model_save = os.path.join(config.exp_sub_dir, "model_save")
        mkdir(config.model_save)

        save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
        yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
        log_save_name = f"log_exp_{config.seed}.txt"
        
    elif inference:
        log_save_name = 'Inference_log_exp'
        
    else:
        config.train_resume = True
        log_save_name = 'Train_Resume_log_exp'

    log_file = os.path.join(config.exp_sub_dir, log_save_name)
    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))


    try:
        runner = Runner(config=config, logger=logger)        
        if not inference:

            # Get Dataset 1
            train_dataset, val_dataset = get_datasets(config, dtpye='train')
            # Get Dataset 2
            train_dataset = pd.read_csv(config.data.dir+'train.csv')

            train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)    
            val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
        
            runner.train(train_loader, val_loader)
            
        test_dataset = get_datasets(config, dtpye='test')
        test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

        preds = runner.test(test_loader)
        

    except:
        logger.error(traceback.format_exc())
        
if __name__ == '__main__':
    main()
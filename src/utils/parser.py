from argparse import ArgumentParser
from pathlib import Path
from utils.config import Config
import os
import yaml
import torch


class Parser:
    def __init__(self):
        self.parser = ArgumentParser()

        # General mode args
        self.parser.add_argument('-c', '--config', type=Path, help='Configuration Path', required=True, dest='CONFIG')
        self.parser.add_argument('--cpu', help='Set CPU as device', action='store_true', dest='CPU')

        # General data args
        self.parser.add_argument('--lr', type=float, help='Learning rate', dest='LR')
        self.parser.add_argument('--epochs', type=int, help='Number of epochs', dest='EPOCHS')
        self.parser.add_argument('--pretrained', type=str, help='Path to pretrained model checkpoint', dest='PRETRAINED')
        self.parser.add_argument('--only_test', help='Set True to skip training', action='store_true', dest='ONLY_TEST')

        # Logger args
        self.parser.add_argument('--experiment_name', type=str, help='Experiment name', dest='EXPERIMENT_NAME')
        self.parser.add_argument('--version', type=int, help='Version number', dest='VERSION_NUMBER')

    def parse_args(self) -> [Config, str]:
        self.args = self.parser.parse_args()

        # Resolve Warning: 
        # oneDNN custom operations are on. 
        # You may see slightly different numerical results due to floating-point round-off errors from different computation orders. 
        # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Check if CUDA devices are available
        available_devs = torch.cuda.device_count()
        if self.args.CPU:
            device = 'cpu'
        else:
            if available_devs >= 1:
                device = 'gpu'
            else:
                print('Couldn\'t find a GPU device, running on cpu...')
                device = 'cpu'

        with open(self.args.CONFIG) as f:
            d = yaml.safe_load(f)
            config = Config(**d)

        if self.args.LR is not None:
            config.model.learning_rate = self.args.LR

        if self.args.EPOCHS is not None:
            config.model.epochs = self.args.EPOCHS

        if self.args.PRETRAINED is not None:
            config.model.pretrained = self.args.PRETRAINED

        if self.args.ONLY_TEST is not None:
            config.model.only_test = self.args.ONLY_TEST
        
        if self.args.EXPERIMENT_NAME is not None:
            config.logger.experiment_name = self.args.EXPERIMENT_NAME

        if self.args.VERSION_NUMBER is not None:
            config.logger.version = self.args.VERSION_NUMBER

        return config, device

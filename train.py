import torch
import yaml
import argparse
import os
import lightning as L

import datasets
from utils import plot_metrics


from models import (
    Llama,
    LlamaHF,
    Wrapper
)

class TrainingLlama:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
        self.configs = configs
    def run(self, max_epochs: int = None, ckpt_path: str = None, limit_batches: bool = False):
        llama_config = self.configs['Llama_config']
        dataset_config = self.configs['dataset_config']
        training_config = self.configs['training_config']
        if max_epochs:
            training_config['trainer_config']['max_epochs'] = max_epochs
        if ckpt_path:
            training_config['fit_config']['ckpt_path'] = ckpt_path

        # dataset
        datamodule = datasets.LitNextTokenPredictionDataModule(**dataset_config)

        # Wrapper model
        optimizer_config = training_config['optimizer_config']
        wrapper_model = Wrapper(
            model = llama_config,
            **optimizer_config
        )

        # trainer config
        trainer_config = training_config['trainer_config']
        # ---logger
        if trainer_config['logger']:
            trainer_config['logger'] = L.pytorch.loggers.CSVLogger(
                save_dir = trainer_config['default_root_dir'],
                name = 'logs'
            )

        # for reproducibility: seed_everything, deterministic
        L.seed_everything(42, workers=True)
        if limit_batches: # for testing purpose
            limit_batches = {
                'limit_train_batches': 2,
                'limit_test_batches': 2,
                'limit_val_batches': 2
            }
        else:
            limit_batches = {}
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
            logging_interval = 'step'
        )
        trainer = L.Trainer(
            **trainer_config,
            deterministic=True,
            enable_checkpointing = False, 
            callbacks = [lr_monitor],
            **limit_batches
        )

        # training phase
        fit_config = training_config['fit_config']
        trainer.fit(
            model = wrapper_model,
            train_dataloaders = datamodule,
            **fit_config
        )
        # ---plot metric curves
        if trainer_config['logger']:
            plot_metrics(trainer.log_dir)

        # testing phase
        trainer.test(
            model = wrapper_model,
            dataloaders = datamodule
        )

        # save the last checkpoint manually if set enable_checkpointing = False
        self.save_checkpoint(trainer_config['default_root_dir'], trainer)
        '''
        default_root_dir = trainer_config['default_root_dir']
        checkpoint_dir = os.path.join(default_root_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok = True)
        epoch = trainer.current_epoch-1
        step = trainer.global_step
        checkpoint_name = f'epoch={epoch}-step={step}.ckpt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        trainer.save_checkpoint(checkpoint_path)
        '''
    def save_checkpoint(self, root, trainer):
        checkpoint_dir = os.path.join(root, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok = True)
        epoch = trainer.current_epoch-1
        step = trainer.global_step
        checkpoint_name = f'epoch={epoch}-step={step}.ckpt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str)
    parser.add_argument('--max_epochs', type = int, default = None)
    parser.add_argument('--ckpt_path', type = str, default = None)
    parser.add_argument('--limit_batches', type = bool, default = False)
    
    args = parser.parse_args()
    training = TrainingLlama(args.config_file)
    training.run(
        max_epochs = args.max_epochs,
        ckpt_path = args.ckpt_path,
        limit_batches = args.limit_batches
    )

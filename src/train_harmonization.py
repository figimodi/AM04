from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import Parser
from datasets import HarmonizationDataset
from modules import HarmonizationModule
import yaml
import os

if __name__ == '__main__':
    # Get configuration arguments
    parser = Parser()
    config, device = parser.parse_args()

    # Load sources
    harmonization_dataset = HarmonizationDataset(defects_folder=config.dataset.defects_folder)

    # Create train-val-test splits
    splits = harmonization_dataset.create_splits(config.dataset.splits)
    train_split, val_split, test_split = splits

    # Instantiate Dataloaders for each split
    train_dataloader = DataLoader(train_split, batch_size=config.model.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_split, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_split, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)

    # Instantiate logger, logs goes into {config.logger.log_dir}/{config.logger.experiment_name}/version_{config.logger.version}
    logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=config.logger.version, name=config.logger.experiment_name)

    # Load pretrained model or else start from scratch
    if config.model.pretrained is None:
        module = HarmonizationModule(
            epochs=config.model.epochs,
            lr=config.model.learning_rate, 
            optimizer=config.model.optimizer, 
            scheduler=config.model.scheduler,
        )
    else:
        module = HarmonizationModule.load_from_checkpoint(
            map_location='cpu',
            checkpoint_path=config.model.pretrained,
            epochs=config.model.epochs,
            lr=config.model.learning_rate,
            optimizer=config.model.optimizer,
            scheduler=config.model.scheduler,
        )

    # Set callback function to save checkpoint of the model
    checkpoint_cb = ModelCheckpoint(
        monitor=config.checkpoint.monitor,
        dirpath=os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{config.logger.version}'),
        filename='epoch={epoch:03d}_loss={config.checkpoint.monitor:.4f}',
        save_weights_only=True,
        save_top_k=config.checkpoint.save_top_k,
        mode=config.checkpoint.mode,
    )

    # Instantiate a trainer
    trainer = Trainer(
        logger=logger,
        accelerator=device,
        devices=[0] if device == "gpu" else "auto",
        default_root_dir=config.logger.log_dir,
        max_epochs=config.model.epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
        num_sanity_val_steps=0, # Validation steps at the very beginning to check bugs without waiting for training
        reload_dataloaders_every_n_epochs=1,  # Reload the dataset to shuffle the order
    )

    # Train
    if not config.model.only_test:
        trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test
    trainer.test(model=module, dataloaders=test_dataloader)

    # Log into yaml file the config
    with open(os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{config.logger.version}', 'hparams.yaml'), 'w') as yaml_file:
        hparams = {
            'epochs': config.model.epochs,
            'batch_size': config.model.batch_size,
            'learning_rate': config.model.learning_rate,
            'optimizer': config.model.optimizer,
            'scheduler': config.model.scheduler,
        }
        yaml.dump(hparams, yaml_file)
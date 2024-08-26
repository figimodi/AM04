import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from torchvision.utils import save_image
from pathlib import Path
from typing import Tuple
from pydantic import BaseModel
from models import TSAINetworkV1, TSAINetworkV2


class HarmonizationModule(LightningModule):
    def __init__(
            self,
            name: str,
            epochs: int,
            lr: float, 
            optimizer: str, 
            scheduler: str, 
            save_images: Path = None,
        ):
        super().__init__()
        self.save_hyperparameters()

        # Network
        if name == 'tsaiv1':
            self.net = TSAINetworkV1()
        elif name == 'tsaiv2':
            self.net = TSAINetworkV2()
        else:
            raise ValueError(f'Network {name} not available.')

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Additional params
        self.save_images = save_images
        if self.save_images:
            if os.path.exists(self.save_images):
                shutil.rmtree(self.save_images)
            os.mkdir(self.save_images)

        # Test outputs
        self.test_outputs = []

    def forward(self, x):
        y = self.net(x)
        return y

    def loss_function(self, y, original_image):
        return torch.nn.functional.mse_loss(y, original_image)

    def training_step(self, batch):
        original_image, fake_image, defect_type, mask = batch
        batch_size = original_image.shape[0]
        y = self.net(fake_image)
        loss = self.loss_function(y, original_image)
        self.log('train_loss', loss.item(), batch_size=batch_size, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch):
        original_image, fake_image, defect_type, mask = batch
        batch_size = original_image.shape[0]
        y = self.net(fake_image)
        loss = self.loss_function(y, original_image)
        self.log('val_loss', loss.item(), batch_size=batch_size, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def test_step(self, batch):
        original_image, fake_image, defect_type, mask = batch
        y = self.net(fake_image)

        for i in range(original_image.shape[0]):
            self.test_outputs.append((
                mask[i],          
                fake_image[i],     
                y[i],              
                original_image[i],
                defect_type[i],  
                self.loss_function(y[i], fake_image[i]),
                self.loss_function(y[i], original_image[i])    
            ))        

        return None

    def on_test_epoch_end(self):
        # Sort examples by loss, to plot the best results obtained by the network
        self.test_outputs.sort(key=lambda x: x[-2], reverse=True)

        for i, (mask, fake_image, y, original_image, defect_type, change, loss) in enumerate(self.test_outputs[:10]):
            if self.save_images:
                save_image(y, os.path.join(self.save_images, f'image_{i}_{defect_type}.png'))
            
            fig, axes = plt.subplots(1, 4, figsize=(15, 6))

            # Display the mask
            im1 = axes[0].imshow(mask.permute(1, 2, 0)[:, :, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('Mask')

            # Display the artifact image 
            im2 = axes[1].imshow(fake_image.permute(1, 2, 0)[:, :, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Color Manipulated Image (input)')
            
            # Display the original image 
            im3 = axes[2].imshow(y.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            axes[2].set_title('Harmonized Image (output)')

            # Display the target image
            im3 = axes[3].imshow(original_image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            axes[3].set_title('Target Image (ground truth)')

            plt.suptitle(f'Comparison {i+1}')

            fig.canvas.draw()
            
            # Copy the buffer to make it writable
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.copy()
            plot_image = torch.from_numpy(plot_image)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Log the image to TensorBoard
            self.logger.experiment.add_image(f'Comparison_{i+1}.png', plot_image, self.current_epoch, dataformats='HWC')

            plt.close(fig)

        loss = torch.tensor([sample[-1] for sample in self.test_outputs]).mean()
        self.log('test_loss:', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        scheduler = None
        
        if self.optimizer == 'adam':
            print("Using Adam optimizer")
            optimizers = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        elif self.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizers = torch.optim.AdamW(self.parameters(), lr = self.lr)
        
        elif self.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizers = torch.optim.SGD(self.parameters(), lr = self.lr)
        
        if self.scheduler == 'cosine':
            print("Using CosineAnnealingLR scheduler")
            scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=self.epochs)]
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = [torch.optim.lr_scheduler.StepLR(optimizers, step_size=10, gamma=0.1)]
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', min_lr=1e-8, factor=0.1, patience=5, verbose=True)
            return  {
                        'optimizer': optimizers,
                        'scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], scheduler
        return [optimizers]
        
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            previous_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']
            scheduler.step(metric)
            new_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']
            if new_lr != previous_lr:
                self.log('lr', new_lr, prog_bar=True, logger=True)
        else:
            super().lr_scheduler_step(scheduler, optimizer_idx, metric)

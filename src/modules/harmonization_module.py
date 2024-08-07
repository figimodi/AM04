import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from torchvision.utils import save_image
from pathlib import Path
from typing import Tuple
from pydantic import BaseModel
from models import TSAINetwork, NewTSAINetwork


class HarmonizationModule(LightningModule):
    def __init__(
            self, 
            epochs: int,
            lr: float, 
            optimizer: str, 
            scheduler: str, 
            save_images: Path = None,
        ):
        super().__init__()
        self.save_hyperparameters()

        # Network
        self.net = TSAINetwork()

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Additional params
        self.save_images = save_images

        # Test outputs
        self.test_outputs = []

    def forward(self, x):
        y = self.net(x)
        return y

    def loss_function(self, y, original_image):
        return torch.nn.functional.mse_loss(y, original_image)

    def training_step(self, batch):
        original_image, fake_image, defect_type = batch
        batch_size = original_image.shape[0]
        y = self.net(fake_image)
        loss = self.loss_function(y, original_image)
        self.log('train_loss', loss.item(), batch_size=batch_size, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch):
        original_image, fake_image, defect_type = batch
        batch_size = original_image.shape[0]
        y = self.net(fake_image)
        loss = self.loss_function(y, original_image)
        self.log('val_loss', loss.item(), batch_size=batch_size, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def test_step(self, batch):
        original_image, fake_image, defect_type = batch
        y = self.net(fake_image)
        self.test_outputs.append((torch.Tensor(y), torch.Tensor(original_image), torch.Tensor(fake_image), defect_type))
        return None

    def on_test_epoch_end(self):
        for i, (y, original_image, fake_image, defect_type) in enumerate(self.test_outputs):
            if self.save_images:
                # Save each image in the batch
                for z in range(y.shape[0]):
                    save_image(y[z], os.path.join(self.save_images, f'image_{i}_{z}_{defect_type[z]}.png'))
            
            for sample in range(min(y.shape[0], 10)):
                fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                # Display the artifact image (assuming y is in [batch_size, 3, height, width] format)
                im1 = axes[0].imshow(y[sample].permute(1, 2, 0).cpu().numpy())
                axes[0].set_title('Artifact Image')

                # Display the original image (assuming original_image is in [batch_size, 3, height, width] format)
                im2 = axes[1].imshow(original_image[sample].permute(1, 2, 0).cpu().numpy())
                axes[1].set_title('Original Image')
                
                im3 = axes[2].imshow(fake_image[sample].permute(1, 2, 0)[:, :, 0].cpu().numpy())
                axes[2].set_title('Fake Image')

                plt.suptitle(f'Comparison {i+1}_{sample+1}')

                fig.canvas.draw()
                
                # Copy the buffer to make it writable
                plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_image = plot_image.copy()  # Make the buffer writable
                plot_image = torch.from_numpy(plot_image)
                plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Log the image to TensorBoard
                self.logger.experiment.add_image(f'Comparison_{i+1}_{sample+1}.png', plot_image, self.current_epoch, dataformats='HWC')

                plt.close(fig)

        loss = torch.tensor([self.loss_function(y, original_image) for y, original_image, _, _ in self.test_outputs]).mean()
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
        
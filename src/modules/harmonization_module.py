from lightning.pytorch import LightningModule
from typing import Tuple
from pydantic import BaseModel
from models import TSAIEncoder, TSAIDecoder
import numpy as np
import matplotlib.pyplot as plt
import torch


class HarmonizationModule(LightningModule):
    def __init__(
            self, 
            epochs: int,
            lr: float, 
            optimizer: str, 
            scheduler: str, 
        ):
        super().__init__()

        # Network
        self.encoder = TSAIEncoder()
        self.decoder = TSAIDecoder()

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Test outputs
        self.test_outputs = []

    def forward(self, x):
        l = self.encoder(x)
        y = self.decoder(l)
        return y

    def loss_function(self, y, original_image):
        return torch.nn.functional.mse_loss(y, original_image)

    def training_step(self, batch):
        original_image, fake_image = batch
        l = self.encoder(fake_image)
        y = self.decoder(l)
        loss = self.loss_function(y, original_image)
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch):
        original_image, fake_image = batch
        l = self.encoder(fake_image)
        y = self.decoder(l)
        loss = self.loss_function(y, original_image)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def test_step(self, batch):
        original_image, fake_image = batch
        l = self.encoder(fake_image)
        y = self.decoder(l)
        self.test_outputs.append((torch.Tensor(y), torch.Tensor(original_image)))
        return None

    def on_test_epoch_end(self):
        for i, (y, original_image) in enumerate(self.test_outputs):
            for sample in range(y.shape[0]):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Display the artifact image (assuming y is in [batch_size, 3, height, width] format)
                im1 = axes[0].imshow(y[sample].permute(1, 2, 0).cpu().numpy())
                axes[0].set_title('Artifact Image')

                # Display the original image (assuming original_image is in [batch_size, 3, height, width] format)
                im2 = axes[1].imshow(original_image[sample].permute(1, 2, 0).cpu().numpy())
                axes[1].set_title('Original Image')

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

        loss = torch.tensor([self.loss_function(y, original_image) for y, original_image in self.test_outputs]).mean()
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
            scheduler = [torch.optim.scheduler.CosineAnnealingLR(optimizers, T_max=self.epochs)]
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = [torch.optim.scheduler.StepLR(optimizers, step_size=10, gamma=0.1)]
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.1, patience=10, verbose=True)
            return  {
                        'optimizer': optimizers,
                        'scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], scheduler
        return [optimizers]
        
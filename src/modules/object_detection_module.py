import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from typing import Tuple
from pydantic import BaseModel
from models import MyFasterRCNN
from datasets import Defect

class ObjectDetectionModule(LightningModule):
    def __init__(
            self, 
            name: str,
            epochs: int,
            lr: float, 
            optimizer: str, 
            scheduler: str, 
        ):
        super().__init__()
        self.save_hyperparameters()

        # Network
        self.model = MyFasterRCNN(num_classes=5)  # Adjust num_classes as needed

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Test outputs
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self((images, targets))
        
        # FasterRCNN returns a list of dicts, one per image
        loss_dict = self.model.roi_heads.losses(predictions, targets)  # Extract losses from the model
        total_loss = sum(loss for loss in loss_dict.values())  # Sum up all losses
        
        self.log('train_loss', total_loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total_loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        loss_dict = self.model.roi_heads.losses(predictions, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        self.log('val_loss', total_loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)

        for i in range(len(images)):
            self.test_outputs.append((
                images[i],
                predictions[i],
                targets[i],
                sum(self.model.roi_heads.losses(predictions, targets).values())  # Total loss
            ))

        return None

    def on_test_epoch_end(self):
        for i, (image, prediction, target, loss) in enumerate(self.test_outputs):
            # Extract bounding boxes and labels
            boxes = prediction['boxes'].cpu().detach().numpy()
            labels = prediction['labels'].cpu().detach().numpy()
            scores = prediction['scores'].cpu().detach().numpy()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')

            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5:  # Threshold for displaying boxes
                    x_min, y_min, x_max, y_max = box
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_min, y_min, f'{Defect(label).name} ({score:.2f})', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

            plt.suptitle(f'Result {i+1}')
            fig.canvas.draw()

            # Copy the buffer to make it writable
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.copy()  # Make the buffer writable
            plot_image = torch.from_numpy(plot_image)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Log the image to TensorBoard
            self.logger.experiment.add_image(f'Comparison_{i+1}.png', plot_image, self.current_epoch, dataformats='HWC')

            plt.close(fig)

        # Calculate the mean test loss
        loss = torch.tensor([sample[-1] for sample in self.test_outputs]).mean()
        self.log('test_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        
        if self.optimizer == 'adam':
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        elif self.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        elif self.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        
        if self.scheduler == 'cosine':
            print("Using CosineAnnealingLR scheduler")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8)
            return  {
                        'optimizer': optimizer,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizer], [scheduler]
        return [optimizer]

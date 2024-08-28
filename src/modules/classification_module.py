import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from typing import Tuple
from pydantic import BaseModel
from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, LeNet5, GrayVGG16_BN, GrayVGG16, GoogLeNet, AlexNet
from datasets import Defect

resnet_architectures = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}

vgg_architectures = {
    'vgg': GrayVGG16,
    'vgg_bn': GrayVGG16_BN,
}


class ClassificationModule(LightningModule):
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
        if 'resnet' in name:
            self.model = resnet_architectures[name]()
        elif 'vgg' in name:
            self.model = vgg_architectures[name]()
        elif name == 'googlenet':
            self.model = GoogLeNet()
        elif name == 'alexnet':
            self.model = AlexNet()
        elif name == 'lenet5':
            self.model = LeNet5()
        else:
            raise ValueError(f'Network {name} not available.')

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Test outputs
        self.test_outputs = []

    def forward(self, x):
        y = self.model(x)
        return y

    def loss_function(self, prediction, label):
        return torch.nn.functional.cross_entropy(prediction, label)

    def training_step(self, batch):
        image, label = batch
        prediction = self(image)
        loss = self.loss_function(prediction, label)
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_epoch=True)

    def validation_step(self, batch):
        image, label = batch
        prediction = self(image)
        loss = self.loss_function(prediction, label)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def test_step(self, batch):
        image, label = batch
        prediction = self(image)

        for i in range(image.shape[0]):
            self.test_outputs.append((
                image[i],
                prediction[i],
                label[i],
                self.loss_function(prediction[i], label[i])    
            ))  

        return None

    def on_test_epoch_end(self):
        for i, (image, prediction, label, loss) in enumerate(self.test_outputs):
            # Determine if prediction is correct
            predicted_class = torch.argmax(prediction).item()
            correct_class = label.item()
            correct_prediction = (predicted_class == correct_class)

            # Display prediction and label with color based on correctness
            if correct_prediction:
                continue
            else:
                fig, ax = plt.subplots(figsize=(6, 6))

                # Display the image
                ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                ax.set_title('Input Image')
                prediction_text_color = 'red'

                prediction_text = Defect(predicted_class).name
                label_text = Defect(correct_class).name

                ax.text(0.5, -0.1, f'Prediction: {prediction_text}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color=prediction_text_color)
                ax.text(0.5, -0.2, f'Label: {label_text}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')

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
        top1 = np.array([(torch.argmax(prediction, dim=0)==label).cpu() for _, prediction, label, _ in self.test_outputs]).mean()*100
        self.log('test_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_top1', top1, logger=True, prog_bar=True, on_step=False, on_epoch=True)

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.1, patience=5, min_lr=1e-8)
            return  {
                        'optimizer': optimizers,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], scheduler
        return [optimizers]

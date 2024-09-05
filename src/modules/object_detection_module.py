import torch
import torchvision 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from pprint import pprint 
from cv2.dnn import NMSBoxes
from typing import Tuple
from pydantic import BaseModel
from models import MyFasterRCNN
from pathlib import Path
from datasets import Defect
from torchvision.ops import box_iou
from sklearn.metrics import auc


class ObjectDetectionModule(LightningModule):
    def __init__(
            self, 
            name: str,
            epochs: int,
            lr: float, 
            optimizer: str, 
            scheduler: str,
            pretrained: Path,
            pretrained_backbone: Path,
        ):
        super().__init__()
        self.save_hyperparameters()

        # Network
        self.model = MyFasterRCNN(pretrained_backbone=pretrained_backbone, num_classes=5)  # Adjust num_classes as needed
        
        checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
        state_dict = {}
        for key in checkpoint['state_dict']:
            new_key = key[6:]
            state_dict[new_key] = checkpoint['state_dict'][key]
        self.model.load_state_dict(state_dict)

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
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total_loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if len(optimizer.param_groups) == 0:
            return
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict , _ = self.model.eval_forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log('val_loss', total_loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)

        for i in range(len(images)):
            self.test_outputs.append((
                images[i],
                predictions[i],
                targets[i]
            ))

        return None

    def on_test_epoch_end(self):
        threshold = 0.5
        nms_threshold = 0.2
        num_classes = 5
        ap_per_class = {k: 0 for k in range(num_classes)}
        occurances_per_class = {k: 0 for k in range(num_classes)}
        predictions_per_class = {
            k: pd.DataFrame({
                'iou': pd.Series(dtype='float64'),        
                'correct': pd.Series(dtype='bool'),      
                'precision': pd.Series(dtype='float64'), 
                'recall': pd.Series(dtype='float64')  
            }) for k in range(num_classes)
        }

        # Check predictions
        for _, prediction, target in self.test_outputs:
            pred_boxes = prediction['boxes'].cpu().detach().numpy()
            pred_labels = prediction['labels'].cpu().detach().numpy()
            pred_scores = prediction['scores'].cpu().detach().numpy()

            target_boxes = target['boxes'].cpu().detach().numpy()
            target_labels = target['labels'].cpu().detach().numpy()

            # For each prediction check if it's correct
            ious = box_iou(torch.Tensor(pred_boxes), torch.Tensor(target_boxes))
            target_idx = np.argmax(ious, axis=1)
            corrects = pred_labels == target_labels[target_idx]
            
            iou_values = ious[np.arange(ious.shape[0]), target_idx]
            ious = iou_values.numpy().reshape(-1, 1).flatten()

            # NMS to eliminate overlapping predictions
            keep = torchvision.ops.nms(torch.Tensor(pred_boxes), torch.Tensor(pred_scores), nms_threshold)

            for p in keep:
                c = pred_labels[p]
                if ious[p] > threshold:
                    new_row = {'iou': ious[p], 'correct': corrects[p], 'precision': 0, 'recall': 0}
                    predictions_per_class[c].loc[len(predictions_per_class[c])] = new_row
                    predictions_per_class[c] = predictions_per_class[c].reset_index(drop=True)

            unique_labels, counts = np.unique(target_labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            for label, count in label_counts.items():
                occurances_per_class[label] += count


        for c in range(num_classes):
            if len(predictions_per_class[c]) < 2:
                ap_per_class[c] = 0
                continue
            predictions_per_class[c] = predictions_per_class[c].sort_values(by='iou', ascending=False)
            TP = 0
            FP = 0

            for i, row in predictions_per_class[c].iterrows():
                TP += row['correct']
                FP += not row['correct']
                predictions_per_class[c].loc[i, 'recall'] = TP / occurances_per_class[c]
                predictions_per_class[c].loc[i, 'precision'] = TP / (TP + FP)

            precision = predictions_per_class[c]['precision'].values
            recall = predictions_per_class[c]['recall'].values
            pr_auc = auc(recall, precision)

            # Average Precision
            ap_per_class[c] = pr_auc
            self.log(f'AP@{threshold} of class {c}', ap_per_class[c], logger=True, prog_bar=True, on_step=False, on_epoch=True)

        mAP = np.array(list(ap_per_class.values())).mean()
        self.log(f'mAP@{threshold}', mAP, logger=True, prog_bar=True, on_step=False, on_epoch=True)

        # Visualize predictions
        for i, (image, prediction, target) in enumerate(self.test_outputs):
            pred_boxes = prediction['boxes'].cpu().detach().numpy()
            pred_labels = prediction['labels'].cpu().detach().numpy()
            pred_scores = prediction['scores'].cpu().detach().numpy()

            target_boxes = target['boxes'].cpu().detach().numpy()
            target_labels = target['labels'].cpu().detach().numpy()

            # For each prediction check if it's correct
            ious = box_iou(torch.Tensor(pred_boxes), torch.Tensor(target_boxes))
            target_idx = np.argmax(ious, axis=1)
            corrects = pred_labels == target_labels[target_idx]
            
            iou_values = ious[np.arange(ious.shape[0]), target_idx]
            ious = iou_values.numpy().reshape(-1, 1).flatten()

            # NMS to eliminate overlapping predictions
            keep = torchvision.ops.nms(torch.Tensor(pred_boxes), torch.Tensor(pred_scores), nms_threshold)

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            ious = ious[keep]
            corrects = corrects[keep]

            if len(keep) == 1:
                pred_boxes = [pred_boxes]
                pred_labels = [pred_labels]
                ious = [ious]
                corrects = [corrects]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')

            for box, label, iou, correct in zip(pred_boxes, pred_labels, ious, corrects):
                if iou > threshold and correct:
                    x_min, y_min, x_max, y_max = box
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='blue', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_min, y_min, f'{Defect(label).name} (iou={iou:.2f})', bbox=dict(facecolor='yellow', alpha=0.3), fontsize=6, color='blue')

            for box, label in zip(target_boxes, target_labels):
                x_min, y_min, x_max, y_max = box
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=1)
                ax.add_patch(rect)
                ax.text(x_max, y_max, f'{Defect(label).name}', bbox=dict(facecolor='yellow', alpha=0.3), fontsize=6, color='red')

            plt.suptitle(f'Result {i+1}')
            fig.canvas.draw()

            # Copy the buffer to make it writable
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.copy()
            plot_image = torch.from_numpy(plot_image)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Log the image to TensorBoard
            self.logger.experiment.add_image(f'Comparison_{i+1}.png', plot_image, self.current_epoch, dataformats='HWC')

            plt.close(fig)

    def configure_optimizers(self):
        scheduler = None
        
        if self.optimizer == 'adam':
            print("Using Adam optimizer")
            optimizers = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        elif self.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizers = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        elif self.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizers = torch.optim.SGD(self.parameters(), lr=self.lr)
        
        if self.scheduler == 'cosine':
            print("Using CosineAnnealingLR scheduler")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=self.epochs)
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=10, gamma=0.1)
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.1, patience=5, min_lr=1e-8)
            return  {
                        'optimizer': optimizers,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], [scheduler]
        else:
            return [optimizers]

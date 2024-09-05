import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from typing import Tuple, List, Dict, Optional
from torch import Tensor
from models import ResNet18
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers


class MyFasterRCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MyFasterRCNN, self).__init__()

        resnet_net = ResNet18()
        
        checkpoint = torch.load('C:/Users/grfil/Documents/GitHub/AM04/src/log/train_resnet_18_5c/version_2/epoch=148_val_loss=0.399691.ckpt', map_location=torch.device('cpu'))
        state_dict = {}
        for key in checkpoint['state_dict']:
            new_key = key.replace('model.', '')
            state_dict[new_key] = checkpoint['state_dict'][key]
        # resnet_net.load_state_dict(checkpoint['state_dict'], strict=False)
        resnet_net.load_state_dict(state_dict)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        
        backbone_with_fpn = BackboneWithFPN(
            backbone, 
            return_layers={'4': '0', '5': '1', '6': '2', '7': '3'},
            in_channels_list=[64, 128, 256, 512],
            out_channels=512
        )

        # Define the Region Proposal Network (RPN) anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes= ((8,), (16,), (32,), (64,), (128,)),  # Sizes for each feature map
            aspect_ratios=((0.03, 1.0, 5, 10),) * 5  # Aspect ratios for each feature map
        )

        # Define the ROI Pooling feature extractor
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
            box_detections_per_img=50,
        )
        
        mean = [0]
        std = [1]
        transforms = GeneralizedRCNNTransform(800,1333,mean,std)
        self.model.transform = transforms

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
        
    def eval_forward(self, images, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        self.model.eval()

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        self.model.rpn.training=True
        #self.model.roi_heads.training=True


        #####proposals, proposal_losses = self.model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = self.model.rpn.head(features_rpn)
        anchors = self.model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = self.model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = self.model.roi_heads.select_training_samples(proposals, targets)
        box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        self.model.rpn.training=False
        self.model.roi_heads.training=False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections
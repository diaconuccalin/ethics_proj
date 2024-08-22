import os

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from constants import *


def frcnn_model(num_classes):
    backbone_vgg = torchvision.models.vgg16(pretrained=False).features
    out_channels = 512
    in_channels_list = [128, 256, 512, 512]
    return_layers = {"9": "0", "16": "1", "23": "2", "30": "3"}
    backbone = BackboneWithFPN(
        backbone_vgg, return_layers, in_channels_list, out_channels
    )
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ),
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_detections_per_img=DETECTIONS_PER_IMAGE,
        box_score_thresh=BOX_SCORE_THRESHOLD,
    )

    return model


def frcnn_model_pretrained(dataset, device):
    if dataset == "COCO":
        model = frcnn_model(len(COCO_CLASSES))
    else:
        model = frcnn_model(len(VOC_CLASSES))

    model.to(device)

    if dataset == "COCO":
        checkpoint = torch.load(os.path.join("weights", "coco-FRCNN-vgg16.pth"))
    else:
        checkpoint = torch.load(os.path.join("weights", "voc-FRCNN-vgg16.pth"))

    model.load_state_dict(checkpoint["model_state_dict"])

    return model

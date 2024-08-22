import torchvision
import torchvision.ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    BackboneWithFPN,
)

from models.ModelTypes import ModelTypes


def get_default_backbone(model_type, pretrained):
    if model_type == ModelTypes.RESNET50:
        backbone = resnet_fpn_backbone(
            model_type, pretrained=pretrained, trainable_layers=5
        )
        backbone.out_channels = 256
    elif model_type == ModelTypes.RESNET18:
        backbone = resnet_fpn_backbone(
            model_type, pretrained=pretrained, trainable_layers=5
        )
        backbone.out_channels = 256
    elif model_type == ModelTypes.VGG16:
        backbone = torchvision.models.vgg16(pretrained=pretrained).features
        out_channels = 512
        in_channels_list = [128, 256, 512, 512]
        return_layers = {"9": "0", "16": "1", "23": "2", "30": "3"}
        backbone = BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels
        )
        backbone.out_channels = 512
    else:
        backbone = None

    return backbone


def get_default_anchor_generator():
    return AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ),
    )


def get_default_roi_pooler():
    return torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )


def get_train_model(num_classes, model_type: ModelTypes = ModelTypes.NONE):
    assert model_type != ModelTypes.NONE, "Model type not compatible."

    backbone = get_default_backbone(model_type=model_type, pretrained=True)
    anchor_generator = get_default_anchor_generator()
    roi_pooler = get_default_roi_pooler()

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


def get_eval_model(
    num_classes,
    detections_per_image,
    box_score_threshold,
    model_type: ModelTypes = ModelTypes.NONE,
):
    assert model_type in [
        ModelTypes.RESNET50,
        ModelTypes.VGG16,
    ], "Model type not supported."

    backbone = get_default_backbone(model_type=model_type, pretrained=False)
    anchor_generator = get_default_anchor_generator()
    roi_pooler = get_default_roi_pooler()

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_detections_per_img=detections_per_image,
        box_score_thresh=box_score_threshold,
    )

    return model

import os
from typing import Optional, Callable, Tuple, Any

import torch
import torchvision.transforms.functional as FT
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset


class COCODataset(VisionDataset):
    def __init__(
        self,
        root: str,
        annotation_file: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(COCODataset, self).__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, item: int) -> Tuple[Any, Any, Any, Any]:
        coco = self.coco
        img_ids = self.ids[item]
        ann_ids = coco.getAnnIds(imgIds=img_ids)
        target = coco.loadAnns(ann_ids)
        corrupted = False
        path = coco.loadImgs(img_ids)[0]["file_names"]

        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        bboxes = list()
        labels = list()
        areas = list()

        for el in target:
            bbox_corrected = bbox = el["bbox"]
            bbox_corrected[2] = bbox[0] + bbox[2]
            bbox_corrected[3] = bbox[1] + bbox[3]

            bboxes.append(bbox_corrected)
            labels.append(el["category_id"])
            areas.append(el["area"])

            if (
                bbox_corrected[0] == bbox_corrected[2]
                or bbox_corrected[1] == bbox_corrected[3]
            ):
                corrupted = True
                break

        if len(target) == 0 or corrupted:
            return None, None, None, None
        else:
            return (
                img,
                torch.FloatTensor(bboxes),
                torch.LongTensor(labels),
                torch.FloatTensor(areas),
            )

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        batch = [(a, b, c, d) for (a, b, c, d) in batch if a is not None]

        images = list()
        boxes = list()
        labels = list()
        areas = list()

        for b in batch:
            images.append(FT.to_tensor(b[0]))
            boxes.append(b[1])
            labels.append(b[2])
            areas.append(b[3])

        targets = list()
        for x in range(len(images)):
            d = dict()

            d["boxes"] = boxes[x]
            d["labels"] = labels[x]
            d["areas"] = areas[x]

            targets.append(d)

        return images, targets

import os

import torch.utils.data
import torchvision
import torchvision.transforms as T

from constants import VOC_CLASSES
from data.COCODataset import COCODataset
from data.VOCDataset import voc_preparations, VOCDataset


def voc_collate_fn(batch):
    images = list()
    boxes = list()
    labels = list()

    for image, annotation in batch:
        images.append(image)

        local_boxes = list()
        local_labels = list()

        for el in annotation["annotation"]["object"]:
            local_boxes.append(el["bndbox"])
            local_labels.append(VOC_CLASSES.index(el["name"]))

        boxes.append(local_boxes)
        labels.append(local_labels)

    targets = []
    images2 = list(image for image in images)

    for x in range(len(images2)):
        dic = {}
        dic["boxes"] = boxes[x]
        dic["labels"] = labels[x]
        targets.append(dic)

    return images2, targets


def voc_data_loaders(required_data_loaders, batch_size, batch_size_test, num_workers):
    # Prepare necessary files
    voc_path = os.path.join("data", "VOC2007")
    if not os.path.exists(os.path.join(voc_path, "train_images.json")):
        print("Generating necessary VOC files...")
        voc_preparations()

    # Generate train data loader
    data_loaders = tuple()
    if "train" in required_data_loaders:
        voc_train_dataset = VOCDataset(split="train")
        voc_train_data_loader = torch.utils.data.DataLoader(
            voc_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=voc_train_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        data_loaders += (voc_train_data_loader,)

    # Generate validation data loader
    if "val" in required_data_loaders:
        voc_val_dataset = VOCDataset(split="val")
        voc_val_data_loader = torch.utils.data.DataLoader(
            voc_val_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=voc_val_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        data_loaders += (voc_val_data_loader,)

    # Generate test data loader
    if "test" in required_data_loaders:
        voc_test_dataset = VOCDataset(split="test")
        voc_test_data_loader = torch.utils.data.DataLoader(
            voc_test_dataset,
            batch_size=batch_size_test,
            shuffle=True,
            collate_fn=voc_test_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        data_loaders += (voc_test_data_loader,)

    return data_loaders


def coco_data_loaders(required_data_loaders, batch_size, batch_size_test, num_workers):
    coco_path = os.path.join("data", "COCO2014")
    data_loaders = tuple()

    # Train dataset
    coco_train_img_file = os.path.join(coco_path, "train2014")
    coco_train_ann_file = os.path.join(
        coco_path, "annotations", "instances_train2014.json"
    )
    coco_train_dataset = COCODataset(
        root=coco_train_img_file, annotation_file=coco_train_ann_file
    )

    # Validation dataset
    coco_val_img_file = os.path.join(coco_path, "val2014")
    coco_val_ann_file = os.path.join(coco_path, "annotations", "instances_val2014.json")
    coco_val_dataset = COCODataset(
        root=coco_val_img_file, annotation_file=coco_val_ann_file
    )

    # Obtain split
    coco_combo_dataset = coco_val_dataset + coco_train_dataset
    coco_minival_1k, coco_minival_4k, coco_train_dataset_final = (
        torch.utils.data.random_split(
            coco_combo_dataset,
            [1000, 4000, 118287],
            generator=torch.Generator().manual_seed(42),
        )
    )
    torch.Generator.initial_seed()

    # Generate data loaders
    if "train" in required_data_loaders:
        coco_train_data_loader = torch.utils.data.DataLoader(
            coco_train_dataset_final,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=COCODataset.collate_fn,
        )
        data_loaders += (coco_train_data_loader,)

    if "val" in required_data_loaders:
        coco_val_data_loader = torch.utils.data.DataLoader(
            coco_minival_1k,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=COCODataset.collate_fn,
        )
        data_loaders += (coco_val_data_loader,)

    if "test" in required_data_loaders:
        coco_test_data_loader = torch.utils.data.DataLoader(
            coco_minival_4k,
            batch_size=batch_size_test,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=COCODataset.collate_fn,
        )
        data_loaders += (coco_test_data_loader,)

    return data_loaders


def get_voc_test_data_loader():
    try:
        dataset = torchvision.datasets.VOCDetection(
            root="data/voc",
            year="2007",
            image_set="test",
            download=False,
            transform=T.ToTensor(),
            # transform/target_transform/transforms
        )
    except RuntimeError:
        dataset = torchvision.datasets.VOCDetection(
            root="data/voc",
            year="2007",
            image_set="test",
            download=True,
            transform=T.ToTensor(),
            # transform/target_transform/transforms
        )

    return torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=voc_collate_fn, pin_memory=True
    )

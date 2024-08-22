import json
import os.path
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset

from constants import VOC_LABEL_MAP


def voc_preparations():
    create_voc_data_list("train")
    create_voc_data_list("val")
    create_voc_data_list("test")


def parse_annotation(annotation_path):
    # Read XML tree
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Prepare variables
    boxes = list()
    labels = list()
    difficulties = list()

    # Iterate through objects and append to lists
    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")

        label = object.find("name").text.lower().strip()
        if label not in VOC_LABEL_MAP:
            continue

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_LABEL_MAP[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}


# subset must be one of train, val, test
def create_voc_data_list(subset: str):
    # Prepare variables
    image_list = list()
    object_list = list()
    n_objects = 0
    root_path = os.path.abspath(os.path.join("data", "VOC2007"))

    # Get ids of images in current subset
    with open(os.path.join(root_path, "ImageSets", "Main", subset + ".txt")) as f:
        ids = f.read().splitlines()

    # Get paths of images and objects in current subset
    for id in ids:
        objects = parse_annotation(os.path.join(root_path, "Annotations", id + ".xml"))
        if len(objects["boxes"]) == 0:
            continue

        n_objects += len(objects["boxes"])
        object_list.append(objects)
        image_list.append(os.path.join(root_path, "JPEGImages", id + ".jpg"))

    assert len(image_list) == len(object_list)

    # Save to files
    with open(os.path.join(root_path, subset + "_images.json"), "w") as f:
        json.dump(image_list, f)
    with open(os.path.join(root_path, subset + "_objects.json"), "w") as f:
        json.dump(object_list, f)
    with open(os.path.join(root_path, subset + "_label_map.json"), "w") as f:
        json.dump(VOC_LABEL_MAP, f)


class VOCDataset(Dataset):
    def __init__(self, split):
        with open(os.path.join("data", "VOC2007", split + "_images.json"), "r") as f:
            self.images = json.load(f)

        with open(os.path.join("data", "VOC2007", split + "_objects.json"), "r") as f:
            self.objects = json.load(f)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, item):
        # Read image
        image = Image.open(self.images[item], mode="r")
        image = image.convert("RGB")

        # Read objects in image
        objects = self.objects[item]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.LongTensor(objects["labels"])
        difficulties = torch.ByteTensor(objects["difficulties"])

        image = FT.to_tensor(image)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for image, box, label, difficulty in batch:
            images.append(image)
            boxes.append(box)
            labels.append(label)
            difficulties.append(difficulty)

        targets = list()
        for i in range(len(images)):
            d = dict()
            d["boxes"] = boxes[i]
            d["labels"] = labels[i]
            d["difficulties"] = difficulties[i]
            targets.append(d)

        return images, targets

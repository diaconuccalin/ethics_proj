import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.nn.functional import pad
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from constants import *
from data.SetTypes import SetTypes


def find_top_k(predictions, boxes, k, device):
    """
    Find the top k highest scoring predictions
    :param predictions: tensor of prediction scores
    :param boxes: tensor of bounding boxes
    :param k: (maximum) number of object to return
    :return predictions2: k amount of highest scoring predictions
    :return boxes2: k amount of bounding boxes corresponding to the highest scoring predictions
    :return labels2: k amount of labels corresponding to the highest scoring predictions
    :return scores2: k amount of scores corresponding to the highest scoring predictions
    """

    if predictions.shape[0] == 0:
        predictions2 = torch.Tensor([]).to(device)
        labels2 = torch.Tensor([]).to(device)
        boxes2 = torch.Tensor([]).to(device)
        scores2 = torch.Tensor([]).to(device)

    else:
        predictions0 = predictions
        scores0 = torch.max(predictions0, dim=1)[0]
        labels0 = torch.argmax(predictions0, dim=1)
        boxes0 = boxes

        sort = torch.argsort(scores0, descending=True)
        boxes1, labels1, scores1, predictions1 = (
            boxes0[sort],
            labels0[sort],
            scores0[sort],
            predictions0[sort],
        )

        boxes2, labels2, scores2, predictions2 = (
            boxes1[:k],
            labels1[:k] + 1,
            scores1[:k],
            predictions1[:k],
        )

    return predictions2, boxes2, labels2, scores2


def find_p_hat(boxes, predictions, bk, lk, S, num_iterations, epsilon, device):
    """
    Compute the knowledge aware predictions, based on the object detector's output and semantic consistency.
    :param boxes: tensor of bounding boxes
    :param predictions: tensor of prediction scores
    :param bk: number of neighbouring bounding boxes to consider for p_hat
    :param lk: number of largest semantic consistent classes to consider for p_hat
    :param S: semantic consistency matrix
    :param num_iterations: number of iterations to calculate p_hat
    :param epsilon: trade-off parameter for traditional detections and knowledge aware detections
    :return p_hat: tensor of knowledge aware predictions
    """

    num_boxes = predictions.shape[0]
    num_classes = predictions.shape[1]

    if num_boxes <= 1:
        return predictions

    if num_boxes <= bk:
        bk = num_boxes - 1

    if num_classes <= lk:
        lk = num_classes

    box_centers = torch.empty(size=(num_boxes, 2), dtype=torch.double).to(device)
    box_centers[:, 0] = ((boxes[:, 2] - boxes[:, 0]) / 2) + boxes[:, 0]
    box_centers[:, 1] = ((boxes[:, 3] - boxes[:, 1]) / 2) + boxes[:, 1]

    box_nearest = torch.empty(size=(num_boxes, bk), dtype=torch.long).to(device)
    for i in range(len(boxes)):
        box_center = box_centers[i]
        distances = torch.sqrt(
            (box_center[0] - box_centers[:, 0]) ** 2
            + (box_center[1] - box_centers[:, 1]) ** 2
        )
        distances[i] = float("inf")
        box_nearest[i] = torch.argsort(distances)[0:bk]

    S_highest = torch.zeros(size=(num_classes, num_classes), dtype=torch.double).to(
        device
    )
    for i in range(len(S)):
        S_args = torch.argsort(S[i])[-lk:]
        S_highest[i, S_args] = S[i, S_args]

    p_hat_init = torch.full(
        size=(num_boxes, num_classes), fill_value=(1 / num_classes), dtype=torch.double
    ).to(device)
    p_hat = p_hat_init
    for i in range(num_iterations):
        p_hat_temp = torch.clone(p_hat)
        for b in range(num_boxes):
            p = predictions[b]
            num = torch.sum(
                torch.mm(S_highest, torch.transpose(p_hat_temp[box_nearest[b]], 0, 1)),
                1,
            )
            denom = torch.sum(S_highest, dim=1) * bk
            p_hat[b] = (1 - epsilon) * torch.squeeze(
                torch.div(num, denom)
            ) + epsilon * p
            p_hat[b] = torch.nan_to_num(p_hat[b])

    return p_hat


def eval_knowledge_graph(
    data_loader,
    model,
    device,
    num_classes,
    bk,
    lk,
    s,
    num_iterations,
    epsilon,
    top_k,
    set_type: SetTypes = SetTypes.NONE,
):
    assert set_type != SetTypes.NONE, "Set type not provided."

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    true_areas = list()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            if len(images) == 0:
                continue

            predictions = model(images)

            for j in range(len(predictions) - 1):
                true_boxes.append(targets[j]["boxes"].to(device))
                true_labels.append(targets[j]["labels"].to(device))

                if set_type == SetTypes.VOC2007:
                    true_difficulties.append(targets[j]["difficulties"].to(device))
                    true_areas.append(torch.zeros(len(targets[j]["boxes"])).to(device))
                elif set_type == SetTypes.COCO2014:
                    true_difficulties.append(
                        torch.zeros(len(targets[j]["boxes"])).to(device)
                    )
                    true_areas.append(targets[j]["areas"].to(device))

            boxes_temp = predictions[0]["boxes"]
            labels_temp = predictions[0]["labels"]
            scores_temp = predictions[0]["scores"]

            new_predictions = torch.zeros((boxes_temp.shape[0], num_classes)).to(device)

            for j in range(new_predictions.shape[0]):
                label = labels_temp[j] - 1
                new_predictions[j, label] = scores_temp[j]

            p_hat = find_p_hat(
                boxes=boxes_temp,
                predictions=new_predictions,
                bk=bk,
                lk=lk,
                s=s,
                num_iterations=num_iterations,
                epsilon=epsilon,
                device=device,
            )

            pred_k, box_k, label_k, score_k = find_top_k(
                predictions=p_hat, boxes=boxes_temp, k=top_k, device=device
            )

            det_boxes.append(box_k)
            det_labels.append(label_k)
            det_scores.append(score_k)

            del predictions
            torch.cuda.empty_cache()

    return (
        det_boxes,
        det_labels,
        det_scores,
        true_boxes,
        true_labels,
        true_difficulties,
        true_areas,
    )


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))

    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


def coco_metrics(
    det_boxes, det_labels, det_scores, true_boxes, true_labels, true_areas, device
):
    n_classes = len(COCO_CLASSES)

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))

    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_areas = torch.cat(true_areas, dim=0)

    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))

    det_images = torch.LongTensor(det_images).to(device)
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    iou = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    classwise_recall = torch.zeros((n_classes - 1), dtype=torch.float)
    classwise_recall_small = torch.zeros((n_classes - 1), dtype=torch.float)
    classwise_recall_medium = torch.zeros((n_classes - 1), dtype=torch.float)
    classwise_recall_large = torch.zeros((n_classes - 1), dtype=torch.float)
    n_all_objects_small = torch.zeros((n_classes - 1), dtype=torch.float)
    n_all_objects_medium = torch.zeros((n_classes - 1), dtype=torch.float)
    n_all_objects_large = torch.zeros((n_classes - 1), dtype=torch.float)
    ap_class = torch.zeros((n_classes - 1), dtype=torch.float)

    # Skip background
    for c in range(1, n_classes):
        ap_iou = torch.zeros(len(iou), dtype=torch.float)
        recall_iou = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_small = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_medium = torch.zeros(len(iou), dtype=torch.float)
        recall_iou_large = torch.zeros(len(iou), dtype=torch.float)

        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_areas = true_areas[true_labels == c]
        n_class_objects = true_class_images.size(0)

        true_class_boxes_detected = torch.zeros(
            (n_class_objects, len(iou)), dtype=torch.uint8
        ).to(device)

        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)

        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(
            det_class_scores, dim=0, descending=True
        )
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(
            (n_class_detections, len(iou)), dtype=torch.float
        ).to(device)
        false_positives = torch.zeros(
            (n_class_detections, len(iou)), dtype=torch.float
        ).to(device)
        tp_small = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(
            device
        )
        tp_medium = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(
            device
        )
        tp_large = torch.zeros((n_class_detections, len(iou)), dtype=torch.float).to(
            device
        )

        n_class_objects_small = 0
        n_class_objects_medium = 0
        n_class_objects_large = 0

        for i in range(len(true_class_areas)):
            if true_class_areas[i] < 32**2:
                n_class_objects_small += 1
            elif true_class_areas[i] > 96**2:
                n_class_objects_large += 1
            else:
                n_class_objects_medium += 1

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_images[d]

            object_boxes = true_class_boxes[true_class_images == this_image]
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[
                true_class_images == this_image
            ][ind]

            for iou_th in range(len(iou)):
                if max_overlap.item() > iou[iou_th]:
                    if true_class_boxes_detected[original_ind, iou_th] == 0:
                        true_positives[d, iou_th] = 1

                        if true_class_areas[original_ind] < 32**2:
                            tp_small[d, iou_th] = 1
                        elif true_class_areas[original_ind] > 96**2:
                            tp_large[d, iou_th] = 1
                        else:
                            tp_medium[d, iou_th] = 1

                        true_class_boxes_detected[original_ind, iou_th] = 1
                    else:
                        false_positives[d, iou_th] = 1
                else:
                    false_positives[d, iou_th] = 1

        n_all_objects_small[c - 1] = n_class_objects_small
        n_all_objects_medium[c - 1] = n_class_objects_medium
        n_all_objects_large[c - 1] = n_class_objects_large

        cumul_tp_all = torch.cumsum(true_positives, dim=0)
        cumul_tp_small = torch.cumsum(tp_small, dim=0)
        cumul_tp_medium = torch.cumsum(tp_medium, dim=0)
        cumul_tp_large = torch.cumsum(tp_large, dim=0)
        cumul_fp_all = torch.cumsum(false_positives, dim=0)

        cumul_tp_all_transpose = torch.transpose(cumul_tp_all, 0, 1)
        cumul_tp_small_transpose = torch.transpose(cumul_tp_small, 0, 1)
        cumul_tp_medium_transpose = torch.transpose(cumul_tp_medium, 0, 1)
        cumul_tp_large_transpose = torch.transpose(cumul_tp_large, 0, 1)
        cumul_fp_all_transpose = torch.transpose(cumul_fp_all, 0, 1)

        cumul_recall_all = torch.zeros(
            (len(iou), n_class_detections), dtype=torch.float
        ).to(device)
        cumul_recall_small = torch.zeros(
            (len(iou), n_class_detections), dtype=torch.float
        ).to(device)
        cumul_recall_medium = torch.zeros(
            (len(iou), n_class_detections), dtype=torch.float
        ).to(device)
        cumul_recall_large = torch.zeros(
            (len(iou), n_class_detections), dtype=torch.float
        ).to(device)
        cumul_precision_all = torch.zeros(
            (len(iou), n_class_detections), dtype=torch.float
        ).to(device)

        for iou_th in range(len(iou)):
            cumul_recall_all[iou_th] = cumul_tp_all_transpose[iou_th] / n_class_objects
            cumul_recall_small[iou_th] = (
                cumul_tp_small_transpose[iou_th] / n_class_objects_small
            )
            cumul_recall_medium[iou_th] = (
                cumul_tp_medium_transpose[iou_th] / n_class_objects_medium
            )
            cumul_recall_large[iou_th] = (
                cumul_tp_large_transpose[iou_th] / n_class_objects_large
            )
            cumul_precision_all[iou_th] = cumul_tp_all_transpose[iou_th] / (
                cumul_tp_all_transpose[iou_th] + cumul_fp_all_transpose[iou_th]
            )

            cumul_recall_all[iou_th][
                cumul_recall_all[iou_th] != cumul_recall_all[iou_th]
            ] = 0

            recall_thresholds = torch.arange(start=0, end=1.01, step=0.01).tolist()
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(
                device
            )
            for i, t in enumerate(recall_thresholds):
                recalls_above_t_all = cumul_recall_all[iou_th] >= t
                if recalls_above_t_all.any():
                    precisions[i] = cumul_precision_all[iou_th][
                        recalls_above_t_all
                    ].max()
                else:
                    precisions[i] = 0.0

            ap_iou[iou_th] = precisions.mean()
            recall_iou[iou_th] = cumul_recall_all[iou_th, -1]
            recall_iou_small[iou_th] = cumul_recall_small[iou_th, -1]
            recall_iou_medium[iou_th] = cumul_recall_medium[iou_th, -1]
            recall_iou_large[iou_th] = cumul_recall_large[iou_th, -1]

        ap_class[c - 1] = ap_iou.mean()
        classwise_recall[c - 1] = recall_iou.mean()
        classwise_recall_small[c - 1] = recall_iou_small.mean()
        classwise_recall_medium[c - 1] = recall_iou_medium.mean()
        classwise_recall_large[c - 1] = recall_iou_large.mean()

    ap_class_corrected = torch.cat(
        [
            ap_class[0:11],
            ap_class[12:25],
            ap_class[26:28],
            ap_class[30:44],
            ap_class[45:65],
            ap_class[66:67],
            ap_class[69:70],
            ap_class[71:82],
            ap_class[83:90],
        ]
    )

    classwise_recall_corrected = torch.cat(
        [
            classwise_recall[0:11],
            classwise_recall[12:25],
            classwise_recall[26:28],
            classwise_recall[30:44],
            classwise_recall[45:65],
            classwise_recall[66:67],
            classwise_recall[69:70],
            classwise_recall[71:82],
            classwise_recall[83:90],
        ]
    )

    classwise_recall_small_corrected = torch.cat(
        [
            classwise_recall_small[0:11],
            classwise_recall_small[12:25],
            classwise_recall_small[26:28],
            classwise_recall_small[30:44],
            classwise_recall_small[45:65],
            classwise_recall_small[66:67],
            classwise_recall_small[69:70],
            classwise_recall_small[71:82],
            classwise_recall_small[83:90],
        ]
    )

    classwise_recall_medium_corrected = torch.cat(
        [
            classwise_recall_medium[0:11],
            classwise_recall_medium[12:25],
            classwise_recall_medium[26:28],
            classwise_recall_medium[30:44],
            classwise_recall_medium[45:65],
            classwise_recall_medium[66:67],
            classwise_recall_medium[69:70],
            classwise_recall_medium[71:82],
            classwise_recall_medium[83:90],
        ]
    )

    classwise_recall_large_corrected = torch.cat(
        [
            classwise_recall_large[0:11],
            classwise_recall_large[12:25],
            classwise_recall_large[26:28],
            classwise_recall_large[30:44],
            classwise_recall_large[45:65],
            classwise_recall_large[66:67],
            classwise_recall_large[69:70],
            classwise_recall_large[71:82],
            classwise_recall_large[83:90],
        ]
    )

    classwise_recall_small_corrected = torch.unsqueeze(
        classwise_recall_small_corrected, dim=1
    )
    classwise_recall_medium_corrected = torch.unsqueeze(
        classwise_recall_medium_corrected, dim=1
    )
    classwise_recall_large_corrected = torch.unsqueeze(
        classwise_recall_large_corrected, dim=1
    )

    classwise_recall_small_corrected = classwise_recall_small_corrected[
        ~torch.any(classwise_recall_small_corrected.isnan(), dim=1)
    ]
    classwise_recall_medium_corrected = classwise_recall_medium_corrected[
        ~torch.any(classwise_recall_medium_corrected.isnan(), dim=1)
    ]
    classwise_recall_large_corrected = classwise_recall_large_corrected[
        ~torch.any(classwise_recall_large_corrected.isnan(), dim=1)
    ]

    all_recall_by_average = classwise_recall_corrected.mean().item()
    recall_small_by_average = classwise_recall_small_corrected.mean().item()
    recall_medium_by_average = classwise_recall_medium_corrected.mean().item()
    recall_large_by_average = classwise_recall_large_corrected.mean().item()

    mean_average_precisioin = ap_class_corrected.mean().item()
    classwise_recall = {
        COCO_REVERSE_LABEL_MAP[c + 1]: v
        for c, v in enumerate(classwise_recall.tolist())
    }
    average_precisions = {
        COCO_REVERSE_LABEL_MAP[c + 1]: v for c, v in enumerate(ap_class.tolist())
    }

    return (
        average_precisions,
        mean_average_precisioin,
        classwise_recall,
        all_recall_by_average,
        recall_small_by_average,
        recall_medium_by_average,
        recall_large_by_average,
    )


def voc_metrics(
    true_labels,
    true_boxes,
    true_difficulties,
    det_labels,
    det_boxes,
    det_scores,
    device,
    set_type: SetTypes = SetTypes.NONE,
):
    assert set_type != SetTypes.NONE, "Set type not defined"
    if set_type == SetTypes.VOC2007:
        n_classes = len(VOC_CLASSES)
    else:
        n_classes = len(COCO_CLASSES)

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_difficulties = torch.cat(true_difficulties, dim=0)

    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)
    classwise_recall = torch.zeros((n_classes - 1), dtype=torch.float)
    total_tp = torch.zeros((n_classes - 1), dtype=torch.float)
    total_fp = torch.zeros((n_classes - 1), dtype=torch.float)
    n_easy_all_objects = (1 - true_difficulties).sum().item()

    # Exclude background
    for c in range(1, n_classes):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_difficulties = true_difficulties[true_labels == c]
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()

        true_class_boxes_detected = torch.zeros(
            (true_class_difficulties.size(0)), dtype=torch.uint8
        ).to(device)

        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(
            det_class_scores, dim=0, descending=True
        )
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_images[d]

            object_boxes = true_class_boxes[true_class_images == this_image]
            object_difficulties = true_class_difficulties[
                true_class_images == this_image
            ]

            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[
                true_class_images == this_image
            ][ind]

            if (
                max_overlap.item() > 0.5
                and object_difficulties[ind] == 0
                and true_class_boxes_detected[original_ind] == 0
            ):
                true_positives[d] = 1
                true_class_boxes_detected[original_ind] = 1
            else:
                false_positives[d] = 1

        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)

        cumul_precision = cumul_true_positives / (
            cumul_true_positives + cumul_false_positives + 1e-10
        )
        cumul_recall = cumul_true_positives / n_easy_class_objects

        classwise_recall[c - 1] = cumul_recall[-1]
        total_tp[c - 1] = cumul_true_positives[-1]
        total_fp[c - 1] = cumul_false_positives[-1]

        recall_thresholds = torch.arange(start=0, end=1.01, step=0.01).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0

        average_precisions[c - 1] = precisions.mean()

    mean_average_precision = average_precisions.mean().item()

    if set_type == SetTypes.VOC2007:
        REVERSE_LABEL_MAP = VOC_REVERSE_LABEL_MAP
    else:
        REVERSE_LABEL_MAP = COCO_REVERSE_LABEL_MAP

    average_precisions = {
        REVERSE_LABEL_MAP[c + 1]: v for c, v in enumerate(average_precisions.tolist())
    }
    classwise_recalls = {
        REVERSE_LABEL_MAP[c + 1]: v for c, v in enumerate(classwise_recall.tolist())
    }

    all_recall = (torch.sum(total_tp, dim=0) / n_easy_all_objects).item()
    all_recall_by_average = classwise_recall.mean().item()

    return (
        average_precisions,
        mean_average_precision,
        classwise_recalls,
        all_recall,
        all_recall_by_average,
    )


def mAP(preds, targets):
    # Compute mean average precision
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, targets)
    map_value = metric.compute()

    return map_value


def preprocess_labels(preds, targets):

    # Determine the maximum number of labels across all images
    max_length = max(
        max(len(pred["labels"]) for pred in preds),
        max(len(true["labels"]) for true in targets),
    )

    # Define a padding value that does not conflict with existing labels
    padding_value = -1

    # Pad predictions and targets to ensure they have the same length
    padded_preds = []
    padded_targets = []

    for pred, true in zip(preds, targets):
        pred_labels = pred["labels"]
        true_labels = true["labels"]

        # Pad the labels to the maximum length
        padded_pred_labels = pad(
            pred_labels, (0, max_length - len(pred_labels)), value=padding_value
        )
        padded_true_labels = pad(
            true_labels, (0, max_length - len(true_labels)), value=padding_value
        )

        padded_preds.extend(padded_pred_labels.tolist())
        padded_targets.extend(padded_true_labels.tolist())

    # Convert lists to tensors
    all_preds_tensor = torch.tensor(padded_preds)
    all_targets_tensor = torch.tensor(padded_targets)

    # Filter out padding values from both predictions and targets
    valid_indices = (all_preds_tensor != padding_value) & (
        all_targets_tensor != padding_value
    )
    all_preds_tensor = all_preds_tensor[valid_indices]
    all_targets_tensor = all_targets_tensor[valid_indices]

    return all_preds_tensor, all_targets_tensor


def accuracy(all_preds_tensor, all_targets_tensor, num_classes):
    results = {}

    # Initialize the Accuracy metric with `average=None` to get per-class accuracy
    acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)

    # Compute the accuracy for each class
    per_class_accuracy = acc(all_preds_tensor, all_targets_tensor)

    # Print the accuracy for each class
    for i, acc in enumerate(per_class_accuracy):
        results[i] = acc.item() * 100

    return results


def conf_matrix(all_preds_tensor, all_targets_tensor, num_classes, save_path):

    # Initialize the ConfusionMatrix metric
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    # Compute the confusion matrix
    confusion_matrix_result = confmat(all_preds_tensor, all_targets_tensor)

    # Print the confusion matrix
    cm = confusion_matrix_result.cpu().numpy()  # Convert to NumPy array for plotting

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)

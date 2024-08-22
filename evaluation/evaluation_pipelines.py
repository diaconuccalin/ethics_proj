import json
import os

import numpy as np
import torch

from constants import VOC_CLASSES
from data.data_loaders import get_voc_test_data_loader
from evaluation.eval_utils import (
    preprocess_labels,
    conf_matrix,
    mAP,
    accuracy,
)
from evaluation.voc_evaluation_loop import (
    evaluate_standard_model,
    evaluate_standard_model_kg,
)
from models.frcnn_models import frcnn_model_pretrained


def voc_evaluation(session_name, device):
    # Define model
    voc_model = frcnn_model_pretrained("VOC", device)
    voc_model.eval()

    # If it doesn't exist, create results directory
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/" + session_name):
        os.makedirs("results/" + session_name)

    # Get VOC test data loader
    voc_dataloader = get_voc_test_data_loader()

    # Evaluate model
    if session_name == "frcnn_standard":
        pred_boxes, pred_labels, pred_scores, true_boxes, true_labels = (
            evaluate_standard_model(voc_model, voc_dataloader, len(VOC_CLASSES), device)
        )
    else:
        if session_name == "frcnn_kg_57":
            kg_path = "data/SemanticConsistencyMatrices/CM_kg_57_info.json"
            key = "KG_VOC_info"
        else:
            kg_path = "data/SemanticConsistencyMatrices/CM_freq_info.json"

            if "all" in session_name:
                key = "KF_All_VOC_info"
            else:
                key = "KF_500_VOC_info"

        if os.path.isfile(kg_path):
            print("Loading knowledge based consistency matrix")
            with open(kg_path, "r") as j:
                info = json.load(j)
            KG_VOC_info = info[key]
            S_KG_57_VOC = np.asarray(KG_VOC_info["S"])
        else:
            print("No matrix available")

        S = torch.from_numpy(S_KG_57_VOC).to(device)

        pred_boxes, pred_labels, pred_scores, true_boxes, true_labels = (
            evaluate_standard_model_kg(
                voc_model,
                voc_dataloader,
                len(VOC_CLASSES),
                device,
                5,
                5,
                S,
                10,
                1.0,
                100,
            )
        )

    # Construct preds and target
    preds = []
    targets = []

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        preds.append(
            dict(
                boxes=box,
                scores=score,
                labels=label.int(),
            )
        )

    for box, label in zip(true_boxes, true_labels):
        targets.append(
            dict(
                boxes=box,
                labels=label.int(),
            )
        )

    # Preprocess labels
    all_preds_tensor, all_targets_tensor = preprocess_labels(preds, targets)

    # Compute confusion matrix
    conf_matrix(
        all_preds_tensor,
        all_targets_tensor,
        20,
        "results/" + session_name + "/confusion_matrix.png",
    )

    # Compute mAP
    map_results = mAP(preds, targets)

    # Compute accuracy
    accuracy_results = accuracy(all_preds_tensor, all_targets_tensor, 20)

    with open("results/" + session_name + "/results.txt", "w") as f:
        for key, value in map_results.items():
            f.write(key + ": " + str(value.numpy()) + "\n")

        f.write("\n\nPer-class accuracy results:\n")

        for key, value in accuracy_results.items():
            f.write(str(key) + ": " + str(value) + "\n")

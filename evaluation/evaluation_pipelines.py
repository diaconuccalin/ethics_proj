import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

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
    evaluate_model_yolo_freq,
)
from models.frcnn_models import frcnn_model_pretrained


def voc_evaluation(session_name, device):
    # Define model
    if "frcnn" in session_name:
        voc_model = frcnn_model_pretrained("VOC", device)
    else:
        voc_model = YOLO("weights/best.pt")
    voc_model.eval()

    # If it doesn't exist, create results directory
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/" + session_name):
        os.makedirs("results/" + session_name)
    if not os.path.exists("results/yolo"):
        os.makedirs("results/yolo")

    # Get VOC test data loader
    voc_dataloader = get_voc_test_data_loader()

    # Evaluate model
    if session_name == "frcnn_standard":
        pred_boxes, pred_labels, pred_scores, true_boxes, true_labels = (
            evaluate_standard_model(voc_model, voc_dataloader, len(VOC_CLASSES), device)
        )
    elif "frcnn" in session_name:
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
    else:
        # Load knowledge graph
        kg_path = "data/SemanticConsistencyMatrices/CM_freq_info.json"
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

        # Prepare evaluation variables
        epsilon_values = [0.7, 0.8, 0.9]
        results_by_epsilon = {}
        map_values_by_epsilon = {}
        accuracy_values_by_epsilon = {}
        confmat_values_by_epsilon = {}
        results_by_epsilon_500 = {}
        map_values_by_epsilon_500 = {}
        accuracy_values_by_epsilon_500 = {}
        confmat_values_by_epsilon_500 = {}

        # Iterate over each epsilon value
        for epsilon in epsilon_values:
            print(f"Evaluating with epsilon = {epsilon}")

            # Call the evaluation function with the current epsilon
            yolo_results_freq = evaluate_model_yolo_freq(
                voc_model,
                voc_dataloader,
                20,
                device,
                5,
                5,
                S,
                10,
                epsilon,
                100,
            )

            # Store the results in the dictionary
            results_by_epsilon[epsilon] = yolo_results_freq

        for epsilon, yolo_results_freq in results_by_epsilon.items():
            # Construct preds and target, and move them to the correct device
            preds_yolo_freq = []
            target_yolo_freq = []

            for result in yolo_results_freq:
                preds_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["pred_boxes"]).to(device),
                        scores=torch.Tensor(result["pred_scores"]).to(device),
                        labels=torch.Tensor(result["pred_labels"]).int().to(device),
                    )
                )

                target_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["true_boxes"]).to(device),
                        labels=torch.Tensor(result["true_labels"]).int().to(device),
                    )
                )

            # Compute mean average precision
            metric = MeanAveragePrecision(class_metrics=True).to(device)
            metric.update(preds_yolo_freq, target_yolo_freq)
            map_value_yolo_freq = metric.compute()
            map_values_by_epsilon[epsilon] = map_value_yolo_freq

        for epsilon, yolo_results_freq in results_by_epsilon.items():
            # Construct preds and target, and move them to the correct device
            preds_yolo_freq = []
            target_yolo_freq = []

            for result in yolo_results_freq:
                preds_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["pred_boxes"]).to(device),
                        scores=torch.Tensor(result["pred_scores"]).to(device),
                        labels=torch.Tensor(result["pred_labels"]).int().to(device),
                    )
                )

                target_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["true_boxes"]).to(device),
                        labels=torch.Tensor(result["true_labels"]).int().to(device),
                    )
                )

            # Determine the maximum number of labels across all images
            max_length = max(
                max(len(pred["labels"]) for pred in preds_yolo_freq),
                max(len(true["labels"]) for true in target_yolo_freq),
            )

            # Define a padding value that does not conflict with existing labels
            padding_value = -1

            # Pad predictions and targets to ensure they have the same length
            padded_preds = []
            padded_targets = []

            for pred, true in zip(preds_yolo_freq, target_yolo_freq):
                pred_labels = pred["labels"]
                true_labels = true["labels"]

                # Pad the labels to the maximum length
                padded_pred_labels = torch.nn.functional.pad(
                    pred_labels, (0, max_length - len(pred_labels)), value=padding_value
                )
                padded_true_labels = torch.nn.functional.pad(
                    true_labels, (0, max_length - len(true_labels)), value=padding_value
                )

                padded_preds.extend(padded_pred_labels.tolist())
                padded_targets.extend(padded_true_labels.tolist())

            # Convert lists to tensors
            all_preds_tensor = torch.tensor(padded_preds).to(device)
            all_targets_tensor = torch.tensor(padded_targets).to(device)

            # Filter out padding values from both predictions and targets
            valid_indices = (all_preds_tensor != padding_value) & (
                all_targets_tensor != padding_value
            )
            all_preds_tensor = all_preds_tensor[valid_indices]
            all_targets_tensor = all_targets_tensor[valid_indices]

            # Determine the number of unique classes, excluding the padding value
            num_classes = 20

            # Initialize the Accuracy metric with `average=None` to get per-class accuracy
            acc = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(
                device
            )

            # Compute the accuracy for each class
            per_class_accuracy = acc(all_preds_tensor, all_targets_tensor)

            # Store the accuracy values for this epsilon
            accuracy_values_by_epsilon[epsilon] = per_class_accuracy
            # Print the accuracy for each class
            print(f"Epsilon = {epsilon}")
            for i, acc in enumerate(per_class_accuracy):
                print(f"Accuracy for class {i}: {acc.item() * 100:.2f}%")

        for epsilon, yolo_results_freq in results_by_epsilon.items():
            # Construct preds and target, and move them to the correct device
            preds_yolo_freq = []
            target_yolo_freq = []

            for result in yolo_results_freq:
                preds_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["pred_boxes"]).to(device),
                        scores=torch.Tensor(result["pred_scores"]).to(device),
                        labels=torch.Tensor(result["pred_labels"]).int().to(device),
                    )
                )

                target_yolo_freq.append(
                    dict(
                        boxes=torch.Tensor(result["true_boxes"]).to(device),
                        labels=torch.Tensor(result["true_labels"]).int().to(device),
                    )
                )

            # Determine the maximum number of labels across all images
            max_length = max(
                max(len(pred["labels"]) for pred in preds_yolo_freq),
                max(len(true["labels"]) for true in target_yolo_freq),
            )

            # Define a padding value that does not conflict with existing labels
            padding_value = -1

            # Pad predictions and targets to ensure they have the same length
            padded_preds = []
            padded_targets = []

            for pred, true in zip(preds_yolo_freq, target_yolo_freq):
                pred_labels = pred["labels"]
                true_labels = true["labels"]

                # Pad the labels to the maximum length
                padded_pred_labels = torch.nn.functional.pad(
                    pred_labels, (0, max_length - len(pred_labels)), value=padding_value
                )
                padded_true_labels = torch.nn.functional.pad(
                    true_labels, (0, max_length - len(true_labels)), value=padding_value
                )

                padded_preds.extend(padded_pred_labels.tolist())
                padded_targets.extend(padded_true_labels.tolist())

            # Convert lists to tensors
            all_preds_tensor = torch.tensor(padded_preds).to(device)
            all_targets_tensor = torch.tensor(padded_targets).to(device)

            # Filter out padding values from both predictions and targets
            valid_indices = (all_preds_tensor != padding_value) & (
                all_targets_tensor != padding_value
            )
            all_preds_tensor = all_preds_tensor[valid_indices]
            all_targets_tensor = all_targets_tensor[valid_indices]

            # Determine the number of unique classes, excluding the padding value
            num_classes = 20

            # Initialize the ConfusionMatrix metric
            confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
                device
            )

            # Compute the confusion matrix
            confusion_matrix_result = confmat(all_preds_tensor, all_targets_tensor)

            # Store the confusion matrix for this epsilon
            confmat_values_by_epsilon[epsilon] = confusion_matrix_result

            # Print the confusion matrix
            cm = (
                confusion_matrix_result.cpu().numpy()
            )  # Convert to NumPy array for plotting

            print(f"Confusion Matrix for epsilon = {epsilon}:\n")
            # Plot the confusion matrix
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.title("Confusion Matrix")
            plt.savefig("results/yolo/conf_mat.png")

        for epsilon, yolo_results_freq in results_by_epsilon_500.items():
            # Construct preds and target, and move them to the correct device
            preds_yolo_freq_500 = []
            target_yolo_freq_500 = []

            for result in yolo_results_freq:
                preds_yolo_freq_500.append(
                    dict(
                        boxes=torch.Tensor(result["pred_boxes"]).to(device),
                        scores=torch.Tensor(result["pred_scores"]).to(device),
                        labels=torch.Tensor(result["pred_labels"]).int().to(device),
                    )
                )

                target_yolo_freq_500.append(
                    dict(
                        boxes=torch.Tensor(result["true_boxes"]).to(device),
                        labels=torch.Tensor(result["true_labels"]).int().to(device),
                    )
                )

            # Compute mean average precision
            metric = MeanAveragePrecision(class_metrics=True).to(device)
            metric.update(preds_yolo_freq_500, target_yolo_freq_500)
            map_value_yolo_freq_500 = metric.compute()
            map_values_by_epsilon_500[epsilon] = map_value_yolo_freq_500

        for epsilon, yolo_results_freq in results_by_epsilon_500.items():
            # Construct preds and target, and move them to the correct device
            preds_yolo_freq_500 = []
            target_yolo_freq_500 = []

            for result in yolo_results_freq:
                preds_yolo_freq_500.append(
                    dict(
                        boxes=torch.Tensor(result["pred_boxes"]).to(device),
                        scores=torch.Tensor(result["pred_scores"]).to(device),
                        labels=torch.Tensor(result["pred_labels"]).int().to(device),
                    )
                )

                target_yolo_freq_500.append(
                    dict(
                        boxes=torch.Tensor(result["true_boxes"]).to(device),
                        labels=torch.Tensor(result["true_labels"]).int().to(device),
                    )
                )

            # Determine the maximum number of labels across all images
            max_length = max(
                max(len(pred["labels"]) for pred in preds_yolo_freq_500),
                max(len(true["labels"]) for true in target_yolo_freq_500),
            )

            # Define a padding value that does not conflict with existing labels
            padding_value = -1

            # Pad predictions and targets to ensure they have the same length
            padded_preds = []
            padded_targets = []

            for pred, true in zip(preds_yolo_freq_500, target_yolo_freq_500):
                pred_labels = pred["labels"]
                true_labels = true["labels"]

                # Pad the labels to the maximum length
                padded_pred_labels = torch.nn.functional.pad(
                    pred_labels, (0, max_length - len(pred_labels)), value=padding_value
                )
                padded_true_labels = torch.nn.functional.pad(
                    true_labels, (0, max_length - len(true_labels)), value=padding_value
                )

                padded_preds.extend(padded_pred_labels.tolist())
                padded_targets.extend(padded_true_labels.tolist())

            # Convert lists to tensors
            all_preds_tensor = torch.tensor(padded_preds).to(device)
            all_targets_tensor = torch.tensor(padded_targets).to(device)

            # Filter out padding values from both predictions and targets
            valid_indices = (all_preds_tensor != padding_value) & (
                all_targets_tensor != padding_value
            )
            all_preds_tensor = all_preds_tensor[valid_indices]
            all_targets_tensor = all_targets_tensor[valid_indices]

            # Determine the number of unique classes, excluding the padding value
            num_classes = 20

            # Initialize the Accuracy metric with `average=None` to get per-class accuracy
            acc = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(
                device
            )

            # Compute the accuracy for each class
            per_class_accuracy = acc(all_preds_tensor, all_targets_tensor)

            # Store the accuracy values for this epsilon
            accuracy_values_by_epsilon_500[epsilon] = per_class_accuracy
            # Print the accuracy for each class
            print(f"Epsilon = {epsilon}")
            for i, acc in enumerate(per_class_accuracy):
                print(f"Accuracy for class {i}: {acc.item() * 100:.2f}%")

            # Initialize the ConfusionMatrix metric
            confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
                device
            )

            # Compute the confusion matrix
            confusion_matrix_result = confmat(all_preds_tensor, all_targets_tensor)

            # Store the confusion matrix for this epsilon
            confmat_values_by_epsilon_500[epsilon] = confusion_matrix_result

            # Print the confusion matrix
            cm = (
                confusion_matrix_result.cpu().numpy()
            )  # Convert to NumPy array for plotting

            # Plot the confusion matrix
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.title("Confusion Matrix")
            plt.savefig("results/yolo/conf_mat_500.png")

        return (
            results_by_epsilon,
            map_values_by_epsilon,
            accuracy_values_by_epsilon,
            confmat_values_by_epsilon,
            results_by_epsilon_500,
            map_values_by_epsilon_500,
            accuracy_values_by_epsilon_500,
            confmat_values_by_epsilon_500,
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

    return None

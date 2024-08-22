import numpy as np
import torch.utils.data
from tqdm import tqdm

from data.SemanticConsistencyMatrices.KnowledgeGraphTypes import KnowledgeGraphTypes
from data.SemanticConsistencyMatrices.knowledge_graph_loader import get_knowledge_graph
from data.SetTypes import SetTypes
from data.VOCDataset import VOCDataset
from evaluation.eval_utils import (
    eval_knowledge_graph,
    voc_metrics,
    find_p_hat,
    find_top_k,
)
from models.ModelTypes import ModelTypes
from models.model_utils import get_eval_model


def voc_evaluation(
    batch_size,
    num_workers,
    num_classes,
    detections_per_image,
    box_score_threshold,
    model_path,
    bk,
    lk,
    num_iterations,
    epsilon,
    top_k,
    device,
    model_type: ModelTypes = ModelTypes.NONE,
    knowledge_graph: KnowledgeGraphTypes = KnowledgeGraphTypes.NONE,
):
    # Create data loader
    test_data_loader = torch.utils.data.DataLoader(
        VOCDataset(split="test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VOCDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get knowledge graph
    s = torch.from_numpy(np.asarray(get_knowledge_graph(knowledge_graph))).to(device)

    # Get model
    model = get_eval_model(
        num_classes=num_classes,
        detections_per_image=detections_per_image,
        box_score_threshold=box_score_threshold,
        model_type=model_type,
    )

    model.to(device)

    # Load model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Perform evaluation
    (
        det_boxes,
        det_labels,
        det_scores,
        true_boxes,
        true_labels,
        true_difficulties,
        true_areas,
    ) = eval_knowledge_graph(
        data_loader=test_data_loader,
        model=model,
        device=device,
        num_classes=num_classes,
        bk=bk,
        lk=lk,
        s=s,
        num_iterations=num_iterations,
        epsilon=epsilon,
        top_k=top_k,
        set_type=SetTypes.VOC2007,
    )

    return voc_metrics(
        true_labels=true_labels,
        true_boxes=true_boxes,
        true_difficulties=true_difficulties,
        det_labels=det_labels,
        det_boxes=det_boxes,
        det_scores=det_scores,
        device=device,
        set_type=SetTypes.VOC2007,
    )


def evaluate_standard_model(model, test_loader, num_classes, device):
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    true_boxes = []
    true_labels = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader):

            # Move to default device.
            images = [im.to(device) for im in images]

            if len(images) == 0:
                continue

            prediction = model(images)

            for p in range(len(prediction)):
                true_boxes.append(
                    torch.Tensor(
                        [list(int(y) for y in i.values()) for i in targets[p]["boxes"]]
                    ).to("cpu")
                )
                true_labels.append(torch.Tensor(targets[p]["labels"]).to("cpu"))

                boxes = prediction[p]["boxes"].to("cpu")
                labels = (prediction[p]["labels"] - 1).to("cpu")
                scores = prediction[p]["scores"].to("cpu")

                pred_boxes.append(boxes)
                pred_labels.append(labels)
                pred_scores.append(scores)

            del prediction
            del images
            torch.cuda.empty_cache()

    return pred_boxes, pred_labels, pred_scores, true_boxes, true_labels


def evaluate_standard_model_kg(
    model, test_loader, num_classes, device, bk, lk, S, num_iters, epsilon, topk
):
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    true_boxes = []
    true_labels = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            # Move images to the device
            images = [im.to(device) for im in images]

            # Skip images with no objects
            if len(images) == 0:
                continue

            # Get predictions from the model
            prediction = model(images)

            for p in range(len(prediction)):
                # Append true boxes and labels
                true_boxes.append(
                    torch.Tensor(
                        [list(int(y) for y in i.values()) for i in targets[p]["boxes"]]
                    ).to("cpu")
                )
                true_labels.append((torch.Tensor(targets[p]["labels"]) + 1).to("cpu"))

                # Get predicted boxes, labels, and scores
                pred_boxes_temp = prediction[p]["boxes"].to(device)
                pred_labels_temp = (prediction[p]["labels"] - 1).to(device)
                pred_scores_temp = prediction[p]["scores"].to(device)

                # Initialize new_predictions tensor for knowledge-aware processing
                new_predictions = torch.zeros(
                    (pred_boxes_temp.shape[0], num_classes)
                ).to(device)
                for l in range(pred_boxes_temp.shape[0]):
                    label = pred_labels_temp[l].item()
                    new_predictions[l, label] = pred_scores_temp[l]

                # Compute knowledge-aware predictions using the consistency matrix
                p_hat = find_p_hat(
                    pred_boxes_temp,
                    new_predictions,
                    bk,
                    lk,
                    S,
                    num_iters,
                    epsilon,
                    device,
                )

                # Find top-k predictions
                predk, boxk, labk, scok = find_top_k(
                    p_hat, pred_boxes_temp, topk, device
                )

                # Append processed predictions
                pred_boxes.append(boxk.to("cpu"))
                pred_labels.append(labk.to("cpu"))
                pred_scores.append(scok.to("cpu"))
        # Clean up memory
        del prediction
        del images
        torch.cuda.empty_cache()

    return pred_boxes, pred_labels, pred_scores, true_boxes, true_labels

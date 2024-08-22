import numpy as np
import torch

from data.SemanticConsistencyMatrices.KnowledgeGraphTypes import KnowledgeGraphTypes
from data.SemanticConsistencyMatrices.knowledge_graph_loader import get_knowledge_graph
from data.SetTypes import SetTypes
from data.data_loaders import coco_data_loaders
from evaluation.eval_utils import eval_knowledge_graph, coco_metrics
from models.ModelTypes import ModelTypes
from models.model_utils import get_eval_model


def coco_evaluation(
    batch_size,
    num_workers,
    num_classes,
    detections_per_image,
    box_score_threshold,
    model_path,
    lk,
    bk,
    num_iterations,
    epsilon,
    top_k,
    device,
    model_type: ModelTypes = ModelTypes.NONE,
    knowledge_graph: KnowledgeGraphTypes = KnowledgeGraphTypes.NONE,
):
    # Create data loader
    test_data_loader = coco_data_loaders(
        required_data_loaders=["test"],
        batch_size=0,
        batch_size_test=batch_size,
        num_workers=num_workers,
    )[0]

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
    det_boxes, det_labels, det_scores, true_boxes, true_labels, _, true_areas = (
        eval_knowledge_graph(
            data_loader=test_data_loader,
            model=model,
            device=device,
            num_classes=num_classes - 1,
            bk=bk,
            lk=lk,
            s=s,
            num_iterations=num_iterations,
            epsilon=epsilon,
            top_k=top_k,
            set_type=SetTypes.COCO2014,
        )
    )

    return coco_metrics(
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_areas, device
    )

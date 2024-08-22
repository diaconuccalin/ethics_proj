import torch

from constants import VOC_CLASSES
from data.SemanticConsistencyMatrices.KnowledgeGraphTypes import KnowledgeGraphTypes
from evaluation.voc_evaluation_loop import voc_evaluation
from models.ModelTypes import ModelTypes

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    voc_evaluation(
        batch_size=1,
        num_workers=1,
        num_classes=len(VOC_CLASSES) + 1,
        detections_per_image=500,
        box_score_threshold=1e-5,
        model_path="weights/voc-FRCNN-vgg16.pth",
        bk=5,
        lk=5,
        num_iterations=10,
        epsilon=1.0,
        top_k=100,
        device=device,
        model_type=ModelTypes.VGG16,
        knowledge_graph=KnowledgeGraphTypes.KF_ALL_COCO,
    )

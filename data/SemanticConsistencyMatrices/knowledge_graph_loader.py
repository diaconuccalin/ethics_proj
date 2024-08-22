import json
import os

from data.SemanticConsistencyMatrices.KnowledgeGraphTypes import KnowledgeGraphTypes


def get_knowledge_graph(
    knowledge_graph: KnowledgeGraphTypes = KnowledgeGraphTypes.NONE,
):
    assert (
        knowledge_graph != KnowledgeGraphTypes.NONE
    ), "No knowledge graph type provided."

    if knowledge_graph == KnowledgeGraphTypes.KF_ALL_COCO:
        matrix_path = os.path.join(
            "data", "SemanticConsistencyMatrices", "CM_freq_info.json"
        )
        with open(matrix_path, "r") as f:
            info = json.load(f)
        kg_info = info["KF_All_COCO_info"]
    elif knowledge_graph == KnowledgeGraphTypes.KF_500_COCO:
        matrix_path = os.path.join(
            "data", "SemanticConsistencyMatrices", "CM_freq_info.json"
        )
        with open(matrix_path, "r") as f:
            info = json.load(f)
        kg_info = info["KF_500_COCO_info"]
    elif knowledge_graph == KnowledgeGraphTypes.KG_CNET_57_COCO:
        matrix_path = os.path.join(
            "data", "SemanticConsistencyMatrices", "CM_kg_57_info.json"
        )
        with open(matrix_path, "r") as f:
            info = json.load(f)
        kg_info = info["KG_COCO_info"]
    elif knowledge_graph == KnowledgeGraphTypes.KG_CNET_55_COCO:
        matrix_path = os.path.join(
            "data", "SemanticConsistencyMatrices", "CM_kg_55_info.json"
        )
        with open(matrix_path, "r") as f:
            info = json.load(f)
        kg_info = info["KG_COCO_info"]
    else:
        kg_info = None

    return kg_info["S"]

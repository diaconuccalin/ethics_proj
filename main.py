import torch

from evaluation.evaluation_pipelines import voc_evaluation


def main(session_name, device):
    return voc_evaluation(session_name, device)


if __name__ == "__main__":
    # Set cuda device
    cuda_device = None
    if torch.cuda.is_available():
        if cuda_device is None:
            cuda_device = torch.cuda.device_count() - 1
        device = torch.device("cuda:" + str(cuda_device))
        print("DEVICE SET TO GPU " + str(cuda_device) + "!\n")
    else:
        print("DEVICE SET TO CPU!\n")
        device = torch.device("cpu")

    # main("frcnn_standard", device)
    # main("frcnn_kg_57", device)
    # main("frcnn_freq_info_all", device)
    # main("frcnn_freq_info_500", device)
    results = main("yolo", device)
    with open("results/yolo_results.txt", "w") as f:
        for el in results:
            for key, value in el.items():
                f.write(str(key) + " " + str(value) + "\n")

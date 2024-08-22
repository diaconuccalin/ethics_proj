import torch

from evaluation.evaluation_pipelines import voc_evaluation


def main(session_name):
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

    voc_evaluation(session_name, device)


if __name__ == "__main__":
    main("frcnn_standard")
    main("frcnn_kg_57")
    main("frcnn_freq_info_all")
    main("frcnn_freq_info_500")

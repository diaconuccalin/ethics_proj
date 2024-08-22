import torch

from data.SetTypes import SetTypes
from data.data_loaders import voc_data_loaders, coco_data_loaders
from models.model_utils import get_train_model
from training.training_loop import train_epoch


def training_pipeline(
    model_type,
    batch_size,
    batch_size_test,
    num_workers,
    lr,
    lr_decay,
    momentum,
    weight_decay,
    session_name,
    device,
    set_name: SetTypes = SetTypes.NONE,
):
    assert set_name != SetTypes.NONE, "Set type not provided."
    if set_name == SetTypes.VOC2007:
        epochs = 48
        epochs_decay = 40
        num_classes = 21

        train_loader = voc_data_loaders(
            required_data_loaders=["train"],
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            num_workers=num_workers,
        )[0]
    elif set_name == SetTypes.COCO2014:
        epochs = 8
        epochs_decay = 6
        num_classes = 92

        train_loader = coco_data_loaders(
            required_data_loaders=["train"],
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            num_workers=num_workers,
        )[0]
    else:
        num_classes = None
        epochs = None
        epochs_decay = None
        train_loader = None

    model = get_train_model(num_classes=num_classes, model_type=model_type)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    model.to(device)
    model.train()

    for e in range(epochs):
        print("Epoch:", e + 1)

        if e >= epochs_decay:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr_decay,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=e,
            session_name=session_name,
            device=device,
        )

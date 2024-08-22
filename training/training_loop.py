import os.path

import torch


def train_epoch(train_loader, model, optimizer, epoch, session_name, device):
    dir_path = "weights"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    losses_list = list()
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        if len(targets) == 0:
            continue

        # Forward
        output = model(images, targets)
        losses = sum(loss for loss in output.values())

        # Backprop
        optimizer.zero_grad()
        losses.backward()

        # Update model
        optimizer.step()

        losses_list.append(losses.item())

        if i % 100 == 0:
            print(
                "Epoch {epoch} [{current_el}/{total_len}]\tLoss {loss}\t".format(
                    epoch=epoch, current_el=i, total_len=len(train_loader), loss=losses
                )
            )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": losses_list,
        },
        os.path.join(
            dir_path, "checkpoint_" + session_name + "_e" + str(epoch) + ".pth"
        ),
    )

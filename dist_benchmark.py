import logging
import click

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def test(rank, world_size, data_root, batch_size, epochs):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    logging.info("Create network ... ")
    model = torchvision.models.resnet50(pretrained=False).to(rank)
    logging.info("ok")
    
    model = DistributedDataParallel(model, device_ids=[rank])

    # define loss function and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # data set
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    is_valid = lambda x: True if x.endswith(".JPEG") else False
    logging.info("Prepare dataset ... ")
    dataset = torchvision.datasets.ImageFolder(data_root, transform=transform, is_valid_file=is_valid)
    logging.info("ok")

    sampler = DistributedSampler(dataset, shuffle=True, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, sampler=sampler)
    

    for e in range(epochs):
        sampler.set_epoch(e)
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(rank), data[1].to(rank)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i > 10:
                break


@click.command()
@click.option("--data_root", default="/home/vidnerova/image_net/raw-data/train")
@click.option("--batch_size", default=64)
def main(data_root, batch_size):
    world_size = 2
    epochs = 1
    mp.spawn(
        test,
        args=(world_size, data_root, batch_size, epochs),
        nprocs=world_size,
        join=True
    )



if __name__ == "__main__":
    main() 

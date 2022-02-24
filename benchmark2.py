import click
import io
import os
import shutil

import tqdm
import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader


def work(model, data_loader, epochs, device):
    for epoch in range(epochs):
        data_loader.batch_sampler.sampler.set_epoch(epoch)

        model.train()
        
        with tqdm(total=len(data_loader)) as t:
            for i, (images, target) in enumerate(data_loader):

                images = images.cuda(device, non_blocking=True)
                target = target.cuda(device, non_blocking=True)
                
                output = model(images)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.update()

    
@click.command()
@click.option("--data_root", default="./data/")
@click.option("--workers", default=1) 
@click.option("--epochs", default=1)
@click.option("--batch_size", default=64) # per worker, GPU
@click.option("--lr", default=1e-3)
@click.option("--momentum", default=0.9) 
@click.option("--weight_decay", default=1e-4) 
def main(data_root, workers, epochs, batch_size, lr, momentum, weight_decay):
    
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"Set cuda device to {device_id}")

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )

    model = model.resnet50(pretrained=False)
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    
    transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    is_valid = lambda x: True if x.endswith(".JPEG") else False
    print("Prepare dataset ... ", end="", flush=True)
    dataset = torchvision.datasets.ImageFolder(data_root, transform=transform,
                                               is_valid_file=is_valid)
    data_sampler = ElasticDistributedSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers,
                             pin_memory=True, sampler=data_sampler)
    print("ok", flush=True)
    
    work(model, data_loader, epochs, device_id)




if __name__ == "__main__":
    main()

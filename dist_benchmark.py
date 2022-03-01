import os
import click

from tqdm import tqdm 
import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group() 


def real_work(rank, world_size, model, dataloader, sampler, optimizer, criterion, epochs):
    for e in range(epochs):
        sampler.set_epoch(e)
        total = len(dataloader)
        if rank == 0:
            pbar = tqdm(total=len(dataloader))
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(rank), data[1].to(rank)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:
                pbar.update()
        if rank == 0:
            pbar.close()
                
        dist.barrier()
        if rank == 0:
            losses = [None for _ in range(world_size)]
            dist.gather_object(
                loss,
                losses
            )
        else:
            dist.gather_object(loss, None)
        if rank == 0:
            print(f"Epoch {e} finished, loss: {losses}") 
    

def test(rank, world_size, data_root, batch_size, epochs):
    setup(rank, world_size)
    
    print("Create network ... ", end="", flush=True)
    model = torchvision.models.resnet50(pretrained=False).to(rank)
    print("ok", flush=True)
    
    model = DistributedDataParallel(model, device_ids=[rank])

    # define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # data set
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    is_valid = lambda x: True if x.endswith(".JPEG") else False
    print("Prepare dataset ... ", end="", flush=True)
    dataset = torchvision.datasets.ImageFolder(data_root, transform=transform, is_valid_file=is_valid)
    print("ok")

    sampler = DistributedSampler(dataset, shuffle=True, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, sampler=sampler)


    num_threads = 1
    stmt = "real_work(rank, world_size, model, dataloader, sampler, optimizer, criterion, epochs)"
    timer = benchmark.Timer(stmt=stmt, 
                            setup="from __main__ import real_work",
                            globals={
                                "rank": rank,
                                "world_size": world_size,
                                "model": model,
                                "sampler": sampler,
                                "epochs": epochs,
                                "optimizer": optimizer,
                                "criterion": criterion,
                                "dataloader": dataloader,
                            },
                            num_threads=num_threads,
                            label="UI Benchmark",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(1)

    dist.barrier()
    if rank == 0:
        results = [None for _ in range(world_size)]
        dist.gather_object(
            profile_result.mean,
            results
        )
    else:
        dist.gather_object(profile_result.mean, None)

    if rank == 0:
        print("Benchmark result:", results) 
        
    cleanup()


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

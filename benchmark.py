import click 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision
import torch.utils.benchmark as benchmark
import torchvision.transforms as transforms 

from dataset import BenchmarkDataset

def run_net(model, epochs, optimizer, criterion, dataloader, device):

    NUM_BATCHES = 5000
    
    for e in range(epochs):

        with tqdm(total=len(dataloader)) as t:
            for i, data in enumerate(dataloader):
                if i > NUM_BATCHES:
                    break
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                t.update()
            
        print(f"Epoch {e} finished") 
            
        
@click.command()
@click.option('--data_root', default="./image_net/")
@click.option('--batch_size', default=64)
@click.option('--num_epochs', default=1)
@click.option('--num_repeats', default=1)
@click.option('--single_gpu/--double_gpu', default=True, is_flag=True)
@click.option('--use_tfrecord', default=False, is_flag=True)
def main(data_root, batch_size, num_epochs, num_repeats, single_gpu, use_tfrecord):

    if single_gpu:
        print("Running on *ONE* GPU.")
    else:
        print("Running on *TWO* GPUs.")
        
    device = torch.device("cuda")

    print("Create network ... ", end="", flush=True)
    model = torchvision.models.resnet50(pretrained=False)
    print("ok", flush=True)

    if not single_gpu:
        model = nn.DataParallel(model, device_ids=[0,1])
    
    model.to(device)
    #    model.eval()

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    
    if use_tfrecord:

        # record_patern = data_root + "/train-{}-of-01024"
        # description = {"image/encoded": "byte", "image/class/label": "int"}
        # splits = {
        #     str(x).zfill(5): 1.0
        #     for x in range(1024)
        # }
        dataset = BenchmarkDataset(data_root)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        # Data set
        transform = transforms.Compose(
            [
                transforms.Resize(size=(224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        is_valid = lambda x: True if x.endswith(".JPEG") else False
        print("Prepare dataset ... ", end="", flush=True)
        dataset = torchvision.datasets.ImageFolder(data_root, transform=transform, is_valid_file=is_valid)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                 shuffle=True)
    print("ok", flush=True)
        

    print("Latency Measurement (Using PyTorch Benchmark)", flush=True)
    print(" ... ", flush=True)
    num_threads = 1
    timer = benchmark.Timer(stmt="run_net(model, epochs, optimizer, loss, dataloader, device)",
                            setup="from __main__ import run_net",
                            globals={
                                "model": model,
                                "epochs": num_epochs,
                                "optimizer": optimizer,
                                "loss": loss,
                                "dataloader": dataloader,
                                "device": device,
                            },
                            num_threads=num_threads,
                            label="UI Benchmark",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    print(f"Latency: {profile_result.mean :.5f} s")


if __name__ == "__main__":

    main()

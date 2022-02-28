import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel


def test(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = nn.Linear(10, 10).to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()


def main():
    world_size = 2
    mp.spawn(
        test,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )



if __name__ == "__main__":
    main() 

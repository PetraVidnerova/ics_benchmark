from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision
import torch.utils.benchmark as benchmark
import torchvision.transforms as transforms 

def run_net(model, optimizer, criterion, dataloader, device):

    epochs = 10
    
    for e in range(epochs):

        with tqdm(total=len(dataloader)) as t:
            for i, data in enumerate(dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                t.update()
            
        print(f"Epoch {e} finished") 
            
        

def main():

    num_warmups = 1
    num_repeats = 1
    input_shape = (1, 3, 224, 224)

    device = torch.device("cuda")

    model = torchvision.models.resnet50(pretrained=False)
    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(3, 6, 5)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(6, 16, 5)
    #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, 10)

    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = torch.flatten(x, 1) # flatten all dimensions except batch
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
        
    # model = Net()


#model = nn.Conv2d(in_channels=input_shape[1],
    #                  out_channels=256,
    #                  kernel_size=(5, 5))

    model.to(device)
#    model.eval()

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    
    
    # Data set
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    is_valid = lambda x: True if x.endswith(".JPEG") else False
    dataset = torchvision.datasets.ImageFolder("/opt/data/image_net/raw-data/train/", transform=transform, is_valid_file=is_valid)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True)

        

    print("Latency Measurement (Using PyTorch Benchmark) ... ", flush=True, end="")
    num_threads = 1
    timer = benchmark.Timer(stmt="run_net(model, optimizer, loss, dataloader, device)",
                            setup="from __main__ import run_net",
                            globals={
                                "model": model,
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

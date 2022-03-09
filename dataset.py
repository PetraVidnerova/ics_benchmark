import torch
from torch.utils.data import IterableDataset
from tfrecord.torch.dataset import TFRecordDataset
import cv2

def decode_image(features):
    features["image/encoded"] = cv2.imdecode(features["image/encoded"], -1)
    features["image/encoded"].resize((3,224,224))
    return features["image/encoded"].astype("float32"), features["image/class/label"][0]


class BenchmarkDataset(IterableDataset):
    def __init__(self, data_root):
        self.data_root = data_root            
        self.length = 6400

        
    def load(self):
        description = {"image/encoded": "byte", "image/class/label": "int"}
        self.dataset = iter(TFRecordDataset(
            self.data_root + f"/train-{str(self.current).zfill(5)}-of-01024",
            None,
            description,
            transform=decode_image
        ))

        self.current += 1

            
    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        self.current = 0
        self.load()
        return self
        
    def __next__(self):
        self.index += 1 
        if self.index > self.length:
            raise StopIteration
        while True:
            try:
                return next(self.dataset)
            except StopIteration:
                self.load()
            

# tfrecord_path = "/tmp/data.tfrecord"
# index_path = None
# description = {"image": "byte", "label": "float"}
# dataset = TFRecordDataset(tfrecord_path, index_path, description)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# data = next(iter(loader))
# print(data)

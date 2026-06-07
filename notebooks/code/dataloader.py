import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 
from typing import Optional, Callable, Any, Union, List, Tuple
from pathlib import Path
from PIL import Image

RESIZE = (224,224)
PER_CHANNEL_MEAN = [0.485, 0.456, 0.406]
PER_CHANNEL_STD = [0.229, 0.224, 0.225]

#https://github.com/pytorch/vision/blob/9bf794dd6f786b869010f8df2a4fc6886108f03d/torchvision/datasets/folder.py#L287
class TwoDYogaDataset(ImageFolder):
    def __init__(self, root: Union[str, Path], 
                 transform: Optional[Callable] = None,):
        
        if transform == None:
            # Fallback to default imagenet vals
            transform = transforms.Compose([
                transforms.Resize(RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=PER_CHANNEL_MEAN, std=PER_CHANNEL_STD)
            ])

        super().__init__(
            root,
            transform=transform,
        )

        # This is a list of tuples with directly accesible paths and indices
        self.pathlist: List[Tuple[str,int]] = self.imgs 

def make_loader(ds: TwoDYogaDataset,
                shuffle: bool,
                batch_size: int = 32,
                num_workers: int = 4):

    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

#load dataset
dataset_path = '/fhome/vis3d02/dataset'
dataset_yoga = TwoDYogaDataset(dataset_path)

print(dataset_yoga.pathlist[0])

#create dataloader
train_loader = make_loader(dataset_yoga, shuffle=True)

#verify number of poses (example command)
print(f"Detected poses: {dataset_yoga.classes}")

img,labels = next(iter(train_loader))
for i in range(len(labels)):
    class_idx = labels[i].item()
    class_name = dataset_yoga.classes[class_idx]
    print(f"Sample [{i}]:")
    print(f"  - Pose Name: {class_name}")





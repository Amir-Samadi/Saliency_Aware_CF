from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import json
import glob
import numpy as np
import torchvision.datasets as datasets
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')
    

class myMNIST (data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False):
        self.mnistData = torch.utils.data.ConcatDataset([datasets.MNIST(root, True, transform, target_transform, download),
                                   datasets.MNIST(root, False, transform, target_transform, download)])
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.data = list(self.mnistData[index])
        # print(self.data[0].shape)
        # self.data[0] = self.data[0].unsqueeze(0)
        self.data[0] = self.data[0].repeat(3, 1, 1)
        self.data[1] = F.one_hot(torch.tensor(self.data[1]),\
                                 num_classes=len(self.mnistData.datasets[0].classes))
        return self.data[0], self.data[1].type(torch.float32)
    
    def __len__(self):
        """Return the number of images."""
        return len(self.mnistData)


class BDDDataset(torch.utils.data.Dataset):
    def __init__(self, imageRoot, gtRoot, transform=None, augment=False):

        super(BDDDataset, self).__init__()
        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.augment = augment
        self.transform = transform

        with open(gtRoot) as json_file:
            data = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        
        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = os.path.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        target = np.array(self.targets[ind], dtype=np.int64)

        image = self.transform(raw_image) 

        # Create a black image with the target size
        # padded_image = torch.zeros((image.size(0), max(image.size()), max(image.size())), dtype=torch.float32)
        
        # # Calculate the number of rows/columns to add as padding
        # new_height = int((max(image.size()) -  image.size(1))/2)
        # new_width  =  int((max(image.size()) - image.size(2))/2)
        
        # # Add the resized image to the padded image, with padding on the left and right sides
        # padded_image[:, new_height:image.size(1)+new_height, new_width:image.size(2)+new_width] = image
    
        target = torch.FloatTensor(target)[0:4]
        return image, target

class BDD100kDataset(torch.utils.data.Dataset):
    def __init__(self, imageRoot, gtRoot, transform=None, augment=False):

        super(BDD100kDataset, self).__init__()
        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.augment = augment
        self.transform = transform
        
        with open(gtRoot) as json_file:
            self.imgNames = json.load(json_file)

        # self.imgNames = glob.glob(self.imageRoot+*.jpg")
        # self.imgNames = glob.glob("dataset/bdd100k/train/*.jpg") + glob.glob("dataset/bdd100k/test/*.jpg")
        # self.imgNames = glob.glob("dataset/bdd100k/val/*.jpg")
        # with open('dataset/bdd100k/labels/val/bdd100k_images_val.json', 'w') as f:
        #     json.dump(self.imgNames, f)
        
        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))


    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        
        image = self.transform(raw_image) 
        # Create a black image with the target size
        # padded_image = torch.zeros((image.size(0), max(image.size()), max(image.size())), dtype=torch.float32)
        
        # # Calculate the number of rows/columns to add as padding
        # new_height = int((max(image.size()) -  image.size(1))/2)
        # new_width  =  int((max(image.size()) - image.size(2))/2)
        
        # # Add the resized image to the padded image, with padding on the left and right sides
        # padded_image[:, new_height:image.size(1)+new_height, new_width:image.size(2)+new_width] = image
    
        
        return image, 0
    
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, image_size, crop_size=178, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, image_root=None, gt_root_train=None):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    if (dataset in ['BDD', 'BDD100k']):
        # transform.append(T.Pad(image_size[1]))
        # transform.append(SquarePad()),
        # transform.append(T.CenterCrop((image_size[1],crop_size[0])))
        transform.append(T.Resize(image_size))
    else:
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    if (dataset=='MNIST'):
        transform.append(T.Normalize(mean=(0.5), std=(0.5)))
    else:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    elif dataset == 'BDD':
        dataset = BDDDataset(   imageRoot = image_root,
                                gtRoot = gt_root_train,
                                transform = transform
                            )
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=(mode=='train'),
                                                   num_workers=num_workers, drop_last=True)
    elif dataset == 'BDD100k':
        dataset = BDD100kDataset(   imageRoot = image_root,
                                gtRoot = gt_root_train,
                                transform = transform
                            )
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=(mode=='train'),
                                                   num_workers=num_workers, drop_last=True)
    elif dataset == 'MNIST':
        dataset = myMNIST(root=image_dir, download=True, transform=transform)
        # dataset ["image"], dataset ["target"] = dataset.data, dataset.targets.unsqueeze(dim=1)
        # dataset = { "image": dataset.data,
        #             "target":dataset.targets.unsqueeze(dim=1)}
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),
                                                    num_workers=num_workers, drop_last=True)
    return data_loader



def decision_model_get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, image_root=None, gt_root_train=None):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop((crop_size[1],crop_size[0])))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
        

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
        
        train_length=int(0.9*len(dataset))
        test_length=len(dataset)-train_length
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset,(train_length,test_length))
        
        data_loader_train = data.DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
        
        data_loader_val = data.DataLoader(dataset=dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

        train_length=int(0.9*len(dataset))
        test_length=len(dataset)-train_length
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset,(train_length,test_length))
        
        data_loader_train = data.DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
        
        data_loader_val = data.DataLoader(dataset=dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True)
        
    elif dataset == 'BDD':
        dataset = BDDDataset(   imageRoot = image_root,
                                gtRoot = gt_root_train,
                                transform = transform
                            )
        
        train_length=int(0.9*len(dataset))
        test_length=len(dataset)-train_length
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset,(train_length,test_length))
        
        data_loader_train = data.DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
        
        data_loader_val = data.DataLoader(dataset=dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True)

    elif dataset == 'BDD100k':
        dataset = BDD100kDataset(   imageRoot = image_root,
                                gtRoot = gt_root_train,
                                transform = transform
                            )
        
        train_length=int(0.9*len(dataset))
        test_length=len(dataset)-train_length
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset,(train_length,test_length))
        
        data_loader_train = data.DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
        
        data_loader_val = data.DataLoader(dataset=dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True)

    return data_loader_train, data_loader_val
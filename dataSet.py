import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import os

def preprocess():
    transform_train = transform.Compose([
        transform.Resize(224),
        transform.RandomCrop(224, padding=4),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    transform_val = transform.Compose([
        transform.Resize(224),
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    return transform_train,transform_val

def load_data(args):
    dataModel=args.data_model
    data_path=args.data_path
    batch_size=args.batch_size
    train_path = os.path.join(data_path,"train")
    val_path=os.path.join(data_path,"test")
    assert dataModel in ["Cifar10","Cifar100","ImageNet"]
    if dataModel=="Cifar10":
        transform_train,transform_val=preprocess()

        traindataSet = torchvision.datasets.CIFAR10(train_path, train=True,
                                                    download=False, transform=transform_train)
        train_loader = DataLoader(traindataSet, batch_size=batch_size, shuffle=True)

        valdataSet = torchvision.datasets.CIFAR10(val_path, train=False, download=False,
                                                  transform=transform_val)
        val_loader = DataLoader(valdataSet, batch_size=batch_size, shuffle=True)

        return train_loader,val_loader
    elif dataModel=="Cifar100":
        transform_train, transform_val = preprocess()

        traindataSet = torchvision.datasets.CIFAR10(train_path, train=True,
                                                    download=False, transform=transform_train)
        train_loader = DataLoader(traindataSet, batch_size=batch_size, shuffle=True)

        valdataSet = torchvision.datasets.CIFAR10(val_path, train=False, download=False,
                                                  transform=transform_val)
        val_loader = DataLoader(valdataSet, batch_size=batch_size, shuffle=True)

        return train_loader,val_loader
    elif dataModel=="ImageNet":
        transform_train, transform_val = preprocess()
        traindataSet=torchvision.datasets.ImageFolder(train_path,transform=transform_train)
        valdataSet=torchvision.datasets.ImageFolder(val_path,transform=transform_val)
        train_loader = DataLoader(traindataSet, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valdataSet, batch_size=batch_size, shuffle=True)

        return train_loader,val_loader
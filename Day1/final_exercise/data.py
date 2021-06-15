import torch
from torchvision import datasets, transforms

def mnist():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                   ])
    
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    return train, test
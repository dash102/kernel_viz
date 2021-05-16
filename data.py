import torchvision
import torchvision.transforms as transforms

def get_data(data_name):
    if data_name == 'fashion':
        return torchvision.datasets.FashionMNIST('./data',
                download=True,
                train=True,
                transform=transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))]))
    if data_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10('./data',
                download=True,
                train=True,
                transform=transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))]))

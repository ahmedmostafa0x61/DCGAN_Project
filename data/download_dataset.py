import torchvision.datasets as datasets
import torchvision.transforms as transforms


def download_cifar10(data_path='./data'):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.CIFAR10(root=data_path,train=True,download=True,transform=transform)
    return dataset


if __name__ == '__main__':
    dataset = download_cifar10()
    print('Dataset Download and transformed Successfully!')



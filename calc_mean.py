# 用于计算图片色素均值，用于归一化

from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
from tqdm import tqdm

transform = T.Compose([
     T.RandomResizedCrop(224),
     T.ToTensor(),
])

def getStat(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean() # N, C, H ,W
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'enhance_dataset', transform=transform)
    print(getStat(train_dataset))
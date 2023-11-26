import torch
from PIL import Image

import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset

# 数据归一化与标准化
transform_BZ= transforms.Normalize(
    mean=[0.4666638, 0.43481198, 0.38117275],# 取决于数据集
    std=[0.21779788, 0.2145783, 0.2144127]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.img_size = 512

        self.train_tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                transform_BZ#图片标准化的步骤
            ])
        self.val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transform_BZ#标准化操作
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info#返回图片信息

    def padding_black(self, img):   # 如果尺寸太小可以扩充
        w, h  = img.size
        scale = self.img_size / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = self.img_size
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):#返回真正想返回的东西
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)#打开图片
        img = img.convert('RGB')#转换为RGB 格式
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print("image.shape = ", image.shape)
        # print("image = ",image)
        print("label = ",label)


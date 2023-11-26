'''
    单图测试
'''

import torch
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms as transforms
import os

transform_BZ= transforms.Normalize(
    mean=[0.4666638, 0.43481198, 0.38117275],  # 取决于数据集
    std=[0.21779788, 0.2145783, 0.2144127]
)


def padding_black(img,img_size = 512):  # 如果尺寸太小可以扩充
    w, h = img.size
    scale = img_size / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = img_size
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img

def predict():
    img_path = r'images/target.png'

    val_tf = transforms.Compose([  ##简单把图片压缩了变成Tensor模式
        transforms.Resize(512),
        transforms.ToTensor(),
        transform_BZ  # 标准化操作
    ])

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    finetune_net = resnet18(num_classes=5).to(device)

    state_dict = torch.load(r"output/resnet18_e_best.pth")
    # print("state_dict = ",state_dict)
    finetune_net.load_state_dict(state_dict)
    finetune_net.eval()
    with torch.no_grad():

        # finetune_net.to(device)
        img = Image.open(img_path)  # 打开图片
        img = img.convert('RGB')  # 转换为RGB 格式
        img = padding_black(img)
        img = val_tf(img)
        img_tensor = torch.unsqueeze(img, 0)  # N,C,H,W, ; C,H,W

        img_tensor = img_tensor.to(device)
        result = finetune_net(img_tensor)
        # print("result = ",result.argmax(1))

        id = result.argmax(1).item()
        return id

if __name__=='__main__':

    img_path = r'horse.png'

    val_tf = transforms.Compose([  ##简单把图片压缩了变成Tensor模式
        transforms.Resize(512),
        transforms.ToTensor(),
        transform_BZ  # 标准化操作
    ])



    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    finetune_net = resnet18(num_classes=5).to(device)

    state_dict = torch.load(r"output/resnet18_e_best.pth")
    # print("state_dict = ",state_dict)
    finetune_net.load_state_dict(state_dict)
    finetune_net.eval()
    with torch.no_grad():

        # finetune_net.to(device)
        img = Image.open(img_path)  # 打开图片
        img = img.convert('RGB')  # 转换为RGB 格式
        img = padding_black(img)
        img = val_tf(img)
        img_tensor = torch.unsqueeze(img, 0)    # N,C,H,W, ; C,H,W

        img_tensor = img_tensor.to(device)
        result = finetune_net(img_tensor)
        # print("result = ",result.argmax(1))


        id = result.argmax(1).item()

        file_list=[]
        for a,b,c in os.walk("dataset"):
            if len(b) != 0:
                file_list = b
                print("预测结果为：",file_list[id])

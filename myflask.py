from flask import Flask, render_template, request, jsonify
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=5).to(device)
model.load_state_dict(torch.load("output/resnet18_e_best.pth"))
model.eval()

# 图片预处理
transform_BZ = transforms.Normalize(
    mean=[0.4666638, 0.43481198, 0.38117275],
    std=[0.21779788, 0.2145783, 0.2144127]
)


def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = transform_BZ(img)
    img = torch.unsqueeze(img, 0)
    return img.to(device)


# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = file.read()
    img_tensor = preprocess_image(image)
    print(img_tensor)

    with torch.no_grad():
        result = model(img_tensor)
        print(result)

    class_index = result.argmax(1).item()
    class_names = ['猫', '牛', '狗', '马', '羊']
    class_name = class_names[class_index]
    print(class_name)

    return jsonify({'result': class_name})


if __name__ == '__main__':
    app.run(debug=True)
import torch
from torchvision import transforms
from keras.utils import load_img

SIZE = 224, 224
label = ['Not dishwasher-safe','Dishwasher-safe']
alexnet = torch.load('alexnet.pt')

def preprocess_img(img_pth):
    img = load_img(img_pth, target_size=(SIZE))
    X = transforms.ToTensor()(img)
    return X.unsqueeze_(0)

def predict_result(X):
    with torch.no_grad():
        alexnet.eval()
        outputs = alexnet(X)
        preds = outputs.argmax(1)
        return label[preds[0].item()]


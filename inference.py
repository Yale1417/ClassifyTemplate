import torch
from modeling.alexnet import AlexNet
from data.build import val_transform
from PIL import Image
from config.config import Caltech_class

cuda_ = torch.cuda.is_available()


if __name__ == '__main__':
    model = AlexNet(num_classes=257)
    model.load_state_dict(torch.load("output/weight_saved/Alexnet-9-1.5478705212275188-62.533333251953124.pt"))
    model.eval()
    img = Image.open(
        "/data/yang/openData/Caltech256/256_ObjectCategories/253.faces-easy-101/253_0129.jpg").convert(
        "RGB")
    img = val_transform(img)
    img = img.unsqueeze(0)
    if cuda_:
        model.cuda()
        img = img.cuda()
    output = model(img).squeeze()
    _, index = torch.max(output, 0)
    print("predict:", Caltech_class[index])


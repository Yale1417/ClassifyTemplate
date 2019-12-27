import torch
# from modeling.alexnet import AlexNet
# from modeling.resnet import resnet50
from modeling.resnet import wide_resnet101_2
from data import getData
from engine.valRun import val_run

cuda_ = torch.cuda.is_available()


def test():
    # model = AlexNet(num_classes=257)
    # model.load_state_dict(torch.load("output/weight_saved/Alexnet-9-1.5478705212275188-62.533333251953124.pt"))

    # model = resnet50(num_classes=257)
    # model.load_state_dict(torch.load("output/weight_saved/Resnet50-9-0.6717463796933492-83.76666666666667.pt"))

    model = wide_resnet101_2(num_classes=257)
    model.load_state_dict(torch.load("output/weight_saved/WideResnet101-9-0.8427028404871623-80.26666666666667.pt"))

    if cuda_:
        model.cuda()
    testData = getData("test")
    test_loader = torch.utils.data.DataLoader(
        testData,
        batch_size=16,
        shuffle=False,
        num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    _, test_acc = val_run(test_loader, model, criterion, device=cuda_)

    print(test_acc)  # AlexNet 64.04059933133415
                    # resnet50 84.40470179709448


if __name__ == '__main__':
    test()

import torch
import os
# from modeling.resnet import resnet50
# from modeling.alexnet import AlexNet
# from modeling.resnet import wide_resnet101_2
from modeling.efficientnet_pytorch import EfficientNet
from utils import Plog, summary
from data import getData
from config import train_config, val_config
from engine.trainRun import train_run
from engine.valRun import val_run

logger = Plog(os.path.join(train_config.logDir, "train_" + train_config.saveName))
cuda_ = torch.cuda.is_available()


def train():
    # model = AlexNet(num_classes=257)
    # model_dict = model.state_dict()
    # paras = torch.load("pre_weights/alexnet-owt-4df8aa71.pth")
    # paras = {k: v for k, v in paras.items() if k not in ["classifier.6.weight", "classifier.6.bias"]}
    # model_dict.update(paras)
    # model.load_state_dict(model_dict)

    # model = resnet50(num_classes=257)
    # model_dict = model.state_dict()
    # paras = torch.load("pre_weights/resnet50-19c8e357.pth")
    # paras = {k: v for k, v in paras.items() if k not in ["fc.weight", "fc.bias"]}
    # model_dict.update(paras)
    # model.load_state_dict(model_dict)

    # model = wide_resnet101_2(num_classes=257)
    # model_dict = model.state_dict()
    # paras = torch.load("pre_weights/wide_resnet101_2-32ee1156.pth")
    # paras = {k: v for k, v in paras.items() if k not in ["fc.weight", "fc.bias"]}
    # model_dict.update(paras)
    # model.load_state_dict(model_dict)

    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=257)
    # model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=257)
    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=257)
    # model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=257)


    if cuda_:
        model.cuda()
    #
    # summary(model, (3, 224, 224))

    # data
    trainData = getData('train')
    valData = getData('val')
    train_loader = torch.utils.data.DataLoader(
        trainData,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        valData,
        batch_size=val_config.batch_size,
        shuffle=False,
        num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()),
                                  'lr': 0.01}],
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=0.0005,
                                nesterov=False,
                                )

    epoch_iters = len(trainData) / train_config.batch_size
    max_iters = train_config.epoch * epoch_iters

    start_epoch = train_config.start_epoch

    best_acc = 0
    for epoch in range(start_epoch, train_config.epoch):
        train_run(train_loader, model, criterion, optimizer, epoch, epoch_iters, max_iters, logger, device=cuda_)
        test_loss, test_acc = val_run(val_loader, model, criterion, logger, device=cuda_)
        # append logger file
        best_acc = max(test_acc, best_acc)
        torch.save(model.state_dict(),
                   os.path.join(train_config.weightDir, f"{train_config.saveName}-{epoch}-{test_loss}-{test_acc}.pt"))

    logger.info(f'Best acc:{best_acc}')


if __name__ == '__main__':
    train()

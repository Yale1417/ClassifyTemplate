import time
from utils.averageMeter import AverageMeter
from utils.utils import accuracy, adjust_learning_rate, timeChange
import torch
from config.config import train_config


def train_run(trainloader, model, criterion, optimizer, epoch, epoch_iters, max_iters, logger=None, device=False):
    # Training
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()
    cur_iters = epoch * epoch_iters

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if device:
            inputs = inputs.cuda()
            targets = targets.cuda()
        # compute output
        outputs = model(inputs)
        # targets = torch.zeros(8, 257).scatter_(1, targets.reshape(8,1), 1)
        if device:
            targets = targets.type(torch.cuda.LongTensor)
        else:
            targets = targets.type(torch.LongTensor)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        lr = adjust_learning_rate(optimizer, base_lr=train_config.base_lr, max_iters=max_iters,
                                  cur_iters=batch_idx + cur_iters)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        eta = timeChange(batch_time.val * (max_iters - batch_idx - cur_iters))
        end = time.time()
        if logger:
            if batch_idx % 10 == 0:
                logger.info(
                    f"epoch:{epoch},iter:{batch_idx}, prec1:{top1.val},prec3:{top3.val},loss:{losses.val} ETA:{eta}")

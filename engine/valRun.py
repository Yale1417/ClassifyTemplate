from utils.averageMeter import AverageMeter
from utils.utils import accuracy


def val_run(val_loader, model, criterion, logger=None, device=False):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    if logger:
        logger.info("start to val:")
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if device:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
    if logger:
        logger.info(f"top1:{top1.avg} top3:{top3.avg}")

    return losses.avg, top1.avg
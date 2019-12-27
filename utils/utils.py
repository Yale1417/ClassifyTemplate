def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {:.2e}".format(total_num_params))


class Config(object):
    """
    https://github.com/dbolya/yolact
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        生成一个新的config对象
        """

        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        """
        更新自身的值。
        传入的参数可以是一个字典，也可以是一个config对象
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def timeChange(t):
    t = int(t)
    days, h, m = 0, 0, 0
    if t > 86400:
        days = t // 86400
        t = t % 86400
    if t > 3600:
        h = t // 3600
        t = t % 3600
    if t > 60:
        m = t // 60
        t = t % 60
    return f"{days}days{h}hours{m}minutes"


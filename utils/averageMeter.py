###使用背景：统计一些参数，比如平均时间消耗、平均损失、平均准确率的时候可以使用.

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.max = float("-inf")   # 负无穷
        self.min = float("inf")  # 正无穷

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True


    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
    def info(self):
        print(f"min:{self.min} max:{self.max} avg:{self.avg} count:{self.count}")

if __name__ == "__main__":
    a = AverageMeter()

    for i in range(100):
        a.update(i)

    print(a.sum)
    print(a.avg)
    print(a.count)
    print(a.val)

# 4950
# 49.5
# 100
# 99
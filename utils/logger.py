import pathlib
import logging
import uuid
from logging.handlers import TimedRotatingFileHandler
import time


class Plog(object):
    """docstring for Plog
    级别       何时使用
    DEBUG     详细信息，典型地调试问题时会感兴趣。
    INFO      证明事情按预期工作。
    WARNING   表明发生了一些意外，或者不久的将来会发生问题（如‘磁盘满了’）。软件还是在正常工作。
    ERROR     由于更严重的问题，软件已不能执行一些功能了。
    CRITICAL  严重错误，表明软件已不能继续运行了。

    """
    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] [%(levelname)08s] [%(lineno)03s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    formatter2 = logging.Formatter('%(message)s')

    def __init__(self, log_file, level=logging.DEBUG, stream=True, msgOnly=True):
        pdir = pathlib.Path(log_file).parent
        if not pdir.exists():
            pathlib.Path.mkdir(pdir, parents=True)  # 父文件夹不存在则自动创建。
        self.log_file = log_file+time.strftime("%Y-%m-%d")+".log"
        self.level = level
        self.stream = stream
        self.log_name = str(uuid.uuid1())  # 区分不同日志。

        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(self.level)

        # 日志文件
        handler = TimedRotatingFileHandler(self.log_file, when='D', encoding="utf-8")
        if msgOnly:
            handler.setFormatter(Plog.formatter2)
        else:
            handler.setFormatter(Plog.formatter)
        self.logger.addHandler(handler)

        # 终端流
        if self.stream:
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(Plog.formatter2)
            self.logger.addHandler(streamHandler)

        self.logger.debug(f"==========*****{time.strftime('%Y-%m-%d-%H-%M-%S')}:start to log*****==========")

    def __getattr__(self, item):
        return getattr(self.logger, item)

if __name__ == '__main__':
    log1 = Plog("train")
    log2 = Plog("test")
    log1.debug("ceshi")
    log2.info("sdf")
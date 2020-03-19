# coding=utf-8
import os

import logbook
from logbook import Logger, TimedRotatingFileHandler
from logbook.more import ColorizedStderrHandler


def log_type(record, handler):
    log = "[{date}] [{level}] [{filename}] [{func_name}] [{lineno}] {msg}".format(
        date=record.time,  # 日志时间
        level=record.level_name,  # 日志等级
        filename=os.path.split(record.filename)[-1],  # 文件名
        func_name=record.func_name,  # 函数名
        lineno=record.lineno,  # 行号
        msg=record.message  # 日志内容
    )
    return log


# 获取配置的日志级别
log_level = 9
# 日志存放路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR = os.path.join(root_dir, 'test_resources', 'run-log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
# 日志打印到屏幕
log_std = ColorizedStderrHandler(bubble=True, level=9)
log_std.formatter = log_type
# 日志打印到文件
log_file = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, '%s.log' % 'log'), date_format='%Y-%m-%d', bubble=True, encoding='utf-8',
    level=11)
log_file.formatter = log_type

# 脚本日志
logger = Logger("script_log")


def init_logger():
    logbook.set_datetime_format("local")
    logger.handlers = []
    logger.handlers.append(log_file)
    logger.handlers.append(log_std)


# 实例化，默认调用
init_logger()

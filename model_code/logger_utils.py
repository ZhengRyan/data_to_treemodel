# -*- coding: utf-8 -*-

import logging


# 日志输出
class Logger():
    # 日志级别关系映射
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self, level="info", name=None,
                 fmt="%(asctime)s - %(name)s[line:%(lineno)d] - %"
                     "(levelname)s: %(message)s"):
        logging.basicConfig(level=self.level_relations.get(level), format=fmt)
        self.logger = logging.getLogger(name)

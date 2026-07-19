import logging
import sys


def setup_logger(name: str = "aifitbot", level: str = "INFO") -> logging.Logger:
    """创建统一格式的日志记录器。

    Args:
        name: 日志记录器名称。
        level: 日志级别（DEBUG / INFO / WARNING / ERROR）。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# 全局默认 logger
logger = setup_logger()

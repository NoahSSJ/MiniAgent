"""全局日志模块(基于 loguru,极简配置)。

用法(整个项目统一):
    from core.logger import logger

    logger.info("启动成功")
    logger.error("出错了", exc_info=True)

运行模式(通过环境变量 APP_ENV 控制):
    - development(默认):    控制台可见 + 文件日志
    - production(正式运行):  控制台静默(打印不可见),仅写文件日志

其他环境变量:
    - LOG_LEVEL: 日志级别(开发默认 DEBUG,生产默认 INFO)
    - LOG_DIR:   日志目录(默认 logs/)
"""

import os
import sys

from loguru import logger

APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"

# 移除默认 stderr handler
logger.remove()

# 开发模式:控制台输出(带颜色);生产模式:控制台静默
if not IS_PRODUCTION:
    logger.add(
        sys.stdout,
        level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
    )

# 文件输出(UTF-8,自动轮转) — 两种模式都保留
logger.add(
    os.getenv("LOG_DIR", "logs") + "/app.log",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    rotation="5 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
)
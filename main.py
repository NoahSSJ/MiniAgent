"""MiniAgent 程序入口。"""

from core.logger import logger


def main() -> None:
    logger.info("MiniAgent 启动")
    # 业务逻辑...
    logger.debug("debug 级别日志(需要 LOG_LEVEL=DEBUG 才显示)")
    logger.warning("警告示例")
    logger.success("初始化完成")


if __name__ == "__main__":
    main()
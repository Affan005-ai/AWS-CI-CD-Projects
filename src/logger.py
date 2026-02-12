import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

# Always log to stdout (works well on Elastic Beanstalk)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# Best-effort file logging (don’t prevent app startup if it fails)
try:
    log_dir = os.environ.get("APP_LOG_DIR", "/tmp/app_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    fh = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
except Exception:
    logger.exception("File logging disabled; using stdout only")




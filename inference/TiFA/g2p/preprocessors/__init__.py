import logging
from .base import Preprocessor

logger = logging.getLogger(__name__)

# Explicitly import all built-in preprocessor modules so their @preprocessor
# decorators execute deterministically across all environments (including Android).
try:
    from . import simple
except Exception as exc:
    logger.warning(f"Failed to import simple preprocessors: {exc}")
